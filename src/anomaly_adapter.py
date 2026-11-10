"""9주차: LEAD 1.0(2016) 이상진단 6종 + TSB-AD 평가를 한 곳에 묶은 어댑터.

설계
- 입력: 1주차 17개 건물의 2016년 전력(8,784h) · 라벨은 4주차와 같은 LEAD 조인(src.finetune.load_anomaly_mask)
- 건물마다 단변량으로 따로 점수를 낸다(라벨이 건물별이고, Sub-LOF·Matrix Profile·IForest가 단변량이라서).
- 점수는 (시점 × 건물) DataFrame 으로 저장해 두고, 평가·분석은 CPU에서 점수 파일만 읽어서 한다.
- 저장소 3종(TSB-AD·CATCH·DADA)은 pip 가 아니라 클론 경로로 import (requirements_stage2.txt 참조).
"""
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPOS = {"TSB-AD": ROOT / "TSB-AD", "CATCH": ROOT / "CATCH", "DADA": ROOT / "DADA"}
COMMITS = {"TSB-AD": "303dffb", "CATCH": "3647c69", "DADA": "ea17238"}
METHODS = ["CATCH", "DADA", "Sub-LOF", "MatrixProfile", "IsolationForest", "LightGBM잔차"]

# CATCH: 저장소 기본값 그대로(seq_len 192, 3에폭). 서버 검증 범위가 기본값이라 건드리지 않음
CATCH_CFG = dict(seq_len=192, num_epochs=3, batch_size=128)
DADA_WIN = 100  # DADA 체크포인트의 seq_len (config.json)


def setup():
    """numpy 2 에서 제거된 별칭 복원(CATCH·TSB-AD) + 저장소 경로 등록. 저장소 코드는 고치지 않는다."""
    np.Inf, np.NaN = np.inf, np.nan
    for name in ("TSB-AD", "CATCH"):
        if str(REPOS[name]) not in sys.path:
            sys.path.insert(0, str(REPOS[name]))
    if str(REPOS["DADA"]) not in sys.path:  # 루트에 data_provider·exp 가 있어 뒤에 붙임(다른 패키지를 가리지 않게)
        sys.path.append(str(REPOS["DADA"]))
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        yield buf


def load_2016():
    """(X, y): 2016년 전력 (시점 × 건물, 결측은 ffill/bfill) · 이상 라벨 bool (시점 × 건물)."""
    setup()
    from src.finetune import load_anomaly_mask
    X = pd.read_csv(ROOT / "Data/processed/subset_17_buildings_Impulation.csv", index_col=0, parse_dates=True).loc["2016"]
    y = load_anomaly_mask(X.columns).reindex(X.index, fill_value=False)
    return X.ffill().bfill().fillna(0.0), y


def events(label):
    """bool 시리즈 → 이상 구간 표 (start, end, length). end 는 포함."""
    v = np.asarray(label, bool)
    edge = np.diff(np.r_[0, v.astype(int), 0])
    s, e = np.flatnonzero(edge == 1), np.flatnonzero(edge == -1) - 1
    return pd.DataFrame({"start": s, "end": e, "length": e - s + 1})


def _tiled(score_fn, x, win):
    """win 길이 비중첩 창으로만 점수를 내는 모델용: 앞에서 k*win, 뒤에서 k*win 두 번 돌려 꼬리까지 채움."""
    n = len(x)
    k = n // win * win
    head = score_fn(x[:k])
    s = np.empty((n,) + head.shape[1:])  # 1차원(점수) 또는 2차원(시점 × 채널) 모두 가능
    s[:k] = head
    if k < n:
        s[k:] = score_fn(x[n - k:])[-(n - k):]
    return s


# ── 고전 3종 (CPU) ───────────────────────────────────────────────────────────
def classic_scores(x):
    """x: 1차원 전력. TSB-AD 래퍼로 Sub-LOF · Matrix Profile · IsolationForest(서브시퀀스) 점수와 주기 윈도우."""
    setup()
    from sklearn.preprocessing import MinMaxScaler
    from TSB_AD.model_wrapper import run_MatrixProfile, run_Sub_IForest, run_Sub_LOF
    from TSB_AD.utils.slidingWindows import find_length_rank
    z = MinMaxScaler().fit_transform(np.asarray(x, float).reshape(-1, 1))
    scores, secs = {}, {}
    for name, fn in [("Sub-LOF", run_Sub_LOF), ("MatrixProfile", run_MatrixProfile), ("IsolationForest", run_Sub_IForest)]:
        t0 = time.perf_counter()
        scores[name] = np.nan_to_num(np.asarray(fn(z), float))
        secs[name] = time.perf_counter() - t0
    return scores, int(find_length_rank(z, rank=1)), secs


# ── CATCH (GPU) ──────────────────────────────────────────────────────────────
def catch_scores(df, seed=2021, **cfg):
    """df: DatetimeIndex 의 DataFrame(열 = 건물, 2개 이상). 같은 데이터로 학습·채점(비지도, 라벨 미사용).
    CATCH 는 채널 간 상관을 학습하는 모델이라 채널이 1개면 ch_discover_loss 가 0으로 나눈다(n_vars*(n_vars-1))
    → 17건물을 한 모델에 넣고, 점수는 채널별로 꺼내 건물별 라벨로 평가한다. → (점수 DataFrame n × 채널, 학습시간 s)"""
    setup()
    import torch
    import torch.nn as nn
    from ts_benchmark.baselines.catch.CATCH import CATCH
    from ts_benchmark.baselines.catch.utils.fre_rec_loss import frequency_criterion
    from ts_benchmark.baselines.utils import anomaly_detection_data_provider
    assert df.shape[1] >= 2, "CATCH 는 채널 2개 이상 필요"
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = CATCH(**{**CATCH_CFG, **cfg})
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with quiet():
        m.detect_fit(df, df)
    if cuda:
        torch.cuda.synchronize()
    fit_s = time.perf_counter() - t0

    # CATCH.detect_score 와 같은 계산에서 채널 평균(dim=-1)만 뺀 것: 점수 = 시간영역 MSE + score_lambda × 주파수 오차
    cfg_, C = m.config, df.shape[1]
    m.model.load_state_dict(m.early_stopping.check_point)
    m.model.to(m.device).eval()
    time_err, freq_err = nn.MSELoss(reduction="none"), frequency_criterion(cfg_)

    def channel_scores(d):
        d = pd.DataFrame(m.scaler.transform(d.values), columns=d.columns, index=d.index)
        loader = anomaly_detection_data_provider(d, batch_size=cfg_.batch_size, win_size=cfg_.seq_len, step=1, mode="thre")
        out = []
        with torch.no_grad():
            for bx, _ in loader:
                bx = bx.float().to(m.device)
                rec, _, _ = m.model(bx)
                out.append((time_err(bx, rec) + cfg_.score_lambda * freq_err(bx, rec)).cpu().numpy())  # B × win × C
        return np.concatenate(out).reshape(-1, C)

    with quiet():
        score = _tiled(channel_scores, df, cfg_.seq_len)
    return pd.DataFrame(score, index=df.index, columns=df.columns), fit_s


# ── DADA (GPU, 학습 없음) ────────────────────────────────────────────────────
def load_dada(device="cuda"):
    """requirements_stage2.txt '우회 2': AutoModel 대신 직접 로드."""
    setup()
    from DADA.configuration_DADA import DADAConfig
    from DADA.modeling_DADA import DADA
    DADA.config_class = DADAConfig
    return DADA.from_pretrained(str(REPOS["DADA"] / "DADA")).to(device).eval()


def dada_scores(model, x, norm=0, batch=32, seed=2021):
    """x: 1차원 전력. 시리즈 전체 표준화 후 100h 비중첩 창으로 zero-shot 점수(마스크 복사본 분산). 라벨·학습 없음."""
    import torch
    np.random.seed(seed)  # 마스크가 np.random 을 씀
    device = next(model.parameters()).device
    z = (x - x.mean()) / (x.std() + 1e-8)

    def run(v):
        w = torch.tensor(v.reshape(-1, DADA_WIN, 1), dtype=torch.float32)
        with torch.no_grad():
            return np.concatenate([model.infer(b.to(device), norm=norm).cpu().numpy() for b in w.split(batch)]).reshape(-1)

    return _tiled(run, z, DADA_WIN)


# ── LightGBM 잔차 (4주차 산출물 재사용) ───────────────────────────────────────
def lgbm_residual_scores(index, columns):
    """4주차 정상학습 모델의 |실제-예측|. 라벨로 정상 구간만 골라 학습한 지도 방식이라 비지도 5종과 조건이 다름(비교표 각주).
    건물별 순위만 쓰는 지표(VUS-PR)에서는 건물별 스케일 정규화가 필요 없어 원 잔차를 쓴다."""
    r = pd.read_csv(ROOT / "results/wk4/residuals_full_year.csv", usecols=["building", "timestamp", "resid_clean"],
                    parse_dates=["timestamp"])
    return r.pivot(index="timestamp", columns="building", values="resid_clean").reindex(index=index, columns=columns).fillna(0.0)


# ── 평가 (TSB-AD 와 같은 함수, get_metrics 에서 필요한 3개만) ─────────────────
def evaluate(score, label, win, pred=None):
    """score: 높을수록 이상. pred=None 이면 PA-F1 은 TSB-AD 관례대로 최적 임계값(oracle). pred 를 주면 그 이진 예측 그대로."""
    setup()
    from TSB_AD.evaluation.basic_metrics import basic_metricor, generate_curve
    score, label = np.asarray(score, float), np.asarray(label).astype(int)
    g = basic_metricor()
    *_, vus_roc, vus_pr = generate_curve(label, score, win, "opt", 250)
    return {"PA-F1": g.metric_PointF1PA(label, score, preds=pred), "F1": g.metric_PointF1(label, score, preds=pred),
            "AUC-PR": g.metric_PR(label, score), "VUS-PR": vus_pr}


def random_predictor(n, p, rng):
    """각 시점을 확률 p 로 이상이라 찍는 예측기. 같은 난수에서 이진 예측(pred)과 연속 점수(score)를 함께 만든다
    (score > 1-p 인 시점 = pred 1) → PA-F1 은 pred 로, VUS-PR 은 score 로 같은 예측기를 평가."""
    u = rng.random(n)
    return (u < p).astype(int), 1.0 - u


def evaluable(y):
    """이상이 1개 이상이고 전부는 아닌 건물만 (이상 0건 건물은 PR 계열 지표가 정의되지 않음)."""
    return [c for c in y.columns if 0 < y[c].sum() < len(y)]


def event_recall_by_type(score, label, bins=(1, 24)):
    """상위 q% 컷(q = 그 건물의 실제 이상률)에서 이상 구간별 탐지 여부. 구간 유형은 길이로: 점(≤bins[0]) / 짧은 / 긴.
    → DataFrame(type, length, hit). 임계값 선택에 쓰는 라벨 정보는 '이상 개수' 하나뿐이라 방식 간 비교가 공정함."""
    s, v = np.asarray(score, float), np.asarray(label, bool)
    pred = s >= np.sort(s)[::-1][max(int(v.sum()) - 1, 0)]
    ev = events(v)
    ev["hit"] = [bool(pred[a:b + 1].any()) for a, b in zip(ev["start"], ev["end"])]
    ev["type"] = pd.cut(ev["length"], [0, bins[0], bins[1], np.inf], labels=["점", "짧은 구간", "긴 구간"])
    return ev
