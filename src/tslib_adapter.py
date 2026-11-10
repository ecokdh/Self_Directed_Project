import copy
import importlib
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
# git clone https://github.com/thuml/Time-Series-Library.git && git checkout 4e938a1 (requirements_stage2.txt 핀)
TSLIB_DIR = ROOT / "Time-Series-Library"
MODELS = ["DLinear", "PatchTST", "iTransformer", "TimeXer"]

# 구조 하이퍼파라미터 = TSLib scripts/long_term_forecast/ECL_script/*.sh, 나머지는 run.py 기본값.
# ECL(고객 321명 시간별 전력, 채널 = 고객)은 건물 17개를 채널로 둔 우리 데이터와 같은 구조 → 튜닝 없이 그대로 사용
# TimeXer 는 ECL 스크립트 4개 중 pred_len=192 설정 (168h에 가장 가까움)
BASE_CFG = dict(task_name="long_term_forecast", label_len=48, d_model=512, n_heads=8, e_layers=2, d_layers=1,
                d_ff=2048, moving_avg=25, factor=1, dropout=0.1, embed="timeF", freq="h", activation="gelu",
                use_norm=1, patch_len=16, output_attention=False, num_class=1)
ARCH = {
    "DLinear": {},
    "PatchTST": dict(e_layers=2, factor=3),
    "iTransformer": dict(e_layers=3, d_ff=512, factor=3),
    "TimeXer": dict(e_layers=3, factor=3),
}
LR = {"iTransformer": 5e-4}  # ECL 스크립트가 지정한 경우만. 그 외 run.py 기본 1e-4

# 학습 설정: 교재 8-2 (에폭 상한 10 · patience 3) + run.py 기본값 (Adam · MSE · lradj type1 = 매 에폭 lr 절반).
# batch_size 는 ECL 스크립트마다 다르지만(4~32, 321채널 메모리 사정) 동일 조건 비교를 위해 run.py 기본 32로 통일.
# 1 배치 = 시점 32개 (S/MS 모드에서는 시점 32개 × 건물 17개 → M 모드와 스텝 수·계산량이 같음)
TRAIN = dict(train_epochs=10, patience=3, batch_size=32, val_ratio=0.1)


def time_marks(index):
    # TSLib utils/timefeatures.py time_features(freq='h') 와 같은 4개: HourOfDay·DayOfWeek·DayOfMonth·DayOfYear
    return np.stack([index.hour / 23.0 - 0.5, index.dayofweek / 6.0 - 0.5,
                     (index.day - 1) / 30.0 - 0.5, (index.dayofyear - 1) / 365.0 - 0.5], axis=1).astype(np.float32)


def temperature_frame(data, weather, building_site):
    # 건물별 사이트 외기온 (시점 × 건물). bench_models.load_weather 결과를 와이드로
    wide = weather.pivot(index="timestamp", columns="site_id", values="airTemperature")
    return pd.DataFrame({b: wide[building_site[b]] for b in data.columns}).reindex(data.index)


def to_panel(df, mode, temp=None):
    # (시점 T, 그룹 G, 채널 C). 모델 입력 1개 = 한 그룹의 [seq_len, C]
    #   M : 건물 17개를 채널로 (G=1, C=17)      — TSLib ECL 과 같은 방식
    #   S : 건물마다 따로, 전력만 (G=17, C=1)
    #   MS: 건물마다 따로, [외기온, 전력] (G=17, C=2) — TSLib 'MS' 규약대로 마지막 채널이 타깃
    v = df.to_numpy(np.float32)
    if mode == "M":
        return v[:, None, :]
    if mode == "S":
        return v[:, :, None]
    if mode == "MS":
        return np.stack([temp.loc[df.index, df.columns].to_numpy(np.float32), v], axis=-1)
    raise ValueError(f"unknown mode: {mode}")


def build_model(name, seq_len, horizon, n_channels, features):
    if str(TSLIB_DIR) not in sys.path:
        sys.path.insert(0, str(TSLIB_DIR))
    cfg = SimpleNamespace(**{**BASE_CFG, **ARCH[name]}, seq_len=seq_len, pred_len=horizon, features=features,
                          enc_in=n_channels, dec_in=n_channels, c_out=n_channels)
    return importlib.import_module(f"models.{name}").Model(cfg)


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def make_tslib_fns(model_name, seq_len, horizon, mode="M", temp=None, refit_hours=168, seed=2021, device="cuda"):
    # rolling_backtest 용 (model_fn, predict_fn) + 학습 기록 log.
    # refit_hours: 직전 학습 이후 origin 이 이만큼 지나면 재학습 (168 = 매 창). 재학습하지 않는 창은
    # 직전 모델로 그 창의 최근 seq_len 시간을 입력해 예측 → 하네스 코드는 그대로, train_mode 인자만으로 8-3b 가능
    features = {"M": "M", "S": "S", "MS": "MS"}[mode]
    f_dim = 0 if mode == "M" else -1  # 손실·출력 채널. M 은 전 채널, S/MS 는 마지막(전력)만
    cfg = TRAIN
    log = {"fits": [], "predict_s": 0.0}
    state = {}

    def windows(t, idx, length):
        # t: (T, G, C) → idx 시점에서 시작하는 length 길이 창, (len(idx)·G, length, C)
        w = t[idx[:, None] + torch.arange(length, device=t.device)]
        return w.permute(0, 2, 1, 3).reshape(-1, length, t.shape[-1])

    def fit(train_df):
        raw = to_panel(train_df, mode, temp)
        T, G, C = raw.shape
        n_val = max(int(T * cfg["val_ratio"]), horizon)  # TSLib 처럼 시간순 끝부분을 검증용으로
        mean = np.nanmean(raw[:T - n_val], axis=0)  # 정규화 통계는 학습 구간만 (검증·평가 구간 누설 없음)
        std = np.nanstd(raw[:T - n_val], axis=0) + 1e-6
        observed = ~np.isnan(raw)
        filled = pd.DataFrame(((raw - mean) / std).reshape(T, -1)).ffill().bfill().fillna(0.0).to_numpy()

        x = torch.tensor(filled.reshape(T, G, C), dtype=torch.float32, device=device)
        obs = torch.tensor(observed, dtype=torch.float32, device=device)
        mk = torch.tensor(time_marks(train_df.index), device=device)[:, None, :].expand(T, G, 4)
        starts = torch.arange(T - seq_len - horizon + 1, device=device)
        train_idx = starts[starts + seq_len + horizon <= T - n_val]  # 타깃이 학습 구간 안
        val_idx = starts[starts + seq_len >= T - n_val]              # 타깃이 검증 구간 안 (입력은 학습 구간과 겹쳐도 됨)

        def batch(idx):
            L = seq_len + horizon
            xb, mb, ob = windows(x, idx, L), windows(mk, idx, L), windows(obs, idx, L)
            return xb[:, :seq_len], mb[:, :seq_len], xb[:, seq_len:, f_dim:], ob[:, seq_len:, f_dim:]

        def loss_fn(model, idx):
            xb, mb, yb, ob = batch(idx)
            out = model(xb, mb, None, None)[:, :, f_dim:]
            return (((out - yb) ** 2) * ob).sum() / ob.sum().clamp(min=1)  # 결측 타깃은 손실에서 제외

        _seed_all(seed)
        model = build_model(model_name, seq_len, horizon, C, features).to(device)
        lr = LR.get(model_name, 1e-4)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        _sync(device)
        t0 = time.perf_counter()

        best, best_state, best_epoch, bad, epoch = np.inf, None, 0, 0, 0
        for epoch in range(1, cfg["train_epochs"] + 1):
            model.train()
            perm = train_idx[torch.randperm(len(train_idx), generator=gen).to(device)]
            for idx in perm.split(cfg["batch_size"]):
                opt.zero_grad()
                loss_fn(model, idx).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val = float(np.mean([loss_fn(model, idx).item() for idx in val_idx.split(cfg["batch_size"])]))
            if val < best:
                best, best_epoch, bad = val, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    break
            for g in opt.param_groups:
                g["lr"] = lr * 0.5 ** epoch  # lradj type1

        _sync(device)
        model.load_state_dict(best_state)
        model.eval()
        log["fits"].append({
            "학습 끝 시점": train_df.index[-1], "학습 행 수": T, "학습 창 수": len(train_idx) * G,
            "실행 에폭": epoch, "best 에폭": best_epoch, "검증 MSE(정규화)": best,
            "학습 시간(s)": time.perf_counter() - t0,
            "최대 VRAM(GB)": torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else np.nan,
            "파라미터 수": sum(p.numel() for p in model.parameters()), "batch_size": cfg["batch_size"], "lr": lr,
        })
        return {"model": model, "mean": mean, "std": std, "fit_end": train_df.index[-1]}

    def model_fn(train_df):
        if "model" not in state or train_df.index[-1] - state["fit_end"] >= pd.Timedelta(hours=refit_hours):
            state.update(fit(train_df))
        return {**state, "context_df": train_df.iloc[-seq_len:]}

    def predict_fn(st, h):
        assert h == horizon, f"모델은 horizon={horizon}으로 학습됨 (요청 {h})"
        ctx = st["context_df"]
        _sync(device)
        t0 = time.perf_counter()
        raw = (to_panel(ctx, mode, temp) - st["mean"]) / st["std"]
        filled = pd.DataFrame(raw.reshape(len(ctx), -1)).ffill().bfill().fillna(0.0).to_numpy().reshape(raw.shape)
        x = torch.tensor(filled, dtype=torch.float32, device=device).permute(1, 0, 2)  # (G, seq_len, C)
        mk = torch.tensor(time_marks(ctx.index), device=device).expand(x.shape[0], -1, -1)
        with torch.no_grad():
            out = st["model"](x, mk, None, None)[:, :, f_dim:].cpu().numpy()  # TimeXer(S/MS)는 타깃 1채널만 출력
        _sync(device)
        log["predict_s"] += time.perf_counter() - t0
        out = out * st["std"][:, None, f_dim:] + st["mean"][:, None, f_dim:]
        pred = out[0] if mode == "M" else out[:, :, -1].T  # → (horizon, 건물 17)
        future_index = pd.date_range(ctx.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
        return pd.DataFrame(pred, index=future_index, columns=ctx.columns)

    return model_fn, predict_fn, log
