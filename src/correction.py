"""11주차 (1)(2): 이벤트 구간 식별 · 오차 보정.

- 기준 예측 = 10주차 기본값 모델(Chronos-2 + 공변량)의 윈도우별 저장 예측. 후보 예측은 조회만 하고 다시 추론하지 않는다.
- 누수 방지: 이벤트 임계값(기온 분위수, 오차 분위수)과 보정 파라미터는 모두 앞 26창(FIT_W)의 정보만으로 추정하고,
  평가는 뒤 26창(EVAL_W)에서만 한다. 보정에 쓰는 '상황' 구분은 예측 시점에 이미 아는 값(기온·달력)만 쓴다.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from pandas.tseries.holiday import USFederalHolidayCalendar

from src import routing as rt
from src.metrics import score_backtest

MODEL = "Chronos-2 + 공변량"
CATS = ["공휴일", "극한고온", "극한저온", "평시"]  # 보정 때 쓰는 사전(ex-ante) 상황 구분
MIN_N = 24  # 오프셋을 추정하려면 (건물, 상황) 칸에 학습 관측이 이만큼은 있어야 함


def temp_matrix(data, weather, site):
    """(시간, 건물) 외기온 행렬 — 건물이 속한 사이트의 기온."""
    w = weather.set_index(["site_id", "timestamp"])["airTemperature"]
    return np.column_stack([w.loc[site[b]].reindex(data.index).to_numpy() for b in data.columns])


def build_frame(pool, data, weather, site, state, pred=None):
    """칸 단위가 아니라 '시간 단위' 긴 표: window, b(건물 위치), step, ts, y, pred, resid, temp, hol, hour, dow, recent."""
    W, B, h = pool.Y.shape
    pred = pool.P[MODEL] if pred is None else pred
    T = temp_matrix(data, weather, site)
    ts = np.stack([pool.origin_ts[w] + pd.to_timedelta(np.arange(h), unit="h") for w in range(W)])  # (W, h)
    pos = data.index.get_indexer(ts.reshape(-1)).reshape(W, h)
    hol_days = USFederalHolidayCalendar().holidays(data.index.min(), data.index.max())
    hol = pd.DatetimeIndex(ts.reshape(-1)).normalize().isin(hol_days).reshape(W, h)
    rec = state.pivot(index="window", columns="building", values="recent_naive_mase")[list(data.columns)].to_numpy()  # (W, B)
    w_i, b_i, s_i = np.meshgrid(np.arange(W), np.arange(B), np.arange(h), indexing="ij")
    df = pd.DataFrame({
        "window": w_i.ravel(), "b": b_i.ravel(), "step": s_i.ravel(),
        "ts": ts[w_i, s_i].ravel(), "y": pool.Y.ravel(), "pred": pred.ravel(),
        "temp": T[pos[w_i, s_i], b_i].ravel(), "hol": hol[w_i, s_i].ravel(), "recent": rec[w_i, b_i].ravel(),
    })
    df["building"] = np.array(pool.buildings)[df["b"]]
    df["site"] = df["building"].map(site)
    df["hour"], df["dow"] = df["ts"].dt.hour, df["ts"].dt.dayofweek
    df["denom"] = pool.denom[df["b"]]
    df["resid"] = df["y"] - df["pred"]
    return df


def add_categories(df, data, weather, fit_end_ts):
    """사이트별 기온 분위수(5%·25%·75%·95%)를 fit_end_ts 이전 기온만으로 정하고, 시간마다 상황을 붙인다."""
    w = weather[weather["timestamp"] < fit_end_ts]
    q = w.groupby("site_id")["airTemperature"].quantile([.05, .25, .75, .95]).unstack()
    q.columns = ["q05", "q25", "q75", "q95"]
    df = df.drop(columns=[c for c in q.columns if c in df.columns]).join(q, on="site")
    df["cat"] = np.select([df["hol"], df["temp"] > df["q95"], df["temp"] < df["q05"]], ["공휴일", "극한고온", "극한저온"], "평시")
    df["tbin"] = np.select([df["temp"] < df["q05"], df["temp"] < df["q25"], df["temp"] <= df["q75"], df["temp"] <= df["q95"]], [0, 1, 2, 3], 4)
    return df, q


def label_events(df, fit_mask, pct=95):
    """11-1: 정규화 오차(|오차| / Naive 분모)가 앞 26창의 pct 분위수를 넘는 시간을 '큰 오차'로 보고 원인을 붙인다.
    원인 우선순위: 공휴일 > 극한기온 > 최근 급증(직전 1주 Naive MASE > 3) > 원인 불명."""
    df = df.copy()
    df["err"] = df["resid"].abs() / df["denom"]
    thr = df.loc[fit_mask, "err"].quantile(pct / 100)
    df["big"] = df["err"] > thr
    df["cause"] = np.select([df["cat"] == "공휴일", df["cat"].isin(["극한고온", "극한저온"]), df["recent"] > 3],
                            ["공휴일", "극한기온", "최근 급증"], "원인 불명")
    return df, thr


# ── 11-2. 보정 방식 3가지 ──────────────────────────────────────────────────────
def fit_offset(fit, k=0.0):
    """상황별 오프셋: (건물, 상황) 평균 잔차에 n/(n+k) 수축을 곱함. 학습 관측이 MIN_N 미만이면 0. (k = 1e9 이면 보정 없음)"""
    g = fit.groupby(["building", "cat"])["resid"].agg(["mean", "size"])
    return {key: (r["mean"] * r["size"] / (r["size"] + k) if r["size"] >= MIN_N else 0.0) for key, r in g.iterrows()}


def apply_offset(df, off, events_only=True):
    o = np.array([off.get((b, c), 0.0) for b, c in zip(df["building"], df["cat"])])
    if events_only:
        o = np.where(df["cat"] == "평시", 0.0, o)
    return df["pred"].to_numpy() + o


def fit_scale(fit, k=50, lo=0.5, hi=2.0):
    """조건부 스케일링: (건물, 기온 구간) 배율 = Σy/Σpred 를 1 쪽으로 n/(n+k) 수축, [lo, hi]로 자름."""
    g = fit.groupby(["building", "tbin"]).agg(y=("y", "sum"), p=("pred", "sum"), n=("y", "size"))
    r = np.clip(g["y"] / g["p"], lo, hi)
    return (1 + (r - 1) * g["n"] / (g["n"] + k)).to_dict()


def apply_scale(df, sc):
    return df["pred"].to_numpy() * np.array([sc.get((b, t), 1.0) for b, t in zip(df["building"], df["tbin"])])


RES_FEATS = ["b", "hour", "dow", "temp", "hol", "step", "pred_s"]
RES_PARAMS = dict(objective="regression", num_leaves=15, learning_rate=0.03, n_estimators=150, min_child_samples=100, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, verbosity=-1, random_state=0)


def _res_X(df):
    X = df[["b", "hour", "dow", "temp", "hol", "step"]].copy()
    X["hol"] = X["hol"].astype(int)
    X["pred_s"] = df["pred"] / df["denom"]
    X["b"] = X["b"].astype("category")
    return X[RES_FEATS]


def fit_resmodel(fit):
    """잔차 모델: 정규화 잔차(잔차/분모)를 예측하는 작은 LightGBM."""
    m = lgb.LGBMRegressor(**RES_PARAMS)
    X = _res_X(fit)
    X["b"] = pd.Categorical(fit["b"], categories=range(17))
    m.fit(X, fit["resid"] / fit["denom"], categorical_feature=["b"])
    return m


def apply_resmodel(df, m):
    X = _res_X(df)
    X["b"] = pd.Categorical(df["b"], categories=range(17))
    return df["pred"].to_numpy() + m.predict(X) * df["denom"].to_numpy()


def score_df(df, col, pool, windows):
    """하네스(score_backtest)로 채점: df 의 예측 열 col 을 (window, building, timestamp, y_true, y_pred) 형식으로 바꿔 넣는다."""
    d = df[df["window"].isin(windows)]
    t = pd.DataFrame({"window": d["window"], "building": d["building"], "timestamp": d["ts"], "y_true": d["y"], "y_pred": d[col]})
    pb, overall = score_backtest(t, dict(zip(pool.buildings, pool.denom)))
    return overall["MASE_평균"], pb.set_index("building")["MASE"]


K_GRID = [0, 50, 200, 1000, 1e9]  # 1e9 = 사실상 보정 없음


def select_k(fit, kind, split=13):
    """보정 강도 k 를 앞 26창 '안에서만' 고른다: 앞 split창으로 추정 → 나머지 앞 창에서 검증(정규화 MAE 최소).
    오프셋은 이벤트 시간, 스케일링은 전체 시간으로 검증. 뒤 26창(평가)은 쓰지 않는다."""
    a, b = fit[fit["window"] < split], fit[fit["window"] >= split]
    if kind == "offset":
        b = b[b["cat"] != "평시"]
    rows = []
    for k in K_GRID:
        p = apply_offset(b, fit_offset(a, k)) if kind == "offset" else apply_scale(b, fit_scale(a, k))
        rows.append({"k": k, "검증 정규화 MAE": float((np.abs(b["y"] - p) / b["denom"]).mean())})
    t = pd.DataFrame(rows)
    return float(t.loc[t["검증 정규화 MAE"].idxmin(), "k"]), t
