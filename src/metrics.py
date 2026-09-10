import numpy as np
import pandas as pd


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.nanmean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.nanmean((y_true - y_pred) ** 2))


def quantile_loss(y_true, y_pred, q=0.5):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    return np.nanmean(np.maximum(q * diff, (q - 1) * diff))


def wql(y_true, y_pred_quantiles: dict):
    y_true = np.asarray(y_true, dtype=float)
    total_ql = 0.0
    for q, y_pred in y_pred_quantiles.items():
        y_pred = np.asarray(y_pred, dtype=float)
        diff = y_true - y_pred
        total_ql += np.nansum(np.maximum(q * diff, (q - 1) * diff))
    denom = np.nansum(np.abs(y_true))
    return total_ql / denom if denom else np.nan

#학습 구간만으로 계산한 Seasonal Naive MAE
def _naive_train_mae(train_series: pd.Series, season=168):
    s = train_series.dropna()
    if len(s) <= season:
        return np.nan
    return np.nanmean(np.abs(s.values[season:] - s.values[:-season]))

#MASE의 분모 계산
def compute_naive_denominators(data: pd.DataFrame, train_end: int, season=168):
    train_ref = data.iloc[:train_end]
    return {bid: _naive_train_mae(train_ref[bid], season) for bid in data.columns}

#집계 점수 반환
def score_backtest(results_df: pd.DataFrame, naive_denominators: dict):
    rows = []
    for bid, g in results_df.groupby("building"):
        b_mae = mae(g["y_true"], g["y_pred"])
        b_rmse = rmse(g["y_true"], g["y_pred"])
        b_wql = wql(g["y_true"], {0.5: g["y_pred"]})
        denom = naive_denominators.get(bid, np.nan)
        b_mase = b_mae / denom if denom else np.nan
        rows.append({
            "building": bid, "MAE": b_mae, "RMSE": b_rmse,
            "MASE": b_mase, "wQL": b_wql,
        })
    per_building = pd.DataFrame(rows)

    overall = {
        "MAE_단순평균": per_building["MAE"].mean(),
        "RMSE_단순평균": per_building["RMSE"].mean(),
        "MASE_평균": per_building["MASE"].mean(),
        "wQL_평균": per_building["wQL"].mean(),
    }
    return per_building, overall
