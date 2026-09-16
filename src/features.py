import numpy as np
import pandas as pd


def build_features(
    data: pd.DataFrame,
    horizon: int,
    lags=(24, 48, 168, 336),
    rolling_windows=(24, 168),
    weather: pd.DataFrame = None,
    building_site: dict = None,
    static: pd.DataFrame = None,
) -> pd.DataFrame:

    valid_lags = [l for l in lags if l >= horizon]

    frames = []
    for bid in data.columns:
        s = data[bid]
        shifted = s.shift(horizon)  # rolling은 반드시 horizon만큼 밀어낸 뒤 집계

        df = pd.DataFrame({"building": bid, "timestamp": s.index, "y": s.values})
        for lag in valid_lags:
            df[f"lag_{lag}"] = s.shift(lag).values
        for w in rolling_windows:
            roll = shifted.rolling(w)
            df[f"rolling_mean_{w}"] = roll.mean().values
            df[f"rolling_std_{w}"] = roll.std().values
            df[f"rolling_min_{w}"] = roll.min().values
            df[f"rolling_max_{w}"] = roll.max().values
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)

    result["hour"] = result["timestamp"].dt.hour
    result["dayofweek"] = result["timestamp"].dt.dayofweek
    result["month"] = result["timestamp"].dt.month
    result["is_weekend"] = (result["dayofweek"] >= 5).astype(int)
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["dayofweek_sin"] = np.sin(2 * np.pi * result["dayofweek"] / 7)
    result["dayofweek_cos"] = np.cos(2 * np.pi * result["dayofweek"] / 7)

    if weather is not None and building_site is not None:
        result["site_id"] = result["building"].map(building_site)
        result = result.merge(
            weather[["site_id", "timestamp", "airTemperature", "dewTemperature"]],
            on=["site_id", "timestamp"], how="left",
        ).drop(columns="site_id")

    if static is not None:
        result = result.merge(static, left_on="building", right_index=True, how="left")

    return result