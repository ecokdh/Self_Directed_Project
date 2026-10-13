from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from src.features import build_features

ROOT = Path(__file__).resolve().parents[1]
LGB_PARAMS = dict(objective="regression", num_leaves=127, learning_rate=0.08,
                  n_estimators=300, min_child_samples=10, verbosity=-1)
LGB_PARAMS_LOCAL = dict(objective="regression", num_leaves=31, learning_rate=0.05,
                        n_estimators=100, min_child_samples=20, verbosity=-1)
TTM_PATH = "ibm-granite/granite-timeseries-ttm-r2"
HOURLY_TOKEN = 7  # tsfm_public DEFAULT_FREQUENCY_MAPPING["h"]


def load_data():
    return pd.read_csv(ROOT / "Data/processed/subset_17_buildings_Impulation.csv", index_col=0, parse_dates=True)


def load_weather(data):
    meta = pd.read_csv(ROOT / "Data/raw/bdg2/metadata.csv")
    raw = pd.read_csv(ROOT / "Data/raw/bdg2/weather.csv", parse_dates=["timestamp"])
    hours = pd.date_range(data.index.min(), data.index.max(), freq="h")
    weather = (raw.set_index("timestamp").groupby("site_id")[["airTemperature"]]
               .apply(lambda g: g.reindex(hours).ffill().bfill())
               .rename_axis(index=["site_id", "timestamp"]).reset_index())
    return weather, meta.set_index("building_id")["site_id"].to_dict()


def _future_index(train_df, horizon):
    return pd.date_range(train_df.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")


def make_naive_fns(season=168):
    def model_fn(train_df):
        return train_df.iloc[-season:]

    def predict_fn(state, horizon):
        return state.iloc[:horizon].reset_index(drop=True)

    return model_fn, predict_fn


def make_global_fns(feat_horizon, lgb_params=LGB_PARAMS):
    def model_fn(train_df):
        feat = build_features(train_df, feat_horizon).dropna(subset=["y"])
        X = feat.drop(columns=["timestamp", "y"]).copy()
        X["building"] = X["building"].astype("category")
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X, feat["y"], categorical_feature=["building"])
        return {"model": model, "train_df": train_df, "columns": X.columns.tolist()}

    def predict_fn(state, horizon):
        train_df = state["train_df"]
        future_index = _future_index(train_df, horizon)
        extended = pd.concat([train_df, pd.DataFrame(np.nan, index=future_index, columns=train_df.columns)])
        feat = build_features(extended, feat_horizon)
        feat = feat[feat["timestamp"].isin(future_index)].copy()
        X = feat[state["columns"]].copy()
        X["building"] = X["building"].astype("category")
        feat["pred"] = state["model"].predict(X)
        return feat.pivot(index="timestamp", columns="building", values="pred").reindex(columns=train_df.columns)

    return model_fn, predict_fn


def make_local_fns(feat_horizon, lgb_params=LGB_PARAMS_LOCAL):
    def model_fn(train_df):
        models = {}
        for bid in train_df.columns:
            feat = build_features(train_df[[bid]], feat_horizon).dropna(subset=["y"])
            X = feat.drop(columns=["timestamp", "y", "building"])
            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(X, feat["y"])
            models[bid] = {"model": model, "columns": X.columns.tolist()}
        return {"models": models, "train_df": train_df}

    def predict_fn(state, horizon):
        train_df = state["train_df"]
        future_index = _future_index(train_df, horizon)
        preds = {}
        for bid, m in state["models"].items():
            extended = pd.concat([train_df[[bid]], pd.DataFrame(np.nan, index=future_index, columns=[bid])])
            feat = build_features(extended, feat_horizon)
            feat = feat[feat["timestamp"].isin(future_index)]
            preds[bid] = m["model"].predict(feat[m["columns"]])
        return pd.DataFrame(preds, index=future_index)

    return model_fn, predict_fn


def make_recursive_fns(lgb_params=LGB_PARAMS):
    def model_fn(train_df):
        feat = build_features(train_df, 1).dropna(subset=["y"])
        X = feat.drop(columns=["timestamp", "y"]).copy()
        X["building"] = X["building"].astype("category")
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X, feat["y"], categorical_feature=["building"])
        return {"model": model, "train_df": train_df, "columns": X.columns.tolist()}

    def predict_fn(state, horizon):
        train_df = state["train_df"]
        future_index = _future_index(train_df, horizon)
        extended = pd.concat([train_df, pd.DataFrame(np.nan, index=future_index, columns=train_df.columns)])
        for ts in future_index:
            feat = build_features(extended.loc[:ts], 1)
            row = feat[feat["timestamp"] == ts].copy()
            X = row[state["columns"]].copy()
            X["building"] = X["building"].astype("category")
            extended.loc[ts, row["building"].values] = state["model"].predict(X)
        return extended.loc[future_index]

    return model_fn, predict_fn


def make_local_recursive_fns(lgb_params=LGB_PARAMS_LOCAL):
    def model_fn(train_df):
        models = {}
        for bid in train_df.columns:
            feat = build_features(train_df[[bid]], 1).dropna(subset=["y"])
            X = feat.drop(columns=["timestamp", "y", "building"])
            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(X, feat["y"])
            models[bid] = {"model": model, "columns": X.columns.tolist()}
        return {"models": models, "train_df": train_df}

    def predict_fn(state, horizon):
        train_df = state["train_df"]
        future_index = _future_index(train_df, horizon)
        preds = {}
        for bid, m in state["models"].items():
            extended = pd.concat([train_df[[bid]], pd.DataFrame(np.nan, index=future_index, columns=[bid])])
            for ts in future_index:
                feat = build_features(extended.loc[:ts], 1)
                row = feat[feat["timestamp"] == ts]
                extended.loc[ts, bid] = m["model"].predict(row[m["columns"]])[0]
            preds[bid] = extended.loc[future_index, bid].to_numpy()
        return pd.DataFrame(preds, index=future_index)

    return model_fn, predict_fn


_CACHE = {}


def get_chronos():
    if "chronos" not in _CACHE:
        from chronos import BaseChronosPipeline
        _CACHE["chronos"] = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    return _CACHE["chronos"]


def make_chronos_fns(context_length, use_covariate=False, weather=None, building_site=None):
    def model_fn(train_df):
        return {"context_df": train_df.iloc[-context_length:]}

    def predict_fn(state, horizon):
        context_df = state["context_df"]
        future_index = _future_index(context_df, horizon)
        long_df = context_df.reset_index().melt(id_vars="timestamp", var_name="item_id", value_name="target")
        future_df = None
        if use_covariate:
            long_df["site_id"] = long_df["item_id"].map(building_site)
            long_df = long_df.merge(weather[["site_id", "timestamp", "airTemperature"]],
                                    on=["site_id", "timestamp"], how="left").drop(columns="site_id")
            fut = weather[weather["timestamp"].isin(future_index)][["site_id", "timestamp", "airTemperature"]]
            rows = []
            for bid in context_df.columns:
                w = fut[fut["site_id"] == building_site[bid]].drop(columns="site_id").copy()
                w["item_id"] = bid
                rows.append(w)
            future_df = pd.concat(rows, ignore_index=True)[["item_id", "timestamp", "airTemperature"]]
        fc = get_chronos().predict_df(long_df, future_df=future_df, id_column="item_id", timestamp_column="timestamp",
                                      target="target", prediction_length=horizon, quantile_levels=[0.5])
        return fc.pivot(index="timestamp", columns="item_id", values="predictions").reindex(columns=context_df.columns)

    return model_fn, predict_fn


def ttm_route(context_length, horizon):
    from tsfm_public.toolkit.get_model import get_model
    return get_model(TTM_PATH, context_length=context_length, prediction_length=horizon, freq="h",
                     return_model_key=True)


def get_ttm(context_length, horizon):
    key = ("ttm", context_length, horizon)
    if key not in _CACHE:
        from tsfm_public.toolkit.get_model import get_model
        model = get_model(TTM_PATH, context_length=context_length, prediction_length=horizon, freq="h")
        _CACHE[key] = model.eval()
    return _CACHE[key]


def make_ttm_fns(context_length, pad="zero"):
    def model_fn(train_df):
        context_df = train_df.iloc[-context_length:]
        need = context_length - len(context_df)
        if pad == "tile" and need > 0:
            fill = context_df.iloc[[(j - need) % 168 for j in range(need)]].copy()
            fill.index = pd.date_range(end=context_df.index[0] - pd.Timedelta(hours=1), periods=need, freq="h")
            context_df = pd.concat([fill, context_df])
        return {"context_df": context_df}

    def predict_fn(state, horizon):
        model = get_ttm(context_length, horizon)
        context_df = state["context_df"]
        observed = ~context_df.isna().to_numpy().T
        filled = context_df.ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32).T
        kwargs = dict(past_values=torch.from_numpy(filled).unsqueeze(-1),
                      past_observed_mask=torch.from_numpy(observed.astype(np.float32)).unsqueeze(-1))
        if model.config.resolution_prefix_tuning:
            kwargs["freq_token"] = torch.full((filled.shape[0],), HOURLY_TOKEN, dtype=torch.long)
        with torch.inference_mode():
            out = model(**kwargs).prediction_outputs[..., 0].numpy().T
        return pd.DataFrame(out, index=_future_index(context_df, horizon), columns=context_df.columns)

    return model_fn, predict_fn
