import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# 파인튜닝 구간 = 2016년 (LEAD 라벨 기간). 평가 52창의 첫 origin(2017-01-08)보다 앞이라 누설 없음
FT_START, FT_END = "2016-01-01", "2016-12-31 23:00"
FRACTIONS = [0.01, 0.05, 0.10, 0.50, 1.00]
SEEDS = [0, 1, 2]


def load_anomaly_mask(columns):
    # LEAD 1.0 (2016) → 건물×시점 bool. True = 이상. 4주차 과제 4-1과 같은 kaggle_id 조인
    meta = pd.read_csv(ROOT / "Data/raw/bdg2/metadata.csv")
    lead = pd.read_csv(ROOT / "Data/raw/lead/lead1.0-small.csv", parse_dates=["timestamp"])
    ids = meta.loc[meta["building_id"].isin(columns), ["building_id", "building_id_kaggle"]].dropna()
    to_name = dict(zip(ids["building_id_kaggle"].astype(int), ids["building_id"]))
    lead = lead[lead["building_id"].isin(to_name)].assign(building=lambda d: d["building_id"].map(to_name))
    return lead.pivot(index="timestamp", columns="building", values="anomaly").reindex(columns=columns).eq(1)


def finetune_frame(data, mask=None, frac=1.0):
    # 2016년 중 마지막 frac 구간. mask 가 주어지면 이상 시점을 NaN 으로 가려 학습에서 제외
    df = data.loc[FT_START:FT_END]
    if mask is not None:
        df = df.mask(mask.reindex(index=df.index, columns=df.columns, fill_value=False))
    return df.iloc[-max(1, round(len(df) * frac)):]


def to_fit_inputs(df):
    return [{"target": df[c].to_numpy(dtype="float32")} for c in df.columns if df[c].notna().any()]


# batch_size: fit 기본값 256은 VRAM 17.3GB(20 step 실측)로 8GB 기준 초과 → 64
def lora_fit(pipe, df, horizon, seed, out_dir, context_length=1024, num_steps=1000, learning_rate=1e-5, batch_size=64):
    import torch
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    ft = pipe.fit(to_fit_inputs(df), prediction_length=horizon, finetune_mode="lora", context_length=context_length,
                  num_steps=num_steps, learning_rate=learning_rate, batch_size=batch_size, output_dir=out_dir,
                  seed=seed, remove_printer_callback=True)
    trainable, total = ft.model.get_nb_trainable_parameters()
    return ft, {"학습 파라미터": trainable, "전체 파라미터": total, "학습 비율(%)": 100 * trainable / total,
                "최대 VRAM(GB)": torch.cuda.max_memory_allocated() / 1e9, "학습 시간(s)": time.perf_counter() - t0}
