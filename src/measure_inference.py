import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.disable(logging.WARNING)

import numpy as np
import psutil
import torch
from chronos import BaseChronosPipeline  # noqa: F401  (라이브러리 import를 baseline 이전에 끝내기 위함)
from tsfm_public.toolkit.get_model import get_model  # noqa: F401

from src import bench_models as bm

MB = 1024 ** 2


def build(model, threads, weather, building_site):
    lgb_g = {**bm.LGB_PARAMS, "n_jobs": threads}
    lgb_l = {**bm.LGB_PARAMS_LOCAL, "n_jobs": threads}
    table = {
        "naive": lambda: bm.make_naive_fns(),
        "lgb_local_direct": lambda: bm.make_local_fns(24, lgb_l),
        "lgb_local_recursive": lambda: bm.make_local_recursive_fns(lgb_l),
        "lgb_global_direct": lambda: bm.make_global_fns(24, lgb_g),
        "lgb_global_recursive": lambda: bm.make_recursive_fns(lgb_g),
        "chronos": lambda: bm.make_chronos_fns(1024),
        "chronos_cov": lambda: bm.make_chronos_fns(1024, True, weather, building_site),
        "ttm512": lambda: bm.make_ttm_fns(512),
        "ttm1024": lambda: bm.make_ttm_fns(1024),
    }
    return table[model]()


def preload(model):
    if model.startswith("chronos"):
        bm.get_chronos()
    elif model.startswith("ttm"):
        bm.get_ttm(int(model[3:]), 24)


class RssSampler(threading.Thread):
    def __init__(self, proc, interval=0.002):
        super().__init__(daemon=True)
        self.proc, self.interval, self.peak, self.stop = proc, interval, 0, False

    def run(self):
        while not self.stop:
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=11)
    ap.add_argument("--series", type=int, default=1)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    data = bm.load_data()
    weather, building_site = bm.load_weather(data)
    model_fn, predict_fn = build(a.model, a.threads, weather, building_site)
    proc = psutil.Process()
    result = {"model": a.model, "threads": a.threads, "series": a.series}

    if a.train:
        sampler = RssSampler(proc)
        sampler.start()
        t0 = time.perf_counter()
        model_fn(data.iloc[:-24])
        result["train_s"] = time.perf_counter() - t0
        sampler.stop = True
        result["train_peak_mb"] = sampler.peak / MB
    else:
        if a.series == 1:
            train_df = data[[data.columns[0]]].iloc[:-24]
        else:
            idx = [i % data.shape[1] for i in range(a.series)]
            train_df = data.iloc[:-24, idx].copy()
            train_df.columns = [f"s{i}" for i in range(a.series)]
        result["baseline_mb"] = proc.memory_info().rss / MB
        state = model_fn(train_df)
        preload(a.model)
        result["after_load_mb"] = proc.memory_info().rss / MB
        sampler = RssSampler(proc)
        sampler.start()
        times = []
        for _ in range(a.repeats):
            t0 = time.perf_counter()
            predict_fn(state, 24)
            times.append((time.perf_counter() - t0) * 1000)
        sampler.stop = True
        result["peak_mb"] = sampler.peak / MB
        result["peak_wset_mb"] = getattr(proc.memory_info(), "peak_wset", 0) / MB
        result["times_ms"] = times
        measured = times[1:] if len(times) > 1 else times
        result["median_ms"] = float(np.median(measured))
        result["min_ms"], result["max_ms"] = float(min(measured)), float(max(measured))
        result["per_series_ms"] = result["median_ms"] / a.series

    Path(a.out).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "times_ms"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
