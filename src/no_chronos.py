"""민감도 분석: 사전학습 파운데이션 모델(Chronos-2, TTM)을 뺀 풀로 10주차 라우터 절차를 그대로 다시 수행.

- 풀(POOL_NC): Seasonal Naive, LightGBM local·global, 8주차 딥러닝 4종(PatchTST, iTransformer, TimeXer, DLinear — 우리 데이터로만 학습)
- 절차는 10주차와 같음: 기본값·규칙은 앞 26창(FIT_W)에서만, 평가는 뒤 26창(EVAL_W)에서만. 규칙 탐색 함수·통과 조건도 같음(fit_rules 기본값).
- 원래 결과(results/wk10, wk12)는 건드리지 않고 results/wk12_no_chronos/ 에만 쓴다.

사용:  python -m src.no_chronos
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src import bench_models as bm, routing as rt

OUT = rt.RES / "wk12_no_chronos"


def fit_router(pool, state, models):
    """앞 26창으로 기본값(칸 평균 MASE 최소)과 규칙을 정한다. → (기본값, 규칙 목록, 칸 MASE 표)"""
    cm = pool.cell_mase()
    E = pd.DataFrame({n: cm[n].reshape(-1) for n in models})  # 행 순서 = (윈도우, 건물) = state 행 순서
    fit = state["window"].isin(rt.FIT_W).to_numpy()
    default = E[fit].mean().idxmin()
    rules = rt.fit_rules(state[fit].reset_index(drop=True), {n: E[n][fit].to_numpy() for n in models}, default)
    for r in rules:
        r.pop("gain_sum")
    return default, rules, E


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = bm.load_data()
    state = pd.read_csv(rt.RES / "wk10/state_signals.csv")  # 상태 신호는 풀과 무관(origin 이전 계측값만) → 10주차 것 재사용
    ev = state["window"].isin(rt.EVAL_W).to_numpy()
    orig = pd.read_csv(rt.RES / "wk10/router_comparison.csv")
    config, rule_rows, cmp_rows, share_rows, fit_ev = {}, [], [], [], {}
    for h in rt.HORIZONS:
        pool = rt.Pool(data, h, models=rt.POOL_NC)
        W, B = pool.W, len(pool.buildings)
        default, rules, E = fit_router(pool, state, rt.POOL_NC)
        fit_ev[f"{h}h"] = E[~ev].mean()

        # 점검: TTM 을 풀에 넣어도 규칙이 같은가 (TTM 사전학습 데이터는 미확인이라 기본 풀에서는 뺌)
        p8 = rt.Pool(data, h, models=rt.POOL_NC + ["TTM (ctx=512)"])
        d8, r8, _ = fit_router(p8, state, rt.POOL_NC + ["TTM (ctx=512)"])
        same_with_ttm = d8 == default and [(r["signal"], r["op"], r["model"]) for r in r8] == [(r["signal"], r["op"], r["model"]) for r in rules]

        model, why = rt.apply_rules(state, rules, default)
        log = state.assign(model=model, reason=why)
        log.to_csv(OUT / f"routing_log_{h}h.csv", index=False, encoding="utf-8")
        config[str(h)] = {"default": default, "rules": [{k: r[k] for k in ("signal", "op", "thr", "model")} for r in rules],
                          "same_rules_with_ttm": bool(same_with_ttm)}

        done = np.zeros(len(state), bool)
        for i, r in enumerate(rules):
            v = state[r["signal"]].to_numpy()
            m = ~done & ((v <= r["thr"]) if r["op"] == "<=" else (v > r["thr"]))
            done |= m
            me = m & ev
            rule_rows.append({"horizon": h, "순서": i + 1, **r, "eval_support": int(me.sum()),
                              "eval_rel_gain": float((E[default][me] - E[r["model"]][me]).mean() / E[default][me].mean())})

        single, single_pb = pool.score(np.full((W, B), default, dtype=object), rt.EVAL_W)
        eb = E[ev].assign(b=state["building"][ev].to_numpy()).groupby("b").mean()
        ways = {f"최고 단일 모델 (앞 26창 기준: {default})": np.full((W, B), default, dtype=object),
                "규칙 기반 라우터": model.reshape(W, B),
                "오라클 (건물 단위, 사후)": np.tile(eb.idxmin(axis=1).reindex(pool.buildings).to_numpy(), (W, 1)),
                "오라클 라우터 (칸 단위, 사후)": E.idxmin(axis=1).to_numpy().reshape(W, B),
                **{f"  (풀) {n}": np.full((W, B), n, dtype=object) for n in rt.POOL_NC}}
        for name, ch in ways.items():
            s, pb = pool.score(ch, rt.EVAL_W)
            d = single_pb - pb
            cmp_rows.append({"horizon": h, "방식": name, "MASE": s, "단일 대비 개선(%)": (single - s) / single * 100,
                             "나은 건물": int((d > 1e-12).sum()), "나쁜 건물": int((d < -1e-12).sum()),
                             "Wilcoxon p": wilcoxon(d).pvalue if (d.abs() > 1e-12).any() else np.nan})
        o = orig[orig["horizon"] == h].set_index("방식")["MASE"]
        for name in [n for n in o.index if n.startswith("최고 단일")] + ["규칙 기반 라우터"]:
            cmp_rows.append({"horizon": h, "방식": f"(참고) 원래 풀 — {name}", "MASE": o[name]})
        share_rows += [{"horizon": h, "model": k, "칸 수": int(v), "비율": v / ev.sum()} for k, v in log[ev]["model"].value_counts().items()]

    pd.DataFrame(fit_ev).to_csv(OUT / "fit_window_evidence.csv", encoding="utf-8")
    pd.DataFrame(rule_rows).to_csv(OUT / "learned_rules.csv", index=False, encoding="utf-8")
    pd.DataFrame(cmp_rows).to_csv(OUT / "router_comparison.csv", index=False, encoding="utf-8")
    pd.DataFrame(share_rows).to_csv(OUT / "selection_share.csv", index=False, encoding="utf-8")
    (OUT / "router_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    print(pd.DataFrame(cmp_rows).round(4).to_string(index=False))
    print(json.dumps(config, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
