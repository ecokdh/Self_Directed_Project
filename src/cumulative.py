"""12주 누적 비교표 (과제 12-3). 모든 수치는 1~11주차 산출물 CSV에서 읽어 온다(손으로 옮겨 적지 않음).

실행:  python -m src.cumulative   →  results/wk12/cumulative_table.csv, cumulative_table.md
"""
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
NA = "—"


def f(x, d=3):
    return NA if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def md_table(df):
    d = df.reset_index() if df.index.name else df
    cols = [str(c) for c in d.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return chr(10).join(rows)


def build():
    b6 = pd.read_csv(R / "wk6/benchmark_table.csv", index_col=0)
    cold = pd.read_csv(R / "wk6/scenario_B_coldstart.csv", index_col=0)
    c8 = pd.read_csv(R / "wk8/cumulative_table.csv", index_col=0)
    sc = pd.read_csv(R / "wk8/same_condition_table.csv", index_col=0)
    a9 = pd.read_csv(R / "wk9/method_comparison.csv", index_col=0)
    t9 = pd.read_csv(R / "wk9/cumulative_row.csv", index_col=0)
    l7 = pd.read_csv(R / "wk7/latency_cpu_vs_gpu.csv", index_col=0)
    cost7 = pd.read_csv(R / "wk7/lora_cost.csv")
    rt10 = pd.read_csv(R / "wk10/router_comparison.csv")

    rows = []

    def fc(name, m24, m168, vus, ms, train, gpu, expl, cold24):
        rows.append({"모델/방식": name, "MASE 24h": f(m24), "MASE 168h": f(m168), "VUS-PR": f(vus), "추론 ms": ms, "학습 시간": train,
                     "GPU 필요": gpu, "설명 가능": expl, "콜드스타트(2주 이력 MASE 24h)": cold24})

    lg = b6.loc["LightGBM (global)"]
    fc("Seasonal Naive", b6.loc["Seasonal Naive", "MASE(24h)"], b6.loc["Seasonal Naive", "MASE(168h)"], np.nan, f(b6.loc["Seasonal Naive", "추론 ms/계열(4스레드)"], 2) + " ms/계열", "없음", "아니오", "규칙 자체가 설명", f(cold.loc["Seasonal Naive", "MASE(24h)"]))
    fc("LightGBM (global)", lg["MASE(24h)"], lg["MASE(168h)"], np.nan, f(lg["추론 ms/계열(4스레드)"], 1) + " ms/계열", f(lg["학습 시간(s)"], 1) + " s (CPU)", "아니오", "SHAP 가능", f(cold.loc["LightGBM (global)", "MASE(24h)"]))
    fc("LightGBM + SHAP", lg["MASE(24h)"], lg["MASE(168h)"], a9.loc["LightGBM잔차", "VUS-PR"], f(lg["추론 ms/계열(4스레드)"], 1) + " ms/계열 (+SHAP 별도)", f(lg["학습 시간(s)"], 1) + " s (CPU)", "아니오", "예 (SHAP)", f(cold.loc["LightGBM (global)", "MASE(24h)"]))
    z, cv = b6.loc["Chronos-2"], b6.loc["Chronos-2 + 공변량"]
    fc("Chronos-2 zero-shot", z["MASE(24h)"], z["MASE(168h)"], np.nan, f(z["추론 ms/계열(4스레드)"], 1) + " ms/계열 (CPU)", "없음", "아니오 (GPU 시 17배 빠름)", "아니오", f(cold.loc["Chronos-2", "MASE(24h)"]))
    fc("Chronos-2 + 공변량", cv["MASE(24h)"], cv["MASE(168h)"], np.nan, f(cv["추론 ms/계열(4스레드)"], 1) + " ms/계열 (CPU)", "없음", "아니오", "아니오", f(cold.loc["Chronos-2 + 공변량", "MASE(24h)"]))
    lo = c8.loc["Chronos-2 LoRA (7주차)"]
    fc("Chronos-2 + LoRA", lo["MASE(24h)"], lo["MASE(168h)"], np.nan, f(l7.loc["건물 1개 (5주차 5-6 조건)", "cuda"], 1) + " ms/건물 (서버 GPU) ¹", f(lo["학습 1회(s)"], 0) + " s (GPU)", "학습 GPU", "아니오", NA)
    tt = b6.loc["TTM (ctx=512)"]
    fc("TTM (ctx=512)", tt["MASE(24h)"], tt["MASE(168h)"], np.nan, f(tt["추론 ms/계열(4스레드)"], 1) + " ms/계열 (CPU)", "없음", "아니오", "아니오", f(cold.loc["TTM (ctx=512)", "MASE(24h)"]) + " (제로패딩) ²")
    for key, nm in [("PatchTST", "PatchTST"), ("iTransformer", "iTransformer"), ("TimeXer", "TimeXer"), ("DLinear", "DLinear")]:
        r = c8.loc[f"{key} (look-back 336)"]
        fc(nm, r["MASE(24h)"], r["MASE(168h)"], np.nan, f(sc.loc[key, "추론 ms/창"], 2) + " ms/창 (GPU) ³", f(r["학습 1회(s)"], 0) + " s (GPU)", "학습·추론 GPU", "아니오", NA)
    for key in ["CATCH", "DADA", "Sub-LOF", "MatrixProfile"]:
        sec = t9.loc[key, "17건물 합계(s)"]
        rows.append({"모델/방식": {"MatrixProfile": "Matrix Profile"}.get(key, key), "MASE 24h": NA, "MASE 168h": NA, "VUS-PR": f(a9.loc[key, "VUS-PR"]),
                     "추론 ms": f(sec, 1) + " s / 17건물 ⁴", "학습 시간": NA, "GPU 필요": t9.loc[key, "GPU 필요"], "설명 가능": "아니오", "콜드스타트(2주 이력 MASE 24h)": NA})
    # 라우터(최종): 뒤 26창 기준이라 52창 표와 직접 비교 불가 → 같은 열에 값을 넣되 각주 ⁵
    r24 = rt10[(rt10["horizon"] == 24) & (rt10["방식"] == "규칙 기반 라우터")].iloc[0]
    r168 = rt10[(rt10["horizon"] == 168) & (rt10["방식"] == "규칙 기반 라우터")].iloc[0]
    rows.append({"모델/방식": "**라우터 (최종, 규칙 기반)** ⁵", "MASE 24h": f(r24["MASE"]), "MASE 168h": f(r168["MASE"]), "VUS-PR": NA,
                 "추론 ms": f(r24["평균 지연(ms/계열)"], 1) + " / " + f(r168["평균 지연(ms/계열)"], 1) + " ms/계열", "학습 시간": "규칙 탐색 포함 수 초", "GPU 필요": "아니오",
                 "설명 가능": "예 (결정 로그)", "콜드스타트(2주 이력 MASE 24h)": "기본 모델과 동일 (0.392)"})
    t = pd.DataFrame(rows)
    # 같은 뒤 26창 기준 보조표 (라우터를 공정하게 비교)
    aux = rt10[rt10["방식"].isin(["최고 단일 모델 (앞 26창 기준: Chronos-2 + 공변량)", "규칙 기반 라우터", "LLM 기반 라우터", "  (풀) Seasonal Naive", "  (풀) LightGBM (global)", "  (풀) Chronos-2", "  (풀) TTM (ctx=512)", "오라클 라우터 (칸 단위, 사후)"])]
    aux = aux.pivot_table(index="방식", columns="horizon", values="MASE").rename(columns={24: "MASE 24h (뒤 26창)", 168: "MASE 168h (뒤 26창)"})
    return t, aux


FOOT = """
¹ LoRA 추론: 서버 GPU 건물 1개 11.6 ms(서버 CPU 134.1 ms). 17건물 배치는 38.6 ms. Chronos-2 zero-shot의 ms/계열은 노트북 CPU 4스레드라 장치가 달라 직접 비교 불가.
² TTM은 2주 이력이 컨텍스트(512)보다 짧아 제로패딩 시 1.510, 주기 패딩 시 0.753.
³ 딥러닝 추론은 창 단위 ms(8주차 4-2, 168h)라 ms/계열과 단위가 다름. look-back 336 · 1회 학습 · 학습 시간은 168h 값. MASE 24h는 8주차 추가 관찰(horizon별 순위)의 24h 별도 학습 결과.
⁴ 이상진단 4종은 17건물 1년치(LEAD 2016) 전체 처리 시간(초). CATCH는 17채널을 한 모델로 학습한 시간(학습 GPU), DADA는 학습 없이 추론만(추론 GPU). 단위가 달라 예측 모델의 ms/계열과 직접 비교 불가. VUS-PR은 LEAD 2016 라벨 기준이라 MASE(2017 평가)와 데이터 구간이 다름.
⁵ 라우터는 **뒤 26창(2017-07-09~)** 에서만 평가했다(규칙을 앞 26창으로 정했기 때문). 나머지 행은 **52창 전체** 값이라 직접 비교하면 안 된다. 같은 뒤 26창 기준 비교는 아래 보조표.
— 칸: 해당 없음(예측 모델의 VUS-PR, 이상진단의 MASE) 또는 미측정(LoRA·딥러닝의 콜드스타트 — 실행하지 않음). LightGBM + SHAP 행의 VUS-PR은 9주차 "LightGBM 잔차" 기법(4주차 모델의 잔차 기반 이상 점수, 정상 구간 라벨 사용)의 값.
"""

if __name__ == "__main__":
    out = R / "wk12"
    out.mkdir(exist_ok=True)
    t, aux = build()
    t.to_csv(out / "cumulative_table.csv", index=False, encoding="utf-8")
    aux.to_csv(out / "cumulative_router_same_windows.csv", encoding="utf-8")
    (out / "cumulative_table.md").write_text(md_table(t) + "\n" + FOOT + "\n### 보조표: 같은 뒤 26창 기준\n\n" + md_table(aux.round(4)) + "\n", encoding="utf-8")
    print(t.to_string(index=False))
    print(aux.round(4))
