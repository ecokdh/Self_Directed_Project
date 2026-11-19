import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib import font_manager

from src import pipeline as pl

for f in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
    if f in {x.name for x in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False


@st.cache_resource
def system(h, variant):
    return pl.System(h, variant)


@st.cache_data(show_spinner=False)
def make_report(h, building, time, use_llm, variant):
    return system(h, variant).report(building, time, use_llm=use_llm)


def render(variant="original"):
    nc = variant == "no_chronos"
    st.set_page_config(page_title="건물 에너지 예측 · 근거 설명" + (" (파운데이션 모델 제외)" if nc else ""), layout="wide")
    st.title("건물 전력 수요 예측 · 근거 설명" + (" — 파운데이션 모델 제외" if nc else ""))
    with st.sidebar:
        h = st.radio("예측 구간(horizon)", [24, 168], format_func=lambda x: f"{x}시간")
        sysm = system(h, variant)
        building = st.selectbox("건물", sysm.pool.buildings)
        wins = list(range(26, sysm.pool.W))
        w = st.selectbox("윈도우 시작 시각", wins, format_func=lambda i: str(sysm.pool.origin_ts[i]))
        step = st.slider("설명할 시각(윈도우 안 몇 시간째)", 0, h - 1, min(12, h - 1))
        use_llm = st.checkbox("LLM 설명 사용 (OpenAI 호출)", value=False, help="끄면 코드가 계산한 값만 쓰는 정형 문장(쉬운 말)을 보여 줍니다.")
        st.caption("평가 구간(2017-07-09~)의 저장된 예측을 조회합니다. 새 시점의 실시간 추론은 하지 않습니다."
                   + (" 딥러닝 4종은 체크포인트가 없어 저장된 예측 조회만 가능합니다." if nc else ""))

    t = sysm.pool.origin_ts[w] + pd.Timedelta(hours=step)
    r = make_report(h, building, str(t), use_llm, variant)
    fc, a, v, p = r["forecast"], r["anomaly_status"], r["verification"], r["evidence"]["prediction"]

    c1, c2, c3 = st.columns(3)
    c1.metric("선택된 모델", r["route"]["model"])
    c1.caption(f"이유: {r['route']['reason']}")
    c2.metric("설명 시각 예측값", f"{p['value']} kWh")
    c2.caption(f"과거 오차로 보면 10번 중 8번은 {p['interval'][0]}–{p['interval'][1]} kWh 사이")
    c3.metric("이상 상태", "이상" if a["is_anomaly"] else "정상",
              help="지난 1주와 그 전 주의 같은 요일·시각 사용량 차이가 평소 차이의 3배를 넘으면 이상 (직전 1주 Naive MASE > 3)")
    c3.caption(f"지난주 변동이 평소의 {a['recent_naive_mase']}배 (기준 3배)")

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.fill_between(fc["timestamp"], fc["구간 하한"], fc["구간 상한"], alpha=.25, label="예측구간(근사)")
    ax.plot(fc["timestamp"], fc["예측"], label="예측")
    ax.plot(fc["timestamp"], fc["실측"], color="k", lw=1.2, label="실측")
    ax.axvline(t, color="tab:red", ls="--", lw=1, label="설명 시각")
    if a["is_anomaly"]:
        ax.set_title("지난주 사용 패턴이 평소와 크게 달랐음(이상 상태)", color="tab:red")
    ax.set_ylabel("kWh"); ax.legend(loc="upper left", fontsize=8)
    st.pyplot(fig)

    st.subheader("근거 설명")
    st.markdown(r["explanation"]["text"])
    st.caption(f"생성 방식: {r['explanation']['source']} (LLM 시도 {r['explanation']['llm_attempts']}회). "
               + "'예측에 크게 작용한 것'은 선택된 예측 모델이 아니라 설명용 LightGBM 대리 모델(SHAP) 기준입니다.")
    if v["n_mismatch"] or v["n_unsupported"]:
        st.error(f"검증 실패: 불일치 {v['mismatch_items']}, 근거 없는 주장 {v['unsupported_claims']}")
    else:
        st.success(f"자동 검증 통과: 문장 속 수치 {v['n_numbers']}개가 모두 근거 JSON과 일치, 근거 없는 주장 0건. (의미 오류는 이 검증으로 잡히지 않음)")

    with st.expander("근거 JSON"):
        st.json(r["evidence"])
    with st.expander("라우팅 신호"):
        st.dataframe(pd.DataFrame([r["route"]["signals"]]))


if __name__ == "__main__":
    render("original")
