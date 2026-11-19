"""11주차 (3)~(6): 근거 데이터 구조화 · 설명문 생성(LLM) · 자동 사실성 검증기.

- 근거 JSON은 코드가 계산한 값만 담는다(LLM은 계산하지 않음). 정답(실측)은 담지 않는다: 설명은 예측 시점에 쓸 수 있는 정보만으로 만든다.
- SHAP은 Chronos-2(트리 모델이 아님)를 직접 설명하지 못하므로, 같은 입력(+외기온)으로 학습한 LightGBM 대리 모델의 SHAP이다. JSON에 그렇게 명시한다.
- 검증기: 문장 속 수치를 모두 뽑아 JSON 값과 대조하고, JSON에 없는 외부 원인 어휘를 '근거 없는 주장'으로 센다.
"""
import json
import os
import re
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
import shap

from src import routing as rt
from src.features import build_features

MODEL = "Chronos-2 + 공변량"


# ── 11-3. 근거 구조화 ─────────────────────────────────────────────────────────
def shap_surrogate(data, weather, site, origin_ts, h, ev_ts, building, top=3):
    """origin 이전 데이터로만 학습한 대리 LightGBM(전 건물, 외기온 포함)의 SHAP. 이벤트 시각 1행에 대해 상위 기여 피처."""
    train = data.loc[: origin_ts - pd.Timedelta(hours=1)]
    weather = weather.assign(dewTemperature=np.nan)  # build_features 가 요구하는 열(이슬점은 이 프로젝트에서 쓰지 않음) → 아래에서 버림
    feat = build_features(train, h, weather=weather, building_site=site).dropna(subset=["y"])
    X = feat.drop(columns=["timestamp", "y", "dewTemperature"])
    X["building"] = pd.Categorical(X["building"], categories=list(data.columns))
    m = lgb.LGBMRegressor(objective="regression", num_leaves=127, learning_rate=0.08, n_estimators=300, min_child_samples=10, verbosity=-1)
    m.fit(X, feat["y"], categorical_feature=["building"])
    fut = pd.date_range(origin_ts, periods=h, freq="h")
    ext = pd.concat([train, pd.DataFrame(np.nan, index=fut, columns=train.columns)])
    f = build_features(ext, h, weather=weather, building_site=site)
    f = f[(f["timestamp"] == ev_ts) & (f["building"] == building)]
    Xe = f[X.columns].copy()  # 이벤트 시각 1행
    Xe["building"] = pd.Categorical(Xe["building"], categories=list(data.columns))
    ex = shap.TreeExplainer(m)
    sv = pd.Series(ex.shap_values(Xe)[0], index=X.columns)
    base = float(np.ravel(ex.expected_value)[0])
    sur_pred = float(m.predict(Xe)[0])
    rows = [{"feature": k, "value": round(float(Xe[k].iloc[0]), 2) if k != "building" else str(Xe[k].iloc[0]), "contribution": round(float(sv[k]), 2)}
            for k in sv.abs().sort_values(ascending=False).index[:top]]
    return rows, round(base, 2), round(sur_pred, 2)


def analogue(data, weather, site, df_hol, building, origin_ts, ev_ts, h):
    """같은 사이트의 과거(origin 이전) 날 중, 같은 날 유형(공휴일/주말/평일)이면서 24시간 기온 곡선이 가장 비슷한 날.
    유사도 = 1 − (24시간 평균 |기온차| ÷ (사이트 기온 95% − 5%)), 0~1."""
    s = site[building]
    T = weather[weather["site_id"] == s].set_index("timestamp")["airTemperature"].reindex(data.index)
    day0 = ev_ts.normalize()
    cur = T.loc[day0: day0 + pd.Timedelta(hours=23)].to_numpy()
    hist = T.loc[: origin_ts - pd.Timedelta(hours=1)]
    rng = float(hist.quantile(.95) - hist.quantile(.05))
    days = hist.index.normalize().unique()
    days = [d for d in days if d + pd.Timedelta(hours=23) < origin_ts]

    def kind(d):
        return "공휴일" if d in df_hol else ("주말" if d.dayofweek >= 5 else "평일")

    best = None
    for d in days:
        if kind(d) != kind(day0):
            continue
        v = hist.loc[d: d + pd.Timedelta(hours=23)].to_numpy()
        if len(v) < 24 or np.isnan(v).any():
            continue
        sim = 1 - float(np.nanmean(np.abs(v - cur))) / rng
        if best is None or sim > best[1]:
            best = (d, sim)
    d, sim = best
    return {"date": str(d.date()), "similarity": round(sim, 2), "day_type": kind(day0),
            "load_same_hour": round(float(data[building].loc[d + pd.Timedelta(hours=ev_ts.hour)]), 1)}


def build_evidence(ev, ctx):
    """ev: dict(window, building, step) / ctx: 노트북에서 만든 공용 객체 모음 → 근거 JSON(dict)."""
    w, b, st = ev["window"], ev["building"], ev["step"]
    pool, df, data = ctx["pool"], ctx["df"], ctx["data"]
    j = pool.buildings.index(b)
    origin = pool.origin_ts[w]
    ev_ts = origin + pd.Timedelta(hours=st)
    row = df[(df["window"] == w) & (df["building"] == b) & (df["step"] == st)].iloc[0]
    fit_res = df[(df["window"] < 26) & (df["building"] == b)]["resid"]
    lo, hi = np.quantile(fit_res, [.1, .9])
    route = ctx["route_log"]
    route = route[(route["window"] == w) & (route["building"] == b)].iloc[0]
    sh, base, sur = shap_surrogate(data, ctx["weather"], ctx["site"], origin, pool.h, ev_ts, b)
    rec = float(row["recent"])
    q50, q75, q95 = ctx["recent_q"]
    level = "low" if rec <= q50 else "normal" if rec <= q75 else "elevated" if rec <= q95 else "high"
    q = ctx["temp_q"].loc[ctx["site"][b]]
    pct = "extreme_high" if row["temp"] > q["q95"] else "extreme_low" if row["temp"] < q["q05"] else "normal_range"
    return {
        "building": b, "forecast_time": str(ev_ts), "horizon_h": pool.h,
        "prediction": {"value": round(float(row["pred"]), 1), "unit": "kWh", "interval": [round(float(row["pred"] + lo), 1), round(float(row["pred"] + hi), 1)],
                       "interval_note": "앞 26주 동안 이 건물의 잔차 10~90% 분위수를 더한 근사 구간", "model_used": route["model"], "route_reason": route["reason"]},
        "shap_top_features": sh,
        "shap_note": {"explained_model": "LightGBM 대리 모델(Chronos-2 자체가 아님)", "base_value": base, "surrogate_prediction": sur},
        "covariates": {"airTemperature_forecast": round(float(row["temp"]), 1), "temperature_band": pct, "is_holiday": bool(row["hol"])},
        "anomaly_status": {"is_anomaly": bool(rec > 3), "recent_residual_level": level, "recent_naive_mase": round(rec, 2)},
        "historical_analogue": analogue(data, ctx["weather"], ctx["site"], ctx["hol_days"], b, origin, ev_ts, pool.h),
    }


# ── 11-4. 설명문 생성 ─────────────────────────────────────────────────────────
PROMPTS = {
    "v1": ("당신은 건물 에너지 수요 예측 결과를 설비 담당자에게 설명하는 도우미입니다. "
           "아래 JSON에 제공된 수치만 사용하고, 없는 정보는 언급하지 마세요. 한국어로 5문장 이내로 설명하세요."),
}


PROMPTS["v2"] = (
    "당신은 건물 에너지 수요 예측 결과를 설비 담당자에게 설명하는 도우미입니다. 아래 JSON에 제공된 정보만 사용해 한국어로 5문장 이내로 설명하세요.\n"
    "규칙:\n"
    "1. 수치는 JSON 값을 한 글자도 바꾸지 말고 그대로 옮기세요(반올림·계산·단위 환산·범위 표현 금지). '~이상', '~이하', '약' 같은 한정어를 수치에 붙이지 마세요.\n"
    "2. 피처 이름은 JSON의 영문 이름(예: lag_24)을 그대로 쓰고, 풀어 쓸 때는 이름을 괄호로 함께 적으세요. 이름에 들어 있는 숫자를 다른 숫자로 바꾸지 마세요.\n"
    "3. 날짜·시각은 forecast_time 값을 그대로 쓰세요. '오늘/내일/어제/작년/올해' 같은 상대 시점 표현은 쓰지 마세요.\n"
    "4. JSON에 없는 원인(계절, 설비 상태, 행사, 고장 등)과 JSON에 없는 권고는 쓰지 마세요.\n"
    "5. shap_top_features는 shap_note의 대리 모델 기준이며 예측 모델(model_used)이 아님을 한 문장으로 밝히세요. contribution이 음수면 값을 낮추는 방향, 양수면 높이는 방향으로만 해석하세요.\n"
    "6. is_anomaly, recent_residual_level, temperature_band, is_holiday는 JSON 값 그대로 한 번씩 언급하세요."
)


def template_report(ev):
    """LLM 없이 JSON 값만 끼워 넣는 정형 문장 — 검증 실패 시 대체 출력(수치가 전부 JSON에서 직접 옴)."""
    p, c, a, n = ev["prediction"], ev["covariates"], ev["anomaly_status"], ev["shap_note"]
    top = ", ".join(f"{f['feature']}({f['contribution']})" for f in ev["shap_top_features"])
    return (f"{ev['forecast_time']}에 {ev['building']}의 예측값은 {p['value']} kWh이고, 구간은 {p['interval'][0]}에서 {p['interval'][1]} kWh입니다. "
            f"사용 모델은 {p['model_used']}이며 라우팅 이유는 '{p['route_reason']}'입니다. "
            f"SHAP 기여도는 {n['explained_model']} 기준이며 상위 요인은 {top}입니다. "
            f"외기온은 {c['airTemperature_forecast']}도(temperature_band={c['temperature_band']}), is_holiday={str(c['is_holiday']).lower()}입니다. "
            f"is_anomaly={str(a['is_anomaly']).lower()}, recent_residual_level={a['recent_residual_level']}, recent_naive_mase={a['recent_naive_mase']}입니다.")


def call_llm(system, evidence, model=None, temperature=0, max_tokens=500):
    key = os.environ["OPENAI_API_KEY"]
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    t0 = time.time()
    r = requests.post("https://api.openai.com/v1/chat/completions", timeout=60,
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": model, "temperature": temperature, "max_tokens": max_tokens,
                            "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": json.dumps(evidence, ensure_ascii=False, indent=1)}]})
    r.raise_for_status()
    j = r.json()
    u = j["usage"]
    return {"text": j["choices"][0]["message"]["content"].strip(), "tok_in": u["prompt_tokens"], "tok_out": u["completion_tokens"], "secs": time.time() - t0}


# ── 11-5. 자동 사실성 검증기 ──────────────────────────────────────────────────
NUM_RE = re.compile(r"(?<![A-Za-z0-9_.])[-−]?\d[\d,]*(?:\.\d+)?")
# JSON 에 근거가 없으면 '근거 없는 주장'으로 세는 어휘 (계절·설비·사건·시점)
CLAIM_WORDS = ["여름", "겨울", "봄", "가을", "폭염", "한파", "냉방", "난방", "냉동기", "냉각", "보일러", "공조", "에어컨", "히터",
               "행사", "고장", "정전", "점검", "방학", "공사", "휴가", "작년", "지난해", "전년", "올해", "내일", "오늘", "어제"]


QUAL_RE = re.compile(r"\d[\d,.]*\s*(?:도|°C|kWh|%|시간)?\s*(?:이상|이하|초과|미만|가까이|넘)")  # "40도 이상" 처럼 값을 부풀리는 한정어


def _collect(obj, nums, texts):
    if isinstance(obj, dict):
        for k, v in obj.items():
            texts.append(str(k))
            _collect(v, nums, texts)
    elif isinstance(obj, (list, tuple)):
        nums.update(range(1, len(obj) + 1))
        for v in obj:
            _collect(v, nums, texts)
    elif isinstance(obj, bool):
        texts.append(str(obj))
    elif isinstance(obj, (int, float)):
        nums.add(abs(float(obj)))
        if abs(obj) <= 1.5:
            nums.add(abs(float(obj)) * 100)  # 비율을 %로 쓴 경우
    else:
        texts.append(str(obj))
        for m in re.findall(r"\d+(?:\.\d+)?", str(obj)):  # 날짜·모델명·피처명(lag_168 → 168)의 숫자도 근거로 인정
            nums.add(float(m))


def allowed_from(evidence):
    nums, texts = set(), []
    _collect(evidence, nums, texts)
    return nums, " ".join(texts)


def verify(text, evidence):
    """문장 속 수치를 모두 뽑아 JSON 의 수치와 '표기한 자릿수까지 반올림해서' 같은지 대조한다. 반환: dict(요약 + 불일치/근거없음 목록)."""
    nums, corpus = allowed_from(evidence)
    found, bad = [], []
    for m in NUM_RE.finditer(text):
        raw = m.group().replace(",", "").replace("−", "-")
        x = abs(float(raw))
        dec = len(raw.split(".")[1]) if "." in raw else 0
        ok = any(abs(round(v, dec) - x) < 1e-9 or abs(v - x) <= 0.5 * 10 ** (-dec) + 1e-9 for v in nums)
        found.append((m.group(), ok))
        if not ok:
            bad.append(m.group())
    # 영문 식별자(피처 이름 등)가 JSON 에 없는 경우
    idents = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", text) if t.lower() not in corpus.lower()]
    claims = [w for w in CLAIM_WORDS if w in text and w not in corpus]
    quals = ["한정어: " + m.group() for m in QUAL_RE.finditer(text)]
    claims += quals
    n = len(found)
    return {"n_numbers": n, "n_match": n - len(bad), "n_mismatch": len(bad), "mismatch_rate": (len(bad) / n) if n else 0.0,
            "mismatch_items": bad, "unsupported_claims": claims + idents, "n_unsupported": len(claims) + len(idents)}
