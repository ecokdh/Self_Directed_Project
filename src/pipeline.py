"""12주차: 통합 파이프라인 (단일 진입점).

[데이터 입력] → [상태 수집] → [라우터] → [예측기 선택·실행] → [이상진단] → [근거 구조화] → [설명 생성] → [검증] → [최종 리포트]

사용:  python -m src.pipeline --building Fox_office_Vicki --time "2017-08-27 12:00" [--horizon 24] [--no-llm] [--variant no_chronos] [--style legacy]

제약(정직하게): 예측기 호출은 1~8주차에 저장된 윈도우별 예측을 조회한다(10주차와 같은 방식). 따라서 입력 시각은
라우터를 평가한 뒤 26창(2017-07-09~) 안에서 해당 horizon 윈도우에 속해야 한다. 새 시점의 실시간 추론은 하지 않는다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from src import bench_models as bm, correction as cr, explain as ex, routing as rt

ROOT = Path(__file__).resolve().parents[1]
MAX_RETRY = 2


class System:
    def __init__(self, horizon=24, variant="original"):
        """variant='original': 10주차 풀·규칙(기본값 Chronos-2 + 공변량).
        variant='no_chronos': 파운데이션 모델을 뺀 풀(rt.POOL_NC)과 src.no_chronos 가 만든 규칙(results/wk12_no_chronos/router_config.json)."""
        rt.load_env()
        self.h, self.variant = horizon, variant
        self.data = bm.load_data()
        self.weather, self.site = bm.load_weather(self.data)
        self.state = pd.read_csv(ROOT / "results/wk10/state_signals.csv")  # 10주차 상태 수집 노드의 산출(52창 × 17건물)
        if variant == "original":
            self.pool = rt.Pool(self.data, horizon)
            lr = pd.read_csv(ROOT / "results/wk10/learned_rules.csv")
            lr = lr[lr["horizon"] == horizon].sort_values("순서")
            self.rules = [dict(signal=r.signal, op=r.op, thr=r.thr, model=r.model) for r in lr.itertuples()]
            self.default = rt.POOL[4]  # 앞 26창으로 정한 최고 단일 모델(Chronos-2 + 공변량)
            pred = None  # 근거 표의 기준 예측 = 기본값 모델 (11주차 그대로)
        else:
            self.pool = rt.Pool(self.data, horizon, models=rt.POOL_NC)
            cfg = json.loads((ROOT / "results/wk12_no_chronos/router_config.json").read_text(encoding="utf-8"))[str(horizon)]
            self.default, self.rules = cfg["default"], cfg["rules"]
            choice = rt.apply_rules(self.state, self.rules, self.default)[0].reshape(self.pool.W, -1)
            # 근거 표의 기준 예측 = 라우터가 칸마다 고른 모델의 예측 (예측구간 잔차도 그 기준)
            pred = np.stack([[self.pool.P[choice[w, j]][w, j] for j in range(choice.shape[1])] for w in range(self.pool.W)])
        df = cr.build_frame(self.pool, self.data, self.weather, self.site, self.state, pred=pred)
        self.df, self.temp_q = cr.add_categories(df, self.data, self.weather, self.pool.origin_ts[26])
        self.hol_days = USFederalHolidayCalendar().holidays(self.data.index.min(), self.data.index.max())
        self.recent_q = self.state[self.state["window"] < 26]["recent_naive_mase"].quantile([.5, .75, .95]).tolist()

    # ── 입력 해석 ────────────────────────────────────────────────────────────
    def locate(self, building, time):
        if building not in self.pool.buildings:
            raise ValueError(f"건물 ID 없음: {building}. 가능: {self.pool.buildings}")
        t = pd.Timestamp(time)
        for w in range(26, self.pool.W):
            o = self.pool.origin_ts[w]
            if o <= t < o + pd.Timedelta(hours=self.h):
                return w, int((t - o) / pd.Timedelta(hours=1))
        ok = [str(self.pool.origin_ts[w]) for w in (26, 27, 28)]
        raise ValueError(f"{time} 은(는) 평가 윈도우 밖입니다. {self.h}시간 윈도우는 {ok[0]}, {ok[1]}, ... (매주, 마지막 {self.pool.origin_ts[-1]}) 부터 시작합니다.")

    # ── 파이프라인 ───────────────────────────────────────────────────────────
    def route(self, w, building):
        s = self.state[(self.state["window"] == w) & (self.state["building"] == building)]
        model, why = rt.apply_rules(s, self.rules, self.default)
        return s.iloc[0], model[0], why[0]

    def matched_rule(self, sig):
        """apply_rules 와 같은 순서·비교로, 적용된 규칙(+신호 값)을 돌려준다. 기본값이면 None (쉬운 말 근거용)."""
        for r in self.rules:
            v = float(sig[r["signal"]])
            if (v <= r["thr"]) if r["op"] == "<=" else (v > r["thr"]):
                return dict(r, value=v)
        return None

    def forecast_frame(self, w, building, model):
        j = self.pool.buildings.index(building)
        fit_res = self.df[(self.df["window"] < 26) & (self.df["building"] == building)]["resid"]
        lo, hi = np.nanquantile(fit_res, [.1, .9])  # 앞 26창 잔차에 결측이 섞이면 quantile 이 NaN → nan 무시
        pred = self.pool.P[model][w, j]
        ts = self.pool.origin_ts[w] + pd.to_timedelta(np.arange(self.h), unit="h")
        return pd.DataFrame({"timestamp": ts, "실측": self.pool.Y[w, j], "예측": pred, "구간 하한": pred + lo, "구간 상한": pred + hi})

    def report(self, building, time, use_llm=True, style="plain"):
        """style='plain': 일반인용 쉬운 말 근거(plain_report / PROMPTS v3). style='legacy': 11·12주차 정형 문장(template_report / PROMPTS v2)."""
        w, step = self.locate(building, time)
        sig, model, why = self.route(w, building)
        fc = self.forecast_frame(w, building, model)
        # 근거 구조화는 10주차 기본값(Chronos-2 + 공변량) 예측을 기준으로 한 11주차 스키마를 그대로 사용 (라우터가 다른 모델을 고르면 route_log 에 그 모델이 기록됨)
        ctx = dict(pool=self.pool, df=self.df, data=self.data, weather=self.weather, site=self.site, recent_q=self.recent_q, temp_q=self.temp_q,
                   hol_days=self.hol_days, route_log=pd.DataFrame([{"window": w, "building": building, "model": model, "reason": why}]))
        evidence = ex.build_evidence(dict(window=w, building=building, step=step), ctx)
        if self.variant != "original":
            evidence["shap_note"]["explained_model"] = "LightGBM 대리 모델(예측 모델 자체가 아님)"
        elif model != self.default:
            evidence["prediction"]["value"] = round(float(fc["예측"].iloc[step]), 1)  # 선택된 모델의 값으로 교체(구간은 기본값 모델 잔차 기준이므로 같이 교체)
            evidence["prediction"]["interval"] = [round(float(fc["구간 하한"].iloc[step]), 1), round(float(fc["구간 상한"].iloc[step]), 1)]
        if style == "plain":
            ex.add_plain(evidence, ctx, w, building, self.matched_rule(sig))
        prompt, template, check = (ex.PROMPTS["v3"], ex.plain_report, ex.verify_plain) if style == "plain" else (ex.PROMPTS["v2"], ex.template_report, ex.verify)
        text, source, attempts, ver = None, "template", 0, None
        if use_llm:
            import os
            if os.environ.get("OPENAI_API_KEY"):
                for attempts in range(1, MAX_RETRY + 2):
                    o = ex.call_llm(prompt, evidence, max_tokens=900)
                    ver = check(o["text"], evidence)
                    if ver["n_mismatch"] == 0 and ver["n_unsupported"] == 0:
                        text, source = o["text"], "llm"
                        break
        if text is None:  # LLM 미사용 / 키 없음 / 검증 계속 실패 → 정형 문장으로 대체
            text = template(evidence)
            ver = check(text, evidence)
        return {"building": building, "forecast_time": str(pd.Timestamp(time)), "window": w, "step": step,
                "route": {"model": model, "reason": why, "signals": {k: (round(float(sig[k]), 3)) for k in rt.SIGNALS}},
                "anomaly_status": evidence["anomaly_status"], "forecast": fc, "evidence": evidence,
                "explanation": {"text": text, "source": source, "llm_attempts": attempts if use_llm else 0},
                "verification": {k: ver[k] for k in ["n_numbers", "n_match", "n_mismatch", "n_unsupported", "mismatch_items", "unsupported_claims"]}}


def format_report(r):
    p, a = r["evidence"]["prediction"], r["anomaly_status"]
    v = r["verification"]
    return "\n".join([
        f"# 예측 리포트 — {r['building']} @ {r['forecast_time']}",
        f"- 예측값: {p['value']} kWh (구간 {p['interval'][0]} ~ {p['interval'][1]})",
        f"- 선택 모델: {r['route']['model']} (이유: {r['route']['reason']})",
        f"- 이상 상태: {'이상' if a['is_anomaly'] else '정상'} (지난주 변동이 평소의 {a['recent_naive_mase']}배, 기준 {ex.ANOMALY_X}배 초과면 이상)",
        f"## 설명 ({r['explanation']['source']}, LLM 시도 {r['explanation']['llm_attempts']}회)", "", r["explanation"]["text"], "",
        f"## 검증: 수치 {v['n_numbers']}개 중 일치 {v['n_match']} / 불일치 {v['n_mismatch']}, 근거 없는 주장 {v['n_unsupported']}"
        + (f" {v['unsupported_claims']}" if v["n_unsupported"] else "")])


def main(argv=None):
    ap = argparse.ArgumentParser(description="건물 ID + 시점 → 예측 리포트")
    ap.add_argument("--building", required=True)
    ap.add_argument("--time", required=True, help='예: "2017-08-27 12:00"')
    ap.add_argument("--horizon", type=int, default=24, choices=[24, 168])
    ap.add_argument("--no-llm", action="store_true", help="LLM 호출 없이 정형 문장으로 설명")
    ap.add_argument("--json", help="전체 결과를 JSON 으로 저장할 경로")
    ap.add_argument("--variant", default="original", choices=["original", "no_chronos"], help="no_chronos: 파운데이션 모델 제외 풀")
    ap.add_argument("--style", default="plain", choices=["plain", "legacy"], help="legacy: 11·12주차 정형 문장(JSON 값 나열)")
    a = ap.parse_args(argv)
    r = System(a.horizon, a.variant).report(a.building, a.time, use_llm=not a.no_llm, style=a.style)
    print(format_report(r))
    if a.json:
        out = {k: v for k, v in r.items() if k != "forecast"}
        out["forecast"] = r["forecast"].assign(timestamp=r["forecast"]["timestamp"].astype(str)).to_dict("records")
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
