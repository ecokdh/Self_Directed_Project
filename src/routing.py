"""10주차: 예측기 풀 · 통일 인터페이스 · 상태 신호 · 규칙/LLM 라우터 · LangGraph 그래프.

설계
- 라우터는 (윈도우, 건물)마다 풀에서 예측기 하나를 고른다. 모든 후보의 윈도우별 예측이 이미 저장돼 있어(1~8주차),
  평가는 '고른 예측을 조회'해 2주차 하네스(score_backtest)로 채점한다. 학습·추론을 다시 돌리지 않는다.
- 누수 방지: 상태 신호는 윈도우 origin 이전 데이터로만 계산. 규칙·LLM 프롬프트의 근거는 앞 26창(FIT_W)에서만 만들고,
  평가는 뒤 26창(EVAL_W)에서만 한다 (규칙을 정한 창으로 평가하면 2주차에 배운 누수와 같은 문제).
"""
import time
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import pandas as pd

from src.metrics import compute_naive_denominators, score_backtest

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
POOL = ["Seasonal Naive", "LightGBM (local)", "LightGBM (global)", "Chronos-2", "Chronos-2 + 공변량", "TTM (ctx=512)"]
HORIZONS = (24, 168)
FIT_W, EVAL_W = np.arange(0, 26), np.arange(26, 52)  # 앞 26창 = 규칙 근거, 뒤 26창 = 평가
SIGNALS = ["hist_len", "miss_28d", "has_weather", "acf168", "acf24", "recent_naive_mase"]


def pred_path(name, h):
    w10 = RES / "wk10/preds"
    return {
        "Seasonal Naive": RES / f"wk2/backtest_expanding_{h}h_w52_step168.csv",
        "LightGBM (global)": RES / f"wk3/global_{h}h_w52_step168.csv",
        "Chronos-2": RES / f"wk5/chronos_{h}h_zero.csv",
        "Chronos-2 + 공변량": RES / f"wk5/chronos_{h}h_cov.csv",
        "LightGBM (local)": w10 / f"lightgbm_local_{h}h.csv",
        "TTM (ctx=512)": w10 / f"ttm512_{h}h.csv",
    }[name]


def latency_ms():
    """모델별 추론 지연(ms/계열) — 6주차 측정값(4스레드 CPU)."""
    t = pd.read_csv(RES / "wk6/benchmark_table.csv", index_col=0)
    return t["추론 ms/계열(4스레드)"].reindex(POOL).to_dict()


def make_missing_preds(data, n_windows=52, step_size=168):
    """윈도우별 예측이 없는 LightGBM(local)·TTM(512)을 2주차 하네스로 만들어 저장 (6주차와 같은 설정)."""
    from src import bench_models as bm
    from src.backtest import rolling_backtest
    out = RES / "wk10/preds"
    out.mkdir(parents=True, exist_ok=True)
    for h in HORIZONS:
        for name, mk in [("LightGBM (local)", lambda h: bm.make_local_fns(h)), ("TTM (ctx=512)", lambda h: bm.make_ttm_fns(512))]:
            p = pred_path(name, h)
            if not p.exists():
                mf, pf = mk(h)
                rolling_backtest(data, mf, pf, n_windows=n_windows, horizon=h, train_mode="expanding",
                                 step_size=step_size).to_csv(p, index=False, encoding="utf-8")


class Pool:
    """한 horizon 의 후보 예측 전체: P[모델] 모양 (윈도우, 건물, h), 정답 Y 같은 모양."""

    def __init__(self, data, h):
        self.h, self.buildings = h, list(data.columns)
        arrs = {}
        for name in POOL:
            df = pd.read_csv(pred_path(name, h), parse_dates=["timestamp"])
            df["building"] = pd.Categorical(df["building"], categories=self.buildings, ordered=True)
            df = df.sort_values(["window", "building", "timestamp"])
            W = df["window"].nunique()
            arrs[name] = (df["y_true"].to_numpy().reshape(W, len(self.buildings), h), df["y_pred"].to_numpy().reshape(W, len(self.buildings), h))
            if name == POOL[0]:
                self.origin_ts = pd.DatetimeIndex(df.groupby("window")["timestamp"].min().to_numpy())
        self.Y = arrs[POOL[0]][0]
        for n, (y, _) in arrs.items():
            assert np.allclose(y, self.Y, equal_nan=True), f"{n}: 정답이 다름"
        self.P = {n: p for n, (_, p) in arrs.items()}
        self.W = self.Y.shape[0]
        self.denom = np.array(list(compute_naive_denominators(data, data.index.get_loc(self.origin_ts[0]), 168).values()))

    def cell_mase(self):
        """모델별 (윈도우, 건물) 칸의 MASE: 그 창 평균 절대오차 / 건물별 Naive 분모 (하네스와 같은 분모)."""
        return {n: np.nanmean(np.abs(self.Y - p), axis=2) / self.denom[None, :] for n, p in self.P.items()}

    def tidy(self, choice, windows):
        """choice: (윈도우, 건물) 모델 이름 배열 → 하네스 입력 형식(window, building, timestamp, y_true, y_pred)."""
        rows = []
        for w in windows:
            ts = pd.date_range(self.origin_ts[w], periods=self.h, freq="h")
            for j, b in enumerate(self.buildings):
                rows.append(pd.DataFrame({"window": w, "building": b, "timestamp": ts, "y_true": self.Y[w, j], "y_pred": self.P[choice[w, j]][w, j]}))
        return pd.concat(rows, ignore_index=True)

    def score(self, choice, windows):
        """하네스 score_backtest 로 채점 → 건물 평균 MASE, 건물별 표."""
        pb, overall = score_backtest(self.tidy(choice, windows), dict(zip(self.buildings, self.denom)))
        return overall["MASE_평균"], pb.set_index("building")["MASE"]


def load_env(path=ROOT / ".env"):
    """프로젝트 루트의 .env 를 환경변수로 읽는다(값이 비어 있거나 이미 설정된 변수는 건드리지 않음). 키는 코드·노트북에 쓰지 않는다."""
    import os
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and not k.startswith("#") and v and k not in os.environ:
                os.environ[k] = v


# ── 10-1. 통일 인터페이스 ────────────────────────────────────────────────────
class Forecaster:
    """(학습 데이터, horizon) → 예측 DataFrame(h × 건물). 라우터는 어떤 예측기든 이 한 가지 방식으로 호출한다.
    backend='cached': 저장된 윈도우별 예측을 origin 시각으로 조회 / 'live': bench_models 의 실제 모델을 호출."""

    def __init__(self, name, backend="cached", data=None, pools=None):
        self.name, self.backend, self.latency = name, backend, latency_ms()[name]
        if backend == "cached":
            self._cache = {h: pools[h] for h in pools}
        else:
            self._live = {}
            self.data = data

    def __call__(self, train_df, horizon):
        t0 = train_df.index[-1] + pd.Timedelta(hours=1)
        if self.backend == "cached":
            pool = self._cache[horizon]
            w = int(np.flatnonzero(pool.origin_ts == t0)[0])
            return pd.DataFrame(pool.P[self.name][w].T, index=pd.date_range(t0, periods=horizon, freq="h"), columns=pool.buildings)
        if (self.name, horizon) not in self._live:
            self._live[(self.name, horizon)] = self._build(horizon)
        mf, pf = self._live[(self.name, horizon)]
        return pf(mf(train_df), horizon)

    def _build(self, h):
        from src import bench_models as bm
        n = self.name
        if n == "Seasonal Naive":
            return bm.make_naive_fns()
        if n == "LightGBM (local)":
            return bm.make_local_fns(h)
        if n == "LightGBM (global)":
            return bm.make_global_fns(h)
        if n == "TTM (ctx=512)":
            return bm.make_ttm_fns(512)
        weather, site = bm.load_weather(self.data)
        return bm.make_chronos_fns(1024, n.endswith("공변량"), weather, site)


# ── 10-2. 상태 수집 ──────────────────────────────────────────────────────────
def weather_ok(data, origin_ts):
    """건물의 사이트 외기온이 origin 직전 28일에 얼마나 있는가(0~1). 공변량 유무 신호."""
    meta = pd.read_csv(ROOT / "Data/raw/bdg2/metadata.csv").set_index("building_id")["site_id"]
    w = pd.read_csv(ROOT / "Data/raw/bdg2/weather.csv", parse_dates=["timestamp"])
    wide = w.pivot_table(index="timestamp", columns="site_id", values="airTemperature").reindex(data.index)
    sites = [meta[b] for b in data.columns]
    return {t: wide[sites].loc[t - pd.Timedelta(hours=672):t - pd.Timedelta(hours=1)].notna().mean().to_numpy() for t in origin_ts}


def _acf(x, lag):
    if len(x) <= lag + 2 or np.std(x) < 1e-9:
        return 0.0
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def state_table(data, pool):
    """(윈도우, 건물)마다 라우팅 신호 5종(+주기 2개). origin 이전 데이터만 사용."""
    X, wok, rows = data.to_numpy(float), weather_ok(data, pool.origin_ts), []
    for w, t in enumerate(pool.origin_ts):
        o = data.index.get_loc(t)
        for j, b in enumerate(pool.buildings):
            s = X[:o, j]
            x = pd.Series(s[-1344:]).ffill().bfill().to_numpy()  # 직전 8주
            rows.append({"window": w, "building": b, "hist_len": int((~np.isnan(s)).sum()), "miss_28d": float(np.isnan(s[-672:]).mean()),
                         "has_weather": float(wok[t][j]), "acf168": _acf(x, 168), "acf24": _acf(x, 24),
                         "recent_naive_mase": float(np.nanmean(np.abs(s[-168:] - s[-336:-168])) / pool.denom[j])})  # 직전 1주 Naive 오차 = 최근 잔차 수준
    return pd.DataFrame(rows)


# ── 10-3. 규칙 기반 라우터 ───────────────────────────────────────────────────
def check_rule(S, E, mask, target, default):
    """조건에 걸린 칸에서 target 이 default 보다 얼마나 나은가 (앞 26창 근거)."""
    n = int(mask.sum())
    if n == 0:
        return {"발동 칸 수": 0, "default MASE": np.nan, "target MASE": np.nan, "개선율(%)": np.nan}
    d, t = E[default][mask].mean(), E[target][mask].mean()
    return {"발동 칸 수": n, "default MASE": d, "target MASE": t, "개선율(%)": (d - t) / d * 100}


def fit_rules(S, E, default, max_rules=3, min_support=60, min_rel_gain=0.03):
    """앞 26창 칸에서 '신호 임계값 → 대체 모델' 규칙을 탐욕적으로 찾는다. 통과 조건: 지원 칸 수·개선율·앞뒤 절반 모두 개선."""
    models, win = [m for m in E if m != default], S["window"].to_numpy()
    early = win < win.min() + (win.max() - win.min() + 1) // 2
    remaining, rules = np.ones(len(S), bool), []
    for _ in range(max_rules):
        best = None
        for sig in SIGNALS:
            v = S[sig].to_numpy()
            if len(np.unique(v)) < 5:
                continue
            for q in (.2, .35, .5, .65, .8):
                thr = float(np.quantile(v, q))
                for op in ("<=", ">"):
                    m = remaining & ((v <= thr) if op == "<=" else (v > thr))
                    if m.sum() < min_support or (m & early).sum() < 10 or (m & ~early).sum() < 10:
                        continue
                    for alt in models:
                        d = E[default][m] - E[alt][m]
                        rel = d.mean() / E[default][m].mean()
                        r1, r2 = d[early[m]].mean() / E[default][m & early].mean(), d[~early[m]].mean() / E[default][m & ~early].mean()
                        if rel >= min_rel_gain and r1 >= 0.01 and r2 >= 0.01 and (best is None or d.sum() > best["gain_sum"]):
                            best = dict(signal=sig, op=op, thr=thr, model=alt, support=int(m.sum()), rel_gain=float(rel),
                                        rel_gain_first=float(r1), rel_gain_second=float(r2), gain_sum=float(d.sum()), mask=m)
        if best is None:
            break
        remaining &= ~best.pop("mask")
        rules.append(best)
    return rules


def apply_rules(S, rules, default):
    """S 의 각 칸에 첫 번째로 맞는 규칙의 모델, 없으면 default. → (모델, 근거 문자열) 배열"""
    model, why = np.full(len(S), default, dtype=object), np.full(len(S), f"기본값 {default}", dtype=object)
    done = np.zeros(len(S), bool)
    for i, r in enumerate(rules):
        v = S[r["signal"]].to_numpy()
        m = ~done & ((v <= r["thr"]) if r["op"] == "<=" else (v > r["thr"]))
        model[m], why[m] = r["model"], f"규칙{i + 1}: {r['signal']} {r['op']} {r['thr']:.3g}"
        done |= m
    return model, why


# ── 10-4. LLM 라우터 (구조화 출력) ───────────────────────────────────────────
def llm_schema():
    from pydantic import BaseModel, Field

    class Decision(BaseModel):
        building: str
        model: Literal[tuple(POOL)]
        reason: str = Field(description="한 문장, 제공된 수치만 인용")

    class Decisions(BaseModel):
        items: list[Decision]

    return Decisions


def llm_prompt(rows, evidence):
    """rows: 한 윈도우의 건물별 상태. evidence: 앞 26창에서 모델별 평균 MASE·지연 표(문자열)."""
    head = ("당신은 건물 에너지 수요 예측기 라우터입니다. 각 건물의 상태 신호를 보고 후보 중 예측기 하나를 고르세요.\n"
            "아래 '근거 표'는 과거 검증 창에서 측정된 값입니다. 제공된 수치만 사용하고 추측하지 마세요.\n\n"
            f"[근거 표: 모델별 평균 MASE(낮을수록 좋음) · 지연(ms/계열)]\n{evidence}\n\n[건물별 상태]\n")
    body = "\n".join(f"- {r.building}: 이력 {r.hist_len}h, 최근28일 결측률 {r.miss_28d:.3f}, 외기온 가용률 {r.has_weather:.2f}, "
                     f"주간 자기상관 {r.acf168:.2f}, 일간 자기상관 {r.acf24:.2f}, 직전 1주 Naive 오차(MASE) {r.recent_naive_mase:.2f}" for r in rows.itertuples())
    return head + body + f"\n\n모든 건물({len(rows)}개)에 대해 building, model, reason 을 JSON 으로 답하세요."


def llm_route_window(llm, rows, evidence):
    """한 윈도우(건물 17개)를 LLM 한 번 호출로 결정. → (선택 dict, 입력 토큰, 출력 토큰, 지연 s, 파싱 실패 수)"""
    structured = llm.with_structured_output(llm_schema(), include_raw=True)
    t0 = time.perf_counter()
    out = structured.invoke(llm_prompt(rows, evidence))
    dt = time.perf_counter() - t0
    usage = getattr(out["raw"], "usage_metadata", None) or {}
    parsed = out["parsed"].items if out["parsed"] is not None else []
    return ({d.building: (d.model, d.reason) for d in parsed}, usage.get("input_tokens", 0), usage.get("output_tokens", 0), dt,
            int(out["parsed"] is None))


# ── 상태 수집 → 라우터 → 예측기 → 결과 수집 (LangGraph) ──────────────────────
class RouteState(TypedDict, total=False):
    window: int
    states: pd.DataFrame
    choices: dict
    route_ms: float
    log: pd.DataFrame


def build_graph(state_df, router_fn):
    """교재의 그래프: [상태 수집] → [라우터] → [예측기 호출은 조회] → [결과 수집·로그].
    router_fn(states: 한 윈도우의 상태 표) -> ({건물: (모델, 근거)}, route_ms)"""
    from langgraph.graph import END, START, StateGraph

    def collect_state(s):
        return {"states": state_df[state_df["window"] == s["window"]].reset_index(drop=True)}

    def route(s):
        choices, ms = router_fn(s["states"])
        return {"choices": choices, "route_ms": ms}

    def collect(s):
        st = s["states"].copy()
        st["model"] = [s["choices"][b][0] for b in st["building"]]
        st["reason"] = [s["choices"][b][1] for b in st["building"]]
        st["route_ms"] = s["route_ms"] / len(st)
        return {"log": st}

    g = StateGraph(RouteState)
    g.add_node("state", collect_state)
    g.add_node("route", route)
    g.add_node("collect", collect)
    g.add_edge(START, "state")
    g.add_edge("state", "route")
    g.add_edge("route", "collect")
    g.add_edge("collect", END)
    return g.compile()


def run_graph(graph, windows):
    """윈도우마다 그래프 실행 → 결정 로그(window, building, 신호들, model, reason, route_ms)."""
    return pd.concat([graph.invoke({"window": int(w)})["log"] for w in windows], ignore_index=True)


def log_to_choice(log, pool):
    ch = np.full((pool.W, len(pool.buildings)), POOL[0], dtype=object)
    idx = {b: j for j, b in enumerate(pool.buildings)}
    for r in log.itertuples():
        ch[r.window, idx[r.building]] = r.model
    return ch
