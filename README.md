# 시계열 예측·이상진단 현장실습

아주대학교 현장실습 — 시계열 예측 트랙 (지도 주체: (주)블루웍스)

건물 전력 수요를 24h / 168h 앞서 예측하고(Seasonal Naive · LightGBM · Chronos-2 · TTM · 딥러닝 4종), 이상 구간을 진단하며(CATCH · DADA · Sub-LOF · Matrix Profile · IsolationForest), 모델을 고르는 라우터와 예측 근거 설명(+자동 사실성 검증)까지 12주간 하나의 하네스 위에서 비교한 프로젝트.
전체 커리큘럼은 [학생용_교재_v2.md](./학생용_교재_v2.md), 결과 요약은 [results/wk12/최종보고서.md](./results/wk12/최종보고서.md) 참조.

## 환경

- Python 3.11 (conda 환경 이름: `Self_directed_Project`)
- 패키지: `requirements_stage1_v2.txt` (1단계, 1~6주차) / `requirements_stage2.txt` (2단계, 7~12주차, 서버에서 별도 안내). **버전을 임의로 올리지 말 것**
- 8주차: `Time-Series-Library`는 `requirements_stage2.txt`에 적힌 커밋으로 각자 클론(저장소에는 포함하지 않음). 9주차: `TSB-AD`, `CATCH`, `DADA` 폴더 사용
- 모든 파일 입출력은 `encoding="utf-8"` 명시 (한글 Windows 기본 인코딩이 cp949이므로)
- 10·11주차 LLM(gpt-4o-mini) 사용 시 프로젝트 루트에 `.env` (커밋 제외 대상):

```
OPENAI_API_KEY=...           # 값은 직접 입력. 코드·노트북에 쓰지 않음
OPENAI_MODEL=gpt-4o-mini
PRICE_IN_PER_M=0.15          # 달러/100만 토큰. 공식 단가표 확인 후 입력
PRICE_OUT_PER_M=0.60
```

키가 없으면 LLM 단계는 건너뛰고(10주차) 정형 문장으로 대체한다(12주차 파이프라인 `--no-llm`).

## 폴더 구조

```
Data/
├── raw/          # 원본 데이터. 가공하지 않음 (git 제외)
│   ├── bdg2/     # metadata.csv, weather.csv, meters/ (electricity.csv 등)
│   └── lead/     # lead1.0-small.csv
└── processed/    # subset_17_buildings_Impulation.csv (결측 보간본, 모든 예측 실험의 입력)

src/              # 주차 간 재사용 코드
│   ├── backtest.py, metrics.py       # 2주차 하네스 (rolling_backtest, MASE/score_backtest)
│   ├── features.py                   # 3주차 피처 파이프라인
│   ├── bench_models.py, measure_inference.py   # 6주차 모델 래퍼 · 측정
│   ├── finetune.py                   # 7주차 LoRA
│   ├── tslib_adapter.py              # 8주차 Time-Series-Library 어댑터
│   ├── anomaly_adapter.py            # 9주차 이상진단 어댑터
│   ├── routing.py                    # 10주차 풀 · 상태 신호 · 규칙/LLM 라우터
│   ├── correction.py, explain.py     # 11주차 오차 보정 · 근거 구조화 · 설명 · 검증기
│   ├── pipeline.py                   # 12주차 통합 파이프라인 (단일 진입점)
│   └── cumulative.py                 # 12주차 누적 비교표 생성
app/dashboard.py  # 12주차 Streamlit 대시보드 (dashboard_no_chronos.py: 파운데이션 모델 제외 시나리오)
notebooks/        # wk{주차2자리}_{주제}.ipynb
results/wk1~wk12/ # 모든 실험 결과 CSV/그림 + 주차별 보고서(주차별보고서_NN.md), 최종보고서(wk12)
```

## 데이터

| 파일 | 크기 | 파일 수정일(로컬) | 용도 |
|---|---|---|---|
| `Data/raw/bdg2/metadata.csv` | 272,024 B | 2026-09-01 | 건물 · 사이트 메타 |
| `Data/raw/bdg2/weather.csv` | 19,457,782 B | 2026-09-01 | 사이트별 외기온 (공변량) |
| `Data/raw/bdg2/meters/*` | — | 2026-09-08 | 전기 미터 시계열 |
| `Data/raw/lead/lead1.0-small.csv` | 57,680,722 B | 2022-04-27 (원본 파일 시각; zip은 2026-09-19 수령) | 이상 라벨 (2016년만) |

- BDG2: Building Data Genome Project 2, LEAD 1.0 — 교재 4~5절의 경로에서 받았다. 위 날짜는 파일 시스템의 수정일이며, 실제 내려받은 날짜는 다를 수 있으니 직접 확인해 기입할 것.
- 서브셋 17개 건물 선정 과정: [results/wk1/subset_selection_wk01.md](./results/wk1/subset_selection_wk01.md). 12주차까지 변경 없음.

## 재현 순서

모든 노트북은 `notebooks/`에서 열고 위에서 아래로 실행한다(경로는 `..` 기준).

| 주차 | 노트북 | 장비 | 산출물 |
|---|---|---|---|
| 1 | `wk01_bdg2_eda.ipynb` | CPU | 서브셋 17개, 보간본 |
| 2 | `wk02_bdg2_harness_construction.ipynb` | CPU | Seasonal Naive 기준선, 누수 실험 |
| 3~5 | `wk03_lightgbm`, `wk04_SHAP`, `wk05_Chronos-2 zero-shot` | CPU | LightGBM, SHAP, Chronos-2 예측 |
| 6 | `wk06_TTM_benchmark.ipynb` | CPU | 6축 판단표, 시나리오 |
| 7 | `wk07_LoRA.ipynb`, `wk07_analysis.ipynb` | **GPU 서버** | LoRA 체크포인트(저장소 제외), 오염 대조 |
| 8 | `wk08_Time-Series_Library.ipynb`, `wk08_analysis.ipynb` | **GPU 서버** | 딥러닝 4종, 민감도 |
| 9 | `wk09_anomaly_models.ipynb`, `wk09_analysis.ipynb` | GPU(CATCH · DADA) | 이상진단 6종, PA-F1 vs VUS-PR |
| 10 | `wk10_routing.ipynb` | CPU (+OpenAI API) | 라우터 비교 |
| 11 | `wk11_correction_explain.ipynb` | CPU (+OpenAI API 18회, 약 $0.004) | 보정 · 설명 · 검증 |
| 12 | 아래 명령 | CPU | 누적 표 · 파이프라인 · 대시보드 |

```bash
python -m src.cumulative                                     # results/wk12/cumulative_table.{csv,md}
python -m src.pipeline --building Bull_office_Sally --time "2017-11-26 18:00"   # 건물 ID + 시점 → 예측 리포트
python -m src.pipeline --building Fox_office_Vicki --time "2017-08-27 12:00" --no-llm
streamlit run app/dashboard.py                               # 대시보드 (streamlit 설치 필요)

# 민감도 분석: 파운데이션 모델(Chronos-2, TTM) 제외 풀 — 원래 결과는 그대로, results/wk12_no_chronos/ 에만 씀
python -m src.no_chronos                                     # 라우터 규칙 · 비교표 · 결정 로그
python -m src.pipeline --building Bull_office_Sally --time "2017-11-26 18:00" --no-llm --variant no_chronos
streamlit run app/dashboard_no_chronos.py                    # 별도 대시보드
```

- 근거 설명은 기본이 일반인용 쉬운 말(판정 → 왜 이상/정상인지 kWh로 → 예측 → 모델 선택 이유 → 영향 요인 → 참고)이다. 새로 쓰는 숫자도 근거 JSON의 `plain` 블록에 코드가 넣으므로 자동 검증이 그대로 적용된다. 11·12주차의 JSON 값 나열 문장은 `--style legacy`로 재현한다(저장된 `results/wk12/demo_*.json`과 같음).
- 파이프라인은 1~8주차에 저장된 윈도우별 예측을 조회하므로 입력 시각은 평가 구간(2017-07-09~, 24h는 매주 일요일 0시부터 24시간) 안이어야 한다.
- 결과 파일은 모두 UTF-8. 보고서의 수치는 `results/wk*/`의 CSV에서 나온 것이다.
