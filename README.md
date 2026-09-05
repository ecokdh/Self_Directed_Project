# 시계열 예측·이상진단 현장실습

아주대학교 현장실습 — 시계열 예측 트랙 (지도 주체: (주)블루웍스)

전체 커리큘럼은 [학생용_교재.md](./학생용_교재.md) 참조.

## 환경

- Python 3.11 (conda 환경 이름: `Self_directed_Project`)
- 패키지: `requirements_stage1.txt` (1단계, 1~6주차) / `requirements_stage2.txt` (2단계, 7~12주차, 서버에서 별도 안내)
- 모든 파일 입출력은 `encoding="utf-8"` 명시 (한글 Windows 기본 인코딩이 cp949이므로)

## 폴더 구조

```
Data/
├── raw/          # 원본 데이터. 가공하지 않음
│   ├── bdg2/     # metadata.csv, electricity.csv 등, weather.csv
│   └── lead/     # lead1.0-small.csv
└── processed/    # 서브셋 선정 결과 등 가공 산출물

src/              # 주차 간 재사용 코드 (하네스, 지표, 피처 파이프라인)
notebooks/        # wk{주차2자리}_{주제}.ipynb
results/          # 모든 실험 결과 CSV
reports/          # 주간 보고서 (부록 A 양식)
```

## 데이터 경로·재현 절차

(1주차 진행하면서 채워 넣을 것: 데이터 입수 경로, 받은 날짜, 파일 크기, 실행 순서)
