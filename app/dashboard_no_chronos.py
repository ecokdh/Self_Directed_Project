"""파운데이션 모델(Chronos-2, TTM) 제외 시나리오 대시보드. 먼저 `python -m src.no_chronos` 로 results/wk12_no_chronos/ 를 만든다."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from dashboard import render

render("no_chronos")
