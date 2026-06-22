"""
webapp/app.py — 가입 고객 이탈 예측 대시보드 (Streamlit 진입점)

실행:  streamlit run webapp/app.py

페이지(사이드바 그룹):
  개요  · 홈 / 개요
  실행  · 우선 대응 고객
  분석  · 분석 A · 세그먼트 / 분석 B · 위험속성 / 서브트랙 Q · 위험신호

산출물 인터페이스:
  segment_discovery/outputs/segment_rules.json  (경계·위험속성·검증 통계)
  churn_prediction/outputs/latest/              (학습된 모델이 있으면 이탈확률에 사용)
  data/WA_FnUseC_TelcoCustomerChurn.csv         (분포·집계)
"""
import sys
from pathlib import Path

import streamlit as st

# webapp/ 을 import 경로에 추가 (lib, views 패키지)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import theme as T  # noqa: E402
from lib import data as D  # noqa: E402
from views import overview, priority, analysis  # noqa: E402

st.set_page_config(page_title="고객 이탈 예측 대시보드",
                   page_icon="📉", layout="wide")
T.inject()

# ---- 사이드바: 브랜드 (nav 위) ----
st.sidebar.markdown('<div class="sb-brand">이탈 예측</div>', unsafe_allow_html=True)

# ---- 네비게이션 (그룹형) ----
PG_OVERVIEW = st.Page(overview.render, title="홈 / 개요", icon="🏠",
                      url_path="overview", default=True)
PG_PRIORITY = st.Page(priority.render, title="우선 대응 고객", icon="🎯",
                      url_path="priority")
PG_ANALYSIS = st.Page(analysis.render, title="분석", icon="🧭",
                      url_path="analysis")
pages = {
    "개요": [PG_OVERVIEW],
    "실행": [PG_PRIORITY],
    "분석": [PG_ANALYSIS],
}
# 홈의 진입 카드(st.page_link)가 참조할 수 있도록 Page 객체 노출
st.session_state["_pages"] = {"priority": PG_PRIORITY, "analysis": PG_ANALYSIS}
nav = st.navigation(pages, position="sidebar")

# ---- 사이드바: 푸터 (nav 아래) ----
try:
    _, meta = D.get_scored()
    n = meta["n"]
    src = "학습 모델(latest)" if meta["source"] == "trained" else "자체 학습(데모)"
except Exception:
    n, src = 7043, "—"
st.sidebar.markdown(
    f'<div class="sb-foot">데이터 <b>{n:,}건</b><br>'
    f'이탈확률 <b>{src}</b><br><br>'
    f'분석 규칙 &nbsp;<b>연 1회</b><br>'
    f'모델 재학습 &nbsp;<b>월 1회</b><br>'
    f'추론 &nbsp;<b>수시</b></div>',
    unsafe_allow_html=True,
)

nav.run()
