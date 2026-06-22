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

import pandas as pd
import streamlit as st

# webapp/ 을 import 경로에 추가 (lib, views 패키지)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import theme as T  # noqa: E402
from lib import data as D  # noqa: E402
from views import overview, priority, analysis, whatif, roi  # noqa: E402


def _sidebar_datasource() -> None:
    """사이드바 데이터소스 전환 UI.

    실행 순서가 중요:
    1) 파일 처리 먼저 → session_state 갱신
    2) 그 후 라디오가 최신 state를 읽어 렌더
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div style="font-size:.8rem;font-weight:700;color:#667085;'
        'text-transform:uppercase;letter-spacing:.04em;margin-bottom:.5rem">'
        '데이터 소스</div>',
        unsafe_allow_html=True)

    # 1) CSV 파일 업로드 — 라디오보다 먼저 실행해 session_state를 갱신한다
    csv_file = st.sidebar.file_uploader(
        "📂 CSV 업로드",
        type=["csv"],
        key="sidebar_csv_upload",
        help="IBM Telco 형식 CSV (21개 컬럼 필수)",
    )
    if csv_file is not None:
        file_id = f"{csv_file.name}_{csv_file.size}"
        if st.session_state.get("uploaded_file_id") != file_id:
            try:
                raw = pd.read_csv(csv_file)
                with st.spinner("업로드 데이터 처리 중…"):
                    df_up, err = D.score_uploaded_csv(raw)
                if df_up is not None:
                    st.session_state["uploaded_df"] = df_up
                    st.session_state["uploaded_file_id"] = file_id
                    st.session_state["use_uploaded"] = True
                else:
                    st.sidebar.error(f"⚠️ {err}")
            except Exception as e:
                st.sidebar.error(f"CSV 읽기 실패: {e}")

    # 2) 업로드된 데이터 확인 (파일 처리 후 최신 state 읽기)
    uploaded_df = st.session_state.get("uploaded_df", None)
    up_label = (f"업로드 데이터 ({len(uploaded_df):,}명)"
                if uploaded_df is not None else "업로드 데이터 (없음)")

    # 3) 데이터 소스 라디오 — 최신 uploaded_df 기준으로 렌더
    src_idx = 1 if (st.session_state.get("use_uploaded", False)
                    and uploaded_df is not None) else 0
    src_sel = st.sidebar.radio(
        "데이터 소스 선택",
        ["기본 데이터 (7,043명)", up_label],
        index=src_idx,
        key="sidebar_src_radio",
        label_visibility="collapsed",
        disabled=(uploaded_df is None),
    )
    st.session_state["use_uploaded"] = (
        src_sel == up_label and uploaded_df is not None
    )

    # 4) 업로드 상태 표시 + 초기화
    if uploaded_df is not None:
        fid   = st.session_state.get("uploaded_file_id", "")
        fname = fid.rsplit("_", 1)[0] if "_" in fid else "업로드됨"
        st.sidebar.caption(f"📄 {fname} · {len(uploaded_df):,}명")
        if st.sidebar.button("✕ 업로드 데이터 초기화", key="sidebar_clear_upload"):
            st.session_state.pop("uploaded_df", None)
            st.session_state.pop("uploaded_file_id", None)
            st.session_state["use_uploaded"] = False


st.set_page_config(page_title="고객 이탈 예측 시스템",
                   page_icon="📉", layout="wide")
T.inject()

# ---- 사이드바: 브랜드 (nav 위) ----
st.sidebar.markdown(
    '<div class="sb-brand">👥 고객 이탈 예측 시스템'
    '<span class="sb-sub">Customer Churn Prediction</span></div>',
    unsafe_allow_html=True)

# ---- 네비게이션 (그룹형) ----
PG_OVERVIEW = st.Page(overview.render, title="홈 / 개요", icon="🏠",
                      url_path="overview", default=True)
PG_PRIORITY = st.Page(priority.render, title="우선 대응 고객", icon="🎯",
                      url_path="priority")
PG_ANALYSIS = st.Page(analysis.render, title="분석", icon="🧭",
                      url_path="analysis")
PG_WHATIF = st.Page(whatif.render, title="위험 고객 맞춤 프로모션 시뮬레이터", icon="🔮",
                    url_path="whatif")
PG_ROI = st.Page(roi.render, title="이탈 방어 비용 효과 분석", icon="💹",
                 url_path="roi")
pages = {
    "개요": [PG_OVERVIEW],
    "실행": [PG_PRIORITY],
    "분석": [PG_ANALYSIS],
    "시뮬레이션": [PG_WHATIF, PG_ROI],
}
# 홈의 진입 카드(st.page_link)가 참조할 수 있도록 Page 객체 노출
st.session_state["_pages"] = {"priority": PG_PRIORITY, "analysis": PG_ANALYSIS, "whatif": PG_WHATIF, "roi": PG_ROI}


nav = st.navigation(pages, position="sidebar")

# ---- 사이드바: 모델 요약 + 푸터 (nav 아래) ----
try:
    _, meta = D.get_active_df()
    n = meta["n"]
    src = "학습 모델(latest)" if meta["source"] == "trained" else "자체 학습(데모)"
except Exception:
    n, src = 7043, "—"

st.sidebar.markdown(
    '<div class="sb-model">'
    '<div class="m-h">모델 요약</div>'
    '<div class="m-row">· Best Model <b>XGBoost</b><br>'
    '· CV Score (F1) <b>0.727</b><br>'
    '· 주요 Feature<br>'
    '&nbsp;&nbsp;계약기간 · 월정액 요금<br>'
    '&nbsp;&nbsp;기술지원 · 온라인 보안</div></div>',
    unsafe_allow_html=True)

st.sidebar.markdown(
    f'<div class="sb-foot">데이터 <b>{n:,}건</b> · 이탈확률 <b>{src}</b><br>'
    f'분석 규칙 <b>연 1회</b> · 모델 <b>월 1회</b> · 추론 <b>수시</b>'
    f'<br><br>SKN32 2nd 4Team</div>',
    unsafe_allow_html=True)

# ---- 사이드바: 데이터 소스 전환 ----
_sidebar_datasource()

nav.run()
