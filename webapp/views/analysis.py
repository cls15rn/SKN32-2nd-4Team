"""
분석 (통합) — A·B·Q를 한 페이지로.

구성:
  상단  · 분석 요약 스윕 (세그먼트·위험속성·위험신호 핵심을 한눈에)
  하단  · 세부 3개를 expander로 펼쳐 보기
          - 세그먼트 (분석 A)   : analysis_a.section()
          - 위험속성 (분석 B)   : analysis_b.section()
          - 위험신호 (서브트랙 Q): subtrack_q.section()

데이터·계산은 lib/data.py 그대로 재사용하고, 각 세부는 기존 분석 로직(section)을 호출만 한다.
"""
from __future__ import annotations

import streamlit as st

from lib import data as D
from lib import theme as T
from views import analysis_a, analysis_b, subtrack_q

# 요약 스윕에서 쓸 속성 한글 약칭 (분석 B top_attributes 표시용)
_ATTR_KR = {
    "MonthlyCharges": "월요금",
    "InternetService": "인터넷",
    "Contract": "계약형태",
    "OnlineSecurity": "온라인보안",
    "TechSupport": "기술지원",
    "PaymentMethod": "결제수단",
}


def _summary(df, rules):
    """상단 요약 스윕 — A/B/Q 핵심을 3개 카드로 나란히."""
    prof = D.segment_profile(df)
    dist = D.risk_count_distribution(df)
    mean_rate = df["churn"].mean()
    ab = {item["segment"]: item for item in rules["analysis_b"]}

    line_style = 'style="font-size:.86rem; line-height:1.9; color:#2b2b2b"'

    # A — 세그먼트별 이탈률 한 줄
    seg_line = " · ".join(
        f'{r["name"]} <b>{r["rate"]*100:.0f}%</b>' for _, r in prof.iterrows())
    card_a = (T.card_title("세그먼트 (A)", "생애주기 4구간 · 이탈률 단조 감소")
              + f'<div {line_style}>{seg_line}</div>')

    # B — 세그먼트별 핵심 위험속성
    b_lines = []
    for _, r in prof.iterrows():
        seg = int(r["segment"])
        tops = ab.get(seg, {}).get("top_attributes", [])
        kr = "·".join(_ATTR_KR.get(c, c) for c in tops) if tops else "—"
        b_lines.append(f'{r["name"]} <b>{kr}</b>')
    card_b = (T.card_title("위험속성 (B)", "구간마다 핵심 요인이 다름")
              + f'<div {line_style}>' + "<br>".join(b_lines) + "</div>")

    # Q — risk_count 상승 요약
    r5 = dist[dist["risk_count"] == 5]["rate"].iloc[0] * 100
    mult = r5 / (mean_rate * 100)
    card_q = (T.card_title("위험신호 (Q)", "신호가 쌓일수록 이탈 급증")
              + f'<div {line_style}>신호 0~2개 <b>3~8%</b> → 5개 <b>{r5:.0f}%</b><br>'
              + f'최고위험 = 전체 평균의 <b>{mult:.1f}배</b></div>')

    c1, c2, c3 = st.columns(3)
    with c1:
        T.html(T.card(card_a))
    with c2:
        T.html(T.card(card_b))
    with c3:
        T.html(T.card(card_q))


def render():
    df, _ = D.get_active_df()
    rules = D.load_rules()

    T.html(T.page_header(
        "분석 · 세그먼트 · 위험속성 · 위험신호",
        "전체 데이터를 한눈에 훑은 뒤, 아래에서 각 분석을 펼쳐 자세히 봅니다."))

    # 상단: 요약 스윕
    _summary(df, rules)

    # 하단: 세부 (펼쳐 보기)
    with st.expander("세그먼트 · 생애주기 구간 도출 (분석 A)", expanded=True):
        analysis_a.section(df, rules)
    with st.expander("위험속성 · 세그먼트별 핵심 요인 (분석 B)", expanded=True):
        analysis_b.section(df, rules)
    with st.expander("위험신호 · 누적 위험 정량화 (서브트랙 Q)", expanded=True):
        subtrack_q.section(df, rules)
