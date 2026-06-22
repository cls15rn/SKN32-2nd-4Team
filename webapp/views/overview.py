"""홈 / 개요 — 대시보드.

목업 구조:
  헤더(제목) · 본문[KPI → 도넛·요인 TOP5·확률분포(3분할) → tenure 추이] · 우측[필터·임계값·활용·인사이트]
  그 아래 운영 핵심: 예상손실 높은 순 고위험 명단(행 선택) → 선택 고객 인과 해설.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from lib import data as D
from lib import theme as T

BLUE, CORAL, MUTED, TRACK = T.MAROON, T.CORAL, T.MUTED, T.TRACK
LIST_N = 30


# ---------------- 차트 ----------------
def _donut(stats) -> alt.Chart:
    d = pd.DataFrame({"label": ["이탈", "유지"],
                      "value": [stats["churned"], stats["retained"]]})
    arc = alt.Chart(d).mark_arc(innerRadius=58, outerRadius=88).encode(
        theta=alt.Theta("value:Q", stack=True),
        color=alt.Color("label:N",
                        scale=alt.Scale(domain=["이탈", "유지"], range=[CORAL, BLUE]),
                        legend=alt.Legend(orient="bottom", title=None)))
    txt = alt.Chart(pd.DataFrame({"t": [f"{stats['churn_rate']*100:.1f}%"]})).mark_text(
        fontSize=24, fontWeight="bold", color=CORAL).encode(text="t:N")
    return (arc + txt).properties(height=230).configure_view(strokeWidth=0)


def _factor_bars(dff) -> alt.Chart:
    fac = D.factor_churn_rates(dff).sort_values("rate", ascending=False).head(5)
    return alt.Chart(fac).mark_bar(color=BLUE).encode(
        x=alt.X("rate:Q", axis=alt.Axis(format="%", title="이탈률", grid=False)),
        y=alt.Y("label:N", sort="-x", title=None),
        tooltip=["label", alt.Tooltip("rate:Q", format=".1%"), "n"],
    ).properties(height=230).configure_view(strokeWidth=0)


def _prob_hist(dff) -> alt.Chart:
    pb = D.prob_band_distribution(dff)
    return alt.Chart(pb).mark_bar(color=BLUE).encode(
        x=alt.X("band:N", sort=list(pb["band"]), title="이탈 확률",
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y("count:Q", title="고객 수", axis=alt.Axis(grid=False)),
        tooltip=["band", "count"],
    ).properties(height=230).configure_view(strokeWidth=0)


def _tenure_line(dff, mean_rate, boundaries) -> alt.Chart:
    curve = D.tenure_churn_curve(dff)
    tmax, ticks = D.tenure_axis(curve)
    bnd = alt.Chart(pd.DataFrame({"x": list(boundaries)})).mark_rule(
        color="#c2cbe0").encode(x="x:Q")
    line = alt.Chart(curve).mark_line(color=CORAL, strokeWidth=2.5).encode(
        x=alt.X("tenure:Q", title="경과월(tenure)", scale=alt.Scale(domain=[0, tmax]),
                axis=alt.Axis(values=ticks, grid=False)),
        y=alt.Y("rate:Q", title="이탈률", axis=alt.Axis(format="%", grid=False)))
    mean_rule = alt.Chart(pd.DataFrame({"y": [mean_rate]})).mark_rule(
        color=MUTED, strokeDash=[4, 4]).encode(y="y:Q")
    return (bnd + line + mean_rule).properties(height=250).configure_view(strokeWidth=0)


def _sel_rows(event) -> list:
    try:
        return list(event.selection.rows)
    except Exception:
        try:
            return list(event["selection"]["rows"])
        except Exception:
            return []


def render():
    df, meta = D.get_scored()
    rules = D.load_rules()
    boundaries = rules["analysis_a"]["boundaries"]
    seg_names = [D.SEGMENT_NAMES[i] for i in range(len(D.SEGMENT_NAMES))]

    T.html(T.page_header(
        "고객 이탈 예측 대시보드",
        "통신사 가입 고객 데이터로 이탈을 예측·분석하고, 한정 예산으로 가장 효과적인 "
        "대응 지점을 찾습니다."))

    main, rail = st.columns([3.3, 1], gap="large")

    # ── 우측 패널 (필터·안내) — 필터값을 먼저 받는다 ──
    with rail:
        seg_sel = st.selectbox("세그먼트 필터", ["전체"] + seg_names)
        thr = st.slider("이탈 확률 임계값", 0.0, 1.0, 0.50, 0.05)
        T.html('<div class="info-box"><div class="ib-h">서비스 활용 방법</div>'
               '<div class="ib-body">① 세그먼트·임계값으로 화면을 좁혀 보고<br>'
               '② 아래 명단에서 고객을 클릭해<br>'
               '③ 인과 해설로 위험 유형을 확인하세요.</div></div>')
        T.html('<div class="good-box"><div class="ib-h">인사이트</div>'
               '<div class="ib-body">✓ 가입 초기일수록 이탈률이 높습니다<br>'
               '✓ 세그먼트마다 핵심 위험요인이 다릅니다<br>'
               '&nbsp;&nbsp;(신규=인터넷·요금, 장기=계약형태)<br>'
               '✓ 위험신호가 쌓일수록 이탈이 급증합니다</div></div>')

    dff = df if seg_sel == "전체" else df[df["segment"] == seg_names.index(seg_sel)]

    # ── 본문 (KPI + 차트) ──
    with main:
        s = D.overview_stats(dff)
        exp_loss = float(dff["예상손실"].sum()) if "예상손실" in dff.columns else 0.0
        hr = int((dff["이탈확률"] >= thr).sum()) if "이탈확률" in dff.columns else 0
        T.html(T.kpi_row([
            T.kpi("전체 고객 수", f"{s['total']:,}", icon="👥"),
            T.kpi("이탈 고객 수", f"{s['churned']:,}", accent=True,
                  suffix=f"({s['churn_rate']*100:.1f}%)", icon="📉"),
            T.kpi("이탈 시 월매출 노출", f"${exp_loss/1000:,.0f}K", suffix="/월", icon="💸"),
            T.kpi(f"고위험(≥{thr:.0%})", f"{hr:,}", icon="⚠️"),
        ]))

        c1, c2, c3 = st.columns(3)
        with c1:
            with st.container(border=True):
                T.html('<div class="ch-t">이탈 여부 분포</div>')
                st.altair_chart(_donut(s), use_container_width=True)
        with c2:
            with st.container(border=True):
                T.html('<div class="ch-t">주요 이탈 요인 TOP 5</div>')
                st.altair_chart(_factor_bars(dff), use_container_width=True)
        with c3:
            with st.container(border=True):
                T.html('<div class="ch-t">이탈 확률 분포</div>')
                st.altair_chart(_prob_hist(dff), use_container_width=True)

        with st.container(border=True):
            T.html('<div class="ch-t">경과월(tenure)별 이탈률 추이 '
                   '<span class="ch-sub">— 달력 월이 아닌 가입 후 경과월 기준</span></div>')
            st.altair_chart(_tenure_line(dff, s["churn_rate"], boundaries),
                            use_container_width=True)

    # ── 운영 핵심: 고위험 명단 + 인과 해설 ──
    pri = D.priority_customers(dff, top=LIST_N).reset_index(drop=True)
    disp = pd.DataFrame({
        "순위": range(1, len(pri) + 1),
        "고객 ID": pri["customerID"],
        "세그먼트": pri["segment"].map(lambda i: D.SEGMENT_NAMES[i]),
        "예상손실": pri["예상손실"].round(0),
        "이탈확률": (pri["이탈확률"] * 100).round(0),
        "핵심 원인": pri["핵심원인"],
    })
    with st.expander(f"🎯 예상손실 높은 순 고위험 고객 상위 {len(pri)}명 (행 클릭 → 인과 해설)",
                     expanded=True):
        event = st.dataframe(
            disp, hide_index=True, use_container_width=True,
            on_select="rerun", selection_mode="single-row",
            column_config={
                "예상손실": st.column_config.NumberColumn("예상손실($/월)", format="$%d"),
                "이탈확률": st.column_config.NumberColumn("이탈확률", format="%d%%"),
            },
        )
        T.html('<div class="note" style="margin-top:-.2rem">정렬 기준은 <b>예상손실</b>'
               '(이탈확률 아님) · 열 머리글로 임시 재정렬 가능.</div>')

    if len(pri) == 0:
        T.html('<div class="callout">선택한 세그먼트에 표시할 고객이 없습니다.</div>')
        return

    idx_list = _sel_rows(event)
    idx = idx_list[0] if idx_list else 0

    T.html('<div class="eyebrow">인과관계 해설 — 선택 고객</div>')
    p70 = float(df["MonthlyCharges"].quantile(0.70))
    det = D.customer_risk_detail(pri.iloc[idx], rules, high_charge_threshold=p70)
    core = "".join(f'<span class="tag sig">{x}</span>' for x in det["core_risks"]) \
        or '<span class="csub">이 세그먼트의 검증된 핵심 드라이버는 보유하지 않음</span>'
    other = "".join(f'<span class="tag">{x}</span>' for x in det["other_risks"]) \
        or '<span class="csub">추가 보유 위험 신호 없음</span>'
    val = (f'세그먼트 단독 판별 AUC {det["seg_auc"]:.3f} · 순열검정 p={det["seg_p"]:.3f} (분석 B 검증)'
           if det["seg_auc"] is not None else "")
    T.html(
        '<div class="card">'
        f'<div class="cust-head"><span class="cust-id">{det["customerID"]}</span>'
        '<span class="cust-loss">'
        f'<span class="big">${det["loss"]:,.0f}</span>'
        f'<span class="calc">/월 · 이탈확률 {det["prob"]*100:.0f}%</span></span></div>'
        '<div class="note" style="margin:.5rem 0 1rem">'
        f'<b>{det["segment_name"]}</b> ({det["range"]}) · tenure {det["tenure"]}개월 · '
        f'보유 위험신호 {det["risk_count"]}개</div>'
        '<div style="margin-bottom:.9rem"><div class="csub" style="margin-bottom:.45rem">'
        '핵심 위험 유형 — 이 세그먼트에서 통계 검증된 드라이버</div>'
        f'{core}</div>'
        '<div><div class="csub" style="margin-bottom:.45rem">그 외 보유 위험 유형</div>'
        f'{other}</div>'
        f'<div class="note" style="margin-top:.9rem">{val}</div></div>'
    )

    src = "학습된 모델(outputs/latest)" if meta["source"] == "trained" else "대시보드 자체 학습(데모)"
    T.html(f'<div class="note">예상손실 = 월요금 × 이탈확률(현재 {src} 기준) · '
           f'세그먼트·위험속성·위험신호는 순열·부트스트랩 검증 통과 · 전체 데이터 {len(df):,}건</div>')
