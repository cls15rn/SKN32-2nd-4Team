"""
우선 대응 고객 (이미지 6~9)

예상 손실(월 요금 × 이탈확률) 기준 우선순위. 4개 탭:
전체 / 세그먼트별 / 위험속성별 / 위험신호별.
※ 요청에 따라 '추천 기획(리텐션 오퍼)' 칩은 모두 제거하고, '핵심 원인'만 남긴다.
"""
from __future__ import annotations

import streamlit as st

from lib import data as D
from lib import theme as T


def _summary_cards(t, third_label, third_value):
    return T.kpi_row([
        T.kpi("대상 고객 (이탈확률 50%+)", f"{len(t):,}명"),
        T.kpi("대상 총 예상 월손실", f"${t['예상손실'].sum():,.0f}", accent=True),
        T.kpi(third_label, third_value),
    ])


def _tab_overall(t):
    conc = D.loss_concentration(t)
    T.html(_summary_cards(t, "손실 집중도", f"상위 20% → {conc*100:.0f}%"))

    # 상위 5명
    rows = []
    p = D.priority_customers(t.sort_values("예상손실", ascending=False), top=5)
    for i, (_, r) in enumerate(p.iterrows(), 1):
        rows.append(
            f'<div class="cust"><div class="cust-head">'
            f'<span class="cust-rank">{i}</span>'
            f'<span class="cust-id">{r.customerID}</span>'
            f'<span class="tag">{r["range"]}</span>'
            f'<span class="tag sig">위험신호 {int(r.risk_count)}</span>'
            f'<span class="cust-loss"><span class="big">${r["예상손실"]:.1f}</span>'
            f'<div class="calc">월 ${r.MonthlyCharges:.1f} × 이탈 {r["이탈확률"]*100:.0f}%</div></span>'
            f'</div>'
            f'<div class="cust-cause">핵심 원인: <b>{r["핵심원인"]}</b></div></div>'
        )
    T.html(T.card(
        T.card_title("예상 월손실 상위 고객", "월 요금 × 이탈확률이 큰 순 — 한정 예산으로 매출 유출을 가장 크게 막을 고객")
        + "".join(rows)
        + f'<hr class="soft"><div class="note">대상 전체 {len(t):,}명 · 핵심 원인은 검증된 위험속성·세그먼트 기준</div>'
    ))

    # 주요 기여 요소 Top 3
    contrib = D.top_contributors(t, top=3)
    maxloss = contrib["loss"].max() if len(contrib) else 1
    bars = "".join(
        T.hbar(r["label"], r["loss"], f"${r['loss']/1000:.0f}k",
               meta=f"이탈 예상 {int(r['n']):,}명", maxpct=maxloss)
        for _, r in contrib.iterrows()
    )
    T.html(T.card(
        T.card_title("주요 기여 요소 · 예상 손실 기준 Top 3",
                     "대상 고객의 예상 손실을 가장 많이 만드는 위험요소") + bars))


def _tab_segment(t):
    seg = D.loss_by_segment(t)
    top_seg = seg.loc[seg["loss"].idxmax()]
    T.html(_summary_cards(t, "최대 손실 구간", top_seg["range"]))

    bars = "".join(
        T.hbar(r["range"], r["loss"], f"${r['loss']/1000:.1f}k",
               meta=f"대상 {int(r['n']):,}명", conc=f"집중도 {r['share']*100:.0f}%",
               maxpct=seg["loss"].max())
        for _, r in seg.iterrows()
    )
    T.html(T.card(
        T.card_title("세그먼트별 예상 손실 분포",
                     "각 생애주기 구간이 대상 고객·예상 손실에서 차지하는 비중 (막대 = 손실 집중도)")
        + bars))
    T.html(f'<div class="callout">신규 구간(0–10개월)이 대상의 가장 큰 손실원이며, '
           f'성숙 구간(23–54개월)도 고객 수 대비 손실 비중이 높습니다(요금이 높은 장기 고객). '
           f'세그먼트는 서로 겹치지 않아 대상 수·손실 합계가 전체'
           f'({len(t):,}명 · ${t["예상손실"].sum():,.0f})와 정확히 일치합니다.</div>')


def _tab_attribute(t):
    attr = D.loss_by_risk_attribute(t)
    top_attr = attr.iloc[0]["label"]
    T.html(_summary_cards(t, "최대 손실 요소", top_attr))

    # ※ 추천 기획(오퍼) 칩 제거 — 예상 손실·이탈률만 표시
    bars = "".join(
        T.hbar(r["label"], r["loss"], f"${r['loss']/1000:.0f}k",
               meta=f"보유 {int(r['n']):,}명 · 이탈률 {r['churn']*100:.0f}%",
               maxpct=attr["loss"].max())
        for _, r in attr.iterrows()
    )
    T.html(T.card(
        T.card_title("위험속성별 예상 손실",
                     "분석 B에서 검증된 위험속성 · 보유 수 · 예상 손실 · 이탈률 (막대 = 예상 손실)")
        + bars))
    T.html('<div class="note">위험속성 목록은 분석 B 검증 결과를 따릅니다(속성 중복 가능). '
           '통화 단위는 데이터 기준(USD).</div>')


def _tab_signal(t):
    sig = D.loss_by_risk_signal(t)
    if len(sig) == 0:
        T.html('<div class="callout">대상 고객이 없습니다.</div>')
        return
    top_bucket = sig.iloc[0]["bucket"]
    T.html(_summary_cards(t, "최고위험 구간", f"위험신호 {top_bucket}"))

    blocks = []
    for _, r in sig.iterrows():
        chips = "".join(f'<span class="chip">{lbl} {int(round(sh*100))}%</span>'
                        for lbl, sh in r["top_signals"])
        blocks.append(
            T.hbar(f"위험신호 {r['bucket']}", r["loss"],
                   f"${r['loss']/1000:.1f}k",
                   meta=f"대상 {int(r['n']):,}명 · 이탈률 {r['churn']*100:.0f}%",
                   conc=f"집중도 {r['share']*100:.0f}%", maxpct=sig["loss"].max())
            + f'<div style="margin:-.2rem 0 .9rem">{chips}</div>')
    T.html(T.card(
        T.card_title("위험신호 개수별 분포",
                     "보유 위험신호 수로 구간을 나눔 · 막대 = 손실 집중도 · 칩 = 구간 내 가장 흔한 속성")
        + "".join(blocks)))

    high = sig[sig["bucket"].isin(["6개+", "5개"])]
    if len(high):
        share = high["share"].sum()
        nn = int(high["n"].sum())
        T.html(f'<div class="callout">위험신호 5개 이상 고객(약 {nn:,}명)이 '
               f'예상 손실의 <b>{share*100:.0f}%</b>를 차지하고 이탈률도 가장 높습니다 — '
               f'신호가 쌓일수록 위험이 커지는 구조라 최우선 타깃입니다.</div>')


def render():
    df, _ = D.get_scored()
    t = D.targets(df)

    T.html(T.page_header("우선 대응 고객",
                         "예상 손실(월 요금 × 이탈확률) 순 — 한정된 예산으로 매출 유출을 "
                         "가장 크게 막을 고객부터"))

    tab1, tab2, tab3, tab4 = st.tabs(["전체", "세그먼트별", "위험속성별", "위험신호별"])
    with tab1:
        _tab_overall(t)
    with tab2:
        _tab_segment(t)
    with tab3:
        _tab_attribute(t)
    with tab4:
        _tab_signal(t)

    T.html('<div class="note">※ 이탈확률·예상손실은 모델 산출값입니다. 구간은 중복 없음(합산 일치). '
           '통화 단위는 데이터 기준(USD).</div>')
