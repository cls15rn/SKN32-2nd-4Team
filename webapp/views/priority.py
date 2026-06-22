"""
우선 대응 고객 — 재무지표 분해.

예상 손실(월 요금 × 이탈확률) 기준. 상단 토글로 분석 '기준'을 고른다:
  · 전체 데이터          → df 전체 (팀장님 요구: 전체 데이터로 재무지표 출력)
  · 대응 대상(이탈확률 50%+) → targets(df)

각 기준에 대해 4개 탭으로 분해: 전체 / 세그먼트별(A) / 위험속성별(B) / 위험신호별(Q).
※ '추천 기획(리텐션 오퍼)' 칩은 모두 제거하고 '핵심 원인'만 남긴다.
"""
from __future__ import annotations

import streamlit as st

from lib import data as D
from lib import theme as T


def _summary_cards(t, third_label, third_value, pop_label):
    return T.kpi_row([
        T.kpi(pop_label, f"{len(t):,}명"),
        T.kpi("총 예상 월손실", f"${t['예상손실'].sum():,.0f}", accent=True),
        T.kpi(third_label, third_value),
    ])


def _tab_overall(t, pop_label):
    conc = D.loss_concentration(t)
    T.html(_summary_cards(t, "손실 집중도", f"상위 20% → {conc*100:.0f}%", pop_label))

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
        + f'<hr class="soft"><div class="note">{pop_label} {len(t):,}명 · 핵심 원인은 검증된 위험속성·세그먼트 기준</div>'
    ))

    contrib = D.top_contributors(t, top=3)
    maxloss = contrib["loss"].max() if len(contrib) else 1
    bars = "".join(
        T.hbar(r["label"], r["loss"], f"${r['loss']/1000:.0f}k",
               meta=f"이탈 예상 {int(r['n']):,}명", maxpct=maxloss)
        for _, r in contrib.iterrows()
    )
    T.html(T.card(
        T.card_title("주요 기여 요소 · 예상 손실 기준 Top 3",
                     f"{pop_label}의 예상 손실을 가장 많이 만드는 위험요소") + bars))


def _tab_segment(t, pop_label):
    seg = D.loss_by_segment(t)
    top_seg = seg.loc[seg["loss"].idxmax()]
    T.html(_summary_cards(t, "최대 손실 구간", top_seg["range"], pop_label))

    bars = "".join(
        T.hbar(f'{r["name"]} ({r["range"]})', r["loss"], f"${r['loss']/1000:.1f}k",
               meta=f"{int(r['n']):,}명", conc=f"집중도 {r['share']*100:.0f}%",
               maxpct=seg["loss"].max())
        for _, r in seg.iterrows()
    )
    T.html(T.card(
        T.card_title("세그먼트별 예상 손실 분포 (분석 A · tenure)",
                     "각 생애주기 구간이 예상 손실에서 차지하는 비중 (막대 = 손실 집중도)")
        + bars))
    T.html(f'<div class="callout"><b>{top_seg["name"]}</b>({top_seg["range"]})이 '
           f'{pop_label} 예상손실에서 가장 큰 비중(<b>{top_seg["share"]*100:.0f}%</b>)을 차지합니다. '
           f'세그먼트는 서로 겹치지 않아 인원·손실 합계가 {pop_label} 전체'
           f'({len(t):,}명 · ${t["예상손실"].sum():,.0f})와 정확히 일치합니다.</div>')


def _tab_attribute(t, pop_label):
    attr = D.loss_by_risk_attribute(t)
    top_attr = attr.iloc[0]["label"]
    T.html(_summary_cards(t, "최대 손실 요소", top_attr, pop_label))

    maxv = attr["loss"].max() or 1
    bars = "".join(
        T.hbar(r["label"], r["loss"], f"${r['loss']/1000:.0f}k",
               meta=f"보유 {int(r['n']):,}명 · 이탈률 {r['churn']*100:.0f}%",
               maxpct=maxv)
        for _, r in attr.iterrows()
    )
    T.html(T.card(
        T.card_title("위험속성별 예상 손실 (분석 B · 유형)",
                     "분석 B에서 검증된 위험속성 · 보유 수 · 예상 손실 · 이탈률 (막대 = 예상 손실)")
        + bars))
    T.html('<div class="note">위험속성 목록은 분석 B 검증 결과를 따릅니다(속성 중복 가능). '
           '통화 단위는 데이터 기준(USD).</div>')


def _tab_signal(t, pop_label):
    sig = D.loss_by_risk_signal(t)
    if len(sig) == 0:
        T.html('<div class="callout">해당 고객이 없습니다.</div>')
        return
    top_bucket = sig.iloc[0]["bucket"]
    T.html(_summary_cards(t, "최고위험 구간", f"위험신호 {top_bucket}", pop_label))

    blocks = []
    for _, r in sig.iterrows():
        chips = "".join(f'<span class="chip">{lbl} {int(round(sh*100))}%</span>'
                        for lbl, sh in r["top_signals"])
        blocks.append(
            T.hbar(f"위험신호 {r['bucket']}", r["loss"],
                   f"${r['loss']/1000:.1f}k",
                   meta=f"{int(r['n']):,}명 · 이탈률 {r['churn']*100:.0f}%",
                   conc=f"집중도 {r['share']*100:.0f}%", maxpct=sig["loss"].max())
            + f'<div style="margin:-.2rem 0 .9rem">{chips}</div>')
    T.html(T.card(
        T.card_title("위험신호 개수별 분포 (서브트랙 Q · 유형)",
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
    df, _ = D.get_active_df()

    T.html(T.page_header("우선 대응 · 재무지표 분해",
                         "예상 손실(월 요금 × 이탈확률)을 전체 데이터와 대응 대상으로 나눠, "
                         "tenure(A)·유형(B·Q)별로 분해합니다."))

    basis = st.radio(
        "재무지표 기준",
        ["전체 데이터", "대응 대상 (이탈확률 50%+)"],
        horizontal=True,
    )
    if basis.startswith("전체"):
        t, pop_label = df, "전체 고객"
    else:
        t, pop_label = D.targets(df), "대응 대상(이탈확률 50%+)"

    tab1, tab2, tab3, tab4 = st.tabs(["전체", "세그먼트별 (A)", "위험속성별 (B)", "위험신호별 (Q)"])
    with tab1:
        _tab_overall(t, pop_label)
    with tab2:
        _tab_segment(t, pop_label)
    with tab3:
        _tab_attribute(t, pop_label)
    with tab4:
        _tab_signal(t, pop_label)

    T.html('<div class="note">※ 이탈확률·예상손실은 모델 산출값입니다. 세그먼트·위험신호 구간은 '
           '중복 없음(합산 일치). 통화 단위는 데이터 기준(USD).</div>')
