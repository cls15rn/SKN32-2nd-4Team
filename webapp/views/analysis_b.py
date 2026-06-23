"""분석 B · 위험속성 (이미지 11) — 세그먼트 선택 → 속성·범주별 이탈률."""
from __future__ import annotations

import streamlit as st

from lib import data as D
from lib import theme as T


def _type_segment_view(df, sel_label):
    """선택한 위험 유형의 세그먼트별 이탈률 막대."""
    rows = {r["label"]: r for r in D.risk_attribute_by_segment(df)}
    r = rows[sel_label]
    valid = [sc["churn"] for sc in r["segs"] if sc["churn"] is not None]
    maxv = (max(valid) * 100) if valid else 1.0
    bars = ""
    for sc in r["segs"]:
        nm = D.SEGMENT_NAMES[sc["seg"]]
        rng = D.SEGMENT_RANGES[sc["seg"]].replace("개월", "")
        seg_color = T.seg_emphasis_color(int(sc["seg"]), len(D.SEGMENT_NAMES))
        if sc["churn"] is None:
            bars += T.hbar(f"{nm} ({rng})", 0, "—", meta="보유 0명", maxpct=maxv,
                           color=seg_color)
        else:
            bars += T.hbar(f"{nm} ({rng})", sc["churn"] * 100, f"{sc['churn']*100:.0f}%",
                           meta=f"보유 {sc['n']:,}명", maxpct=maxv, color=seg_color)
    return T.card(T.card_title(
        f"‘{sel_label}’ 유형의 세그먼트별 이탈률",
        f"이 유형 보유 고객 {r['overall_n']:,}명 · 전체 이탈률 {r['overall_churn']*100:.0f}% "
        "· 같은 유형이라도 생애주기 구간에 따라 이탈률이 달라짐") + bars)


def section(df=None, rules=None):
    if df is None:
        df, _ = D.get_active_df()
    if rules is None:
        rules = D.load_rules()
    prof = D.segment_profile(df).set_index("segment")

    # 유형 선택 → 그 유형의 세그먼트별 이탈률 (세그먼트별 상세 탭 위)
    options = [r["label"] for r in D.risk_attribute_by_segment(df)]
    sel = st.selectbox("위험 유형 선택", options, key="b_type_seg")
    T.html(_type_segment_view(df, sel))
    T.html('<div class="note" style="margin:-.4rem 0 1rem">유형을 바꿔가며 생애주기 구간별 이탈률을 비교해 보세요 '
           '— 같은 유형이라도 가입 초기 구간에서 이탈률이 크게 높아지는 경향이 보입니다. '
           '아래 탭에서는 각 세그먼트의 속성·범주별 상세를 확인할 수 있습니다.</div>')

    labels = [f"{D.SEGMENT_NAMES[s]} {D.SEGMENT_RANGES[s].replace('개월','')}"
              for s in range(len(D.SEGMENT_NAMES))]
    tabs = st.tabs(labels)

    for seg, tab in enumerate(tabs):
        with tab:
            row = prof.loc[seg]
            val = D.segment_validation(rules, seg)
            auc = val.get("attribute_auc")
            pval = val.get("p_value", 0.0)
            pstr = "p<0.001" if pval < 0.001 else f"p={pval:.3f}"
            auc_str = f"세그먼트 검증 AUC {auc:.3f} ({pstr})" if auc else ""
            T.html(f'<div class="page-sub" style="margin-top:.2rem">'
                   f'고객 {int(row["count"]):,}명 · 평균 이탈 {row["rate"]*100:.0f}% · {auc_str}</div>')

            # 이 탭(=세그먼트)의 강조색. 첫/마지막 세그먼트 탭이면 막대 전체를 그 색으로.
            tab_color = T.seg_emphasis_color(int(seg), len(D.SEGMENT_NAMES))

            groups = D.segment_attribute_breakdown(df, seg)
            # 사용자 지정 3열 배치: 기술지원|월요금|온라인보안 / 결제수단|인터넷|계약
            order = ["기술지원", "월요금 수준", "온라인보안", "결제수단", "인터넷 종류", "계약 형태"]
            ordered = [g for g in order if g in groups] + [g for g in groups if g not in order]
            blocks = []
            for group in ordered:
                rows = groups.get(group)
                if not rows:
                    continue
                bars = "".join(
                    T.hbar(r["label"], r["rate"] * 100, f"{r['rate']*100:.0f}%",
                           meta=f"n {r['n']:,}", color=tab_color)
                    for r in rows
                )
                blocks.append(
                    f'<div class="attr-group"><div class="ag-title">{group}</div>{bars}</div>'
                )
            grid = ('<div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));'
                    'gap:.3rem 1.1rem;align-items:start">' + "".join(blocks) + '</div>')
            T.html(T.card(T.card_title(f"속성·범주별 이탈률 — {D.SEGMENT_NAMES[seg]} 구간")
                          + grid))

            # 재무 축 — 이 세그먼트 안에서 위험속성별 예상손실·이탈률
            seg_df = df[df["segment"] == seg]
            attr = D.loss_by_risk_attribute(seg_df)
            maxv = attr["loss"].max() or 1
            fbars = "".join(
                T.hbar(r["label"], r["loss"], f"${r['loss']/1000:.1f}k",
                       meta=f"보유 {int(r['n']):,}명 · 이탈률 {r['churn']*100:.0f}%",
                       maxpct=maxv, color=tab_color)
                for _, r in attr.iterrows()
            )
            T.html(T.card(T.card_title(
                f"위험속성별 예상손실 — {D.SEGMENT_NAMES[seg]} 구간",
                "같은 위험속성이라도 세그먼트마다 보유 규모·예상손실·이탈률이 다름 (막대 = 예상손실)")
                + fbars))

            top = val.get("top_attributes", [])
            top_kr = ", ".join(D.RISK_LABELS.get(c, c) for c in top) if top else "—"
            T.html(f'<div class="callout">이 구간에서 분석 B가 검증한 핵심 위험속성: <b>{top_kr}</b>. '
                   '같은 속성이라도 생애주기 맥락에 따라 영향력이 달라지는 것을 범주별 이탈률 차이로 확인할 수 있습니다.</div>')

    T.html('<div class="note">※ 이탈률·표본수는 실제 데이터 기준입니다. 표본이 작은 범주는 수치 변동이 클 수 있어 n을 함께 표기했습니다. '
           '세그먼트 검증(AUC·순열검정)은 segment_rules.json을 따릅니다.</div>')


def render():
    """독립 페이지로 쓸 때(현재는 통합 '분석' 페이지의 한 섹션)."""
    T.html(T.page_header("분석 B · 위험속성",
                         "세그먼트를 선택해 그 구간의 속성·범주별 이탈률을 자세히 확인"))
    section()