"""분석 B · 위험속성 (이미지 11) — 세그먼트 선택 → 속성·범주별 이탈률."""
from __future__ import annotations

import streamlit as st

from lib import data as D
from lib import theme as T


def section():
    df, _ = D.get_scored()
    rules = D.load_rules()
    prof = D.segment_profile(df).set_index("segment")

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

            groups = D.segment_attribute_breakdown(df, seg)
            blocks = []
            for group, rows in groups.items():
                if not rows:
                    continue
                bars = "".join(
                    T.hbar(r["label"], r["rate"] * 100, f"{r['rate']*100:.0f}%",
                           meta=f"n {r['n']:,}")
                    for r in rows
                )
                blocks.append(
                    f'<div class="attr-group"><div class="ag-title">{group}</div>{bars}</div>'
                )
            T.html(T.card(T.card_title(f"속성·범주별 이탈률 — {D.SEGMENT_NAMES[seg]} 구간")
                          + "".join(blocks)))

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
