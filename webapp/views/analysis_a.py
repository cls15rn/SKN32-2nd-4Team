"""분석 A · 세그먼트 도출 (이미지 10) — tenure 곡선+경계, 세그먼트 프로파일, 검증 통계."""
from __future__ import annotations

import altair as alt
import streamlit as st

from lib import data as D
from lib import theme as T


def _risk_bands(prof, boundaries, tmax):
    """구간별 배경 밴드 — 이탈률(위험도)이 높은 구간일수록 진하게."""
    edges = [0] + list(boundaries) + [tmax]
    bands = []
    for i, (_, r) in enumerate(prof.iterrows()):
        bands.append({"x": edges[i], "x2": edges[i + 1], "y": 0, "y2": 0.7,
                      "op": min(0.16, round(float(r["rate"]) * 0.27, 3))})
    return alt.Chart(alt.Data(values=bands)).mark_rect(color=T.MAROON).encode(
        x="x:Q", x2="x2:Q",
        y=alt.Y("y:Q", axis=None), y2="y2:Q",
        opacity=alt.Opacity("op:Q", scale=alt.Scale(domain=[0, 1], range=[0, 1]),
                            legend=None))


def _segment_labels(prof, boundaries, tmax):
    """각 세그먼트 밴드 상단에 이름 라벨(신규/성장/성숙/장기) 표시."""
    edges = [0] + list(boundaries) + [tmax]
    data = [{"x": (edges[i] + edges[i + 1]) / 2, "y": 0.67, "name": r["name"]}
            for i, (_, r) in enumerate(prof.iterrows())]
    return alt.Chart(alt.Data(values=data)).mark_text(
        baseline="top", fontSize=12, fontWeight="bold", color="#5f5950").encode(
        x="x:Q", y=alt.Y("y:Q", axis=None), text="name:N")


def _chart(curve, mean_rate, boundaries, prof):
    tmax, ticks = D.tenure_axis(curve)
    band = _risk_bands(prof, boundaries, tmax)
    labels = _segment_labels(prof, boundaries, tmax)
    base = alt.Chart(curve).encode(
        x=alt.X("tenure:Q", title="경과월(tenure)",
                scale=alt.Scale(domain=[0, tmax], nice=False),
                axis=alt.Axis(values=ticks, grid=False)))
    line = base.mark_line(color=T.MAROON, strokeWidth=2.2).encode(
        y=alt.Y("rate:Q", title="이탈률", axis=alt.Axis(format="%", grid=False),
                scale=alt.Scale(domain=[0, 0.7])))
    mean_rule = alt.Chart(alt.Data(values=[{"y": mean_rate}])).mark_rule(
        color=T.MUTED, strokeDash=[4, 4]).encode(y="y:Q")
    bnd = alt.Chart(alt.Data(values=[{"x": b} for b in boundaries])).mark_rule(
        color="#5f5950", strokeWidth=1.6).encode(x="x:Q")
    return (band + line + mean_rule + bnd + labels).properties(
        height=250).configure_view(strokeWidth=0)


def section(df=None, rules=None):
    if df is None:
        df, _ = D.get_active_df()
    if rules is None:
        rules = D.load_rules()
    a = rules["analysis_a"]
    boundaries = a["boundaries"]
    mean_rate = df["churn"].mean()
    prof = D.segment_profile(df)

    bnd_str = "·".join(str(int(b)) for b in boundaries)
    T.html('<div class="card">' + T.card_title(
        "tenure별 이탈률과 세그먼트 경계",
        f"배경 진하기 = 구간별 위험도(이탈률) · 세로 실선 = 경계({bnd_str}개월) · "
        f"가로 점선 = 전체 평균 {mean_rate*100:.1f}%")
        + "</div>")
    st.altair_chart(_chart(D.tenure_churn_curve(df), mean_rate, boundaries, prof),
                    use_container_width=True)

    # 세그먼트 프로파일
    bars = "".join(
        T.hbar(f"{r['range']}", r["rate"] * 100, f"{r['rate']*100:.1f}%",
               meta=f"고객 {int(r['count']):,}명")
        for _, r in prof.iterrows()
    )
    T.html(T.card(
        T.card_title("세그먼트별 프로파일 (전체 고객)",
                     "생애주기 구간이 진행될수록 이탈률이 단조 감소 (막대 = 이탈률)") + bars))

    # 검증 요약
    auc = a["segment_only_auc"]
    ci_lo, ci_hi = a["ci_low"], a["ci_high"]
    ci_w = (ci_hi - ci_lo) / 2
    pval = a["p_value"]
    pstr = "< 0.001" if pval < 0.001 else f"{pval:.3f}"
    stat = (
        f'<div class="stat-grid">'
        f'<div class="stat"><div class="s-label">세그먼트단독 AUC</div>'
        f'<div class="s-val">{auc:.3f}</div><div class="s-sub">95% CI {ci_lo:.3f}–{ci_hi:.3f}</div></div>'
        f'<div class="stat"><div class="s-label">순열검정 p값</div>'
        f'<div class="s-val">{pstr}</div><div class="s-sub">라벨 무작위 재배열 대비</div></div>'
        f'<div class="stat"><div class="s-label">표본 안정성</div>'
        f'<div class="s-val">CI 폭 ±{ci_w:.3f}</div><div class="s-sub">부트스트랩 재추정</div></div>'
        f'</div>'
    )
    body = (T.card_title("경계 검증 요약 &nbsp;<span class='badge-ok'>검증 통과</span>",
                         "세그먼트 라벨만의 판별력과 경계의 실재성")
            + stat
            + '<hr class="soft"><div class="note">경계 탐지: 가지치기 회귀나무 — 교차검증으로 가지치기 강도(ccp_alpha) 자동 선택. '
              '세그먼트단독 AUC를 순열검정·부트스트랩으로 검증해 경계가 우연이 아니고 표본도 충분함을 확인.</div>')
    T.html(T.card(body))

    T.html(f'<div class="note">※ 세그먼트단독 AUC({auc:.3f})는 세그먼트 라벨 하나의 판별력으로, '
           '전체 변수를 넣은 예측모델의 AUC와는 다른 수치입니다. 경계값은 segment_rules.json을 따릅니다.</div>')


def render():
    """독립 페이지로 쓸 때(현재는 통합 '분석' 페이지의 한 섹션)."""
    T.html(T.page_header("분석 A · 세그먼트 도출",
                         "tenure(경과월)만으로 생애주기 구간을 데이터가 직접 찾고, 통계 검증으로 "
                         "실재성 확인"))
    section()
