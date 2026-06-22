"""서브트랙 Q · 위험신호 (이미지 12) — risk_count 개수별 이탈률 + 검증 통계."""
from __future__ import annotations

from lib import data as D
from lib import theme as T


def section(df=None, rules=None):
    if df is None:
        df, _ = D.get_active_df()
    if rules is None:
        rules = D.load_rules()
    q = rules["subtrack_q"]
    mean_rate = df["churn"].mean()

    dist = D.risk_count_distribution(df)
    signals = D.risk_count_signals(df, top=3)
    rows = []
    for _, r in dist.iterrows():
        k = int(r["risk_count"])
        bar = T.hbar(f"신호 {k}개", r["rate"] * 100, f"{r['rate']*100:.0f}%",
                     meta=f"n {int(r['count']):,}", maxpct=70)
        chips = "".join(f'<span class="chip">{lbl} {int(round(sh*100))}%</span>'
                        for lbl, sh in signals.get(k, []))
        chip_div = f'<div style="margin:-.35rem 0 .85rem">{chips}</div>' if chips else ""
        rows.append(bar + chip_div)
    T.html(T.card(
        T.card_title("위험신호 개수별 이탈률 (전체 고객)",
                     f"위험속성 5개(계약·결제·인터넷·보안·기술지원) 중 보유 개수 · 전체 평균 {mean_rate*100:.1f}%"
                     " · 칩 = 그룹 내 주요 위험속성 보유율 Top 3")
        + "".join(rows)
        + '<hr class="soft">'
          '<div class="note">막대는 그 신호 개수 고객의 이탈률, 칩은 그 그룹에서 가장 많이 보유한 위험속성 Top 3입니다. '
          '<b>칩의 %는 그 그룹 고객 중 해당 속성을 가진 비율</b>이에요 '
          '(예: "월단위 계약 90%" = 그 그룹의 90%가 월단위 계약자). 신호 수가 많을수록 대부분 속성을 함께 보유합니다.</div>'))

    # 상승 서사 callout — top_risk_count_value 기준으로 동적 산출
    top_v = q["top_risk_count_value"]
    _sorted = dist.sort_values("risk_count")
    # 절반 지점(중간)과 최고위험 구간 이탈률을 동적으로 산출
    _mid_v = max(1, top_v // 2)
    def _rate(k):
        rows = dist[dist["risk_count"] == k]["rate"]
        return rows.iloc[0] * 100 if len(rows) else None
    def _count(k):
        rows = dist[dist["risk_count"] == k]["count"]
        return int(rows.iloc[0]) if len(rows) else 0
    r_mid  = _rate(_mid_v)
    r_top  = _rate(top_v)
    n_top  = _count(top_v)
    if r_mid is not None and r_top is not None:
        T.html(f'<div class="callout">신호 0~2개는 이탈 3~8%로 낮지만, <b>{_mid_v}개를 넘어서면 '
               f'{r_mid:.0f}%→{r_top:.0f}%</b>로 가파르게 상승합니다. '
               f'신호 {top_v}개 고객(n {n_top:,})은 '
               f'이탈률 {r_top:.0f}%로 전체 평균의 {r_top/(mean_rate*100):.1f}배. 단, 이 상승은 신호들의 '
               '누적 효과이지 곱셈적 시너지는 아닌 것으로 검증됐습니다(가법·곱셈 모델 비교).</div>')

    # 검증 요약
    auc = q["risk_count_only_auc"]
    pval = q["p_value"]
    pstr = "< 0.001" if pval < 0.001 else f"{pval:.3f}"
    ci_lo, ci_hi = q["ci_low"], q["ci_high"]
    stat = (
        f'<div class="stat-grid">'
        f'<div class="stat"><div class="s-label">risk_count단독 AUC</div>'
        f'<div class="s-val">{auc:.3f}</div><div class="s-sub">신호 개수 하나만으로</div></div>'
        f'<div class="stat"><div class="s-label">순열검정 p값</div>'
        f'<div class="s-val">{pstr}</div><div class="s-sub">라벨 무작위 재배열 대비</div></div>'
        f'<div class="stat"><div class="s-label">최고위험({top_v}개) 이탈률</div>'
        f'<div class="s-val">{(r_top or 0):.0f}%</div><div class="s-sub">부트스트랩 CI {ci_lo*100:.0f}–{ci_hi*100:.0f}%</div></div>'
        f'</div>'
    )
    T.html(T.card(
        T.card_title("검증 요약 &nbsp;<span class='badge-ok'>검증 통과</span>",
                     "risk_count와 이탈의 관계가 우연이 아님을 데이터 기반으로 확인")
        + stat
        + '<hr class="soft"><div class="note">순열검정(라벨 무작위 재배열)으로 관계의 실재성을, '
          '부트스트랩으로 최고위험 구간의 안정성을 확인. K-means로 risk_count가 놓칠 수 있는 미발견 위험 조합도 보조 탐색.</div>'))

    T.html('<div class="note">※ risk_count는 검증된 위험속성 5개의 보유 개수로, 해석용 지표입니다'
           '(기존 변수들의 합이라 예측모델 입력으로는 증분 기여가 미미해 사용하지 않음). '
           '검증값은 segment_rules.json 기준입니다.</div>')


def render():
    """독립 페이지로 쓸 때(현재는 통합 '분석' 페이지의 한 섹션)."""
    T.html(T.page_header("서브트랙 Q · 위험신호",
                         "검증된 위험속성을 몇 개나 동시에 가졌는지(risk_count)로 이탈 위험을 정량화"))
    section()
