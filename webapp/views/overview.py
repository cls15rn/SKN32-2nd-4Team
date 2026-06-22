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
    arc = alt.Chart(d).mark_arc(innerRadius=42, outerRadius=64).encode(
        theta=alt.Theta("value:Q", stack=True),
        color=alt.Color("label:N",
                        scale=alt.Scale(domain=["이탈", "유지"], range=[CORAL, BLUE]),
                        legend=alt.Legend(orient="bottom", title=None)))
    txt = alt.Chart(pd.DataFrame({"t": [f"{stats['churn_rate']*100:.1f}%"]})).mark_text(
        fontSize=21, fontWeight="bold", color=CORAL).encode(text="t:N")
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


SEG_PAL = ["#cdd9ff", "#c9efe1", "#ffe2bf", "#e4dcfb"]  # 세그먼트 1~4 구분용 라이트 톤


def _tenure_line(df_full, dff, mean_rate, boundaries, seg_sel, seg_names) -> alt.Chart:
    full = D.tenure_churn_curve(df_full)
    tmax, _ = D.tenure_axis(full)
    edges = [0] + list(boundaries) + [tmax]
    yaxis = alt.Axis(format="%", grid=False)

    # 세그먼트별 정수 시작·끝 (세그먼트1: 0~10, 2: 11~22, 3: 23~54, 4: 55~72)
    def _intrange(i):
        lo = 0 if i == 0 else int(boundaries[i - 1] + 0.5)
        hi = int(boundaries[i] - 0.5) if i < len(boundaries) else int(tmax)
        return lo, hi
    seg_ranges = [_intrange(i) for i in range(len(seg_names))]

    if seg_sel == "전체":
        curve = full
        x_domain = [0, tmax]
        xticks = sorted({v for r in seg_ranges for v in r})   # 0,10,11,22,23,54,55,72
        ymax = float(full["rate"].max()) * 1.12
        bands = alt.Chart(pd.DataFrame([
            {"x0": edges[i], "x1": edges[i + 1], "y0": 0.0, "y1": ymax, "seg": seg_names[i]}
            for i in range(len(seg_names))])).mark_rect(opacity=0.5).encode(
            x=alt.X("x0:Q", scale=alt.Scale(domain=x_domain, nice=False), title="경과월(tenure)"),
            x2="x1:Q",
            y=alt.Y("y0:Q", axis=None, scale=alt.Scale(domain=[0, ymax])), y2="y1:Q",
            color=alt.Color("seg:N", scale=alt.Scale(domain=seg_names, range=SEG_PAL),
                            legend=None))
        labels = alt.Chart(pd.DataFrame([
            {"x": (edges[i] + edges[i + 1]) / 2, "y": ymax * 0.93, "t": seg_names[i]}
            for i in range(len(seg_names))])).mark_text(
            fontWeight="bold", fontSize=11, color="#4b5566").encode(
            x="x:Q", y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[0, ymax])), text="t:N")
        extra, pt = [bands, labels], False
    else:
        sidx = seg_names.index(seg_sel)
        curve = D.tenure_churn_curve(dff)
        lo, hi = seg_ranges[sidx]
        x_domain = [lo, hi]
        xticks = [lo, hi]
        ymax = (float(curve["rate"].max()) * 1.15) if len(curve) else 1.0
        band = alt.Chart(pd.DataFrame([{"x0": lo, "x1": hi, "y0": 0.0, "y1": ymax}])).mark_rect(
            opacity=0.5, color=SEG_PAL[sidx]).encode(
            x=alt.X("x0:Q", scale=alt.Scale(domain=x_domain, nice=False), title="경과월(tenure)"),
            x2="x1:Q",
            y=alt.Y("y0:Q", axis=None, scale=alt.Scale(domain=[0, ymax])), y2="y1:Q")
        label = alt.Chart(pd.DataFrame([
            {"x": (lo + hi) / 2, "y": ymax * 0.93, "t": seg_names[sidx]}])).mark_text(
            fontWeight="bold", fontSize=11, color="#4b5566").encode(
            x="x:Q", y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[0, ymax])), text="t:N")
        extra, pt = [band, label], True

    line = alt.Chart(curve).mark_line(color=CORAL, strokeWidth=2.5, point=pt).encode(
        x=alt.X("tenure:Q", title="경과월(tenure)", scale=alt.Scale(domain=x_domain, nice=False),
                axis=alt.Axis(values=xticks, grid=False, labelOverlap=False, labelFontSize=10)),
        y=alt.Y("rate:Q", title="이탈률", scale=alt.Scale(domain=[0, ymax]), axis=yaxis))
    mean_rule = alt.Chart(pd.DataFrame({"y": [mean_rate]})).mark_rule(
        color=MUTED, strokeDash=[4, 4]).encode(
        y=alt.Y("y:Q", scale=alt.Scale(domain=[0, ymax])))
    return alt.layer(*extra, line, mean_rule).properties(height=250).configure_view(strokeWidth=0)


def _risk_body(scope_df, rules, seg_sel, seg_names) -> str:
    """현재 스코프(전체 또는 선택 세그먼트)의 위험 요소별 이탈률 + 위험 신호 조합."""
    if seg_sel == "전체":
        top, scope_label = set(), "전체 고객"
        sub_a = "세그먼트마다 핵심 요소가 다릅니다 — 상단에서 세그먼트를 고르면 ‘핵심’ 표시"
    else:
        sidx = seg_names.index(seg_sel)
        top = set(D.segment_validation(rules, sidx).get("top_attributes", []))
        scope_label = seg_sel
        sub_a = "해당 요소를 가진 고객의 이탈률 · ‘핵심’ = 분석 B 검증 요소"

    # ① 위험 요소별 이탈률
    attr = D.loss_by_risk_attribute(scope_df).sort_values("churn", ascending=False)
    amax = (float(attr["churn"].max()) * 100) or 1.0
    abars = ""
    for _, r in attr.iterrows():
        star = ' <span class="tag sig" style="margin:0">핵심</span>' if r["attr"] in top else ""
        abars += T.hbar(f'{r["label"]}{star}', r["churn"] * 100, f'{r["churn"]*100:.0f}%',
                        meta=f'보유 {int(r["n"]):,}명', maxpct=amax)
    card_a = T.card(T.card_title(f"위험 요소별 이탈률 — {scope_label}", sub_a) + abars)

    # ② 위험 신호 조합 · 누적 (서브트랙 Q)
    rc = D.risk_count_distribution(scope_df)
    sig = D.risk_count_signals(scope_df, top=3)
    cmax = (float(rc["rate"].max()) * 100) or 1.0
    cbars = ""
    for _, r in rc.iterrows():
        k = int(r["risk_count"])
        chips = "".join(f'<span class="chip">{lbl} {int(round(sh*100))}%</span>'
                        for lbl, sh in sig.get(k, []))
        cbars += (T.hbar(f"위험신호 {k}개", r["rate"] * 100, f'{r["rate"]*100:.0f}%',
                         meta=f'{int(r["count"]):,}명', maxpct=cmax)
                  + (f'<div style="margin:-.2rem 0 .75rem">{chips}</div>' if chips else ""))
    card_b = T.card(T.card_title(
        f"위험 신호 조합 · 누적 (서브트랙 Q) — {scope_label}",
        "신호가 쌓일수록 이탈률 상승 · 칩 = 그 그룹에 흔한 신호(어떤 조합이 누적되는지)") + cbars)
    return card_a + card_b


def _loss_reference(df_full, dff, seg_sel) -> str:
    """필터와 무관하게 항상 전체(7,043) 기준 손실 지표 — 비교 앵커."""
    e_all = float(df_full["예상손실"].sum()) if "예상손실" in df_full.columns else 0.0
    c_all = D.loss_concentration(df_full, 0.20)
    s = D.overview_stats(df_full)
    items = (
        f'<span class="lr-i">고객 <b>{s["total"]:,}</b></span>'
        f'<span class="lr-i">이탈률 <b>{s["churn_rate"]*100:.1f}%</b></span>'
        f'<span class="lr-i">예상 월손실 <b>${e_all/1000:,.0f}K</b></span>'
        f'<span class="lr-i">손실 집중도(상위20%) <b>{c_all*100:.0f}%</b></span>')
    cmp = ""
    if seg_sel != "전체":
        e_seg = float(dff["예상손실"].sum()) if "예상손실" in dff.columns else 0.0
        share = (e_seg / e_all * 100) if e_all else 0.0
        cmp = (f'<div class="lr-cmp">현재 보기 <b>{seg_sel}</b> · 예상 월손실 '
               f'<b>${e_seg/1000:,.0f}K</b> · 고객 <b>{len(dff):,}명</b> '
               f'→ 전체 손실의 <b>{share:.0f}%</b></div>')
    return f'<div class="loss-ref"><span class="lr-tag">전체 기준 · 필터 무관</span>{items}{cmp}</div>'


def _seg_loss_card(dff) -> str:
    """세그먼트별(생애주기) 예상손실 카드."""
    seg = D.loss_by_segment(dff)
    maxv = float(seg["loss"].max()) or 1.0
    bars = "".join(
        T.hbar(f'{r["name"]} ({r["range"]})', r["loss"], f'${r["loss"]/1000:.1f}k',
               meta=f'{int(r["n"]):,}명', conc=f'{r["share"]*100:.0f}%', maxpct=maxv)
        for _, r in seg.iterrows())
    return T.card(T.card_title(
        "세그먼트별 예상손실 (분석 A · tenure)",
        "생애주기 구간별 예상손실 (막대=손실, 우측=비중)") + bars)


def _attr_loss_card(dff, title_suffix="") -> str:
    """위험 요소별 예상손실 카드."""
    attr = D.loss_by_risk_attribute(dff)
    maxv = float(attr["loss"].max()) or 1.0
    bars = "".join(
        T.hbar(r["label"], r["loss"], f'${r["loss"]/1000:.1f}k',
               meta=f'보유 {int(r["n"]):,}명 · 이탈률 {r["churn"]*100:.0f}%', maxpct=maxv)
        for _, r in attr.iterrows())
    title = "위험 요소별 예상손실" + (f" — {title_suffix}" if title_suffix else " (분석 B · 유형)")
    return T.card(T.card_title(
        title, "위험 요소를 가진 고객의 예상손실 (막대=손실)") + bars)


def _loss_body(dff, seg_sel) -> str:
    """손실 재무 지표 — 예상손실(월요금×이탈확률) 총액·집중도·분해. 상단 필터 스코프 반영."""
    exp = float(dff["예상손실"].sum()) if "예상손실" in dff.columns else 0.0
    conc = D.loss_concentration(dff, 0.20)
    top20 = exp * conc
    n = len(dff)
    avg = exp / n if n else 0.0

    kpis = T.kpi_row([
        T.kpi("예상 월손실 총액", f"${exp/1000:,.0f}K", accent=True, suffix="/월", icon="💸"),
        T.kpi("손실 집중도", f"{conc*100:.0f}%", suffix="·상위20%", icon="🎯"),
        T.kpi("상위 20% 손실 합", f"${top20/1000:,.0f}K", suffix="/월", icon="📊"),
        T.kpi("고객당 평균 예상손실", f"${avg:,.1f}", suffix="/월", icon="🧾"),
    ])

    if seg_sel == "전체":
        # 왼쪽: 세그먼트별 / 오른쪽: 위험 요소별
        body = (
            '<div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));'
            'gap:1rem;align-items:start">'
            f'{_seg_loss_card(dff)}{_attr_loss_card(dff)}</div>')
    else:
        # 단일 세그먼트: 세그먼트별 분해는 무의미 → 위험 요소별만
        body = _attr_loss_card(dff, seg_sel)

    note = ('<div class="note">예상손실 = 월요금 × 이탈확률(매출 노출 관점) · '
            '비용·이익 데이터가 없어 매출 손실(노출) 기준만 표시합니다.</div>')
    return kpis + body + note


def _sel_rows(event) -> list:
    try:
        return list(event.selection.rows)
    except Exception:
        try:
            return list(event["selection"]["rows"])
        except Exception:
            return []



# ---------------------------------------------------------------------------
# CSV 업로드 처리
# ---------------------------------------------------------------------------
REQUIRED_COLS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]


def _score_uploaded(raw: pd.DataFrame):
    """업로드 CSV → scored DataFrame (get_scored와 동일한 컬럼 구조)."""
    import numpy as np
    from data_loader import clean_raw_data  # shared/

    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        return None, f"필수 컬럼 누락: {', '.join(missing)}"
    try:
        df = clean_raw_data(raw.copy())
    except Exception as e:
        return None, f"전처리 오류: {e}"

    rules = D.load_rules()
    boundaries = rules["analysis_a"]["boundaries"]
    upper = max(df["tenure"].max(), boundaries[-1]) + 1
    bins = [-1] + list(boundaries) + [upper]
    df["segment"] = pd.cut(df["tenure"], bins=bins,
                           labels=range(len(bins) - 1)).astype(int)
    df["segment_name"] = df["segment"].map(D.SEGMENT_NAMES)

    risk_values = rules["subtrack_q"]["risk_attribute_values"]
    masks = pd.DataFrame(index=df.index)
    for col, risky in risk_values.items():
        masks[col] = (df[col] == risky).astype(int)
    df["risk_count"] = masks.sum(axis=1)
    df["churn"] = (df["Churn"] == "Yes").astype(int)

    try:
        clf, feature_cols = D.get_whatif_model()
        from columns import BINARY_MAP_COLS, CATEGORICAL_COLS  # shared/
        work = df.copy()
        for col in BINARY_MAP_COLS:
            work[col] = work[col].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).fillna(0)
        multi = [c for c in CATEGORICAL_COLS if c not in BINARY_MAP_COLS]
        enc = pd.get_dummies(work, columns=multi + ["segment"])
        for fc in feature_cols:
            if fc not in enc.columns:
                enc[fc] = 0
        X = enc[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
        df["이탈확률"] = clf.predict_proba(X)[:, 1]
    except Exception as e:
        return None, f"이탈확률 예측 오류: {e}"

    df["예상손실"] = df["MonthlyCharges"] * df["이탈확률"]
    df["HighCharge"] = np.where(
        df["MonthlyCharges"] >= df["MonthlyCharges"].quantile(D.HIGH_CHARGE_PCTL),
        "High", "Low"
    )
    return df, ""


def _upload_panel():
    """CSV 업로드 UI — 성공 시 scored DataFrame 반환, 아니면 None."""
    T.html(
        '<div style="background:#f3f6fc;border-radius:14px;padding:1.2rem 1.4rem;'
        'margin-bottom:1rem">'
        '<div style="font-weight:700;font-size:.95rem;margin-bottom:.5rem">'
        '📂 새 고객 데이터 업로드</div>'
        '<div style="font-size:.84rem;color:#667085;line-height:1.7">'
        '원본 IBM Telco 형식의 CSV 파일을 업로드하면 동일한 대시보드로 분석합니다.<br>'
        '필수 컬럼: customerID, tenure, MonthlyCharges, TotalCharges, Churn 포함 21개 컬럼<br>'
        '<span style="color:#c0392b;font-weight:600">'
        '※ 개인정보가 포함된 실제 데이터는 업로드하지 마세요.</span>'
        '</div></div>'
    )
    uploaded = st.file_uploader(
        "CSV 파일 선택", type=["csv"],
        key="ov_csv_upload", label_visibility="collapsed",
    )
    if uploaded is None:
        T.html(
            '<div style="text-align:center;padding:2.5rem;color:#667085;font-size:.9rem">'
            '파일을 업로드하면 이 탭에 분석 결과가 표시됩니다.</div>'
        )
        return None
    try:
        raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"CSV 읽기 실패: {e}")
        return None
    with st.spinner("데이터 처리 중…"):
        df_up, err = _score_uploaded(raw)
    if df_up is None:
        st.error(f"⚠️ {err}")
        return None
    st.success(f"✅ {len(df_up):,}명 로드 완료 · 이탈확률 예측 완료")
    return df_up


# ---------------------------------------------------------------------------
# 대시보드 본문 렌더 (기존 render 내용, df·meta를 인자로 받도록 분리)
# ---------------------------------------------------------------------------
def _render_dashboard(df: pd.DataFrame, meta: dict) -> None:
    rules = D.load_rules()
    boundaries = rules["analysis_a"]["boundaries"]
    seg_names = [D.SEGMENT_NAMES[i] for i in range(len(D.SEGMENT_NAMES))]

    main, rail = st.columns([3.3, 1], gap="large")

    with rail:
        seg_sel = st.selectbox("세그먼트 필터", ["전체"] + seg_names,
                               key=f"ov_seg_{meta.get('source','x')}")
        thr = st.slider("이탈 확률 임계값", 0.0, 1.0, 0.50, 0.05,
                        key=f"ov_thr_{meta.get('source','x')}")
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

    with main:
        s = D.overview_stats(dff)
        hr = int((dff["이탈확률"] >= thr).sum()) if "이탈확률" in dff.columns else 0
        T.html(T.kpi_row([
            T.kpi("전체 고객 수", f"{s['total']:,}", icon="👥"),
            T.kpi("이탈 고객 수", f"{s['churned']:,}", accent=True,
                  suffix=f"({s['churn_rate']*100:.1f}%)", icon="📉"),
            T.kpi("유지 고객 수", f"{s['retained']:,}",
                  suffix=f"({(1-s['churn_rate'])*100:.1f}%)", icon="🛡️"),
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
            st.altair_chart(
                _tenure_line(df, dff, s["churn_rate"], boundaries, seg_sel, seg_names),
                use_container_width=True)

    scope_txt = "전체 고객" if seg_sel == "전체" else seg_sel
    T.html(f'<div class="eyebrow">손실 재무 지표 — {scope_txt}</div>')
    T.html(_loss_reference(df, dff, seg_sel))
    T.html(_loss_body(dff, seg_sel))

    with st.expander("🔬 위험 요소 · 위험 신호 조합 — 펼쳐 보기 (상단 세그먼트 필터 적용)",
                     expanded=False, key=f"exp_risk_{meta.get('source','x')}"):
        scope_txt = "전체 고객" if seg_sel == "전체" else seg_sel
        T.html(f'<div class="note" style="margin-bottom:.6rem">현재 보기: <b>{scope_txt}</b> · '
               '위험 요소별 이탈률과, 위험 신호가 누적될수록 이탈이 어떻게 커지는지'
               '(어떤 신호 조합이 쌓이는지)를 봅니다. 상단 세그먼트 필터로 범위가 바뀝니다.</div>')
        T.html(_risk_body(dff, rules, seg_sel, seg_names))

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
                     expanded=True, key=f"exp_pri_{meta.get('source','x')}"):
        event = st.dataframe(
            disp, hide_index=True, use_container_width=True,
            on_select="rerun", selection_mode="single-row",
            key=f"df_pri_{meta.get('source','x')}",
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

    src_label = {
        "trained": "학습된 모델(outputs/latest)",
        "fallback": "대시보드 자체 학습(데모)",
        "uploaded": "업로드 데이터 · 자체 학습 모델 적용",
    }.get(meta.get("source", "fallback"), "대시보드 자체 학습(데모)")
    T.html(f'<div class="note">예상손실 = 월요금 × 이탈확률(현재 {src_label} 기준) · '
           f'세그먼트·위험속성·위험신호는 순열·부트스트랩 검증 통과 · 전체 데이터 {len(df):,}건</div>')


# ---------------------------------------------------------------------------
# 진입점 — 탭으로 기본 데이터 / CSV 업로드 분기
# ---------------------------------------------------------------------------
def render():
    T.html(T.page_header(
        "고객 이탈 예측 대시보드",
        "순열·부트스트랩 자동화 루프로 검증된 생애주기 세그먼트·위험속성·위험신호 분석 체계와 이원화 MLOps 아키텍처를 결합한 고객 이탈 예측 시스템 입니다."
        " 통신사 가입 고객 데이터로 이탈을 예측·분석하고, 한정 예산으로 가장 효과적인 "
        "대응 지점을 찾습니다."))

    df, meta = D.get_active_df()
    _render_dashboard(df, meta)
