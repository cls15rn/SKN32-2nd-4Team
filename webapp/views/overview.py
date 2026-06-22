"""홈 / 개요 — 운영 콘솔.

구성:
  1) 아키텍처 선언문
  2) 이원화 레이어 — 거시 통계(분석 A·B·Q) / 미시 예측(XGBoost) : 펼치면 실제 데이터
  3) 핵심 재무 지표
  4) 예상손실 높은 순 고위험 고객 명단 (접기/펼치기 · 행 선택)
  5) 선택 고객 인과관계 해설 (분석 B·Q 매칭)
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import data as D
from lib import theme as T

ARCH_DECLARATION = ("순열·부트스트랩 자동화 루프로 검증된 생애주기 세그먼트·위험속성·"
                    "위험신호 분석 체계와 이원화 MLOps 아키텍처를 결합한 고객 이탈 예측 시스템")
MACRO_UPDATED = "2026년 1월"     # 거시 통계 레이어 (분석 A·B·Q)
MICRO_UPDATED = "2026년 06월"    # 미시 예측 레이어 (XGBoost)
LIST_N = 30                      # 고위험 명단 표시 인원

ATTR_KO = {"MonthlyCharges": "월요금", "Contract": "계약형태", "InternetService": "인터넷",
           "OnlineSecurity": "온라인보안", "TechSupport": "기술지원", "PaymentMethod": "결제수단"}


def _macro_body(df, rules) -> str:
    """거시 통계 레이어 펼침 내용 — 분석 A·B·Q 결과 직접 표시."""
    a = rules["analysis_a"]
    q = rules["subtrack_q"]
    prof = D.segment_profile(df)
    bnd = "·".join(str(int(b)) for b in a["boundaries"])

    # 분석 A — 세그먼트별 이탈률
    amax = max(prof["rate"]) * 100 * 1.12
    a_bars = "".join(
        T.hbar(f'{r["name"]} ({r["range"]})', r["rate"] * 100, f'{r["rate"]*100:.0f}%',
               meta=f'n {int(r["count"]):,}', maxpct=amax)
        for _, r in prof.iterrows())
    html = ('<div class="card">' + T.card_title(
        "분석 A · 생애주기 세그먼트",
        f'세그먼트단독 AUC {a["segment_only_auc"]:.3f} · 순열검정 p<.001 · 경계 {bnd}개월')
        + a_bars + '</div>')

    # 분석 B — 세그먼트별 핵심 위험속성
    rows = ""
    for it in rules["analysis_b"]:
        attrs = " · ".join(ATTR_KO.get(x, x) for x in it["top_attributes"])
        rows += (f'<tr><td>{D.SEGMENT_NAMES[it["segment"]]}</td><td>{attrs}</td>'
                 f'<td>{it["attribute_auc"]:.3f}</td><td>p&lt;.001</td></tr>')
    html += ('<div class="card">' + T.card_title(
        "분석 B · 세그먼트별 핵심 위험속성", "각 세그먼트에서 이탈을 가르는 검증된 속성")
        + '<table class="mini"><thead><tr><th>세그먼트</th><th>핵심 위험속성</th>'
          '<th>AUC</th><th>검정</th></tr></thead><tbody>'
        + rows + '</tbody></table></div>')

    # 서브트랙 Q — 위험신호 누적
    rc = D.risk_count_distribution(df)
    qmax = max(rc["rate"]) * 100 * 1.12
    q_bars = "".join(
        T.hbar(f'위험신호 {int(r["risk_count"])}개', r["rate"] * 100, f'{r["rate"]*100:.0f}%',
               meta=f'n {int(r["count"]):,}', maxpct=qmax)
        for _, r in rc.iterrows())
    html += ('<div class="card">' + T.card_title(
        "서브트랙 Q · 위험신호 누적",
        f'risk_count 단독 AUC {q["risk_count_only_auc"]:.3f} · 최고위험 {q["top_risk_count_value"]}개 보유')
        + q_bars + '</div>')
    return html


def _micro_body(df, meta) -> str:
    """미시 예측 레이어 펼침 내용 — 예측 분포 직접 표시."""
    src = ("학습된 모델 (churn_prediction/outputs/latest)" if meta.get("source") == "trained"
           else "대시보드 자체 학습(데모) — 학습 산출물 미배포")
    n = len(df)
    hr = int((df["이탈확률"] >= 0.50).sum()) if "이탈확률" in df.columns else 0
    avg_p = df["이탈확률"].mean() * 100 if "이탈확률" in df.columns else 0

    pb = D.prob_band_distribution(df)
    cmax = (max(pb["count"]) * 1.12) if len(pb) else 1
    bars = "".join(
        T.hbar(r["band"], r["count"], f'{int(r["count"]):,}명',
               meta=f'평균손실 ${r["avg_loss"]:.0f}/월', maxpct=cmax)
        for _, r in pb.iterrows())

    head = (f'<div class="note" style="margin:0 0 .7rem">출처: <b>{src}</b> · '
            f'평균 이탈확률 {avg_p:.0f}% · 고위험(≥50%) {hr:,}명({hr/n*100:.0f}%)</div>')
    table = ('<div class="card">' + T.card_title(
        "예측 이탈확률 분포", "구간별 고객수 · 평균 예상손실") + bars + '</div>')
    note = ('<div class="note">정밀 성능지표(ROC-AUC·F2 등)는 학습 산출물에 저장되며, '
            '현재 데모 모드에서는 예측 분포만 표시합니다.</div>')
    return head + table + note


def _sel_rows(event) -> list:
    """st.dataframe 선택 결과를 버전 차이에 안전하게 추출."""
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
    s = D.overview_stats(df)

    T.html(T.page_header(
        "고객 이탈 예측 — 운영 콘솔",
        "검증된 분석 체계(거시) + 자동 재학습 예측(미시)을 결합해, "
        "예상손실이 큰 고객부터 대응합니다."))

    # 1) 아키텍처 선언문 ────────────────────────────────────
    T.html(f'<div class="arch"><div class="arch-decl">{ARCH_DECLARATION}</div></div>')

    # 2) 이원화 레이어 (펼치면 실제 데이터) ───────────────────
    T.html('<div class="eyebrow">시스템 레이어 — 펼치면 실제 데이터</div>')
    with st.expander(
            f"📊 거시 통계 레이어 · 분석 A·B·Q · 최종 갱신 {MACRO_UPDATED} (연 단위 배포)",
            expanded=False):
        T.html(_macro_body(df, rules))
    micro_state = "학습 모델 연결" if meta.get("source") == "trained" else "데모 모드(자체 학습)"
    with st.expander(
            f"⚙️ 미시 예측 레이어 · XGBoost · 최종 갱신 {MICRO_UPDATED} · 월 단위 자동 재학습 · {micro_state}",
            expanded=False):
        T.html(_micro_body(df, meta))

    # 3) 핵심 재무 지표 ──────────────────────────────────────
    exp_loss = float(df["예상손실"].sum()) if "예상손실" in df.columns else 0.0
    conc = round(D.loss_concentration(df, 0.20) * 100)
    top20_loss = exp_loss * D.loss_concentration(df, 0.20)
    hr = int((df["이탈확률"] >= 0.50).sum()) if "이탈확률" in df.columns else 0
    pri = D.priority_customers(df, top=LIST_N).reset_index(drop=True)

    T.html('<div class="eyebrow">핵심 재무 지표</div>')
    T.html(T.kpi_row([
        T.kpi("이탈 시 월매출 노출", f"${exp_loss/1000:,.0f}K", accent=True, suffix="/월"),
        T.kpi("예상손실 상위 20% 비중", f"{conc}%"),
        T.kpi("상위 20% 예상손실 합", f"${top20_loss/1000:,.0f}K", suffix="/월"),
        T.kpi("고위험 고객", f"{hr:,}명", suffix="·확률≥50%"),
    ]))

    # 4) 고위험 고객 명단 (접기/펼치기 · 예상손실 높은 순) ──────
    disp = pd.DataFrame({
        "순위": range(1, len(pri) + 1),
        "고객 ID": pri["customerID"],
        "세그먼트": pri["segment"].map(lambda i: D.SEGMENT_NAMES[i]),
        "예상손실": pri["예상손실"].round(0),
        "이탈확률": (pri["이탈확률"] * 100).round(0),
        "핵심 원인": pri["핵심원인"],
    })
    with st.expander(
            f"🎯 당장 이탈·손실이 예상되는 고위험 고객 상위 {LIST_N}명 — 예상손실 높은 순",
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
               '(이탈확률 아님) · 행을 클릭하면 아래에 인과 해설이 표시됩니다 · '
               '열 머리글로 임시 재정렬 가능.</div>')

    idx_list = _sel_rows(event)
    idx = idx_list[0] if idx_list else 0

    # 5) 인과관계 해설 (선택 고객) ───────────────────────────
    T.html('<div class="eyebrow">인과관계 해설 — 선택 고객</div>')
    p70 = float(df["MonthlyCharges"].quantile(0.70))
    det = D.customer_risk_detail(pri.iloc[idx], rules, high_charge_threshold=p70)

    core = "".join(f'<span class="tag sig">{x}</span>' for x in det["core_risks"]) \
        or '<span class="csub">이 세그먼트의 검증된 핵심 드라이버는 보유하지 않음</span>'
    other = "".join(f'<span class="tag">{x}</span>' for x in det["other_risks"]) \
        or '<span class="csub">추가 보유 위험 신호 없음</span>'
    val = ""
    if det["seg_auc"] is not None:
        val = (f'세그먼트 단독 판별 AUC {det["seg_auc"]:.3f} · '
               f'순열검정 p={det["seg_p"]:.3f} (분석 B 검증)')

    T.html(
        '<div class="card">'
        f'<div class="cust-head"><span class="cust-id">{det["customerID"]}</span>'
        '<span class="cust-loss">'
        f'<span class="big">${det["loss"]:,.0f}</span>'
        f'<span class="calc">/월 · 이탈확률 {det["prob"]*100:.0f}%</span></span></div>'
        '<div class="note" style="margin:.5rem 0 1rem">'
        f'<b>{det["segment_name"]}</b> ({det["range"]}) · tenure {det["tenure"]}개월 · '
        f'보유 위험신호 {det["risk_count"]}개</div>'
        '<div style="margin-bottom:.9rem">'
        '<div class="csub" style="margin-bottom:.45rem">'
        '핵심 위험 유형 — 이 세그먼트에서 통계 검증된 드라이버</div>'
        f'{core}</div>'
        '<div><div class="csub" style="margin-bottom:.45rem">그 외 보유 위험 유형</div>'
        f'{other}</div>'
        f'<div class="note" style="margin-top:.9rem">{val}</div>'
        '</div>'
    )

    src = "학습된 모델(outputs/latest)" if meta["source"] == "trained" else "대시보드 자체 학습(데모)"
    T.html(f'<div class="note">예상손실 = 월요금 × 이탈확률(현재 {src} 기준) · '
           f'세그먼트·위험속성·위험신호는 순열·부트스트랩 검증 통과 · 데이터 {s["total"]:,}건</div>')
