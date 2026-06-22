"""홈 / 개요 — 운영 콘솔.

구성:
  1) 아키텍처 선언 + 이원화 레이어(거시 통계 / 미시 예측) 상태
  2) 핵심 재무 지표
  3) 예상손실 높은 순 고위험 고객 명단 (행 선택)
  4) 선택 고객 인과관계 해설 (분석 B·Q 매칭)
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


def _arch_block(meta) -> str:
    macro = '<span class="l-stat ok">● 규칙 로드됨</span>'
    if meta.get("source") == "trained":
        micro = '<span class="l-stat ok">● 학습 모델 연결</span>'
    else:
        micro = '<span class="l-stat warn">● 데모 모드(자체 학습)</span>'
    return (
        '<div class="arch">'
        f'<div class="arch-decl">{ARCH_DECLARATION}</div>'
        '<div class="layer-grid">'
        '<div class="layer macro"><div class="l-top">'
        '<span class="l-name">거시 통계 레이어</span>'
        '<span class="l-tag">분석 A·B·Q</span></div>'
        f'<div class="l-meta">최종 갱신 <b>{MACRO_UPDATED}</b> · 연 단위 배포 완료</div>'
        f'{macro}</div>'
        '<div class="layer-plus">+</div>'
        '<div class="layer micro"><div class="l-top">'
        '<span class="l-name">미시 예측 레이어</span>'
        '<span class="l-tag">XGBoost</span></div>'
        f'<div class="l-meta">최종 갱신 <b>{MICRO_UPDATED}</b> · 월 단위 자동 재학습</div>'
        f'{micro}</div>'
        '</div></div>'
    )


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

    # 1) 아키텍처 선언 + 이원화 레이어 상태 ──────────────────
    T.html(_arch_block(meta))

    # 2) 핵심 재무 지표 ──────────────────────────────────────
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

    # 3) 고위험 고객 명단 (예상손실 높은 순) ──────────────────
    T.html('<div class="eyebrow">당장 이탈·손실이 예상되는 고위험 고객 — 예상손실 높은 순</div>')
    disp = pd.DataFrame({
        "순위": range(1, len(pri) + 1),
        "고객 ID": pri["customerID"],
        "세그먼트": pri["segment"].map(lambda i: D.SEGMENT_NAMES[i]),
        "예상손실": pri["예상손실"].round(0),
        "이탈확률": (pri["이탈확률"] * 100).round(0),
        "핵심 원인": pri["핵심원인"],
    })
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

    # 4) 인과관계 해설 (선택 고객) ───────────────────────────
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
