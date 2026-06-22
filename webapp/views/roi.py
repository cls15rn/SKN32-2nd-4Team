"""
ROI 시뮬레이션 — 리텐션 오퍼 비용 대비 보존 LTV 비교.

설계 원칙 (기획구현.md 5·6번 반영):
  - FP비용(오퍼 단가)은 데이터에 정보가 없어 추정 불가 → 사용자가 직접 입력
  - 전환율도 실측값 없는 가정값 → 사용자 입력, UI에 명시
  - 잔존기간: max_tenure − 세그먼트 평균 tenure (기획구현.md 9번)
  - "예산 최적 배분 알고리즘" 없음 — 세그먼트별 비교표로 사람이 판단
  - 스냅샷 데이터 한계(절대 날짜 없음)를 결과 하단에 명시
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from lib import data as D
from lib import theme as T

# ---------------------------------------------------------------------------
# 색상 헬퍼
# ---------------------------------------------------------------------------
def _net_color(v: float) -> str:
    if math.isnan(v):
        return T.MUTED
    return T.GOOD if v >= 0 else T.CORAL


def _roi_color(v: float) -> str:
    if math.isnan(v) or math.isinf(v):
        return T.MUTED
    if v >= 200:
        return T.GOOD
    if v >= 0:
        return "#f5a623"
    return T.CORAL


# ---------------------------------------------------------------------------
# 개별 세그먼트 입력 컨트롤
# ---------------------------------------------------------------------------
def _seg_input(r: pd.Series, df_full: pd.DataFrame) -> tuple[int, bool]:
    """세그먼트 1행 — 체크박스 + 개입 인원 슬라이더. (n_intervene, selected) 반환."""
    seg = int(r["segment"])
    n_high = int(r["n_high"])
    selected = st.checkbox(
        f'**{r["name"]}** ({r["range"]})',
        value=True,
        key=f"roi_seg_chk_{seg}",
    )
    if not selected:
        st.markdown(" ")
        return 0, False

    n = st.slider(
        f"고위험군 개입 인원 (이탈확률 50%+, 최대 {n_high:,}명)",
        min_value=1,
        max_value=max(n_high, 1),
        value=min(n_high, max(n_high, 1)),
        step=1,
        key=f"roi_seg_n_{seg}",
        help=f"이탈확률 50%+ 고객 {n_high:,}명 중 실제 오퍼를 제안할 인원",
    )
    # 세그먼트 참고 수치 (소형 배지 스타일)
    T.html(
        f'<div style="display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:.3rem">'
        f'<span style="background:#f3f6fc;border-radius:6px;padding:.15rem .5rem;'
        f'font-size:.76rem;color:#667085">평균 월요금 <b>${r["avg_mc"]:.0f}</b></span>'
        f'<span style="background:#f3f6fc;border-radius:6px;padding:.15rem .5rem;'
        f'font-size:.76rem;color:#667085">잔존추정 <b>{r["remaining"]:.0f}개월</b></span>'
        f'<span style="background:#f3f6fc;border-radius:6px;padding:.15rem .5rem;'
        f'font-size:.76rem;color:#667085">평균 이탈확률 <b>{r["avg_prob"]*100:.0f}%</b></span>'
        f'</div>'
    )
    return n, True


# ---------------------------------------------------------------------------
# 결과 카드: 세그먼트별 행
# ---------------------------------------------------------------------------
def _result_row(r: pd.Series) -> str:
    net_c = _net_color(r["net"])
    roi_c = _roi_color(r["roi_pct"])
    be    = r["breakeven_cr"]
    be_str = f'{be*100:.1f}%' if not (math.isnan(be) or math.isinf(be)) else '—'
    roi_str = (f'{r["roi_pct"]:+.0f}%'
               if not (math.isnan(r["roi_pct"]) or math.isinf(r["roi_pct"]))
               else '—')

    return (
        f'<div style="display:grid;grid-template-columns:1.6fr 1fr 1fr 1fr 1fr 1fr;'
        f'align-items:center;gap:.4rem;padding:.65rem .5rem;'
        f'border-bottom:1px solid #f0f2f8;font-size:.88rem">'

        # 세그먼트
        f'<div>'
        f'<div style="font-weight:700">{r["name"]}</div>'
        f'<div style="color:#667085;font-size:.76rem">{r["range"]} · {int(r["n_intervene"]):,}명</div>'
        f'</div>'

        # 개입 비용
        f'<div style="text-align:right">'
        f'<div style="color:#667085;font-size:.74rem">개입 비용</div>'
        f'<div style="font-weight:700">${r["offer_cost"]:,.0f}</div>'
        f'</div>'

        # 보존 LTV
        f'<div style="text-align:right">'
        f'<div style="color:#667085;font-size:.74rem">보존 LTV</div>'
        f'<div style="font-weight:700;color:{T.GOOD}">${r["saved_ltv"]:,.0f}</div>'
        f'</div>'

        # 순이익
        f'<div style="text-align:right">'
        f'<div style="color:#667085;font-size:.74rem">순이익</div>'
        f'<div style="font-weight:800;color:{net_c}">${r["net"]:+,.0f}</div>'
        f'</div>'

        # ROI
        f'<div style="text-align:right">'
        f'<div style="color:#667085;font-size:.74rem">ROI</div>'
        f'<div style="font-weight:800;font-size:1.05rem;color:{roi_c}">{roi_str}</div>'
        f'</div>'

        # 손익분기 전환율
        f'<div style="text-align:right">'
        f'<div style="color:#667085;font-size:.74rem">손익분기 전환율</div>'
        f'<div style="font-weight:700">{be_str}</div>'
        f'</div>'

        f'</div>'
    )


def _result_header() -> str:
    cols = ["세그먼트 / 개입 인원", "개입 비용", "보존 LTV", "순이익", "ROI", "손익분기 전환율"]
    cells = "".join(
        f'<div style="text-align:{"left" if i==0 else "right"};'
        f'font-size:.76rem;font-weight:700;color:#667085;'
        f'text-transform:uppercase;letter-spacing:.03em">{c}</div>'
        for i, c in enumerate(cols)
    )
    return (
        f'<div style="display:grid;grid-template-columns:1.6fr 1fr 1fr 1fr 1fr 1fr;'
        f'gap:.4rem;padding:.4rem .5rem;border-bottom:2px solid #e6e9f2">'
        f'{cells}</div>'
    )


# ---------------------------------------------------------------------------
# 총합 요약 KPI
# ---------------------------------------------------------------------------
def _summary_kpis(res: pd.DataFrame) -> str:
    total_cost  = res["offer_cost"].sum()
    total_ltv   = res["saved_ltv"].sum()
    total_net   = res["net"].sum()
    total_roi   = (total_net / total_cost * 100) if total_cost > 0 else float("nan")
    n_total     = int(res["n_intervene"].sum())

    net_c = _net_color(total_net)
    roi_c = _roi_color(total_roi)
    roi_str = f'{total_roi:+.0f}%' if not math.isnan(total_roi) else '—'

    return T.kpi_row([
        T.kpi("총 개입 인원", f"{n_total:,}명", suffix="이탈확률 50%+ 고위험군"),
        T.kpi("총 개입 비용",  f"${total_cost:,.0f}"),
        T.kpi("총 보존 LTV",   f"${total_ltv:,.0f}", accent=True),
        T.kpi("총 순이익",     f"${total_net:+,.0f}"),
        T.kpi("전체 ROI",      roi_str),
    ])


# ---------------------------------------------------------------------------
# 손익분기 전환율 해설
# ---------------------------------------------------------------------------
def _breakeven_callout(res: pd.DataFrame, conversion_rate: float) -> str:
    need_be = res[res["breakeven_cr"] > conversion_rate]
    ok_be   = res[res["breakeven_cr"] <= conversion_rate]
    parts   = []
    if len(ok_be):
        names = " · ".join(ok_be["name"].tolist())
        parts.append(
            f'✅ <b>{names}</b>: 입력한 전환율({conversion_rate*100:.0f}%)이 손익분기점을 '
            f'넘어 <b style="color:{T.GOOD}">흑자</b>가 예상됩니다.'
        )
    if len(need_be):
        for _, r in need_be.iterrows():
            be = r["breakeven_cr"]
            be_str = f'{be*100:.1f}%' if not (math.isnan(be) or math.isinf(be)) else '—'
            parts.append(
                f'⚠️ <b>{r["name"]}</b>: 전환율이 최소 <b>{be_str}</b> 이상이어야 '
                f'손익분기점을 넘습니다. (현재 가정: {conversion_rate*100:.0f}%)'
            )
    if not parts:
        return ''
    inner = '<br>'.join(parts)
    return (
        f'<div style="background:#f0f7f4;border-left:3px solid {T.GOOD};'
        f'border-radius:0 10px 10px 0;padding:.8rem 1rem;'
        f'font-size:.86rem;line-height:1.8;margin-top:.5rem">'
        f'{inner}</div>'
    )


# ---------------------------------------------------------------------------
# 세그먼트별 순이익 가로 막대
# ---------------------------------------------------------------------------
def _net_bars(res: pd.DataFrame) -> str:
    if res.empty:
        return ''
    max_abs = res["net"].abs().max() or 1
    bars = ''
    for _, r in res.iterrows():
        pct   = abs(r["net"]) / max_abs * 100
        color = _net_color(r["net"])
        label = f'${r["net"]:+,.0f}'
        bars += (
            f'<div style="margin-bottom:.55rem">'
            f'<div style="display:flex;justify-content:space-between;'
            f'font-size:.82rem;margin-bottom:.2rem">'
            f'<span style="font-weight:600">{r["name"]}</span>'
            f'<span style="font-weight:700;color:{color}">{label}</span></div>'
            f'<div style="background:#f0f2f8;border-radius:6px;height:10px">'
            f'<div style="width:{pct:.1f}%;background:{color};'
            f'border-radius:6px;height:10px"></div></div>'
            f'</div>'
        )
    return bars


# ---------------------------------------------------------------------------
# 메인 렌더
# ---------------------------------------------------------------------------
def render():
    df, _ = D.get_scored()
    seg_stats = D.roi_segment_stats(df)

    T.html(T.page_header(
        "ROI 시뮬레이션",
        "리텐션 오퍼 비용 대비 보존 LTV를 세그먼트별로 비교합니다. "
        "오퍼 단가와 전환율은 가정값으로 직접 입력하세요."
    ))

    # ── 레이아웃 ──────────────────────────────────────────────────────────
    col_ctrl, col_result = st.columns([1, 1.8], gap="large")

    # ── 왼쪽: 입력 컨트롤 ──────────────────────────────────────────────────
    with col_ctrl:
        T.html(
            f'<div style="font-size:.82rem;font-weight:700;color:#667085;'
            f'text-transform:uppercase;letter-spacing:.04em;margin-bottom:.6rem">'
            f'시뮬레이션 설정</div>'
        )

        # ① 오퍼 설계
        T.html(T.card(T.card_title("① 오퍼 설계", "1인당 비용과 예상 전환율을 설정하세요.")))

        offer_type = st.radio(
            "오퍼 종류",
            ["고정 금액 ($)", "월 요금 할인율 (%)"],
            horizontal=True,
            key="roi_offer_type",
        )

        if offer_type == "고정 금액 ($)":
            offer_fixed = st.slider(
                "1인당 오퍼 금액 ($)",
                min_value=1, max_value=500, value=30, step=1,
                key="roi_offer_fixed",
            )
            # 세그먼트별로 offer_cost_per는 동일 (고정)
            offer_cost_fn = lambda avg_mc: float(offer_fixed)
            offer_label = f"고정 ${offer_fixed}/인"
        else:
            offer_pct = st.slider(
                "월 요금 할인율 (%)",
                min_value=1, max_value=50, value=10, step=1,
                key="roi_offer_pct",
            )
            # 세그먼트 평균 월요금 기준으로 환산 (1개월치 할인)
            offer_cost_fn = lambda avg_mc: avg_mc * offer_pct / 100
            offer_label = f"월 요금 {offer_pct}% 할인"

        conversion_rate = st.slider(
            "예상 전환율 (%) — 오퍼 수락 후 실제 이탈 방지",
            min_value=1, max_value=100, value=30, step=1,
            key="roi_conversion",
            help="오퍼를 받은 고객 중 실제로 이탈하지 않을 비율 (가정값)",
        ) / 100

        st.markdown("---")

        # ② 세그먼트 선택 & 개입 인원
        T.html(T.card(T.card_title(
            "② 개입 대상 세그먼트",
            "세그먼트를 선택하고 오퍼를 제안할 인원 수를 설정하세요."
        )))

        n_intervene: dict[int, int] = {}
        selected_segs: list[int] = []

        for _, r in seg_stats.iterrows():
            seg = int(r["segment"])
            n, sel = _seg_input(r, df)
            if sel:
                n_intervene[seg] = n
                selected_segs.append(seg)
            st.markdown(" ")

    # ── 오른쪽: 결과 ────────────────────────────────────────────────────────
    with col_result:
        T.html(
            f'<div style="font-size:.82rem;font-weight:700;color:#667085;'
            f'text-transform:uppercase;letter-spacing:.04em;margin-bottom:.6rem">'
            f'시뮬레이션 결과</div>'
        )

        if not selected_segs:
            T.html(
                '<div style="background:#f3f6fc;border-radius:12px;'
                'padding:2rem;text-align:center;color:#667085">'
                '왼쪽에서 세그먼트를 하나 이상 선택하면<br>결과가 표시됩니다.</div>'
            )
            return

        # 세그먼트별 offer_cost_per 계산 (할인율인 경우 avg_mc 기준)
        rows_calc = []
        for _, r in seg_stats[seg_stats["segment"].isin(selected_segs)].iterrows():
            seg = int(r["segment"])
            cost_per = offer_cost_fn(r["avg_mc"])
            rows_calc.append((seg, cost_per))

        # roi_calc 호출 — 세그먼트마다 offer_cost_per가 다를 수 있어 개별 계산 후 합침
        result_rows = []
        for seg, cost_per in rows_calc:
            r_stat = seg_stats[seg_stats["segment"] == seg].iloc[0]
            n = n_intervene.get(seg, 0)
            if n == 0:
                continue
            ltv_per   = float(r_stat["avg_mc"]) * float(r_stat["remaining"])
            saved_ltv = n * conversion_rate * ltv_per
            cost      = n * cost_per
            net       = saved_ltv - cost
            roi_pct   = (net / cost * 100) if cost > 0 else float("nan")
            breakeven = (cost / (n * ltv_per)) if (n * ltv_per) > 0 else float("nan")
            result_rows.append({
                "segment":     seg,
                "name":        r_stat["name"],
                "range":       r_stat["range"],
                "n_intervene": n,
                "offer_cost":  cost,
                "saved_ltv":   saved_ltv,
                "net":         net,
                "roi_pct":     roi_pct,
                "breakeven_cr": breakeven,
                "avg_mc":      r_stat["avg_mc"],
                "remaining":   r_stat["remaining"],
            })
        res = pd.DataFrame(result_rows)

        if res.empty:
            T.html('<div class="callout">개입 인원이 0명입니다. 슬라이더를 조정해 주세요.</div>')
            return

        # ── 총합 KPI ────────────────────────────────────────────────────
        T.html(_summary_kpis(res))

        st.markdown(" ")

        # ── 세그먼트별 상세 테이블 ───────────────────────────────────────
        T.html(T.card(
            T.card_title(
                "세그먼트별 비교",
                f"오퍼: {offer_label} · 전환율 가정: {conversion_rate*100:.0f}%"
            )
            + _result_header()
            + "".join(_result_row(r) for _, r in res.iterrows())
        ))

        # ── 손익분기 해설 ────────────────────────────────────────────────
        T.html(_breakeven_callout(res, conversion_rate))

        st.markdown(" ")

        # ── 순이익 막대 ──────────────────────────────────────────────────
        T.html(T.card(
            T.card_title("세그먼트별 순이익", "보존 LTV − 개입 비용")
            + _net_bars(res)
        ))

    # ── 하단 주석 ──────────────────────────────────────────────────────────
    st.markdown("---")
    T.html(
        f'<div class="note">'
        f'※ 보존 LTV = 개입인원 × 전환율 × 세그먼트 평균 월요금 × 잔존기간 추정값. '
        f'잔존기간은 (데이터 최대 tenure {int(df.tenure.max())}개월 − 세그먼트 평균 tenure)로 추정한 근사값입니다. '
        f'이 데이터는 절대 가입일이 없는 스냅샷 데이터이므로 잔존기간은 실제 계약 잔여 기간이 아닌 통계적 추정치입니다. '
        f'오퍼 단가·전환율은 가정값이며 실제 캠페인 결과와 다를 수 있습니다. '
        f'통화 단위는 데이터 기준(USD).'
        f'</div>'
    )
