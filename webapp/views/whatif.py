"""
위험 고객 맞춤 프로모션 시뮬레이터 — 고객 속성을 가상으로 바꾸면 이탈 확률이 어떻게 달라지는가.

구조:
  왼쪽  · 고객 검색/선택 + 현재 프로필 카드
  오른쪽 · 속성 변경 컨트롤 → 이탈 확률 변화 실시간 표시

설계 원칙 (기획구현.md 5·7번 반영):
  - 모델 재학습 없음: 저장된 모델(또는 fallback 모델)의 predict_proba만 호출
  - 변경 가능 속성은 위험속성 5종(서브트랙Q 검증) + 계약 관련 주요 속성
  - "성능이 아니라 구조를 보여준다"는 서사 — 변경 전후 SHAP-like 설명 텍스트 포함
  - 세그먼트는 tenure 원본에서 자동 재산출 (변경 불가 — 분석A 정의와 일관)
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import streamlit as st

from lib import data as D
from lib import theme as T

# ---------------------------------------------------------------------------
# 변경 가능한 속성 정의
# 각 항목: (컬럼명, 표시 라벨, 옵션 목록, 위험값, 안전값 설명)
# ---------------------------------------------------------------------------
EDITABLE_ATTRS = [
    {
        "col": "Contract",
        "label": "계약 형태",
        "options": ["Month-to-month", "One year", "Two year"],
        "labels_kr": ["월단위", "1년 약정", "2년 약정"],
        "risk_val": "Month-to-month",
        "safe_val": "Two year",
        "icon": "📋",
        "effect": "계약 기간이 길수록 이탈 확률이 크게 낮아집니다.",
    },
    {
        "col": "OnlineSecurity",
        "label": "온라인 보안",
        "options": ["No", "Yes"],
        "labels_kr": ["미가입", "가입"],
        "risk_val": "No",
        "safe_val": "Yes",
        "icon": "🔒",
        "effect": "보안 서비스 가입이 이탈 확률을 낮추는 핵심 요인입니다.",
    },
    {
        "col": "TechSupport",
        "label": "기술 지원",
        "options": ["No", "Yes"],
        "labels_kr": ["미가입", "가입"],
        "risk_val": "No",
        "safe_val": "Yes",
        "icon": "🛠️",
        "effect": "기술 지원 서비스 가입이 이탈 확률을 낮춥니다.",
    },
    {
        "col": "PaymentMethod",
        "label": "결제 수단",
        "options": [
            "Electronic check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Mailed check",
        ],
        "labels_kr": ["전자수표", "자동이체", "신용카드", "우편수표"],
        "risk_val": "Electronic check",
        "safe_val": "Bank transfer (automatic)",
        "icon": "💳",
        "effect": "자동화된 결제 수단으로 변경 시 이탈 확률이 낮아지는 경향이 있습니다.",
    },
    {
        "col": "InternetService",
        "label": "인터넷 종류",
        "options": ["Fiber optic", "DSL", "No"],
        "labels_kr": ["광랜(Fiber)", "DSL", "인터넷 없음"],
        "risk_val": "Fiber optic",
        "safe_val": "DSL",
        "icon": "🌐",
        "effect": "광랜 사용자의 이탈 확률이 상대적으로 높게 나타납니다.",
    },
    {
        "col": "PaperlessBilling",
        "label": "전자청구서",
        "options": ["Yes", "No"],
        "labels_kr": ["사용", "미사용"],
        "risk_val": "Yes",
        "safe_val": "No",
        "icon": "📄",
        "effect": "전자청구서 사용 고객의 이탈률이 소폭 높습니다.",
    },
    {
        "col": "MonthlyCharges",
        "label": "월 요금 ($)",
        "options": None,  # 슬라이더
        "risk_val": None,
        "safe_val": None,
        "icon": "💰",
        "effect": "월 요금이 높을수록 이탈 확률이 올라가는 경향이 있습니다.",
    },
]

# 범주형 속성만 (슬라이더 제외)
CAT_ATTRS = [a for a in EDITABLE_ATTRS if a["options"] is not None]
NUM_ATTRS = [a for a in EDITABLE_ATTRS if a["options"] is None]


# ---------------------------------------------------------------------------
# 예측 헬퍼 — 단일 행 변형 후 이탈 확률 재산출
# ---------------------------------------------------------------------------
def _predict_single(row_dict: dict, base_df: pd.DataFrame) -> float:
    """
    row_dict: 원본 컬럼 값 dict (변경된 값 포함)
    base_df:  전체 데이터프레임 (get_scored 반환값) — 인코딩 기준으로만 참조
    반환: 이탈 확률 float (0~1)
    """
    clf, feature_cols = D.get_whatif_model()
    if clf is None:
        return float("nan")

    # 1) 단일 행 DataFrame 구성
    single = pd.DataFrame([row_dict])

    # 2) 원본과 동일한 전처리 적용
    single = _preprocess_single(single, base_df, feature_cols)
    if single is None:
        return float("nan")

    try:
        proba = clf.predict_proba(single)[0, 1]
        return float(proba)
    except Exception:
        return float("nan")


def _preprocess_single(single: pd.DataFrame,
                        base_df: pd.DataFrame,
                        feature_cols: list[str]) -> pd.DataFrame | None:
    """단일 행을 feature_cols 기준으로 전처리.

    get_scored()의 base_df는 load_frame() → clean_raw_data() 이후 상태.
    clean_raw_data()는 "No internet service" / "No phone service" → "No" 통합까지만 하고
    이진/범주형 인코딩은 하지 않는다. 따라서 여기서 동일하게 처리해야 한다.
    """
    try:
        from columns import BINARY_MAP_COLS, CATEGORICAL_COLS  # type: ignore
    except ImportError:
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
            from columns import BINARY_MAP_COLS, CATEGORICAL_COLS  # type: ignore
        except ImportError:
            return None

    work = single.copy()

    # 이진 컬럼 매핑 (Yes/No → 1/0)
    for col in BINARY_MAP_COLS:
        if col in work.columns:
            work[col] = work[col].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).fillna(0).astype(float)

    # SeniorCitizen은 이미 0/1이지만 혹시 object인 경우 대비
    if "SeniorCitizen" in work.columns:
        work["SeniorCitizen"] = pd.to_numeric(work["SeniorCitizen"], errors="coerce").fillna(0)

    # 다중 범주형 원-핫 인코딩
    multi = [c for c in CATEGORICAL_COLS if c not in BINARY_MAP_COLS and c in work.columns]
    if "segment" in work.columns:
        multi_full = multi + ["segment"]
    else:
        multi_full = multi
    work = pd.get_dummies(work, columns=multi_full)

    # feature_cols 기준으로 맞추기 (없는 컬럼 0 채움, 순서 정렬)
    for fc in feature_cols:
        if fc not in work.columns:
            work[fc] = 0
    work = work[feature_cols]

    # 모든 컬럼을 float으로 변환 (bool → float 포함)
    work = work.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
    return work


# ---------------------------------------------------------------------------
# 프로필 카드 HTML
# ---------------------------------------------------------------------------
def _prob_badge(prob: float) -> str:
    if np.isnan(prob):
        return '<span style="color:#667085">—</span>'
    pct = prob * 100
    color = T.CORAL if prob >= 0.5 else ("#f5a623" if prob >= 0.3 else T.GOOD)
    return (
        f'<span style="font-size:2.2rem;font-weight:800;color:{color}">'
        f'{pct:.1f}%</span>'
    )


def _delta_html(orig: float, new: float) -> str:
    if np.isnan(orig) or np.isnan(new):
        return ""
    diff = (new - orig) * 100
    if abs(diff) < 0.5:
        arrow, color = "→", T.MUTED
    elif diff < 0:
        arrow, color = "▼", T.GOOD
    else:
        arrow, color = "▲", T.CORAL
    return (
        f'<span style="font-size:1.1rem;font-weight:700;color:{color};margin-left:.5rem">'
        f'{arrow} {abs(diff):.1f}%p</span>'
    )


def _prob_color(prob: float) -> str:
    """이탈 확률 수준에 따른 색상 — 높을수록 위험색."""
    if np.isnan(prob):
        return T.MUTED
    if prob >= 0.5:
        return T.CORAL          # 빨강 — 고위험
    if prob >= 0.3:
        return "#f5a623"        # 주황 — 중위험
    return T.GOOD               # 초록 — 저위험


def _before_after_card(orig: float, new: float) -> str:
    """변경 전 → 변화량 → 변경 후 3분할 카드."""
    # 변화량 계산
    if np.isnan(orig) or np.isnan(new):
        diff, arrow, delta_color, delta_text = 0.0, "→", T.MUTED, "—"
    else:
        diff = (new - orig) * 100
        if diff < -0.5:
            arrow, delta_color = "▼", T.GOOD      # 감소 → 초록 (리텐션 유리)
        elif diff > 0.5:
            arrow, delta_color = "▲", T.CORAL     # 증가 → 빨강 (위험 상승)
        else:
            arrow, delta_color = "→", T.MUTED     # 변화 없음 → 회색
        delta_text = f"{arrow} {abs(diff):.1f}%p"

    orig_str = f"{orig*100:.1f}%" if not np.isnan(orig) else "—"
    new_str  = f"{new*100:.1f}%"  if not np.isnan(new)  else "—"

    orig_color = _prob_color(orig)
    new_color  = _prob_color(new)

    return (
        f'<div class="card" style="background:#f8f9fc;border:1.5px solid #e6e9f2;padding:1.1rem 1.25rem">'
        f'<div style="display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:.5rem">'

        # 변경 전
        f'<div style="text-align:center">'
        f'<div style="color:#667085;font-size:.78rem;margin-bottom:.4rem;font-weight:600">변경 전</div>'
        f'<div style="font-size:2rem;font-weight:800;color:{orig_color}">{orig_str}</div>'
        f'</div>'

        # 가운데 변화량
        f'<div style="text-align:center;padding:0 .5rem">'
        f'<div style="font-size:1.35rem;font-weight:800;color:{delta_color};line-height:1">{delta_text}</div>'
        f'<div style="font-size:.72rem;color:#667085;margin-top:.3rem">변화량</div>'
        f'</div>'

        # 변경 후
        f'<div style="text-align:center">'
        f'<div style="color:#667085;font-size:.78rem;margin-bottom:.4rem;font-weight:600">변경 후</div>'
        f'<div style="font-size:2rem;font-weight:800;color:{new_color}">{new_str}</div>'
        f'</div>'

        f'</div>'
        f'</div>'
    )


def _profile_card(row: pd.Series, rules: dict) -> str:
    seg = int(row["segment"])
    seg_name = D.SEGMENT_NAMES.get(seg, f"세그먼트 {seg + 1}")
    seg_range = D.SEGMENT_RANGES.get(seg, "")
    rv = rules["subtrack_q"]["risk_attribute_values"]
    risk_chips = "".join(
        f'<span style="background:#fff0ef;color:#c0392b;border-radius:6px;'
        f'padding:.15rem .5rem;font-size:.78rem;margin:.15rem .2rem .15rem 0;display:inline-block">'
        f'{D.RISK_LABELS[c]}</span>'
        for c in D.RISK_PRIORITY
        if row.get(c) == rv[c]
    )
    if not risk_chips:
        risk_chips = '<span style="color:#667085;font-size:.82rem">위험신호 없음</span>'

    return f"""
<div class="card" style="margin-bottom:.8rem">
  <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.8rem">
    <div style="width:38px;height:38px;border-radius:50%;background:#eef2ff;
                display:flex;align-items:center;justify-content:center;
                font-weight:800;font-size:.9rem;color:#3b5bdb">
      {str(row.get("customerID","?"))[:2].upper()}
    </div>
    <div>
      <div style="font-weight:700;font-size:.95rem">{row.get("customerID","—")}</div>
      <div style="color:#667085;font-size:.8rem">{seg_name} · {seg_range} · tenure {int(row.get("tenure",0))}개월</div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.5rem;margin-bottom:.75rem">
    <div style="background:#f8f9fc;border-radius:10px;padding:.55rem .7rem">
      <div style="color:#667085;font-size:.75rem;margin-bottom:.2rem">월 요금</div>
      <div style="font-weight:700;font-size:1.05rem">${float(row.get("MonthlyCharges",0)):.0f}</div>
    </div>
    <div style="background:#f8f9fc;border-radius:10px;padding:.55rem .7rem">
      <div style="color:#667085;font-size:.75rem;margin-bottom:.2rem">계약 형태</div>
      <div style="font-weight:700;font-size:.88rem">{row.get("Contract","—")}</div>
    </div>
    <div style="background:#f8f9fc;border-radius:10px;padding:.55rem .7rem">
      <div style="color:#667085;font-size:.75rem;margin-bottom:.2rem">위험신호</div>
      <div style="font-weight:700;font-size:1.05rem">{int(row.get("risk_count",0))}개</div>
    </div>
  </div>
  <div style="font-size:.8rem;color:#667085;margin-bottom:.35rem;font-weight:600">보유 위험신호</div>
  <div>{risk_chips}</div>
</div>
"""


# ---------------------------------------------------------------------------
# 변경 결과 요약 HTML
# ---------------------------------------------------------------------------
def _changes_summary(changes: list[dict]) -> str:
    if not changes:
        return '<div style="color:#667085;font-size:.88rem;padding:.5rem 0">변경된 속성이 없습니다.</div>'
    rows_html = "".join(
        f'<div style="display:flex;align-items:center;gap:.5rem;padding:.3rem 0;'
        f'border-bottom:1px solid #f0f2f8">'
        f'<span style="font-size:.95rem">{c["icon"]}</span>'
        f'<span style="font-size:.85rem;font-weight:600;min-width:90px">{c["label"]}</span>'
        f'<span style="font-size:.82rem;color:#c0392b;text-decoration:line-through">{c["before_kr"]}</span>'
        f'<span style="color:#667085;font-size:.8rem;margin:0 .3rem">→</span>'
        f'<span style="font-size:.82rem;color:#1d7d56;font-weight:700">{c["after_kr"]}</span>'
        f'</div>'
        for c in changes
    )
    return f'<div style="margin:.2rem 0">{rows_html}</div>'


# ---------------------------------------------------------------------------
# 메인 렌더 함수
# ---------------------------------------------------------------------------
def render():
    df, _ = D.get_active_df()
    rules = D.load_rules()

    T.html(T.page_header(
        "위험 고객 맞춤 프로모션 시뮬레이터",
        "고객의 속성을 가상으로 바꿨을 때 이탈 확률이 어떻게 달라지는지 시뮬레이션합니다. "
        "상담원이나 마케터가 고객별 맞춤 오퍼를 설계하는 데 활용할 수 있습니다."
    ))

    # ── 고객 선택 ─────────────────────────────────────────────────────────
    col_search, col_filter = st.columns([2, 1])
    with col_filter:
        seg_filter = st.selectbox(
            "세그먼트 필터",
            ["전체"] + [f"{D.SEGMENT_NAMES[i]} ({D.SEGMENT_RANGES[i]})"
                        for i in range(len(D.SEGMENT_NAMES))],
            key="wi_seg_filter",
        )
    fdf = df.copy()
    if seg_filter != "전체":
        seg_idx = int(seg_filter.split("세그먼트 ")[1][0]) - 1
        fdf = fdf[fdf["segment"] == seg_idx]

    # 이탈 위험 높은 순으로 정렬한 고객 목록
    fdf_sorted = fdf.sort_values("이탈확률", ascending=False)
    cid_options = fdf_sorted["customerID"].tolist()
    display_opts = [
        f"{cid}  —  이탈 {fdf_sorted.loc[fdf_sorted.customerID==cid,'이탈확률'].values[0]*100:.0f}%  "
        f"| {D.SEGMENT_NAMES[int(fdf_sorted.loc[fdf_sorted.customerID==cid,'segment'].values[0])]} "
        f"| 위험신호 {int(fdf_sorted.loc[fdf_sorted.customerID==cid,'risk_count'].values[0])}개"
        for cid in cid_options[:300]
    ]
    with col_search:
        selected_display = st.selectbox(
            "고객 선택 (이탈 위험 높은 순)",
            display_opts,
            key="wi_customer_select",
        )

    selected_cid = selected_display.split("  —  ")[0].strip()
    row = df[df["customerID"] == selected_cid].iloc[0]
    orig_prob = float(row["이탈확률"])

    # ── 레이아웃: 왼쪽(프로필) / 오른쪽(시뮬레이터) ──────────────────────
    col_left, col_right = st.columns([1, 1.6], gap="large")

    with col_left:
        T.html(f'<div style="font-size:.82rem;font-weight:700;color:#667085;'
               f'text-transform:uppercase;letter-spacing:.04em;margin-bottom:.4rem">'
               f'현재 고객 프로필</div>')
        T.html(_profile_card(row, rules))

        # 현재 이탈 확률 강조
        T.html(
            f'<div class="card" style="text-align:center;padding:1.2rem">'
            f'<div style="color:#667085;font-size:.82rem;margin-bottom:.3rem">현재 이탈 확률</div>'
            f'{_prob_badge(orig_prob)}'
            f'<div style="color:#667085;font-size:.78rem;margin-top:.4rem">'
            f'예상 월손실 <b>${float(row["예상손실"]):.1f}</b></div>'
            f'</div>'
        )

        # 세그먼트별 평균 이탈률 참고
        seg_val = rules["analysis_b"][int(row["segment"])]
        T.html(
            f'<div style="background:#f3f6fc;border-radius:12px;padding:.7rem .9rem;'
            f'font-size:.8rem;color:#667085;margin-top:.2rem">'
            f'📊 이 세그먼트 평균 이탈률: '
            f'<b style="color:#1f2740">{seg_val["churn_rate"]*100:.0f}%</b>'
            f'&nbsp;·&nbsp;검증 AUC: <b style="color:#1f2740">{seg_val["attribute_auc"]:.3f}</b>'
            f'</div>'
        )

    with col_right:
        T.html(f'<div style="font-size:.82rem;font-weight:700;color:#667085;'
               f'text-transform:uppercase;letter-spacing:.04em;margin-bottom:.4rem">'
               f'속성 변경 시뮬레이션</div>')

        # 변경 상태를 session_state에 보관
        state_key = f"wi_vals_{selected_cid}"
        if state_key not in st.session_state:
            st.session_state[state_key] = {
                a["col"]: row.get(a["col"]) for a in EDITABLE_ATTRS
            }
        cur = st.session_state[state_key]

        # ── 범주형 속성 컨트롤 ──────────────────────────────────────────
        T.html(T.card(
            T.card_title("위험속성 변경",
                         "검증된 위험속성 5종 + 결제/청구 관련 속성을 변경해 보세요.")
        ))
        # card 내부를 st 위젯으로 채우기 위해 expander 대신 직접 배치
        changed_attrs: list[dict] = []
        for attr in CAT_ATTRS:
            col_a, col_b = st.columns([1, 2])
            with col_a:
                # 현재 값이 위험값인지 표시
                is_risky = (row.get(attr["col"]) == attr.get("risk_val"))
                badge = (
                    f'<span style="background:#fff0ef;color:#c0392b;border-radius:5px;'
                    f'padding:.1rem .4rem;font-size:.72rem;font-weight:700">위험</span> '
                    if is_risky else ""
                )
                T.html(
                    f'<div style="font-size:.85rem;font-weight:600;margin-top:.35rem">'
                    f'{attr["icon"]} {attr["label"]}</div>'
                    f'<div style="font-size:.75rem;margin-top:.1rem">{badge}</div>'
                )
            with col_b:
                opts = attr["options"]
                labels_kr = attr["labels_kr"]
                cur_val = cur[attr["col"]]
                try:
                    cur_idx = opts.index(cur_val)
                except (ValueError, TypeError):
                    cur_idx = 0

                new_val = st.selectbox(
                    label=attr["label"],
                    options=opts,
                    format_func=lambda v, a=attr: a["labels_kr"][a["options"].index(v)],
                    index=cur_idx,
                    key=f"wi_{attr['col']}_{selected_cid}",
                    label_visibility="collapsed",
                )
                cur[attr["col"]] = new_val
                orig_val = row.get(attr["col"])
                if new_val != orig_val:
                    orig_kr = labels_kr[opts.index(orig_val)] if orig_val in opts else str(orig_val)
                    new_kr = labels_kr[opts.index(new_val)]
                    changed_attrs.append({
                        "icon": attr["icon"],
                        "label": attr["label"],
                        "before_kr": orig_kr,
                        "after_kr": new_kr,
                        "effect": attr["effect"],
                    })

        # ── 월 요금 슬라이더 ──────────────────────────────────────────
        mc_min = float(df["MonthlyCharges"].min())
        mc_max = float(df["MonthlyCharges"].max())
        mc_orig = float(row["MonthlyCharges"])
        mc_new = st.slider(
            "💰 월 요금 ($)",
            min_value=mc_min,
            max_value=mc_max,
            value=mc_orig,
            step=1.0,
            key=f"wi_mc_{selected_cid}",
        )
        cur["MonthlyCharges"] = mc_new
        if abs(mc_new - mc_orig) >= 1.0:
            changed_attrs.append({
                "icon": "💰",
                "label": "월 요금",
                "before_kr": f"${mc_orig:.0f}",
                "after_kr": f"${mc_new:.0f}",
                "effect": EDITABLE_ATTRS[-1]["effect"],
            })

        st.markdown("---")

        # ── 변경 요약 + 새 이탈 확률 ───────────────────────────────────
        T.html(f'<div style="font-size:.85rem;font-weight:700;margin-bottom:.4rem">'
               f'변경 내역</div>')
        T.html(_changes_summary(changed_attrs))

        # 새 이탈 확률 계산
        row_modified = row.to_dict()
        row_modified.update(cur)

        new_prob = _predict_single(row_modified, df)

        st.markdown(" ")
        T.html(_before_after_card(orig_prob, new_prob))

        # ── 개입 효과 해설 ────────────────────────────────────────────
        if changed_attrs and not np.isnan(new_prob):
            diff_pct = (new_prob - orig_prob) * 100
            if diff_pct < -1:
                direction = "낮아"
                color_dir = T.GOOD
                key_effects = " · ".join(c["effect"] for c in changed_attrs[:2])
                summary = (
                    f"속성 변경으로 이탈 확률이 <b>{abs(diff_pct):.1f}%p 낮아집니다</b>. "
                    f"{key_effects}"
                )
            elif diff_pct > 1:
                direction = "높아"
                color_dir = T.CORAL
                summary = (
                    f"이 방향의 변경은 이탈 확률을 <b>{diff_pct:.1f}%p 높입니다</b>. "
                    f"고객 리텐션에 불리한 방향입니다."
                )
            else:
                summary = "이탈 확률 변화가 미미합니다 (±1%p 이내). 다른 속성 변경을 시도해 보세요."

            T.html(
                f'<div style="background:#f0f7f4;border-left:3px solid {T.GOOD};'
                f'border-radius:0 10px 10px 0;padding:.7rem .9rem;'
                f'font-size:.85rem;line-height:1.6;margin-top:.6rem">'
                f'💡 {summary}</div>'
            )

        elif not changed_attrs:
            T.html(
                '<div style="background:#f3f6fc;border-radius:10px;padding:.7rem .9rem;'
                'font-size:.84rem;color:#667085;margin-top:.5rem">'
                '위의 컨트롤로 속성을 변경하면 이탈 확률 변화를 실시간으로 확인할 수 있습니다.'
                '</div>'
            )

        # ── 초기화 버튼 ────────────────────────────────────────────────
        st.markdown(" ")
        if st.button("↺ 원래 값으로 초기화", key=f"wi_reset_{selected_cid}"):
            if state_key in st.session_state:
                del st.session_state[state_key]
            st.rerun()

    # ── 하단: 전체 고객 비교 인사이트 ──────────────────────────────────────
    st.markdown("---")
    T.html(T.card(
        T.card_title(
            "위험속성별 평균 이탈률 비교",
            "전체 데이터 기준 — 각 위험속성 보유 고객과 미보유 고객의 이탈률 차이"
        )
    ))
    col1, col2, col3 = st.columns(3)
    rv = rules["subtrack_q"]["risk_attribute_values"]
    for i, attr in enumerate(CAT_ATTRS[:6]):
        col = [col1, col2, col3][i % 3]
        with col:
            held = df[df[attr["col"]] == attr.get("risk_val", "")]["churn"].mean()
            not_held = df[df[attr["col"]] != attr.get("risk_val", "")]["churn"].mean()
            diff = (held - not_held) * 100
            T.html(
                f'<div style="background:#f8f9fc;border-radius:12px;padding:.7rem .9rem;margin-bottom:.6rem">'
                f'<div style="font-size:.8rem;font-weight:700;margin-bottom:.4rem">'
                f'{attr["icon"]} {attr["label"]}</div>'
                f'<div style="display:flex;gap:.5rem;align-items:baseline">'
                f'<span style="font-size:1.1rem;font-weight:800;color:{T.CORAL}">{held*100:.0f}%</span>'
                f'<span style="font-size:.75rem;color:#667085">위험값 보유</span>'
                f'</div>'
                f'<div style="font-size:.75rem;color:#667085">'
                f'미보유 {not_held*100:.0f}% · 차이 <b style="color:{T.CORAL if diff>0 else T.GOOD}">'
                f'{"+" if diff>0 else ""}{diff:.0f}%p</b></div>'
                f'</div>'
            )

    T.html('<div class="note">※ 이탈 확률은 모델 산출값입니다. '
           '속성 변경은 고객 당사자의 동의·계약 변경이 선행되어야 합니다. '
           '통화 단위는 데이터 기준(USD).</div>')
