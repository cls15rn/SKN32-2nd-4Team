"""
webapp/lib/data.py

대시보드가 읽는 단 하나의 데이터 관문.

원칙(README의 "인터페이스는 코드가 아니라 결과 파일"과 동일):
- 세그먼트 경계·위험속성·검증 통계 → segment_discovery/outputs/segment_rules.json
- 고객별 이탈확률 → churn_prediction/outputs/latest/ 의 학습된 모델이 있으면 그대로 사용,
  없으면 대시보드가 자체적으로 가벼운 모델을 한 번 학습(캐시)해 항상 동작하게 한다.
- 그 외 분포·집계(이탈률, tenure 곡선, risk_count 분포 등)는 원본 CSV에서 직접 계산.

이 파일만 segment_discovery / churn_prediction 의 산출물 경로를 안다.
각 페이지(views/*)는 여기서 만들어진 DataFrame·dict 만 받아 그린다.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# streamlit 캐시 데코레이터 — streamlit 없이도 import/테스트 가능하게 폴백 제공
# ---------------------------------------------------------------------------
def _passthrough(*dargs, **dkw):
    if len(dargs) == 1 and callable(dargs[0]) and not dkw:
        return dargs[0]

    def deco(fn):
        return fn

    return deco


try:  # pragma: no cover
    import streamlit as st

    _cache_data = st.cache_data
    _cache_resource = st.cache_resource
except Exception:  # streamlit 미설치 환경(테스트 등)
    st = None
    _cache_data = _passthrough
    _cache_resource = _passthrough

# ---------------------------------------------------------------------------
# 경로 — webapp/lib/data.py 기준으로 프로젝트 루트를 찾는다
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "WA_FnUseC_TelcoCustomerChurn.csv"
RULES_PATH = ROOT / "segment_discovery" / "outputs" / "segment_rules.json"
MODEL_PATH = ROOT / "churn_prediction" / "outputs" / "latest" / "model.pkl"
TRANSFORMER_PATH = ROOT / "churn_prediction" / "outputs" / "latest" / "feature_transformer.pkl"

# shared/ 모듈을 쓰기 위해 경로 추가 (clean_raw_data, 컬럼 정의 단일 출처)
for p in (ROOT / "shared", ROOT / "churn_prediction" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_loader import clean_raw_data  # noqa: E402
from columns import (  # noqa: E402
    BINARY_MAP_COLS,
    CATEGORICAL_COLS,
    SCALING_NUMERIC_COLS,
)

# ---------------------------------------------------------------------------
# 표시용 세그먼트 이름 / 범위 — 경계(boundaries)에서 자동 생성 (의미 이름 하드코딩 안 함)
# ---------------------------------------------------------------------------
def _build_segment_labels(boundaries: list) -> tuple[dict, dict]:
    """경계로부터 세그먼트 이름·범위를 자동 생성.

    - 이름: 성격을 사전 가정하지 않고 '세그먼트 1·2·…' (경계 개수+1 만큼).
    - 범위: 경계값에서 직접 산출 ('0–10개월' …, 마지막 구간은 'N개월+').
    경계가 바뀌거나 개수가 달라져도 그대로 따라간다.
    """
    n = len(boundaries) + 1
    names = {i: f"세그먼트 {i + 1}" for i in range(n)}
    ranges = {}
    for i in range(n):
        lo = 0 if i == 0 else int(boundaries[i - 1] + 0.5)
        if i < len(boundaries):
            ranges[i] = f"{lo}–{int(boundaries[i] - 0.5)}개월"
        else:
            ranges[i] = f"{lo}개월+"
    return names, ranges


try:
    with open(RULES_PATH, "r", encoding="utf-8") as _f:
        _BOUNDS = json.load(_f)["analysis_a"]["boundaries"]
    SEGMENT_NAMES, SEGMENT_RANGES = _build_segment_labels(_BOUNDS)
except Exception:  # 규칙 파일이 아직 없을 때의 안전 기본값
    SEGMENT_NAMES = {i: f"세그먼트 {i + 1}" for i in range(4)}
    SEGMENT_RANGES = {i: f"구간 {i + 1}" for i in range(4)}

# 서브트랙Q에서 검증된 5개 위험속성 → (한글 라벨, 위험값)
RISK_LABELS = {
    "Contract": "월단위 계약",
    "PaymentMethod": "전자수표 결제",
    "InternetService": "광랜(Fiber)",
    "OnlineSecurity": "온라인보안 미가입",
    "TechSupport": "기술지원 미가입",
    "HighCharge": "높은 월요금(상위 30%)",
}
# 핵심 원인 우선순위 (이탈 신호가 강한 순) — 한 고객이 여러 위험속성을 가질 때 무엇을 대표로 보여줄지
RISK_PRIORITY = ["Contract", "OnlineSecurity", "TechSupport", "PaymentMethod", "InternetService"]

# 위험 신호 누적(risk_count)은 서브트랙 Q가 검증한 위 5종으로 고정.
# 아래 '표시용' 목록은 위험 요소별 이탈률/손실 표시에 검증된 연속형 드라이버(월요금)를 더한 것.
# (risk_count 와 분리: risk_count 에 6번째를 넣으면 segment_rules.json 검증과 어긋남)
HIGH_CHARGE_PCTL = 0.70          # 상위 30% = 70퍼센타일 이상 (인과 해설 패널과 동일 기준)
RISK_PRIORITY_DISPLAY = RISK_PRIORITY + ["HighCharge"]


def _ext_risk_values(rules: dict) -> dict:
    """5종 위험값 + 파생 'HighCharge' 위험값('High')을 합친 매핑 (표시용 함수에서 사용)."""
    return {**rules["subtrack_q"]["risk_attribute_values"], "HighCharge": "High"}


# 분석B 속성·범주 표시 그룹 (이미지11) — 컬럼: [(범주명, 표시라벨)]
ANALYSIS_B_GROUPS = {
    "인터넷 종류": ("InternetService", {"Fiber optic": "광랜", "DSL": "DSL", "No": "없음"}),
    "결제수단": (
        "PaymentMethod",
        {
            "Electronic check": "전자수표",
            "Bank transfer (automatic)": "자동이체",
            "Credit card (automatic)": "신용카드",
            "Mailed check": "우편수표",
        },
    ),
    "온라인보안": ("OnlineSecurity", {"No": "미가입", "Yes": "가입"}),
    "기술지원": ("TechSupport", {"No": "미가입", "Yes": "가입"}),
    "계약 형태": (
        "Contract",
        {"Month-to-month": "월단위", "One year": "1년", "Two year": "2년"},
    ),
    "월요금 수준": (
        "HighCharge",
        {"High": "높은 요금 (상위 30%)", "Low": "그 외 요금"},
    ),
}

TARGET_PROB = 0.50  # "우선 대응 고객" = 이탈확률 50% 이상


# ===========================================================================
# 로더
# ===========================================================================
@_cache_data(show_spinner=False)
def load_rules() -> dict:
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@_cache_data(show_spinner=False)
def load_frame() -> pd.DataFrame:
    """원본 → 공통 전처리 → segment / risk_count / churn 라벨까지 부여한 기본 프레임."""
    rules = load_rules()
    df = pd.read_csv(DATA_PATH)
    df = clean_raw_data(df)

    boundaries = rules["analysis_a"]["boundaries"]
    upper = max(df["tenure"].max(), boundaries[-1]) + 1
    bins = [-1] + list(boundaries) + [upper]
    df["segment"] = pd.cut(df["tenure"], bins=bins, labels=range(len(bins) - 1)).astype(int)
    df["segment_name"] = df["segment"].map(SEGMENT_NAMES)

    risk_values = rules["subtrack_q"]["risk_attribute_values"]
    masks = pd.DataFrame(index=df.index)
    for col, risky in risk_values.items():
        masks[col] = (df[col] == risky).astype(int)
    df["risk_count"] = masks.sum(axis=1)

    df["churn"] = (df["Churn"] == "Yes").astype(int)
    return df


# ===========================================================================
# 이탈확률 — 학습된 모델 우선, 없으면 자체 캐시 모델
# ===========================================================================
def _fallback_classifier():
    """xgboost가 있으면 그걸, 없으면 sklearn HistGradientBoosting을 쓴다(둘 다 트리 계열)."""
    try:
        from xgboost import XGBClassifier  # type: ignore

        # 이탈(소수 클래스) 가중 — F2/Recall 우선 서사와 맞추기 위함
        return XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.1,
            scale_pos_weight=2.0,
            eval_metric="logloss",
            random_state=42,
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.1, max_iter=300, random_state=42,
            class_weight="balanced",
        )


@_cache_data(show_spinner=False)
def get_scored() -> tuple[pd.DataFrame, dict]:
    """
    load_frame()에 '이탈확률', '예상손실'(월) 컬럼을 추가해 반환.

    Returns (df, meta) — meta.source 는 'trained'(저장된 모델) 또는 'fallback'(자체 학습).
    """
    df = load_frame().copy()

    proba, source = _try_trained_model(df)
    if proba is None:
        proba = _train_fallback(df)
        source = "fallback"

    df["이탈확률"] = proba
    df["예상손실"] = df["MonthlyCharges"] * df["이탈확률"]
    # 연속형 '높은 월요금'을 표시용 위험 요소로 쓰기 위한 이진 플래그 (전역 70퍼센타일 기준 = 인과 해설과 동일).
    # risk_count(검증된 5종)에는 넣지 않고, 표시용 RISK_PRIORITY_DISPLAY 에서만 사용.
    df["HighCharge"] = np.where(
        df["MonthlyCharges"] >= df["MonthlyCharges"].quantile(HIGH_CHARGE_PCTL),
        "High", "Low")
    meta = {"source": source, "n": len(df)}
    return df, meta


def _try_trained_model(df: pd.DataFrame):
    """outputs/latest/ 에 학습된 모델이 있으면 그것으로 추론."""
    if not (MODEL_PATH.exists() and TRANSFORMER_PATH.exists()):
        return None, None
    try:
        import joblib
        from feature_engineering import FeatureTransformer  # churn_prediction/src

        model = joblib.load(MODEL_PATH)
        transformer = FeatureTransformer.load(TRANSFORMER_PATH)
        transformed = transformer.transform(df)
        proba = model.predict_proba(transformed["X_tree_3"])[:, 1]
        return proba, "trained"
    except Exception:
        return None, None


def _train_fallback(df: pd.DataFrame) -> np.ndarray:
    """저장된 모델이 없을 때 대시보드가 자체적으로 한 번 학습(3단계 = segment 포함)."""
    work = df.copy()
    for col in BINARY_MAP_COLS:
        work[col] = work[col].map({"Yes": 1, "No": 0})
    multi = [c for c in CATEGORICAL_COLS if c not in BINARY_MAP_COLS]
    enc = pd.get_dummies(work, columns=multi + ["segment"])

    drop = {"customerID", "Churn", "segment_name", "churn", "risk_count"}
    feat3 = [c for c in enc.columns if c not in drop and enc[c].dtype != object]
    X = enc[feat3].astype(float)
    y = work["churn"]

    clf = _fallback_classifier()
    clf.fit(X, y)
    return clf.predict_proba(X)[:, 1]


# ===========================================================================
# 집계 헬퍼 (페이지들이 공통으로 쓰는 계산)
# ===========================================================================
def overview_stats(df: pd.DataFrame) -> dict:
    n = len(df)
    churned = int(df["churn"].sum())
    return {
        "total": n,
        "churned": churned,
        "retained": n - churned,
        "churn_rate": df["churn"].mean(),
    }


def tenure_churn_curve(df: pd.DataFrame) -> pd.DataFrame:
    """tenure(0~72)별 평균 이탈률 + 표본수."""
    g = df.groupby("tenure")["churn"].agg(["mean", "count"]).reset_index()
    g.columns = ["tenure", "rate", "count"]
    return g


def tenure_axis(curve: pd.DataFrame) -> tuple[int, list[int]]:
    """tenure 곡선에서 x축 최댓값과 12개월(1년) 간격 눈금을 데이터로 산출.

    고정값(domain=[0,72], values=[0,12,...,72]) 대신 실제 데이터 범위에 맞춤.
    현재 데이터(0~72)에선 [0,12,24,36,48,60,72]로 동일하게 나오고,
    데이터 범위가 달라지면 그에 맞춰 자동 조정된다.
    """
    tmax = int(curve["tenure"].max())
    ticks = list(range(0, tmax + 1, 12))
    return tmax, ticks


def factor_churn_rates(df: pd.DataFrame) -> pd.DataFrame:
    """요인별 '이탈률이 가장 높은 범주'의 이탈률 (이미지5 하단)."""
    rows = [
        ("결제 · 전자수표", df["PaymentMethod"] == "Electronic check"),
        ("계약 · 월단위", df["Contract"] == "Month-to-month"),
        ("인터넷 · 광랜", df["InternetService"] == "Fiber optic"),
        ("시니어 고객", df["SeniorCitizen"] == 1),
        ("전자청구서", df["PaperlessBilling"] == "Yes"),
        ("온라인보안 미가입", df["OnlineSecurity"] == "No"),
        ("기술지원 미가입", df["TechSupport"] == "No"),
    ]
    out = []
    for label, mask in rows:
        sub = df[mask]
        out.append({"label": label, "rate": sub["churn"].mean(), "n": len(sub)})
    res = pd.DataFrame(out).sort_values("rate", ascending=False).reset_index(drop=True)
    return res


def segment_profile(df: pd.DataFrame) -> pd.DataFrame:
    """세그먼트별 고객수 / 이탈률 (전체 데이터 기준)."""
    g = df.groupby("segment")["churn"].agg(["count", "mean"]).reset_index()
    g.columns = ["segment", "count", "rate"]
    g["name"] = g["segment"].map(SEGMENT_NAMES)
    g["range"] = g["segment"].map(SEGMENT_RANGES)
    return g


def risk_count_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """risk_count(0~5)별 고객수 / 이탈률."""
    g = df.groupby("risk_count")["churn"].agg(["count", "mean"]).reset_index()
    g.columns = ["risk_count", "count", "rate"]
    return g


def risk_count_signals(df: pd.DataFrame, top: int = 3) -> dict:
    """risk_count 값별로, 그 그룹 고객이 보유한 위험속성 비율 Top N.

    반환: {risk_count: [(한글라벨, 보유비율), ...]} — 비율 0인 속성은 제외.
    예) risk_count=3 그룹에서 '월단위 계약'을 가진 고객 비율이 0.9면 ("월단위 계약", 0.9).
    """
    rules = load_rules()
    risk_values = rules["subtrack_q"]["risk_attribute_values"]
    out: dict[int, list] = {}
    for k, sub in df.groupby("risk_count"):
        freqs = [(RISK_LABELS[c], float((sub[c] == risk_values[c]).mean()))
                 for c in RISK_PRIORITY]
        freqs.sort(key=lambda x: x[1], reverse=True)
        out[int(k)] = [(lbl, sh) for lbl, sh in freqs[:top] if sh > 0]
    return out


def core_cause(row) -> str:
    """고객 한 명의 '핵심 원인' 한 줄 (추천 오퍼 없이 원인만)."""
    rules = load_rules()
    risk_values = rules["subtrack_q"]["risk_attribute_values"]
    held = [c for c in RISK_PRIORITY if row.get(c) == risk_values[c]]
    seg = int(row["segment"])
    if seg == 0 and (not held or row["risk_count"] <= 2):
        return f"가입 초기 이탈 위험 ({SEGMENT_NAMES[seg]})"
    if held:
        top = held[0]
        return f"{RISK_LABELS[top]} · {SEGMENT_NAMES[seg]}"
    return f"{SEGMENT_NAMES[seg]} 구간 복합 위험"


def priority_customers(df: pd.DataFrame, top: int = 5) -> pd.DataFrame:
    """예상 월손실 순 상위 고객."""
    cols = ["customerID", "segment", "tenure", "risk_count", "MonthlyCharges",
            "이탈확률", "예상손실"]
    out = df.nlargest(top, "예상손실")[cols + ["Contract", "OnlineSecurity",
                                              "TechSupport", "PaymentMethod",
                                              "InternetService"]].copy()
    out["핵심원인"] = out.apply(core_cause, axis=1)
    out["range"] = out["segment"].map(SEGMENT_RANGES)
    return out


def customer_risk_detail(row, rules: dict | None = None,
                         high_charge_threshold: float | None = None) -> dict:
    """고객 한 명의 인과 해설.

    분석 B(세그먼트별 검증 드라이버)·서브트랙 Q(위험 신호 5종)를 고객에 매칭:
      - core_risks : 보유 위험 신호 중 '이 세그먼트에서 통계 검증된 드라이버'에 해당
      - other_risks: 그 외 보유 위험 신호
    """
    if rules is None:
        rules = load_rules()
    rv = rules["subtrack_q"]["risk_attribute_values"]
    seg = int(row["segment"])
    valb = segment_validation(rules, seg)
    top_attrs = set(valb.get("top_attributes", []))

    core, other = [], []
    for c in RISK_PRIORITY:                         # 위험 신호 5종
        if row.get(c) != rv[c]:
            continue
        (core if c in top_attrs else other).append(RISK_LABELS[c])
    # 연속형 '높은 월요금' — 세그먼트 드라이버에 MonthlyCharges가 있을 때만 핵심에 포함
    if ("MonthlyCharges" in top_attrs and high_charge_threshold is not None
            and float(row["MonthlyCharges"]) >= high_charge_threshold):
        core.append("높은 월요금")

    return {
        "customerID": row["customerID"],
        "segment": seg,
        "segment_name": SEGMENT_NAMES[seg],
        "range": SEGMENT_RANGES[seg],
        "prob": float(row["이탈확률"]),
        "loss": float(row["예상손실"]),
        "monthly": float(row["MonthlyCharges"]),
        "tenure": int(row.get("tenure", 0)),
        "risk_count": int(row.get("risk_count", 0)),
        "core_risks": core,
        "other_risks": other,
        "seg_auc": valb.get("attribute_auc"),
        "seg_p": valb.get("p_value"),
    }


def targets(df: pd.DataFrame, prob: float = TARGET_PROB) -> pd.DataFrame:
    return df[df["이탈확률"] >= prob].copy()


def prob_band_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """이탈확률 구간별 고객수·평균 예상손실 (미시 예측 레이어 표시용)."""
    if "이탈확률" not in df.columns:
        return pd.DataFrame(columns=["band", "count", "avg_loss"])
    edges = [0, .2, .4, .6, .8, 1.01]
    labels = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]
    b = pd.cut(df["이탈확률"], bins=edges, labels=labels,
               right=False, include_lowest=True)
    g = (df.groupby(b, observed=False)
           .agg(count=("이탈확률", "size"), avg_loss=("예상손실", "mean"))
           .reset_index())
    g.columns = ["band", "count", "avg_loss"]
    return g


def loss_concentration(t: pd.DataFrame, top_frac: float = 0.20) -> float:
    """상위 top_frac 고객이 차지하는 예상손실 비중."""
    if len(t) == 0:
        return 0.0
    losses = t["예상손실"].sort_values(ascending=False).values
    k = max(1, int(round(len(losses) * top_frac)))
    return losses[:k].sum() / losses.sum()


def loss_by_segment(t: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby("segment").agg(n=("customerID", "count"),
                                 loss=("예상손실", "sum")).reset_index()
    g["name"] = g["segment"].map(SEGMENT_NAMES)
    g["range"] = g["segment"].map(SEGMENT_RANGES)
    total = g["loss"].sum()
    g["share"] = g["loss"] / total if total else 0
    return g.sort_values("segment")


def loss_by_risk_attribute(t: pd.DataFrame) -> pd.DataFrame:
    """위험속성별: 보유 대상 수 / 예상손실 합 / 보유자 이탈률. (검증 5종 + 높은 월요금)"""
    rules = load_rules()
    rv = _ext_risk_values(rules)
    rows = []
    for col in RISK_PRIORITY_DISPLAY:
        held = t[t[col] == rv[col]]
        rows.append({
            "attr": "MonthlyCharges" if col == "HighCharge" else col,
            "label": RISK_LABELS[col],
            "n": len(held),
            "loss": held["예상손실"].sum(),
            "churn": held["churn"].mean() if len(held) else 0.0,
        })
    res = pd.DataFrame(rows).sort_values("loss", ascending=False).reset_index(drop=True)
    return res


def risk_attribute_by_segment(df: pd.DataFrame) -> list:
    """유형(위험속성)별 × 세그먼트별 보유자 이탈률.
    같은 위험 유형이라도 생애주기(세그먼트)에 따라 이탈률이 어떻게 달라지는지 본다.
    반환: [{attr, label, overall_n, overall_churn, segs:[{seg,n,churn|None}]}], 전체 이탈률 내림차순."""
    rules = load_rules()
    rv = _ext_risk_values(rules)
    n_seg = len(SEGMENT_NAMES)
    out = []
    for col in RISK_PRIORITY_DISPLAY:
        held = df[df[col] == rv[col]]
        segs = []
        for s in range(n_seg):
            h = held[held["segment"] == s]
            segs.append({"seg": s, "n": len(h),
                         "churn": (float(h["churn"].mean()) if len(h) else None)})
        out.append({
            "attr": "MonthlyCharges" if col == "HighCharge" else col,
            "label": RISK_LABELS[col],
            "overall_n": len(held),
            "overall_churn": float(held["churn"].mean()) if len(held) else 0.0,
            "segs": segs,
        })
    out.sort(key=lambda r: r["overall_churn"], reverse=True)
    return out


def loss_by_risk_signal(t: pd.DataFrame) -> pd.DataFrame:
    """위험신호(=risk_count) 구간별 분포 + 구간 내 가장 흔한 속성 Top3."""
    rules = load_rules()
    risk_values = rules["subtrack_q"]["risk_attribute_values"]
    buckets = [("6개+", lambda s: s >= 6), ("5개", lambda s: s == 5),
               ("4개", lambda s: s == 4), ("3개", lambda s: s == 3),
               ("1–2개", lambda s: (s >= 1) & (s <= 2))]
    # risk_count는 0~5 범위이므로 6개+는 비어있을 수 있음 → 5개 이하만 의미. 그래도 구조 유지.
    total_loss = t["예상손실"].sum()
    rows = []
    for name, cond in buckets:
        sub = t[cond(t["risk_count"])]
        if len(sub) == 0:
            continue
        freqs = []
        for col in RISK_PRIORITY:
            share = (sub[col] == risk_values[col]).mean()
            freqs.append((RISK_LABELS[col], share))
        freqs.sort(key=lambda x: x[1], reverse=True)
        rows.append({
            "bucket": name,
            "n": len(sub),
            "churn": sub["churn"].mean(),
            "loss": sub["예상손실"].sum(),
            "share": sub["예상손실"].sum() / total_loss if total_loss else 0,
            "top_signals": freqs[:3],
        })
    return pd.DataFrame(rows)


def top_contributors(t: pd.DataFrame, top: int = 3) -> pd.DataFrame:
    """대상 고객의 예상손실을 가장 많이 만드는 위험요소 Top N (이미지6 하단)."""
    res = loss_by_risk_attribute(t)
    return res.head(top)


def segment_attribute_breakdown(df: pd.DataFrame, segment: int) -> dict:
    """분석B 페이지: 특정 세그먼트의 속성·범주별 이탈률/표본수."""
    sub = df[df["segment"] == segment]
    out = {}
    for group, (col, mapping) in ANALYSIS_B_GROUPS.items():
        rows = []
        for raw_val, label in mapping.items():
            cell = sub[sub[col] == raw_val]
            if len(cell) == 0:
                continue
            rows.append({"label": label, "rate": cell["churn"].mean(), "n": len(cell)})
        rows.sort(key=lambda r: r["rate"], reverse=True)
        out[group] = rows
    return out


def segment_validation(rules: dict, segment: int) -> dict:
    """segment_rules.json 의 analysis_b 검증 통계 (AUC, p값)."""
    for item in rules["analysis_b"]:
        if item["segment"] == segment:
            return item
    return {}


# ===========================================================================
# What-If 분석용 — 단일 고객 재예측에 사용하는 모델과 피처 목록 반환
# ===========================================================================
@_cache_resource(show_spinner=False)
def get_whatif_model():
    """
    What-If 분석에서 단일 행 predict_proba 호출에 쓸 (model, feature_cols) 반환.

    - 학습된 모델(outputs/latest/model.pkl + feature_transformer.pkl)이 있으면 그것 사용.
    - 없으면 get_scored()가 내부적으로 학습한 fallback 모델을 재사용.
    반환: (classifier, feature_cols: list[str])
    """
    # 1) 학습된 모델 시도
    if MODEL_PATH.exists() and TRANSFORMER_PATH.exists():
        try:
            import joblib
            from feature_engineering import FeatureTransformer  # churn_prediction/src

            model = joblib.load(MODEL_PATH)
            transformer = FeatureTransformer.load(TRANSFORMER_PATH)
            return model, transformer.feature_cols_3
        except Exception:
            pass

    # 2) Fallback — get_scored()와 동일한 방식으로 모델 재학습 (캐시되므로 두 번 학습 안 됨)
    df = load_frame().copy()
    for col in BINARY_MAP_COLS:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    multi = [c for c in CATEGORICAL_COLS if c not in BINARY_MAP_COLS]
    enc = pd.get_dummies(df, columns=multi + ["segment"])

    drop = {"customerID", "Churn", "segment_name", "churn", "risk_count"}
    feat3 = [c for c in enc.columns if c not in drop and enc[c].dtype != object]
    X = enc[feat3].astype(float)
    y = (df["Churn"] == "Yes").astype(int)

    clf = _fallback_classifier()
    clf.fit(X, y)
    return clf, feat3


# ===========================================================================
# ROI 시뮬레이션용 집계 함수
# ===========================================================================

def roi_segment_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    세그먼트별 ROI 시뮬레이션 기초 통계.

    반환 컬럼:
      segment, name, range,
      n_total       전체 고객 수
      n_high        이탈확률 50%+ 고객 수
      avg_mc        평균 MonthlyCharges
      avg_tenure    평균 tenure
      remaining     잔존기간 추정 (max_tenure - avg_tenure, 기획구현.md 9번)
      avg_prob      평균 이탈확률
      total_loss    예상 월손실 합계
    """
    max_tenure = float(df["tenure"].max())
    rows = []
    for seg in sorted(df["segment"].unique()):
        sub = df[df["segment"] == seg]
        hi  = sub[sub["이탈확률"] >= TARGET_PROB]
        rows.append({
            "segment":    int(seg),
            "name":       SEGMENT_NAMES.get(int(seg), f"세그먼트 {seg+1}"),
            "range":      SEGMENT_RANGES.get(int(seg), ""),
            "n_total":    len(sub),
            "n_high":     len(hi),
            "avg_mc":     float(sub["MonthlyCharges"].mean()),
            "avg_tenure": float(sub["tenure"].mean()),
            "remaining":  float(max_tenure - sub["tenure"].mean()),
            "avg_prob":   float(sub["이탈확률"].mean()),
            "total_loss": float(sub["예상손실"].sum()),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# 전역 데이터 소스 — session_state 기반 (CSV 업로드 공유)
# ===========================================================================


# 업로드 CSV 필수 컬럼 목록 (단일 출처 — 여기서만 정의)
UPLOAD_REQUIRED_COLS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]


def score_uploaded_csv(raw: pd.DataFrame) -> tuple[pd.DataFrame | None, str]:
    """
    업로드된 원본 CSV DataFrame → (scored_df, error_msg).
    get_scored()와 동일한 컬럼 구조를 반환.
    오류 시 (None, 에러 메시지) 반환.
    """
    missing = [c for c in UPLOAD_REQUIRED_COLS if c not in raw.columns]
    if missing:
        return None, f"필수 컬럼 누락: {', '.join(missing)}"

    try:
        df = clean_raw_data(raw.copy())
    except Exception as e:
        return None, f"전처리 오류: {e}"

    # 세그먼트 할당
    rules = load_rules()
    boundaries = rules["analysis_a"]["boundaries"]
    upper = max(df["tenure"].max(), boundaries[-1]) + 1
    bins = [-1] + list(boundaries) + [upper]
    df["segment"] = pd.cut(df["tenure"], bins=bins,
                           labels=range(len(bins) - 1)).astype(int)
    df["segment_name"] = df["segment"].map(SEGMENT_NAMES)

    # risk_count
    risk_values = rules["subtrack_q"]["risk_attribute_values"]
    masks = pd.DataFrame(index=df.index)
    for col, risky in risk_values.items():
        masks[col] = (df[col] == risky).astype(int)
    df["risk_count"] = masks.sum(axis=1)
    df["churn"] = (df["Churn"] == "Yes").astype(int)

    # 이탈확률 예측
    try:
        clf, feature_cols = get_whatif_model()
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
        df["MonthlyCharges"] >= df["MonthlyCharges"].quantile(HIGH_CHARGE_PCTL),
        "High", "Low"
    )
    return df, ""


def get_active_df() -> tuple[pd.DataFrame, dict]:
    """
    현재 활성 데이터소스를 반환.
    - session_state["use_uploaded"] == True 이고 uploaded_df 가 있으면 업로드 데이터
    - 그 외에는 get_scored() (기본 데이터)
    반환: (df, meta)  — meta["source"] = "trained"/"fallback"/"uploaded"
    """
    try:
        import streamlit as _st
        use_up = _st.session_state.get("use_uploaded", False)
        up_df  = _st.session_state.get("uploaded_df", None)
    except Exception:
        use_up, up_df = False, None

    if use_up and up_df is not None:
        return up_df, {"source": "uploaded", "n": len(up_df)}
    return get_scored()
