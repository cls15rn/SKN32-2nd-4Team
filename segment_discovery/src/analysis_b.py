"""
분석 B - 세그먼트별 위험속성 탐지 (기획_메모.md 4.1-A 참조)

분석 A와 동일한 구조를 세그먼트별 속성 분석에 재사용:
Ⓐ 패턴 탐지: 세그먼트별 가지치기 결정나무로 주요 위험속성 탐지
Ⓑ 적절성 검증: 그 속성들만으로 만든 모델의 AUC + 순열검정
Ⓒ 표본충분성 확인: Ⓑ의 AUC 부트스트랩 신뢰구간
  (분석B는 세그먼트 안에서 또 fold를 나누므로 분석A보다 표본부족 위험이 큼 - 필수 단계)
"""
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
NUMERIC_COLS = ["SeniorCitizen", "MonthlyCharges"]


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df_enc = df.copy()
    for col in CATEGORICAL_COLS:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))
    return df_enc


# ---------------------------------------------------------------------------
# Ⓐ 패턴 탐지
# ---------------------------------------------------------------------------

def find_top_risk_attributes(
    df_segment: pd.DataFrame, top_n: int = 2, random_state: int = 42,
) -> list[str]:
    """
    세그먼트 안에서 가지치기 결정나무로 "어떤 속성이 이탈 분기에 가장 크게
    기여하는지" 확인 (feature_importances_ 기준 상위 N개)
    """
    df_enc = _encode_categoricals(df_segment)
    feature_cols = CATEGORICAL_COLS + NUMERIC_COLS
    X = df_enc[feature_cols]
    y = df_enc["ChurnFlag"]

    min_leaf = max(30, len(df_segment) // 20)
    tree = DecisionTreeClassifier(
        max_depth=2, min_samples_leaf=min_leaf, random_state=random_state
    )
    tree.fit(X, y)

    importances = pd.Series(tree.feature_importances_, index=feature_cols)
    importances = importances[importances > 0].sort_values(ascending=False)
    return importances.head(top_n).index.tolist()


# ---------------------------------------------------------------------------
# Ⓑ 적절성 검증: 속성기반 AUC + 순열검정
# ---------------------------------------------------------------------------

def attribute_based_auc(
    df_segment: pd.DataFrame, attributes: Sequence[str],
    cv_folds: int = 5, random_state: int = 42,
) -> float:
    """그 속성들만으로 만든 모델의 교차검증 AUC"""
    df_enc = _encode_categoricals(df_segment)
    X = df_enc[list(attributes)].values
    y = df_enc["ChurnFlag"].values

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    aucs = []
    for train_idx, test_idx in cv.split(X, y):
        model = RandomForestClassifier(
            n_estimators=100, max_depth=4, random_state=random_state
        )
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], proba))
    return float(np.mean(aucs))


def permutation_test_for_attributes(
    df_segment: pd.DataFrame, attributes: Sequence[str],
    n_permutations: int = 200, random_state: int = 42,
) -> float:
    """분석A의 ②와 동일한 절차 - 라벨 순열검정으로 우연이 아님을 확인"""
    observed = attribute_based_auc(df_segment, attributes, random_state=random_state)
    rng = np.random.RandomState(random_state)
    df_enc = _encode_categoricals(df_segment)
    X = df_enc[list(attributes)].values
    y_values = df_enc["ChurnFlag"].values

    permuted_aucs = []
    for _ in range(n_permutations):
        y_shuffled = rng.permutation(y_values)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        aucs = []
        for train_idx, test_idx in cv.split(X, y_shuffled):
            model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=random_state)
            model.fit(X[train_idx], y_shuffled[train_idx])
            proba = model.predict_proba(X[test_idx])[:, 1]
            aucs.append(roc_auc_score(y_shuffled[test_idx], proba))
        permuted_aucs.append(np.mean(aucs))
    permuted_aucs = np.array(permuted_aucs)
    return float((permuted_aucs >= observed).mean())


# ---------------------------------------------------------------------------
# Ⓒ 표본충분성 확인: Ⓑ의 AUC 부트스트랩 신뢰구간 (분석A의 ③과 동일 절차)
# ---------------------------------------------------------------------------

def bootstrap_attribute_auc_ci(
    df_segment: pd.DataFrame, attributes: Sequence[str],
    n_bootstrap: int = 100, random_state: int = 42,
) -> tuple[float, float, float]:
    """
    ⚠️ 분석B는 세그먼트 안에서 또 5-fold를 나누므로 분석A보다 표본부족 위험이 큼.
    (기획_메모.md 4.1-A 보강 참조)
    """
    rng = np.random.RandomState(random_state)
    n = len(df_segment)
    boot_aucs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        sample = df_segment.iloc[idx]
        if sample["ChurnFlag"].nunique() < 2:
            continue
        auc = attribute_based_auc(sample, attributes, random_state=random_state)
        boot_aucs.append(auc)
    boot_aucs = np.array(boot_aucs)
    ci_low, ci_high = np.percentile(boot_aucs, [2.5, 97.5])
    return float(boot_aucs.mean()), float(ci_low), float(ci_high)


# ---------------------------------------------------------------------------
# 세그먼트별 전체 사이클
# ---------------------------------------------------------------------------

def run_analysis_b(
    df_train_with_segment: pd.DataFrame, top_n: int = 2,
) -> pd.DataFrame:
    """
    각 세그먼트에 대해 Ⓐ→Ⓑ→Ⓒ 를 수행해 결과를 표로 정리.

    Returns
    -------
    result_df : columns = [segment, n, churn_rate, top_attributes,
                            attribute_auc, p_value, ci_low, ci_high]
    """
    rows = []
    for segment_id in sorted(df_train_with_segment["segment"].unique()):
        df_segment = df_train_with_segment[df_train_with_segment["segment"] == segment_id]
        if df_segment["ChurnFlag"].nunique() < 2 or len(df_segment) < 50:
            continue

        top_attrs = find_top_risk_attributes(df_segment, top_n=top_n)
        auc = attribute_based_auc(df_segment, top_attrs)
        p_value = permutation_test_for_attributes(df_segment, top_attrs)
        mean_auc, ci_low, ci_high = bootstrap_attribute_auc_ci(df_segment, top_attrs)

        rows.append({
            "segment": segment_id,
            "n": len(df_segment),
            "churn_rate": df_segment["ChurnFlag"].mean(),
            "top_attributes": top_attrs,
            "attribute_auc": auc,
            "p_value": p_value,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })
    return pd.DataFrame(rows)
