"""
분석 B - 세그먼트별 위험속성 탐지 (기획_메모.md 4.1-A 참조)

분석 A와 동일한 구조를 세그먼트별 속성 분석에 재사용:
Ⓐ 패턴 탐지: 세그먼트별 가지치기 결정나무로 주요 위험속성 탐지
Ⓑ 적절성 검증: 그 속성들만으로 만든 모델의 AUC + 순열검정
Ⓒ 표본충분성 확인: Ⓑ의 AUC 부트스트랩 신뢰구간
  (분석B는 세그먼트 안에서 또 fold를 나누므로 분석A보다 표본부족 위험이 큼 - 필수 단계)
"""
from typing import Sequence
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config  # noqa: E402
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
from columns import ANALYSIS_B_NUMERIC_COLS, CATEGORICAL_COLS  # noqa: E402

# ⚠️ 컬럼 목록을 여기서 다시 정의하지 말 것 - shared/columns.py 가 단일 출처.
# 이 NUMERIC_COLS는 churn_prediction의 스케일링 대상(SCALING_NUMERIC_COLS)과
# 다른, 분석B 모델 입력 전용 목록이다 - 과거에 이름은 같은데 내용이 다른
# 두 NUMERIC_COLS가 따로 존재했던 버그를 막기 위해 ANALYSIS_B_NUMERIC_COLS로
# 명시적으로 구분해서 가져온다.
NUMERIC_COLS = ANALYSIS_B_NUMERIC_COLS


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df_enc = df.copy()
    for col in CATEGORICAL_COLS:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))
    return df_enc


# ---------------------------------------------------------------------------
# Ⓐ 패턴 탐지
# ---------------------------------------------------------------------------

def find_top_risk_attributes(
    df_segment: pd.DataFrame, random_state: int = 42,
) -> list[str]:
    """
    세그먼트 안에서 가지치기 결정나무로 "어떤 속성이 이탈 분기에 가장 크게
    기여하는지" 확인.

    ⚠️ [수정] 예전에는 top_n=2로 "정확히 2개"를 사람이 미리 정해서 잘라냈는데,
    이건 우리가 K-means/분석A에서 거듭 경계해온 "사람이 결과를 미리 정한다"
    패턴과 같은 문제였다. 실제로 세그먼트0은 3번째 속성(OnlineSecurity)이
    추가 판별력(AUC +0.0156)을 주는데 top_n=2가 그걸 잘라내고 있었다.

    그래서 "importance > 0인 속성을 전부" 반환하도록 바꿨다 - 개수를 사람이
    정하지 않고, max_depth=2(트리 자체의 제약)가 자연스럽게 막아주는 만큼만
    데이터가 결정하게 한다. 실데이터 확인 결과 세그먼트마다 1~3개로 자연스럽게
    달라짐(노이즈로 끝없이 늘어나지 않음) - max_depth가 이미 안전장치 역할.
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
    return importances.index.tolist()


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

    [버그 수정, OOB 방식] 부트스트랩(복원추출)으로 뽑은 표본 "안에서" 다시
    5-fold를 나누면, 같은 원본 행의 중복 복사본이 Train fold와 Test fold에
    동시에 들어가 데이터 누수가 생긴다 (실데이터 확인: 점추정 AUC=0.7419인데
    부트스트랩 평균이 0.7699로 체계적으로 더 높게 나옴 - 신뢰구간 하한이
    점추정값보다 높아지는 비정상적 결과로 발견됨).

    해결: Out-of-Bag(OOB) 방식 - 복원추출로 뽑힌 행으로만 학습하고, 한 번도
    뽑히지 않은 행(보통 전체의 약 37%)으로만 평가한다. Train/Test가 절대
    겹치지 않는다는 게 구조적으로 보장됨. 부트스트랩(복원추출) 자체는
    그대로 유지 - "표본 변동성을 재현한다"는 핵심 원리는 안 바뀜.
    """
    rng = np.random.RandomState(random_state)
    n = len(df_segment)
    df_enc = _encode_categoricals(df_segment)
    X_all = df_enc[list(attributes)].values
    y_all = df_enc["ChurnFlag"].values

    boot_aucs = []
    for _ in range(n_bootstrap):
        boot_idx = rng.choice(n, n, replace=True)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[np.unique(boot_idx)] = False
        oob_idx = np.where(oob_mask)[0]

        if len(oob_idx) < 10 or len(np.unique(y_all[oob_idx])) < 2:
            continue  # OOB 표본이 너무 적거나 한 클래스만 있으면 AUC 계산 불가
        if len(np.unique(y_all[boot_idx])) < 2:
            continue

        model = RandomForestClassifier(
            n_estimators=100, max_depth=4, random_state=random_state
        )
        model.fit(X_all[boot_idx], y_all[boot_idx])
        proba = model.predict_proba(X_all[oob_idx])[:, 1]
        boot_aucs.append(roc_auc_score(y_all[oob_idx], proba))

    boot_aucs = np.array(boot_aucs)
    ci_low, ci_high = np.percentile(boot_aucs, [2.5, 97.5])
    return float(boot_aucs.mean()), float(ci_low), float(ci_high)


# ---------------------------------------------------------------------------
# 세그먼트별 전체 사이클
# ---------------------------------------------------------------------------

def run_analysis_b(
    df_train_with_segment: pd.DataFrame,
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

        top_attrs = find_top_risk_attributes(df_segment)
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
