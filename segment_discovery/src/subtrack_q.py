"""
서브 트랙 Q - 위험신호 누적 분석 (기획_메모.md 4.1-B 참조)

메인 트랙(분석A·B)과 달리 예측모델 설계에 구조적으로 얽혀있지 않으나,
검증 엄격도(순열검정·부트스트랩·단독AUC)는 메인 트랙과 동등.

목적: "여러 위험신호를 동시에 가진 고객 집단을 찾고 정량화" 한다.
(주의) "위험요인 결합 시 시너지로 더 위험해진다"는 의미가 아님 -
가법/곱셈모델 양쪽 검증 결과 시너지는 없고 중복/상쇄로 확인됨.

① 위험요인 선정: 분석B가 검증한 세그먼트별 주요 위험속성을 그대로 사용
② risk_count 생성: 위험속성 보유 개수를 정수로 집계
③ 검증: 순열검정 + 최고위험구간 부트스트랩 신뢰구간 (분석A의 ②③과 동일 절차)
보조 탐색: K-means 클러스터링 (risk_count 가 다루지 못하는 미발견 위험조합 탐색용)
"""
from typing import Sequence
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config  # noqa: E402

from analysis_b import CATEGORICAL_COLS  # noqa: E402


# ---------------------------------------------------------------------------
# ① 위험요인 선정 + ② risk_count 생성
# ---------------------------------------------------------------------------

def build_risk_factor_masks(
    df: pd.DataFrame, risk_attribute_values: dict[str, str],
) -> pd.DataFrame:
    """
    분석B가 검증한 위험속성 = 특정 값(예: Contract=='Month-to-month')의
    불리언 마스크를 만들어 합산할 수 있게 함.

    Parameters
    ----------
    risk_attribute_values : {컬럼명: "위험으로 판단되는 값"}
        예: {"Contract": "Month-to-month", "OnlineSecurity": "No", ...}
    """
    masks = pd.DataFrame(index=df.index)
    for col, risky_value in risk_attribute_values.items():
        masks[col] = (df[col] == risky_value).astype(int)
    return masks


def compute_risk_count(df: pd.DataFrame, risk_attribute_values: dict[str, str]) -> pd.Series:
    """각 고객이 보유한 위험속성 개수를 정수로 집계"""
    masks = build_risk_factor_masks(df, risk_attribute_values)
    return masks.sum(axis=1).rename("risk_count")


# ---------------------------------------------------------------------------
# ③ 검증: 순열검정 + 부트스트랩 (분석A의 ②③과 동일 절차)
# ---------------------------------------------------------------------------

def risk_count_only_auc(
    risk_count: pd.Series, churn_flag: pd.Series,
    cv_folds: int = config.ANALYSIS_A_CV_FOLDS, random_state: int = config.RANDOM_STATE,
) -> float:
    X = risk_count.values.reshape(-1, 1)
    y = churn_flag.values
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    aucs = []
    for train_idx, test_idx in cv.split(X, y):
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], proba))
    return float(np.mean(aucs))


def permutation_test_for_risk_count(
    risk_count: pd.Series, churn_flag: pd.Series,
    n_permutations: int = config.SUBTRACK_Q_PERMUTATION_COUNT,
    random_state: int = config.RANDOM_STATE,
) -> float:
    """순열검정으로 risk_count~이탈여부 관계가 우연이 아님을 확인"""
    observed_var = churn_flag.groupby(risk_count).mean().var()
    rng = np.random.RandomState(random_state)
    y_values = churn_flag.values
    perm_vars = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(y_values)
        tmp = pd.DataFrame({"risk_count": risk_count.values, "churn": shuffled})
        perm_vars.append(tmp.groupby("risk_count")["churn"].mean().var())
    perm_vars = np.array(perm_vars)
    return float((perm_vars >= observed_var).mean())


def bootstrap_top_risk_group_ci(
    df: pd.DataFrame, risk_count_col: str = "risk_count", churn_col: str = "ChurnFlag",
    n_bootstrap: int = config.SUBTRACK_Q_BOOTSTRAP_COUNT,
    random_state: int = config.RANDOM_STATE,
) -> tuple[int, float, float]:
    """최고위험구간(risk_count 최댓값)의 부트스트랩 신뢰구간으로 안정성 점검"""
    top_value = int(df[risk_count_col].max())
    top_group = df[df[risk_count_col] == top_value][churn_col].values

    rng = np.random.RandomState(random_state)
    boot_means = [
        rng.choice(top_group, len(top_group), replace=True).mean()
        for _ in range(n_bootstrap)
    ]
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return top_value, float(ci_low), float(ci_high)


# ---------------------------------------------------------------------------
# 보조 탐색: K-means 클러스터링 (메인 아님, risk_count 의 미발견조합 보완용)
# ---------------------------------------------------------------------------

def run_kmeans_exploration(
    df: pd.DataFrame, n_clusters: int = config.SUBTRACK_Q_KMEANS_CLUSTERS,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """
    전체데이터 1회, tenure+전체속성으로 클러스터링.
    risk_count 가 다루지 못하는 미발견 위험조합을 발견적으로 탐색하는 용도.
    """
    df_enc = df.copy()
    for col in CATEGORICAL_COLS:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))

    # ⚠️ "tenure+전체속성"이라는 docstring 의도를 코드로 명시적으로 보장.
    # 이전에는 analysis_b.NUMERIC_COLS(분석B 전용, SeniorCitizen+MonthlyCharges
    # 라는 작은 부분집합)를 가져다 썼는데, 우연히 tenure/TotalCharges를 더하면
    # 전체 컬럼과 일치했을 뿐 - analysis_b의 입력 목록이 바뀌면 여기 K-means
    # 피처도 의도와 다르게 같이 바뀌는 위험이 있었다. 여기서 직접
    # "SeniorCitizen, MonthlyCharges, tenure, TotalCharges"를 명시한다.
    other_numeric_cols = ["SeniorCitizen", "MonthlyCharges", "tenure", "TotalCharges"]
    features = CATEGORICAL_COLS + other_numeric_cols
    features = [f for f in features if f in df_enc.columns]
    X_scaled = StandardScaler().fit_transform(df_enc[features])

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df_enc["cluster"] = km.fit_predict(X_scaled)

    summary = df_enc.groupby("cluster").agg(
        n=("ChurnFlag", "size"),
        churn_rate=("ChurnFlag", "mean"),
        avg_tenure=("tenure", "mean"),
    ).sort_values("churn_rate", ascending=False)
    return summary


# ---------------------------------------------------------------------------
# 전체 사이클
# ---------------------------------------------------------------------------

def run_subtrack_q(
    df: pd.DataFrame, risk_attribute_values: dict[str, str],
    n_permutations: int = config.SUBTRACK_Q_PERMUTATION_COUNT,
    n_bootstrap: int = config.SUBTRACK_Q_BOOTSTRAP_COUNT,
    kmeans_clusters: int = config.SUBTRACK_Q_KMEANS_CLUSTERS,
) -> dict:
    """
    Returns
    -------
    result : dict with keys
        risk_count (Series), risk_count_only_auc, p_value,
        top_risk_count_value, ci_low, ci_high, kmeans_summary (보조)
    """
    risk_count = compute_risk_count(df, risk_attribute_values)
    df_with_rc = df.copy()
    df_with_rc["risk_count"] = risk_count

    auc = risk_count_only_auc(risk_count, df["ChurnFlag"])
    p_value = permutation_test_for_risk_count(
        risk_count, df["ChurnFlag"], n_permutations=n_permutations
    )
    top_value, ci_low, ci_high = bootstrap_top_risk_group_ci(
        df_with_rc, n_bootstrap=n_bootstrap
    )
    kmeans_summary = run_kmeans_exploration(df, n_clusters=kmeans_clusters)

    return {
        "risk_count": risk_count,
        "risk_count_only_auc": auc,
        "p_value": p_value,
        "top_risk_count_value": top_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "kmeans_summary": kmeans_summary,
    }
