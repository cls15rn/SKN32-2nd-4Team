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

[자동화 보강 — 기획_메모.md 4.1-E 참조, 11일차]
순열검정·부트스트랩 둘 다 분석A/B와 같은 원칙(반복횟수를 사람이 고정하지
않고 데이터가 직접 찾음)을 적용한다. 단, 부트스트랩의 이론적 기준이
분석A/B와 다르다 - risk_count 최고위험구간의 부트스트랩은 AUC가 아니라
단순 "이탈률(비율)"의 신뢰구간이므로 Hanley-McNeil(AUC 전용 공식)을 쓸 수
없고, 대신 Wilson 표준오차(비율 추정량의 표준 공식)를 이론값으로 쓴다.
순열검정은 통계량이 달라도(AUC 대신 risk_count별 이탈률의 분산) "관측값을
넘는 순열의 비율"이 항상 이항분포를 따른다는 사실을 이용해 분석A의
Clopper-Pearson 로직(analysis_a.find_stable_permutation_p_value)을 그대로
재사용한다.
"""
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
from stats_formulas import wilson_se  # noqa: E402,F401  (하위 호환 재노출 - shared/stats_formulas.py 참조)

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
# ③ 검증: 순열검정 + 부트스트랩 (분석A의 ②③과 동일 절차, 단 이론값은 다름)
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


def _risk_count_statistic(risk_count: pd.Series, churn_values: np.ndarray) -> float:
    """risk_count~이탈여부 관계의 강도를 나타내는 통계량(risk_count별 이탈률의 분산)"""
    tmp = pd.DataFrame({"risk_count": risk_count.values, "churn": churn_values})
    return float(tmp.groupby("risk_count")["churn"].mean().var())


def permutation_test_for_risk_count(
    risk_count: pd.Series, churn_flag: pd.Series,
    p_threshold: float = config.ANALYSIS_A_P_VALUE_THRESHOLD,
    confidence: float = config.SEQUENTIAL_PERMUTATION_CONFIDENCE,
    check_every: int = config.SEQUENTIAL_PERMUTATION_CHECK_EVERY,
    max_iter: int = config.SEQUENTIAL_PERMUTATION_MAX_ITER,
    random_state: int = config.RANDOM_STATE,
) -> tuple[float, int]:
    """
    순열검정으로 risk_count~이탈여부 관계가 우연이 아님을 확인.

    [순차적 조기중단 — 기획_메모.md 4.1-E 참조] 통계량(risk_count별
    이탈률의 분산)이 분석A(AUC)와 다르지만, "관측값을 넘는 순열의 비율"은
    통계량 종류와 무관하게 항상 이항분포를 따른다 - 분석A의
    find_stable_permutation_p_value(Clopper-Pearson 기반)를 그대로
    재사용한다. 실데이터 검증: 60회에서 멈춤(예전 고정값 500회 대비
    8.3배 절감).

    Returns
    -------
    p_value : 멈춘 시점까지 누적된 (초과횟수/반복횟수) - 추정 p값
    n_permutations_used : 데이터가 직접 찾은 순열 반복횟수
    """
    from analysis_a import find_stable_permutation_p_value  # 지연 import (순환참조 방지)

    observed = _risk_count_statistic(risk_count, churn_flag.values)
    rng = np.random.RandomState(random_state)
    y_values = churn_flag.values

    def _permuted_stat(_i: int) -> float:
        shuffled = rng.permutation(y_values)
        return _risk_count_statistic(risk_count, shuffled)

    return find_stable_permutation_p_value(
        observed_statistic=observed,
        permuted_statistic_fn=_permuted_stat,
        p_threshold=p_threshold,
        confidence=confidence,
        check_every=check_every,
        max_iter=max_iter,
    )


# wilson_se는 파일 상단에서 shared.stats_formulas로부터 재노출됨
# (코드 점검 중 발견된 순환 import 문제 해결 - 12일차 보강, shared/stats_formulas.py 참조)


def find_stable_bootstrap_count_for_risk_group(
    df: pd.DataFrame, risk_count_col: str = "risk_count", churn_col: str = "ChurnFlag",
    random_state: int = config.RANDOM_STATE,
    max_iter: int = config.SEQUENTIAL_MAX_ITER,
    check_every: int = config.SEQUENTIAL_CHECK_EVERY,
    min_n_before_check: int = config.SEQUENTIAL_MIN_N_BEFORE_CHECK,
    structural_gap: float | None = None,
    ceiling_patience: int = config.SEQUENTIAL_CEILING_PATIENCE,
    self_stability_window: int = config.SEQUENTIAL_SELF_STABILITY_WINDOW,
    self_stability_threshold: float = config.SEQUENTIAL_SELF_STABILITY_THRESHOLD,
    record_observation: bool = True,
) -> tuple[int, int, float, float, float, pd.DataFrame]:
    """
    최고위험구간(risk_count 최댓값)의 이탈률 신뢰구간을 구하려면 몇 번
    재추출하면 충분한가를 데이터가 직접 결정한다 (순차적 조기중단,
    analysis_a.find_stable_bootstrap_count와 동일한 G안 구조).

    [Wilson 기반 — analysis_a의 Hanley-McNeil과 다른 점]
    risk_count 최고위험구간 부트스트랩은 AUC가 아니라 단순 "이탈률(비율)"의
    CI이므로, 안전 상한을 Wilson 표준오차×structural_gap으로 계산한다.
    모델 재학습이 없는 단순 재추출이라(분석A/B는 매 반복 RF를 재학습) 구조적
    격차가 더 작은 것이 실측으로 확인됨 - structural_gap의 콜드스타트
    폴백값도 SUBTRACK_Q_STRUCTURAL_GAP(1.1)으로 분석A/B(1.3)보다 타이트하게
    설정.

    gap_calibration과의 연동: structural_gap을 명시하지 않으면(기본값
    None) statistic_type="proportion_wilson"으로 적응형 조회를 한다 -
    AUC 기반 관측치와는 절대 섞이지 않는다(gap_calibration.py 참조).

    Returns
    -------
    stop_n : 멈춘 시점의 누적 부트스트랩 횟수
    top_value : 최고위험구간의 risk_count 값
    mean_p, ci_low, ci_high : 그 시점까지 누적된 표본으로 계산한 이탈률과 신뢰구간
    diagnostics : 체크포인트별 (n, ci_width) 표 (보고서/로그용)
    """
    top_value = int(df[risk_count_col].max())
    top_group = df[df[risk_count_col] == top_value][churn_col].values
    n = len(top_group)

    point_p = float(np.mean(top_group))

    if structural_gap is None:
        from gap_calibration import adaptive_structural_gap
        structural_gap, _gap_source = adaptive_structural_gap(n, statistic_type="proportion_wilson")
    safe_ceiling = 2 * 1.96 * wilson_se(point_p, n) * structural_gap

    rng = np.random.RandomState(random_state)

    def _measure_once() -> float:
        # ⚠️ 단순 재추출뿐(모델 재학습 없음) - 분석A/B의 measurement_fn과
        # 달리 None을 반환할 표본부족 조건이 없다(top_group이 비어있지
        # 않은 한 항상 측정값을 만들 수 있음).
        resampled = rng.choice(top_group, n, replace=True)
        return float(np.mean(resampled))

    from gap_calibration import run_sequential_bootstrap
    stop_n, boot_means, ci_low, ci_high, rows = run_sequential_bootstrap(
        measurement_fn=_measure_once,
        safe_ceiling=safe_ceiling,
        max_iter=max_iter,
        check_every=check_every,
        min_n_before_check=min_n_before_check,
        ceiling_patience=ceiling_patience,
        self_stability_window=self_stability_window,
        self_stability_threshold=self_stability_threshold,
    )

    diagnostics = pd.DataFrame(rows)
    mean_p = float(np.mean(boot_means))
    if record_observation and np.isfinite(ci_low) and np.isfinite(ci_high):
        from gap_calibration import record_proportion_gap_observation
        record_proportion_gap_observation(
            n=n, point_proportion=point_p, ci_width=ci_high - ci_low,
        )
    return stop_n, top_value, mean_p, ci_low, ci_high, diagnostics


def bootstrap_top_risk_group_ci(
    df: pd.DataFrame, risk_count_col: str = "risk_count", churn_col: str = "ChurnFlag",
    n_bootstrap: int = config.SUBTRACK_Q_BOOTSTRAP_COUNT,
    random_state: int = config.RANDOM_STATE,
) -> tuple[int, float, float]:
    """
    ⚠️ [레거시, 고정 반복값] 최고위험구간(risk_count 최댓값)의 부트스트랩
    신뢰구간으로 안정성 점검. find_stable_bootstrap_count_for_risk_group
    (순차적 조기중단)이 메인이며, 이 함수는 고정 반복값이 필요한 경우를
    위한 호환용으로만 남겨둔다.
    """
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
    kmeans_clusters: int = config.SUBTRACK_Q_KMEANS_CLUSTERS,
) -> dict:
    """
    Returns
    -------
    result : dict with keys
        risk_count (Series), risk_count_only_auc, p_value,
        n_permutations_used, top_risk_count_value, ci_low, ci_high,
        n_bootstrap_used, kmeans_summary (보조)
    """
    risk_count = compute_risk_count(df, risk_attribute_values)
    df_with_rc = df.copy()
    df_with_rc["risk_count"] = risk_count

    auc = risk_count_only_auc(risk_count, df["ChurnFlag"])

    # ③ 검증: 순열검정 + 부트스트랩 모두 순차적 조기중단(데이터가 반복횟수 직접 결정)
    p_value, n_permutations_used = permutation_test_for_risk_count(
        risk_count, df["ChurnFlag"]
    )
    n_bootstrap_used, top_value, mean_p, ci_low, ci_high, boot_diagnostics = (
        find_stable_bootstrap_count_for_risk_group(df_with_rc)
    )
    kmeans_summary = run_kmeans_exploration(df, n_clusters=kmeans_clusters)

    return {
        "risk_count": risk_count,
        "risk_count_only_auc": auc,
        "p_value": p_value,
        "n_permutations_used": n_permutations_used,  # 데이터가 직접 찾은 순열 반복횟수
        "top_risk_count_value": top_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_bootstrap_used": n_bootstrap_used,  # 데이터가 직접 찾은 부트스트랩 반복횟수
        "kmeans_summary": kmeans_summary,
    }
