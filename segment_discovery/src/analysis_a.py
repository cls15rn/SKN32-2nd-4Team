"""
분석 A - 세그먼트 경계 탐지 (기획_메모.md 3장 참조)

① 경계 탐지: 가지치기 회귀나무 (메인) + 랜덤포레스트 투표 (보조검증)
② 적절성 검증: 세그먼트단독 AUC + 순열검정
③ 표본충분성 확인: AUC 측정값의 부트스트랩 신뢰구간

①②③ 은 완전히 별개 절차 - 혼동하지 말 것 (기획_메모.md 3.3 참조)
"""
from collections import Counter
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.tree import DecisionTreeRegressor


# ---------------------------------------------------------------------------
# ① 경계 탐지: 가지치기 회귀나무 (메인) + 랜덤포레스트 투표 (보조검증)
# ---------------------------------------------------------------------------

def find_boundaries_pruned_tree(
    df_train: pd.DataFrame,
    cv_folds: int = 5,
    random_state: int = 42,
) -> tuple[list[float], float]:
    """
    가지치기 회귀나무로 tenure 기준 경계 탐지.

    ⚠️ 필수 구현 규칙: cost_complexity_pruning_path 가 반환하는 alpha 후보를
    절대 샘플링하지 말고 전체를 다 탐색할 것 - 일부만 샘플링하면 탐색 그리드
    설정에 따라 결과 K값이 미세하게 흔들리는 그리드 민감성이 확인됨.

    Returns
    -------
    boundaries : 확정된 경계(tenure 개월) 리스트
    best_alpha : 교차검증으로 선택된 ccp_alpha
    """
    monthly = df_train.groupby("tenure")["ChurnFlag"].agg(["mean", "count"]).reset_index()
    X = monthly[["tenure"]].values
    y = monthly["mean"].values
    sample_weight = np.sqrt(monthly["count"].values)

    base_tree = DecisionTreeRegressor(random_state=random_state)
    path = base_tree.cost_complexity_pruning_path(X, y, sample_weight=sample_weight)
    ccp_alphas = path.ccp_alphas[path.ccp_alphas > 0]  # 전체 후보 사용 (샘플링 금지)

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        DecisionTreeRegressor(random_state=random_state),
        param_grid={"ccp_alpha": ccp_alphas},
        cv=kf,
        scoring="neg_mean_squared_error",
    )
    grid.fit(X, y, sample_weight=sample_weight)
    best_alpha = grid.best_params_["ccp_alpha"]

    final_tree = DecisionTreeRegressor(random_state=random_state, ccp_alpha=best_alpha)
    final_tree.fit(X, y, sample_weight=sample_weight)
    boundaries = sorted(
        float(t) for t in final_tree.tree_.threshold if t != -2
    )
    return boundaries, float(best_alpha)


def random_forest_boundary_votes(
    df_train: pd.DataFrame,
    n_estimators: int = 250,
    random_state: int = 42,
    top_n: int = 10,
) -> list[tuple[float, int]]:
    """
    랜덤포레스트의 모든 개별 트리에서 분기점(threshold)을 수집해 빈도 집계.
    ①의 보조 검증용 - 단일 결정나무를 대체하지 않음.

    Returns
    -------
    top_candidates : (분기점, 득표수) 리스트, 득표 많은 순
    """
    monthly = df_train.groupby("tenure")["ChurnFlag"].agg(["mean", "count"]).reset_index()
    X = monthly[["tenure"]].values
    y = monthly["mean"].values
    sample_weight = np.sqrt(monthly["count"].values)

    rf = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=3, min_samples_leaf=5,
        random_state=random_state,
    )
    rf.fit(X, y, sample_weight=sample_weight)

    all_thresholds = []
    for estimator in rf.estimators_:
        th = estimator.tree_.threshold[estimator.tree_.threshold != -2]
        all_thresholds.extend(th.round(1))

    return Counter(all_thresholds).most_common(top_n)


def check_boundary_against_rf_votes(
    boundaries: Sequence[float],
    rf_votes: Sequence[tuple[float, int]],
    max_distance: float = 2.0,
) -> bool:
    """
    ①이 찾은 경계가 RF투표 상위 후보와 일치하는지 확인.
    일치(거리 <= max_distance)하면 True (신뢰), 아니면 False (보수적 후보로 내려갈 것)
    """
    vote_positions = [v[0] for v in rf_votes]
    if not boundaries:
        return True  # 경계 없음(K=1)은 비교 대상 없음
    distances = [min(abs(b - v) for v in vote_positions) for b in boundaries]
    return max(distances) <= max_distance


# ---------------------------------------------------------------------------
# ② 적절성 검증: 세그먼트단독 AUC + 순열검정
# ---------------------------------------------------------------------------

def make_segment_column(df: pd.DataFrame, boundaries: Sequence[float]) -> pd.Series:
    """경계로 tenure를 세그먼트 정수 라벨로 변환 (0, 1, 2, ...)"""
    bins = [-1] + list(boundaries) + [df["tenure"].max() + 1]
    labels = list(range(len(bins) - 1))
    return pd.cut(df["tenure"], bins=bins, labels=labels).astype(int)


def segment_only_auc(
    segment: pd.Series, churn_flag: pd.Series,
    cv_folds: int = 5, random_state: int = 42,
) -> float:
    """세그먼트 라벨만으로 분류했을 때의 교차검증 AUC"""
    X = segment.values.reshape(-1, 1)
    y = churn_flag.values
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    aucs = []
    for train_idx, test_idx in cv.split(X, y):
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], proba))
    return float(np.mean(aucs))


def permutation_test_for_segment(
    segment: pd.Series, churn_flag: pd.Series,
    n_permutations: int = 200, random_state: int = 42,
) -> tuple[float, np.ndarray]:
    """
    이탈여부 라벨을 무작위로 섞어 segment_only_auc 를 재계산 반복.
    이론적 분포(카이제곱 등)를 전혀 쓰지 않는 완전 데이터기반 검증.

    ⚠️ 1회당 약 1초 소요 (5-fold×RandomForest) - n_permutations=200이면 약 3~4분.
    1년 단위로 1회 실행하는 게 전제이므로 운영에서는 문제 없으나, 개발/디버깅 중에는
    n_permutations를 줄여서 빠르게 돌려볼 것.

    Returns
    -------
    p_value : 실제 AUC가 무작위 분포보다 높은 비율의 역 (작을수록 유의)
    permuted_aucs : 순열로 얻은 AUC 분포 (참고/시각화용)
    """
    observed = segment_only_auc(segment, churn_flag, random_state=random_state)
    rng = np.random.RandomState(random_state)
    permuted_aucs = []
    y_values = churn_flag.values
    for _ in range(n_permutations):
        shuffled = pd.Series(rng.permutation(y_values))
        permuted_aucs.append(
            segment_only_auc(segment, shuffled, random_state=random_state)
        )
    permuted_aucs = np.array(permuted_aucs)
    p_value = float((permuted_aucs >= observed).mean())
    return p_value, permuted_aucs


# ---------------------------------------------------------------------------
# ③ 표본충분성 확인: AUC 측정값의 부트스트랩 신뢰구간
# ---------------------------------------------------------------------------

def bootstrap_auc_confidence_interval(
    df: pd.DataFrame, segment_col: str = "segment", churn_col: str = "ChurnFlag",
    n_bootstrap: int = 300, random_state: int = 42,
) -> tuple[float, float, float]:
    """
    세그먼트단독 AUC를 부트스트랩으로 재추정해 95% 신뢰구간 계산.
    사전 외부 공식(검정력분석 등) 없이, 측정 자체의 안정성을 사후에 직접 확인.

    [OOB 방식] 분석B(analysis_b.bootstrap_attribute_auc_ci)와 동일한 이유로
    Out-of-Bag 방식 사용 - 복원추출로 뽑힌 행으로 학습, 안 뽑힌 행으로만 평가해
    Train/Test가 겹치지 않게 함. 실데이터 확인 결과 분석A는 입력이 세그먼트
    라벨(범주 몇 개)뿐이라 이전 방식의 편향이 작았지만(점추정 0.7208 vs
    이전방식 부트스트랩 평균 0.7205로 거의 일치), 분석B와 같은 원리로
    일관되게 통일.

    Returns
    -------
    mean_auc, ci_low, ci_high
    """
    rng = np.random.RandomState(random_state)
    n = len(df)
    X_all = df[segment_col].values.reshape(-1, 1)
    y_all = df[churn_col].values

    boot_aucs = []
    for _ in range(n_bootstrap):
        boot_idx = rng.choice(n, n, replace=True)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[np.unique(boot_idx)] = False
        oob_idx = np.where(oob_mask)[0]

        if len(oob_idx) < 10 or len(np.unique(y_all[oob_idx])) < 2:
            continue
        if len(np.unique(y_all[boot_idx])) < 2:
            continue

        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        model.fit(X_all[boot_idx], y_all[boot_idx])
        proba = model.predict_proba(X_all[oob_idx])[:, 1]
        boot_aucs.append(roc_auc_score(y_all[oob_idx], proba))

    boot_aucs = np.array(boot_aucs)
    ci_low, ci_high = np.percentile(boot_aucs, [2.5, 97.5])
    return float(boot_aucs.mean()), float(ci_low), float(ci_high)


# ---------------------------------------------------------------------------
# 검증 통과까지 반복하는 전체 사이클 (기획구현.md 1번 섹션 참조)
# ---------------------------------------------------------------------------

def run_analysis_a(
    df_train: pd.DataFrame,
    p_value_threshold: float = 0.05,
    ci_width_threshold: float = 0.1,
    max_iterations: int = 5,
) -> dict:
    """
    ①→②→③ 사이클. ②③ 미통과 시 인접 구간 통합 후 ①부터 재실행.

    Returns
    -------
    result : dict with keys
        boundaries, alpha, rf_votes, rf_agreement,
        segment_only_auc, p_value, ci_low, ci_high, n_iterations
    """
    current_df = df_train.copy()

    for iteration in range(max_iterations):
        boundaries, alpha = find_boundaries_pruned_tree(current_df)
        rf_votes = random_forest_boundary_votes(current_df)
        rf_agreement = check_boundary_against_rf_votes(boundaries, rf_votes)

        current_df = current_df.copy()
        current_df["segment"] = make_segment_column(current_df, boundaries)

        p_value, _ = permutation_test_for_segment(
            current_df["segment"], current_df["ChurnFlag"]
        )
        mean_auc, ci_low, ci_high = bootstrap_auc_confidence_interval(current_df)
        ci_width = ci_high - ci_low

        passed = (p_value < p_value_threshold) and (ci_width < ci_width_threshold)
        if passed:
            return {
                "boundaries": boundaries,
                "alpha": alpha,
                "rf_votes": rf_votes,
                "rf_agreement": rf_agreement,
                "segment_only_auc": mean_auc,
                "p_value": p_value,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_iterations": iteration + 1,
            }

        # 미통과: 가장 작은 세그먼트를 인접 구간과 통합 (경계 1개 제거) 후 재실행
        if len(boundaries) <= 1:
            break  # 더 통합할 경계가 없으면 중단
        seg_counts = current_df["segment"].value_counts()
        smallest_seg = seg_counts.idxmin()
        # 가장 작은 세그먼트에 인접한 경계를 제거
        boundary_idx = min(smallest_seg, len(boundaries) - 1)
        boundaries.pop(boundary_idx)
        # 다음 반복에서 ①이 다시 처음부터 찾도록 current_df 는 원본으로 되돌림
        current_df = df_train.copy()

    # max_iterations 내 통과 못하면 마지막 결과라도 반환 (경고 표시)
    return {
        "boundaries": boundaries,
        "alpha": alpha,
        "rf_votes": rf_votes,
        "rf_agreement": rf_agreement,
        "segment_only_auc": mean_auc,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_iterations": max_iterations,
        "warning": "검증을 통과하지 못하고 최대 반복 횟수에 도달했습니다.",
    }
