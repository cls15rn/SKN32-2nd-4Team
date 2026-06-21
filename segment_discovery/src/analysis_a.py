"""
분석 A - 세그먼트 경계 탐지 (기획_메모.md 3장 참조)

① 경계 탐지: 가지치기 회귀나무 (메인) + 랜덤포레스트 투표 (보조검증)
② 적절성 검증: 세그먼트단독 AUC + 순열검정
③ 표본충분성 확인: AUC 측정값의 부트스트랩 신뢰구간

①②③ 은 완전히 별개 절차 - 혼동하지 말 것 (기획_메모.md 3.3 참조)
"""
from collections import Counter
from pathlib import Path
from typing import Sequence
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config  # noqa: E402


# ---------------------------------------------------------------------------
# ① 경계 탐지: 가지치기 회귀나무 (메인) + 랜덤포레스트 투표 (보조검증)
# ---------------------------------------------------------------------------

def find_boundaries_pruned_tree(
    df_train: pd.DataFrame,
    cv_folds: int = 5,
    random_state: int = 42,
    max_boundaries: int | None = None,
) -> tuple[list[float], float]:
    """
    가지치기 회귀나무로 tenure 기준 경계 탐지.

    ⚠️ 필수 구현 규칙: cost_complexity_pruning_path 가 반환하는 alpha 후보를
    절대 샘플링하지 말고 전체를 다 탐색할 것 - 일부만 샘플링하면 탐색 그리드
    설정에 따라 결과 K값이 미세하게 흔들리는 그리드 민감성이 확인됨.

    Parameters
    ----------
    max_boundaries : ②③ 검증에 실패해 인접 구간을 통합해야 할 때 쓰는 옵션.
        None(기본값)이면 교차검증 최적 alpha를 그대로 쓴다(평소 동작).
        정수를 주면 "교차검증 성능이 가장 좋으면서도 경계 개수가 그 값
        이하인" 후보를 선택한다 - 같은 df_train으로 재실행해도 항상 같은
        결과만 나오던 문제(통합 로직이 죽은 코드였던 버그)를 해결한다.
        alpha가 클수록(가지치기가 강할수록) 경계가 줄어드는 단조관계를
        이용해, "이전 결과보다 단순한 트리를 강제로 찾는다".

    Returns
    -------
    boundaries : 확정된 경계(tenure 개월) 리스트
    best_alpha : 선택된 ccp_alpha (max_boundaries 적용 시 교차검증 최적값이 아닐 수 있음)
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

    if max_boundaries is None:
        best_alpha = grid.best_params_["ccp_alpha"]
    else:
        # 교차검증 성능 순으로 정렬한 뒤, "경계 개수가 max_boundaries 이하인"
        # 첫 후보를 선택 - alpha가 클수록(가지치기가 강할수록) 경계가 줄어드는
        # 단조관계를 이용해 "이전보다 단순한 트리"를 강제로 찾는다.
        cv_results = pd.DataFrame(grid.cv_results_).sort_values(
            "mean_test_score", ascending=False
        )
        best_alpha = None
        for alpha_candidate in cv_results["param_ccp_alpha"]:
            candidate_tree = DecisionTreeRegressor(
                random_state=random_state, ccp_alpha=alpha_candidate
            )
            candidate_tree.fit(X, y, sample_weight=sample_weight)
            n_boundaries = sum(1 for t in candidate_tree.tree_.threshold if t != -2)
            if n_boundaries <= max_boundaries:
                best_alpha = alpha_candidate
                break
        if best_alpha is None:
            # 가장 강한 가지치기(가장 큰 alpha)로도 max_boundaries를 못 만족하면
            # 그 alpha를 그대로 사용 (경계 0개, 즉 세그먼트 1개로 수렴)
            best_alpha = float(ccp_alphas.max())

    final_tree = DecisionTreeRegressor(random_state=random_state, ccp_alpha=best_alpha)
    final_tree.fit(X, y, sample_weight=sample_weight)
    boundaries = sorted(
        float(t) for t in final_tree.tree_.threshold if t != -2
    )
    return boundaries, float(best_alpha)


def find_stable_rf_n_estimators(
    df_train: pd.DataFrame,
    candidates: Sequence[int] = tuple(config.ANALYSIS_A_RF_N_ESTIMATORS_CANDIDATES),
    seeds: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8),
    improvement_threshold: float = config.ANALYSIS_A_RF_STABILITY_IMPROVEMENT_THRESHOLD,
    patience: int = 2,
) -> tuple[int, pd.DataFrame]:
    """
    "RF 보조검증용 트리 개수가 몇 개면 충분한가"를 데이터가 직접 결정한다.

    분석A의 ccp_alpha(교차검증으로 예측오차가 더 안 줄어드는 지점을 찾음)와
    같은 원리 - 트리 개수를 늘려가며 "1위 분기점의 득표율이 시드를 바꿔도
    얼마나 흔들리는지(표준편차)"를 측정하고, 더 늘려도 표준편차가 의미있게
    줄지 않는 지점에서 멈춘다.

    ⚠️ [수정] 처음에는 "한 단계라도 개선이 없으면 즉시 멈춤"으로 짰는데,
    실데이터 검증 결과 표준편차 자체가 시드 8개만으로는 단조롭게 줄지 않고
    중간에 노이즈로 한두 번 튀는 게 확인됨(예: 100->150에서 살짝 증가했다가
    200부터 다시 감소). 그래서 "연속 patience회 모두 개선이 없을 때"만
    멈추도록 완화 - 단발성 노이즈로 너무 일찍 멈추는 것을 방지.

    ⚠️ K-means의 K값(서브트랙Q)과 다른 점: 여기는 "득표율이 안정적인가"라는
    단 하나의 목표만 있어 트리를 늘릴수록 좋아지거나 그대로일 뿐, 서로
    충돌하는 두 목표(K-means의 결합효과 vs 표본안정성)가 없다. 그래서 단일
    기준으로 완전 자동화가 가능하다.

    Returns
    -------
    stable_n : 충분히 안정적이라고 판단된 트리 개수
    diagnostics : 후보별 (n_estimators, std) 표 (보고서/로그용)
    """
    monthly = df_train.groupby("tenure")["ChurnFlag"].agg(["mean", "count"]).reset_index()
    X = monthly[["tenure"]].values
    y = monthly["mean"].values
    sample_weight = np.sqrt(monthly["count"].values)

    rows = []
    stds = []
    no_improvement_streak = 0
    stable_n = candidates[-1]  # 끝까지 안정화 안 되면 가장 큰 후보를 안전하게 사용

    for n in candidates:
        top1_rates = []
        for seed in seeds:
            rf = RandomForestRegressor(
                n_estimators=n, max_depth=3, min_samples_leaf=5, random_state=seed,
            )
            rf.fit(X, y, sample_weight=sample_weight)
            all_thresholds = []
            for estimator in rf.estimators_:
                th = estimator.tree_.threshold[estimator.tree_.threshold != -2]
                all_thresholds.extend(th.round(1))
            if not all_thresholds:
                continue
            counts = Counter(all_thresholds)
            top1_value, top1_count = counts.most_common(1)[0]
            top1_rates.append(top1_count / n)

        std = float(np.std(top1_rates)) if top1_rates else float("nan")
        rows.append({"n_estimators": n, "top1_rate_std": std})
        stds.append(std)

        if len(stds) >= 2 and stds[-2] > 0:
            improvement = (stds[-2] - stds[-1]) / stds[-2]
            if improvement < improvement_threshold:
                no_improvement_streak += 1
            else:
                no_improvement_streak = 0

            if no_improvement_streak >= patience:
                # 연속 patience회 동안 의미있는 개선이 없었음 -> 이전 단계가 충분히 안정적이었던 지점
                stable_n = candidates[len(stds) - 1 - (patience - 1)]
                break

    diagnostics = pd.DataFrame(rows)
    return stable_n, diagnostics


def random_forest_boundary_votes(
    df_train: pd.DataFrame,
    n_estimators: int | None = None,
    random_state: int = 42,
    top_n: int = 10,
) -> list[tuple[float, int]]:
    """
    랜덤포레스트의 모든 개별 트리에서 분기점(threshold)을 수집해 빈도 집계.
    ①의 보조 검증용 - 단일 결정나무를 대체하지 않음.

    n_estimators=None(기본값)이면 find_stable_rf_n_estimators로 트리 개수를
    데이터에서 직접 찾는다 - 사람이 250 같은 숫자를 미리 고정하지 않는다.
    (실데이터 확인 결과 우리 데이터에서는 n=200~250 부근에서 안정화됨 - 단,
    이건 참고용이고 코드가 매번 그 데이터에 맞게 다시 계산한다)

    Returns
    -------
    top_candidates : (분기점, 득표수) 리스트, 득표 많은 순
    """
    if n_estimators is None:
        n_estimators, _ = find_stable_rf_n_estimators(df_train)

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
    n_permutations: int = config.ANALYSIS_A_PERMUTATION_COUNT,
    random_state: int = config.RANDOM_STATE,
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
    n_bootstrap: int = config.ANALYSIS_A_BOOTSTRAP_COUNT,
    random_state: int = config.RANDOM_STATE,
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
    p_value_threshold: float = config.ANALYSIS_A_P_VALUE_THRESHOLD,
    ci_width_threshold: float = config.ANALYSIS_A_CI_WIDTH_THRESHOLD,
    max_iterations: int = config.ANALYSIS_A_MAX_ITERATIONS,
) -> dict:
    """
    ①→②→③ 사이클. ②③ 미통과 시 인접 구간 통합 후 ①부터 재실행.

    ⚠️ [버그 수정] 예전 코드는 미통과 시 boundaries.pop()으로 경계를 줄여놓고도
    바로 다음 줄에서 current_df를 원본(df_train)으로 되돌렸다. find_boundaries_
    pruned_tree는 같은 입력에 항상 같은 출력을 내는 결정론적 함수라, 다음
    반복에서 ①을 "원본으로" 다시 돌리면 통합 전과 완전히 똑같은 경계가
    다시 나왔다 - 즉 통합 로직이 실제로는 전혀 반영되지 않는 죽은 코드였다
    (실제 검증: 같은 df_train으로 find_boundaries_pruned_tree를 두 번 호출하면
    결과가 항상 동일함을 확인).

    수정: max_boundaries(이전 반복의 경계 개수 - 1)를 ①에 직접 전달해,
    "이전보다 단순한 트리를 강제로 찾으라"고 명시적으로 요구한다. 이러면
    재시도가 실제로 더 적은 경계를 가진 새로운 결과를 만들어낸다.

    Returns
    -------
    result : dict with keys
        boundaries, alpha, rf_votes, rf_agreement,
        segment_only_auc, p_value, ci_low, ci_high, n_iterations
    """
    max_boundaries = None  # 첫 시도는 교차검증 최적값 그대로 (제약 없음)

    for iteration in range(max_iterations):
        boundaries, alpha = find_boundaries_pruned_tree(
            df_train, max_boundaries=max_boundaries
        )
        stable_n_estimators, rf_n_diagnostics = find_stable_rf_n_estimators(df_train)
        rf_votes = random_forest_boundary_votes(df_train, n_estimators=stable_n_estimators)
        rf_agreement = check_boundary_against_rf_votes(boundaries, rf_votes)

        current_df = df_train.copy()
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
                "rf_n_estimators_used": stable_n_estimators,  # 데이터가 직접 찾은 트리 개수
                "segment_only_auc": mean_auc,
                "p_value": p_value,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_iterations": iteration + 1,
            }

        # 미통과: 다음 반복에서 ①이 "경계 1개 더 적은" 트리를 강제로 찾도록 함
        if len(boundaries) <= 1:
            break  # 더 줄일 경계가 없으면 중단 (세그먼트 1개=통합 불가능한 한계)
        max_boundaries = len(boundaries) - 1

    # max_iterations 내 통과 못하면 마지막 결과라도 반환 (경고 표시)
    return {
        "boundaries": boundaries,
        "alpha": alpha,
        "rf_votes": rf_votes,
        "rf_agreement": rf_agreement,
        "rf_n_estimators_used": stable_n_estimators,
        "segment_only_auc": mean_auc,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_iterations": max_iterations,
        "warning": "검증을 통과하지 못하고 최대 반복 횟수에 도달했습니다.",
    }
