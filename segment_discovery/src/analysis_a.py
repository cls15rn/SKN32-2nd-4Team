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
from scipy import stats
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
    cv_folds: int = config.ANALYSIS_A_CV_FOLDS,
    random_state: int = config.RANDOM_STATE,
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


def find_stable_repetition_count(
    measure_fn,
    candidates: Sequence[int],
    seeds: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8),
    improvement_threshold: float = config.ANALYSIS_A_RF_STABILITY_IMPROVEMENT_THRESHOLD,
    patience: int = 2,
) -> tuple[int, pd.DataFrame]:
    """
    "이 반복 작업을 몇 번 해야 충분히 안정적인 측정값을 얻는가"를 데이터가
    직접 결정하는 공통 로직. find_stable_rf_n_estimators(RF 트리개수)와
    부트스트랩/순열검정의 반복횟수 자동탐지가 똑같은 구조(후보를 늘려가며
    여러 시드로 측정 -> 시드 간 표준편차가 더 안 줄어드는 지점에서 멈춤)를
    공유하므로 하나로 합침 - 같은 로직이 여러 곳에 따로 구현되면 한쪽만
    고치고 다른 쪽을 놓치는 위험이 있다(이번에 발견한 config 연결 누락
    버그와 같은 패턴).

    Parameters
    ----------
    measure_fn : (n: int, seed: int) -> float
        n번 반복(또는 n개 트리 등)으로 측정값 하나를 반환하는 함수.
        예: 부트스트랩이면 n=반복횟수, seed=난수시드를 받아 CI폭을 반환.

    Returns
    -------
    stable_n : 충분히 안정적이라고 판단된 반복횟수(또는 개수)
    diagnostics : 후보별 (n, std) 표 (보고서/로그용)
    """
    rows = []
    stds = []
    no_improvement_streak = 0
    stable_n = candidates[-1]  # 끝까지 안정화 안 되면 가장 큰 후보를 안전하게 사용

    for n in candidates:
        measurements = [measure_fn(n, seed) for seed in seeds]
        std = float(np.std(measurements))
        rows.append({"n": n, "measurement_std": std})
        stds.append(std)

        if len(stds) >= 2 and stds[-2] > 0:
            improvement = (stds[-2] - stds[-1]) / stds[-2]
            if improvement < improvement_threshold:
                no_improvement_streak += 1
            else:
                no_improvement_streak = 0

            if no_improvement_streak >= patience:
                stable_n = candidates[len(stds) - 1 - (patience - 1)]
                break

    diagnostics = pd.DataFrame(rows)
    return stable_n, diagnostics


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
    줄지 않는 지점에서 멈춘다. (공통 로직은 find_stable_repetition_count 참조)

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

    def measure_top1_rate(n_estimators: int, seed: int) -> float:
        rf = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=3, min_samples_leaf=5, random_state=seed,
        )
        rf.fit(X, y, sample_weight=sample_weight)
        all_thresholds = []
        for estimator in rf.estimators_:
            th = estimator.tree_.threshold[estimator.tree_.threshold != -2]
            all_thresholds.extend(th.round(1))
        if not all_thresholds:
            return float("nan")
        counts = Counter(all_thresholds)
        _, top1_count = counts.most_common(1)[0]
        return top1_count / n_estimators

    stable_n, diagnostics = find_stable_repetition_count(
        measure_top1_rate, candidates, seeds, improvement_threshold, patience,
    )
    diagnostics = diagnostics.rename(columns={"n": "n_estimators", "measurement_std": "top1_rate_std"})
    return stable_n, diagnostics


def random_forest_boundary_votes(
    df_train: pd.DataFrame,
    n_estimators: int | None = None,
    random_state: int = config.RANDOM_STATE,
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
    cv_folds: int = config.ANALYSIS_A_CV_FOLDS, random_state: int = config.RANDOM_STATE,
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


def find_stable_permutation_p_value(
    observed_statistic: float,
    permuted_statistic_fn,
    p_threshold: float,
    confidence: float = config.SEQUENTIAL_PERMUTATION_CONFIDENCE,
    check_every: int = config.SEQUENTIAL_PERMUTATION_CHECK_EVERY,
    max_iter: int = config.SEQUENTIAL_PERMUTATION_MAX_ITER,
) -> tuple[float, int]:
    """
    순열검정의 Clopper-Pearson 기반 순차적 조기중단 — 통계량 종류에 무관한
    공통 로직 (기획_메모.md 4.1-C 보강, 4.1-E 참조).

    "관측값을 넘는 순열의 비율"은 통계량이 AUC든 분산이든 그 어떤
    것이든 항상 이항분포를 따른다는 사실을 이용한다 - 그래서 이
    멈춤 로직 자체는 통계량을 모른다. permuted_statistic_fn(i)가
    i번째 순열에서 계산된 통계량 하나를 반환하기만 하면 된다
    (호출 측이 "무엇을 어떻게 섞을지"를 결정).

    분석A(permutation_test_for_segment, AUC)와 서브트랙Q
    (permutation_test_for_risk_count, risk_count별 이탈률 분산)가 이
    함수를 공유한다 - 같은 공식을 두 곳에 따로 구현하면 한쪽만 고치고
    다른 쪽을 놓치는 위험이 재발할 수 있다.

    Returns
    -------
    p_value : 멈춘 시점까지 누적된 (초과횟수/반복횟수) - 추정 p값
    n_permutations_used : 데이터가 직접 찾은 순열 반복횟수
    """
    exceed_count = 0

    for i in range(1, max_iter + 1):
        permuted_statistic = permuted_statistic_fn(i)
        if permuted_statistic >= observed_statistic:
            exceed_count += 1

        if i % check_every == 0:
            upper = stats.beta.ppf(confidence, exceed_count + 1, i - exceed_count)
            lower = (
                stats.beta.ppf(1 - confidence, exceed_count, i - exceed_count + 1)
                if exceed_count > 0 else 0.0
            )
            if upper < p_threshold or lower > p_threshold:
                # 어느 쪽으로든 통과기준 대비 결론이 명확해졌으면 멈춤
                return exceed_count / i, i

    return exceed_count / max_iter, max_iter


def permutation_test_for_segment(
    segment: pd.Series, churn_flag: pd.Series,
    p_threshold: float = config.ANALYSIS_A_P_VALUE_THRESHOLD,
    confidence: float = config.SEQUENTIAL_PERMUTATION_CONFIDENCE,
    check_every: int = config.SEQUENTIAL_PERMUTATION_CHECK_EVERY,
    max_iter: int = config.SEQUENTIAL_PERMUTATION_MAX_ITER,
    random_state: int = config.RANDOM_STATE,
) -> tuple[float, int]:
    """
    이탈여부 라벨을 무작위로 섞어 segment_only_auc 를 재계산 반복.
    이론적 분포(카이제곱 등)를 전혀 쓰지 않는 완전 데이터기반 검증.

    [순차적 조기 중단으로 자동화 — 기획_메모.md 4.1-C 보강 참조]
    예전에는 n_permutations=200 고정값을 모두 채울 때까지 반복했다. 실측
    결과 우리 데이터는 거의 항상 압도적으로 우연이 아님(관측 AUC를 넘는
    순열이 200회 중 단 한 번도 안 나옴)이 확인됐는데, 이건 "200회가 필요
    없다"는 뜻이 아니라 "더 적은 반복으로도 같은 결론에 확신을 가질 수
    있다는 걸 자동으로 감지할 수 있다"는 뜻이다.

    순열을 1개씩 누적하면서(이전 계산 안 버림), 매 check_every회마다
    "지금까지 관측값을 넘은 횟수/반복횟수"의 Clopper-Pearson 신뢰상한을
    계산한다(공통 로직은 find_stable_permutation_p_value 참조). 이 상한이
    p_threshold(0.05)보다 확실히 낮으면(또는 신뢰하한이 확실히 높으면)
    "더 반복해도 결론이 안 바뀐다"고 보고 멈춘다 - 이항분포의 표준
    신뢰구간 공식이라 Hanley-McNeil처럼 닫힌 형태로 계산되며, 부트스트랩
    G안과 달리 OOB 모델재학습 같은 구조적 추가변동성이 없어 이론값에 항상
    안정적으로 수렴한다(시드 3개 모두 정확히 일치하는 지점에서 멈춤).

    실데이터 검증: 전체데이터·작은세그먼트 모두 일관되게 60회에서 멈춤
    (200회 대비 3.3배 절감) - 표본크기보다 "효과의 크기(관측 AUC가 우연
    분포에서 얼마나 떨어져 있는지)"가 멈춤 시점을 좌우함(부트스트랩과
    다른 패턴).

    Returns
    -------
    p_value : 멈춘 시점까지 누적된 (초과횟수/반복횟수) - 추정 p값
    n_permutations_used : 데이터가 직접 찾은 순열 반복횟수
    """
    observed = segment_only_auc(segment, churn_flag, random_state=random_state)
    rng = np.random.RandomState(random_state)
    y_values = churn_flag.values

    def _permuted_auc(_i: int) -> float:
        shuffled = pd.Series(rng.permutation(y_values))
        return segment_only_auc(segment, shuffled, random_state=random_state)

    return find_stable_permutation_p_value(
        observed_statistic=observed,
        permuted_statistic_fn=_permuted_auc,
        p_threshold=p_threshold,
        confidence=confidence,
        check_every=check_every,
        max_iter=max_iter,
    )


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


def hanley_mcneil_se(auc: float, n_positive: int, n_negative: int) -> float:
    """
    Hanley & McNeil(1982)의 AUC 표준오차 근사 공식.
    표본크기·이탈률(불균형도)·AUC만으로 "이론적으로 기대되는 측정 불안정성"을
    부트스트랩 없이 즉시 계산한다 - Cohen 효과크기처럼 사전에 닫힌 형태로
    구해지는 공식이지만, 정확히는 AUC 추정량의 분산을 위한 표준 통계 공식이다.

    ⚠️ 이 공식은 "고정된 분류기"의 AUC를 가정한다. 우리는 매 부트스트랩
    반복마다 RandomForest를 새로 학습시키므로(OOB 방식), 실제 측정값은
    이 이론값보다 항상 더 넓게 나오는 구조적 격차가 있다(실측 확인됨) -
    그래서 이 값을 "목표"가 아니라 "안전 상한의 기준선"으로만 쓴다.
    """
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    return float(np.sqrt(
        (auc * (1 - auc) + (n_positive - 1) * (q1 - auc**2) + (n_negative - 1) * (q2 - auc**2))
        / (n_positive * n_negative)
    ))


def find_stable_bootstrap_count(
    df: pd.DataFrame, segment_col: str = "segment", churn_col: str = "ChurnFlag",
    random_state: int = config.RANDOM_STATE,
    max_iter: int = config.SEQUENTIAL_MAX_ITER,
    check_every: int = config.SEQUENTIAL_CHECK_EVERY,
    min_n_before_check: int = config.SEQUENTIAL_MIN_N_BEFORE_CHECK,
    structural_gap: float | None = None,
    ceiling_patience: int = config.SEQUENTIAL_CEILING_PATIENCE,
    self_stability_window: int = config.SEQUENTIAL_SELF_STABILITY_WINDOW,
    self_stability_threshold: float = config.SEQUENTIAL_SELF_STABILITY_THRESHOLD,
    record_observation: bool = True,
) -> tuple[int, float, float, float, pd.DataFrame]:
    """
    "AUC 부트스트랩 신뢰구간을 구하려면 몇 번 재추출하면 충분한가"를
    데이터가 직접, 그리고 효율적으로 결정한다 (순차적 조기 중단,
    Sequential Early Stopping).

    [최종 설계, 여러 시행착오를 거쳐 확정 — 기획_메모.md 4.1-C 참조]
    이전 시도들과 그 한계:
    - 후보(50,100,150...)를 처음부터 따로 계산해 비교하는 방식: 매 후보마다
      처음부터 다시 부트스트랩해야 해서 비용이 운영값(고정 300회)보다
      6~26배 더 들었음 - 탐색 비용이 운영 비용을 정당화하지 못함.
    - "이론값(SE_HM)과의 근접도"만으로 멈추는 방식: 우리 OOB 방식은 모델을
      매번 새로 학습하는 추가 변동성이 있어 이론값에 항상 못 미치거나
      운에 따라 도달여부가 들쭉날쭉함(시드 5개 중 1개만 작동).
    - "측정값 자체의 변화율(자기 안정성)"만으로 멈추는 방식: 노이즈가
      우연히 잠잠해진 구간을 안정으로 착각해 운영값보다 못한 정밀도를 냄.

    최종 해법: 부트스트랩을 1회씩 누적하면서(이전 계산을 버리지 않음),
    매 check_every 회마다 "지금까지 쌓인 표본으로 측정한 CI폭"이
    (1) Hanley-McNeil 이론값×structural_gap 이라는 안전 상한보다 좁아지는지,
    또는 (2) 최근 self_stability_window 번의 측정값들이 서로
    self_stability_threshold 이내로 거의 안 변하는지를 확인한다.
    두 조건 중 하나라도 연속 ceiling_patience(또는 그 이상)회 충족되면 멈춘다.
    실데이터 검증 결과: 표본이 큰 데이터(4,930명)는 평균 40회(기존 300회
    대비 7.5배 절감)로 매우 일관되게 멈췄고, 표본이 작고 불균형한 세그먼트
    (749명, 이탈률 9.8%)는 평균 242회(1.2배 절감)로 더 신중하게 멈췄다 -
    "표본 특성에 맞게 동적으로 반복횟수가 정해진다"는 목표가 실증됨.

    structural_gap 적응형 보정 (3번 항목, gap_calibration.py 참조)
    ----------------------------------------------------------------
    structural_gap을 명시적으로 넘기지 않으면(기본값 None), 과거 실행에서
    누적된 관측치 중 이번 표본크기(n)와 가장 비슷한 것들의 상위 분位수를
    안전 상한 배율로 자동 채택한다 - 새 부트스트랩을 추가로 실행하지
    않으므로 탐색비용이 들지 않는다. 관측치가 충분히 쌓이기 전(콜드스타트)
    에는 기존 고정값(1.3)을 그대로 쓴다. structural_gap을 직접 지정하면
    (예: 기존 테스트 호환을 위해) 그 값을 그대로 쓰고 적응형 조회를 건너뛴다.

    함수가 멈출 때, 이미 계산해놓은 값들(점추정 AUC, 표본구성, 실측 CI폭)
    로부터 "이번 실행의 실측 gap비율"을 나눗셈 한 번으로 구해 기록한다
    (record_observation=True일 때만, 회귀테스트 등에서는 False로 끌 수 있음).

    Returns
    -------
    stop_n : 멈춘 시점의 누적 부트스트랩 횟수
    mean_auc, ci_low, ci_high : 그 시점까지 누적된 표본으로 계산한 최종 결과
    diagnostics : 체크포인트별 (n, ci_width) 표 (보고서/로그용)
    """
    X = df[segment_col].values.reshape(-1, 1)
    y = df[churn_col].values
    n = len(df)

    point_auc = segment_only_auc(df[segment_col], df[churn_col], random_state=random_state)
    n_positive = int(y.sum())
    n_negative = n - n_positive
    se_hm = hanley_mcneil_se(point_auc, n_positive, n_negative)

    if structural_gap is None:
        from gap_calibration import adaptive_structural_gap
        structural_gap, _gap_source = adaptive_structural_gap(n, statistic_type="auc_hanley_mcneil")
    safe_ceiling = 2 * 1.96 * se_hm * structural_gap

    rng = np.random.RandomState(random_state)
    boot_aucs: list[float] = []
    width_history: list[float] = []
    rows = []
    ceiling_streak = 0
    last_ci_low, last_ci_high = float("nan"), float("nan")

    def _finish(stop_n: int) -> tuple[int, float, float, float, pd.DataFrame]:
        diagnostics = pd.DataFrame(rows)
        mean_auc = float(np.mean(boot_aucs))
        ci_low, ci_high = float(last_ci_low), float(last_ci_high)
        if record_observation and np.isfinite(ci_low) and np.isfinite(ci_high):
            from gap_calibration import record_auc_gap_observation
            record_auc_gap_observation(
                n=n, point_auc=point_auc, n_positive=n_positive,
                n_negative=n_negative, ci_width=ci_high - ci_low,
            )
        return stop_n, mean_auc, ci_low, ci_high, diagnostics

    for i in range(1, max_iter + 1):
        boot_idx = rng.choice(n, n, replace=True)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[np.unique(boot_idx)] = False
        oob_idx = np.where(oob_mask)[0]

        if len(oob_idx) < 10 or len(np.unique(y[oob_idx])) < 2 or len(np.unique(y[boot_idx])) < 2:
            continue

        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        model.fit(X[boot_idx], y[boot_idx])
        proba = model.predict_proba(X[oob_idx])[:, 1]
        boot_aucs.append(roc_auc_score(y[oob_idx], proba))

        if i % check_every == 0 and len(boot_aucs) >= min_n_before_check:
            last_ci_low, last_ci_high = np.percentile(boot_aucs, [2.5, 97.5])
            width = last_ci_high - last_ci_low
            width_history.append(width)
            rows.append({"n": i, "ci_width": width})

            # 조건 ①: 안전 상한보다 좁은가
            if width <= safe_ceiling:
                ceiling_streak += 1
            else:
                ceiling_streak = 0

            # 조건 ②: 최근 변화가 충분히 작은가 (자기 안정성, 상한 도달 못해도 안전망)
            self_stable = False
            if len(width_history) >= self_stability_window + 1:
                recent = width_history[-(self_stability_window + 1):]
                rel_changes = [
                    abs(recent[j + 1] - recent[j]) / recent[j]
                    for j in range(len(recent) - 1) if recent[j] > 0
                ]
                self_stable = bool(rel_changes) and all(
                    rc < self_stability_threshold for rc in rel_changes
                )

            if ceiling_streak >= ceiling_patience or self_stable:
                return _finish(i)

    return _finish(max_iter)


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

        # ② 순열검정: 순차적 조기중단 - 부트스트랩과 마찬가지로 반복횟수를
        # 사람이 고정하지 않고 Clopper-Pearson 신뢰상한으로 데이터가 직접 찾음
        p_value, n_permutations_used = permutation_test_for_segment(
            current_df["segment"], current_df["ChurnFlag"]
        )
        # ③ 부트스트랩: 순차적 조기중단(G안) - 반복횟수를 사람이 고정하지 않고
        # 이 데이터의 표본크기·불균형도에 맞게 데이터가 직접 멈춤지점을 찾음
        n_bootstrap_used, mean_auc, ci_low, ci_high, boot_diagnostics = (
            find_stable_bootstrap_count(current_df)
        )
        ci_width = ci_high - ci_low

        passed = (p_value < p_value_threshold) and (ci_width < ci_width_threshold)
        if passed:
            return {
                "boundaries": boundaries,
                "alpha": alpha,
                "rf_votes": rf_votes,
                "rf_agreement": rf_agreement,
                "rf_n_estimators_used": stable_n_estimators,  # 데이터가 직접 찾은 트리 개수
                "n_bootstrap_used": n_bootstrap_used,  # 데이터가 직접 찾은 부트스트랩 반복횟수
                "n_permutations_used": n_permutations_used,  # 데이터가 직접 찾은 순열검정 반복횟수
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
        "n_bootstrap_used": n_bootstrap_used,
        "n_permutations_used": n_permutations_used,
        "segment_only_auc": mean_auc,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_iterations": max_iterations,
        "warning": "검증을 통과하지 못하고 최대 반복 횟수에 도달했습니다.",
    }
