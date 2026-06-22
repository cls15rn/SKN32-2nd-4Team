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
from scipy import stats

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
    df_segment: pd.DataFrame, random_state: int = config.RANDOM_STATE,
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
    cv_folds: int = config.ANALYSIS_A_CV_FOLDS, random_state: int = config.RANDOM_STATE,
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
    p_threshold: float = config.ANALYSIS_A_P_VALUE_THRESHOLD,
    confidence: float = config.SEQUENTIAL_PERMUTATION_CONFIDENCE,
    check_every: int = config.SEQUENTIAL_PERMUTATION_CHECK_EVERY,
    max_iter: int = config.SEQUENTIAL_PERMUTATION_MAX_ITER,
    random_state: int = config.RANDOM_STATE,
) -> tuple[float, int]:
    """
    분석A의 ②와 동일한 절차(라벨 순열검정) + 동일한 순차적 조기중단(Clopper-
    Pearson 신뢰상한 기반). analysis_a.permutation_test_for_segment와 같은
    원리를 그대로 재사용 - 실데이터 검증 결과 작은 세그먼트에서도 일관되게
    60회 부근에서 멈춤(200회 대비 약 3.3배 절감, 기획_메모.md 4.1-C 보강).

    Returns
    -------
    p_value : 멈춘 시점까지 누적된 (초과횟수/반복횟수) - 추정 p값
    n_permutations_used : 데이터가 직접 찾은 순열 반복횟수
    """
    observed = attribute_based_auc(df_segment, attributes, random_state=random_state)
    rng = np.random.RandomState(random_state)
    df_enc = _encode_categoricals(df_segment)
    X = df_enc[list(attributes)].values
    y_values = df_enc["ChurnFlag"].values
    exceed_count = 0

    for i in range(1, max_iter + 1):
        y_shuffled = rng.permutation(y_values)
        cv = StratifiedKFold(
            n_splits=config.ANALYSIS_A_CV_FOLDS, shuffle=True, random_state=random_state
        )
        aucs = []
        for train_idx, test_idx in cv.split(X, y_shuffled):
            model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=random_state)
            model.fit(X[train_idx], y_shuffled[train_idx])
            proba = model.predict_proba(X[test_idx])[:, 1]
            aucs.append(roc_auc_score(y_shuffled[test_idx], proba))
        permuted_auc = np.mean(aucs)
        if permuted_auc >= observed:
            exceed_count += 1

        if i % check_every == 0:
            upper = stats.beta.ppf(confidence, exceed_count + 1, i - exceed_count)
            lower = (
                stats.beta.ppf(1 - confidence, exceed_count, i - exceed_count + 1)
                if exceed_count > 0 else 0.0
            )
            if upper < p_threshold or lower > p_threshold:
                return exceed_count / i, i

    return exceed_count / max_iter, max_iter


# ---------------------------------------------------------------------------
# Ⓒ 표본충분성 확인: Ⓑ의 AUC 부트스트랩 신뢰구간 (분석A의 ③과 동일 절차)
# ---------------------------------------------------------------------------

def bootstrap_attribute_auc_ci(
    df_segment: pd.DataFrame, attributes: Sequence[str],
    n_bootstrap: int = config.ANALYSIS_B_BOOTSTRAP_COUNT,
    random_state: int = config.RANDOM_STATE,
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


def find_stable_bootstrap_count_for_attributes(
    df_segment: pd.DataFrame, attributes: Sequence[str],
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
    분석A의 find_stable_bootstrap_count(G안, 순차적 조기중단)와 완전히 같은
    원리를 Ⓒ(분석B의 표본충분성 확인)에 적용. analysis_a.hanley_mcneil_se를
    그대로 재사용 - 같은 공식을 두 곳에 따로 구현하면 한쪽만 고치고 다른
    쪽을 놓치는 위험이 재발할 수 있다.

    ⚠️ 분석B는 세그먼트 안에서 또 5-fold(Ⓑ)를 나누어 분석A보다 표본부족
    위험이 크다고 누차 강조했는데, 정작 실측에서 작고 불균형한 세그먼트일수록
    순차적 조기중단이 "더 많은 반복이 필요하다"고 자동으로 판단해 더 오래
    실행되는 것으로 확인됨 - 표본부족 위험이 큰 곳에 자동으로 더 신중한
    검증이 적용되는 바람직한 결과.

    structural_gap 적응형 보정 (3번 항목, gap_calibration.py 참조)
    ----------------------------------------------------------------
    분석A의 find_stable_bootstrap_count와 같은 적응형 보정을 그대로
    재사용한다 - 같은 gap_observations.csv에 분석A·B의 관측치가 함께
    누적되므로, 더 빨리 신뢰할 만한 분位수에 도달한다. structural_gap을
    직접 지정하면(기존 테스트 호환) 적응형 조회를 건너뛴다.

    Returns
    -------
    stop_n, mean_auc, ci_low, ci_high, diagnostics : find_stable_bootstrap_count와 동일한 의미
    """
    from analysis_a import hanley_mcneil_se  # noqa: E402  (지연 import - 순환참조 방지)

    df_enc = _encode_categoricals(df_segment)
    X = df_enc[list(attributes)].values
    y = df_enc["ChurnFlag"].values
    n = len(df_segment)

    point_auc = attribute_based_auc(df_segment, attributes, random_state=random_state)
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

        model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=random_state)
        model.fit(X[boot_idx], y[boot_idx])
        proba = model.predict_proba(X[oob_idx])[:, 1]
        boot_aucs.append(roc_auc_score(y[oob_idx], proba))

        if i % check_every == 0 and len(boot_aucs) >= min_n_before_check:
            last_ci_low, last_ci_high = np.percentile(boot_aucs, [2.5, 97.5])
            width = last_ci_high - last_ci_low
            width_history.append(width)
            rows.append({"n": i, "ci_width": width})

            if width <= safe_ceiling:
                ceiling_streak += 1
            else:
                ceiling_streak = 0

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
                            attribute_auc, p_value, ci_low, ci_high, n_bootstrap_used]
    """
    rows = []
    for segment_id in sorted(df_train_with_segment["segment"].unique()):
        df_segment = df_train_with_segment[df_train_with_segment["segment"] == segment_id]
        if df_segment["ChurnFlag"].nunique() < 2 or len(df_segment) < 50:
            continue

        top_attrs = find_top_risk_attributes(df_segment)
        auc = attribute_based_auc(df_segment, top_attrs)
        # Ⓑ: 순열검정도 순차적 조기중단 - 분석A와 동일 원리
        p_value, n_permutations_used = permutation_test_for_attributes(df_segment, top_attrs)
        # Ⓒ: 순차적 조기중단(G안) - 분석A와 동일 원리, 표본부족 위험이 큰
        # 작은 세그먼트일수록 자동으로 더 많은 반복을 신중하게 수행함
        n_bootstrap_used, mean_auc, ci_low, ci_high, _ = (
            find_stable_bootstrap_count_for_attributes(df_segment, top_attrs)
        )

        rows.append({
            "segment": segment_id,
            "n": len(df_segment),
            "churn_rate": df_segment["ChurnFlag"].mean(),
            "top_attributes": top_attrs,
            "attribute_auc": auc,
            "p_value": p_value,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_bootstrap_used": n_bootstrap_used,
            "n_permutations_used": n_permutations_used,
        })
    return pd.DataFrame(rows)
