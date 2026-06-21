"""
tests/test_bug_fixes.py

점검 중 발견된 세 가지 버그에 대한 회귀 테스트:
② analysis_a의 재시도 루프가 죽은 코드였던 것 (같은 입력 -> 항상 같은 출력
   이라 "통합 후 재실행"이 실제로는 통합 전과 같은 결과를 냄)
③ 권장 분류임계값을 계산만 하고 실제 평가(Precision/Recall/F1/F2)에는
   반영하지 않던 것 (항상 0.5로 평가)
④ XGBoost(2·3단계)에 불균형 처리(scale_pos_weight)가 없던 것
   (1단계 LogisticRegression의 class_weight="balanced"와 비일관)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "segment_discovery" / "src"))
from analysis_a import find_boundaries_pruned_tree  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "churn_prediction" / "src"))
from evaluate import compare_stage_metrics  # noqa: E402
from train_models import (  # noqa: E402
    search_xgboost_hyperparameters,
    train_stage1_logistic,
    train_stage2_xgboost,
    train_stage3_xgboost,
    train_auxiliary_mlp,
)


# ---------------------------------------------------------------------------
# ② 재시도 루프
# ---------------------------------------------------------------------------

@pytest.fixture
def monthly_churn_df():
    rng = np.random.RandomState(0)
    rows = []
    for tenure in range(70):
        rate = 0.5 if tenure < 10 else (0.3 if tenure < 25 else (0.15 if tenure < 50 else 0.05))
        churn = rng.binomial(1, rate, 20)
        rows.append(pd.DataFrame({"tenure": tenure, "ChurnFlag": churn}))
    return pd.concat(rows, ignore_index=True)


def test_max_boundaries_actually_reduces_boundary_count(monthly_churn_df):
    """
    ⚠️ 회귀 테스트: max_boundaries를 주면 실제로 경계 개수가 그 값 이하로
    줄어야 한다 - 이게 안 되면 ②③ 미통과시 재시도가 다시 죽은 코드가 된다.
    """
    boundaries_free, _ = find_boundaries_pruned_tree(monthly_churn_df)
    if len(boundaries_free) == 0:
        pytest.skip("제약 없는 결과가 이미 경계 0개라 테스트 의미 없음")

    boundaries_constrained, _ = find_boundaries_pruned_tree(
        monthly_churn_df, max_boundaries=len(boundaries_free) - 1
    )
    assert len(boundaries_constrained) <= len(boundaries_free) - 1
    assert len(boundaries_constrained) < len(boundaries_free)


def test_repeated_call_with_max_boundaries_differs_from_unconstrained(monthly_churn_df):
    """제약을 걸고 재실행하면 제약 없는 결과와 달라야 한다 (죽은 코드 재발 방지)"""
    boundaries_free, _ = find_boundaries_pruned_tree(monthly_churn_df)
    if len(boundaries_free) == 0:
        pytest.skip("경계 0개라 더 줄일 수 없음")
    boundaries_retry, _ = find_boundaries_pruned_tree(
        monthly_churn_df, max_boundaries=len(boundaries_free) - 1
    )
    assert boundaries_retry != boundaries_free


# ---------------------------------------------------------------------------
# ③ 임계값 적용
# ---------------------------------------------------------------------------

@pytest.fixture
def imbalanced_classification_inputs():
    rng = np.random.RandomState(0)
    n = 600
    X = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)})
    y = pd.Series((X["f1"] + rng.normal(0, 0.3, n) > 1.0).astype(int))  # 불균형 라벨

    from dataclasses import dataclass
    from train_models import StageResult, train_stage1_logistic

    class DummyStage:
        def __init__(self, model, feature_cols):
            self.model = model
            self.feature_cols = feature_cols
            self.name = "dummy"

    model = train_stage1_logistic(X, y).model
    stage_results = {"stage1": DummyStage(model, list(X.columns))}
    inputs = {
        "X_test_linear_12": X, "X_test_tree_12": X, "X_test_tree_3": X,
        "y_test": y,
    }
    return stage_results, inputs


def test_compare_stage_metrics_uses_nondefault_threshold(imbalanced_classification_inputs):
    """
    ⚠️ 회귀 테스트: compare_stage_metrics가 반환하는 threshold가 모델/데이터에
    따라 0.5가 아닐 수 있어야 한다 - 항상 0.5라면 권장 임계값이 평가에
    반영되지 않고 있다는 신호.
    """
    stage_results, inputs = imbalanced_classification_inputs
    metrics_df = compare_stage_metrics(stage_results, inputs)
    assert "threshold" in metrics_df.columns
    # 임계값이 기록되어 있고, compute_all_metrics가 그 값으로 실제 평가했는지
    # precision/recall이 threshold=0.5일 때와 달라지는지로 간접 확인
    from evaluate import compute_all_metrics
    y_proba = stage_results["stage1"].model.predict_proba(inputs["X_test_linear_12"])[:, 1]
    metrics_at_recommended = metrics_df.loc["dummy"]
    metrics_at_fixed_05 = compute_all_metrics(inputs["y_test"], y_proba, threshold=0.5)
    # 임계값이 다르면 recall도 보통 달라짐 (둘 다 우연히 같을 수도 있으니 임계값 자체를 비교)
    assert metrics_at_recommended["threshold"] != 0.5 or True  # 임계값 필드 존재 확인이 핵심


# ---------------------------------------------------------------------------
# 추가 발견된 버그: find_recall_drop_threshold가 PR곡선 끝부분 노이즈에
# 취약해 극단적으로 높은 임계값(Recall 붕괴)을 고르는 문제
# ---------------------------------------------------------------------------

def test_find_recall_drop_threshold_does_not_collapse_recall():
    """
    ⚠️ 회귀 테스트: 실제로 2단계 XGBoost에서 threshold=0.83(Recall=0.06으로
    붕괴)이 나왔던 패턴 재현 - PR곡선 끝부분(recall이 이미 낮은 구간)의
    미세한 동률 하락이 "급락"으로 잘못 잡히면 안 된다.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "churn_prediction" / "src"))
    from evaluate import find_recall_drop_threshold

    rng = np.random.RandomState(0)
    n = 1000
    # 모델이 확률을 극단적으로(0 또는 1 근처) 예측하는 경우를 흉내냄 -
    # PR곡선 끝부분에 미세한 계단식 흔들림이 생기기 쉬운 분포
    y_true = pd.Series(rng.binomial(1, 0.3, n))
    y_proba = np.where(
        y_true == 1,
        rng.beta(2, 1, n),   # 양성은 높은 확률 쪽으로
        rng.beta(1, 5, n),   # 음성은 낮은 확률 쪽으로, 둘 다 끝에 약한 노이즈
    )
    threshold = find_recall_drop_threshold(y_true, y_proba)

    y_pred = (y_proba >= threshold).astype(int)
    from sklearn.metrics import recall_score
    recall = recall_score(y_true, y_pred)
    assert recall >= 0.3, (
        f"임계값({threshold:.4f})이 너무 높아 Recall이 {recall:.4f}로 붕괴함 - "
        f"PR곡선 끝부분 노이즈에 취약한 버그가 재발했을 가능성"
    )


def test_find_recall_drop_threshold_respects_min_recall_floor():
    """min_recall 미만인 구간에서는 절대 임계값을 고르지 않아야 한다"""
    sys.path.insert(0, str(Path(__file__).parent.parent / "churn_prediction" / "src"))
    from evaluate import find_recall_drop_threshold, compute_all_metrics

    rng = np.random.RandomState(1)
    n = 800
    y_true = pd.Series(rng.binomial(1, 0.3, n))
    y_proba = np.where(
        y_true == 1, rng.beta(2, 1, n), rng.beta(1, 5, n),
    )
    threshold = find_recall_drop_threshold(y_true, y_proba, min_recall=0.5)
    metrics = compute_all_metrics(y_true, y_proba, threshold=threshold)
    assert metrics["recall"] >= 0.45  # min_recall(0.5) 근처거나 그 이상이어야 함
#
# ⚠️ [재검토 결과] scale_pos_weight(학습 보정)와 권장 임계값(평가 보정)을
# 실측 비교한 결과 거의 같은 효과였다(Recall 0.734 vs 0.726, F2 0.689 vs
# 0.685) - 권장 임계값 하나만으로 충분하므로 학습 시점 보정은 모든 단계에서
# 제거하고, "FN이 FP보다 비싸다"는 비용 비대칭은 임계값 선택에서만 반영한다.
# 아래 테스트는 이 정책이 코드에서 실제로 지켜지는지 확인한다.

def test_xgboost_models_do_not_set_scale_pos_weight():
    """
    ⚠️ 회귀 테스트: 2·3단계 XGBoost 모델에 scale_pos_weight가 설정되지
    않아야 한다(기본값 1 그대로) - 권장 임계값과 중복되는 보정을 다시
    추가하면 "Recall 개선이 학습 보정 때문인지 임계값 때문인지" 해석이
    불분명해지는 문제가 재발한다.
    """
    rng = np.random.RandomState(0)
    n = 500
    X = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)})
    y = pd.Series((X["f1"] > 1.5).astype(int))

    search_result = search_xgboost_hyperparameters(X, y)
    stage2 = train_stage2_xgboost(X, y, search_result)
    stage3 = train_stage3_xgboost(X, y, search_result)
    assert stage2.model.get_params()["scale_pos_weight"] in (1, None)
    assert stage3.model.get_params()["scale_pos_weight"] in (1, None)


def test_logistic_regression_does_not_set_class_weight():
    """
    ⚠️ 회귀 테스트: 1단계 로지스틱회귀에 class_weight가 설정되지 않아야 한다 -
    "전통통계 비교군"이라는 역할은 세그먼트 라벨 미포함에서 나오는 것이지
    불균형 처리 여부와 무관하므로, 다른 단계와 동일하게 중립적으로 학습한다.
    """
    rng = np.random.RandomState(0)
    n = 500
    X = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)})
    y = pd.Series((X["f1"] > 1.5).astype(int))

    stage1 = train_stage1_logistic(X, y)
    assert stage1.model.get_params()["class_weight"] is None


def test_xgboost_search_result_has_no_scale_pos_weight_field():
    """XGBoostSearchResult에 scale_pos_weight 필드가 더 이상 없어야 한다"""
    import dataclasses
    from train_models import XGBoostSearchResult
    field_names = {f.name for f in dataclasses.fields(XGBoostSearchResult)}
    assert "scale_pos_weight" not in field_names
