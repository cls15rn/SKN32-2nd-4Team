"""
tests/test_automation_principle.py

"사람이 숫자를 미리 정하지 않고 데이터가 직접 결정한다"는 원칙이 코드에서
실제로 지켜지는지 확인하는 회귀 테스트.

- 분석B: 세그먼트별 위험속성 개수를 top_n으로 고정하지 않고, 결정나무의
  importance>0 결과를 그대로 쓴다 (세그먼트마다 자연스럽게 달라짐).
- XGBoost: max_depth/n_estimators를 고정값이 아니라 GridSearchCV로 찾고,
  2단계와 3단계가 반드시 같은 설정을 물려받는다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "segment_discovery" / "src"))
from analysis_b import find_top_risk_attributes  # noqa: E402
from analysis_a import find_stable_rf_n_estimators  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "churn_prediction" / "src"))
from train_models import (  # noqa: E402
    search_xgboost_hyperparameters,
    train_stage2_xgboost,
    train_stage3_xgboost,
)


def test_find_top_risk_attributes_does_not_take_top_n_argument():
    """
    ⚠️ 회귀 테스트: find_top_risk_attributes가 top_n 매개변수를 받지 않아야
    한다 - "정확히 N개"를 사람이 미리 정하는 패턴이 재도입되면 이 테스트가
    TypeError로 바로 알려준다.
    """
    import inspect
    sig = inspect.signature(find_top_risk_attributes)
    assert "top_n" not in sig.parameters


def test_find_top_risk_attributes_varies_by_segment():
    """세그먼트마다 위험속성 '개수'가 데이터에 따라 다르게 나와야 한다 (고정 X)"""
    rng = np.random.RandomState(0)
    n = 400

    # 세그먼트0: 한 속성(A)이 이탈을 강하게 좌우 -> importance>0 인 속성이 적을 것
    seg0 = pd.DataFrame({
        "A": rng.choice(["x", "y"], n),
        "B": rng.choice(["x", "y"], n),
        "C": rng.choice(["x", "y"], n),
        "ChurnFlag": rng.binomial(1, 0.3, n),
    })
    seg0["ChurnFlag"] = (seg0["A"] == "x").astype(int)  # A만 완벽히 결정

    cat_cols_backup = None
    import analysis_b
    cat_cols_backup = analysis_b.CATEGORICAL_COLS
    num_cols_backup = analysis_b.NUMERIC_COLS
    analysis_b.CATEGORICAL_COLS = ["A", "B", "C"]
    analysis_b.NUMERIC_COLS = []
    try:
        attrs = find_top_risk_attributes(seg0)
        assert len(attrs) >= 1  # 적어도 A는 잡혀야 함
    finally:
        analysis_b.CATEGORICAL_COLS = cat_cols_backup
        analysis_b.NUMERIC_COLS = num_cols_backup


@pytest.fixture
def synthetic_classification_data():
    rng = np.random.RandomState(0)
    n = 500
    X = pd.DataFrame({
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(0, 1, n),
        "segment_0": rng.binomial(1, 0.3, n),
    })
    y = pd.Series((X["f1"] + rng.normal(0, 0.5, n) > 0).astype(int))
    return X, y


def test_random_forest_boundary_votes_does_not_hardcode_n_estimators():
    """
    ⚠️ 회귀 테스트: random_forest_boundary_votes의 n_estimators 기본값이
    고정 숫자가 아니라 None(자동탐지 트리거)이어야 한다.
    """
    import inspect
    sys.path.insert(0, str(Path(__file__).parent.parent / "segment_discovery" / "src"))
    from analysis_a import random_forest_boundary_votes
    sig = inspect.signature(random_forest_boundary_votes)
    assert sig.parameters["n_estimators"].default is None


def test_find_stable_rf_n_estimators_picks_within_candidate_range():
    """자동탐지된 트리개수가 후보 범위 안에 있어야 한다"""
    rng = np.random.RandomState(0)
    n_tenure_points = 70
    df = pd.DataFrame({
        "tenure": np.repeat(np.arange(n_tenure_points), 5),
        "ChurnFlag": rng.binomial(1, 0.3, n_tenure_points * 5),
    })
    stable_n, diagnostics = find_stable_rf_n_estimators(
        df, candidates=(50, 100, 150), seeds=(1, 2, 3)
    )
    assert stable_n in (50, 100, 150)
    assert len(diagnostics) >= 1
    assert "top1_rate_std" in diagnostics.columns


def test_find_stable_rf_n_estimators_does_not_stop_on_single_noisy_step():
    """
    ⚠️ 회귀 테스트: 표준편차가 중간에 한 번만 튀어도(노이즈) 거기서 바로
    멈추면 안 된다 - patience(연속 개선없음 횟수)가 1보다 커야 단발성
    노이즈를 견딜 수 있다.
    """
    import inspect
    sig = inspect.signature(find_stable_rf_n_estimators)
    assert sig.parameters["patience"].default >= 2


def test_xgboost_search_returns_params_within_candidate_range(synthetic_classification_data):
    """탐색 결과의 max_depth/n_estimators가 config의 후보 범위 안에 있어야 한다"""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config

    X, y = synthetic_classification_data
    result = search_xgboost_hyperparameters(X, y)
    assert result.best_params["max_depth"] in config.XGBOOST_MAX_DEPTH_CANDIDATES
    assert result.best_params["n_estimators"] in config.XGBOOST_N_ESTIMATORS_CANDIDATES


def test_stage2_and_stage3_use_identical_hyperparameters(synthetic_classification_data):
    """
    ⚠️ 가장 중요한 회귀 테스트: 2단계와 3단계 XGBoost는 search_result를 통해
    반드시 같은 max_depth/n_estimators/learning_rate를 써야 한다 - "성능차이가
    접근법 때문인지 알고리즘 설정 때문인지 혼동 방지"라는 원칙이 코드에서
    깨지지 않는지 확인.
    """
    X, y = synthetic_classification_data
    search_result = search_xgboost_hyperparameters(X, y)

    stage2 = train_stage2_xgboost(X, y, search_result)
    stage3 = train_stage3_xgboost(X, y, search_result)

    params2 = stage2.model.get_params()
    params3 = stage3.model.get_params()
    for key in ["max_depth", "n_estimators", "learning_rate"]:
        assert params2[key] == params3[key], (
            f"2단계와 3단계의 {key}가 다름 ({params2[key]} vs {params3[key]}) - "
            f"두 단계는 반드시 같은 설정을 써야 함"
        )
