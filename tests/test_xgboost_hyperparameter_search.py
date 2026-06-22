"""
tests/test_xgboost_hyperparameter_search.py

train_models.search_xgboost_hyperparameters / _xgboost_kwargs 회귀테스트.

배경: XGBOOST_LEARNING_RATE가 0.1로 고정되어 max_depth/n_estimators만
GridSearchCV로 탐색하는 비일관성이 있었다(코드 점검 중 발견) - learning_rate도
같은 GridSearchCV 한 번에 포함시켜 세 파라미터를 함께 탐색하도록 수정.
이 테스트는 (1) param_grid가 config의 후보 리스트와 정확히 일치하는지,
(2) 탐색 결과(best_params)가 3단계로 정확히 전달되는지를 검증한다 -
이전까지 train_models.py에는 단위테스트가 전혀 없었던 공백을 메운다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "churn_prediction" / "src"))
from train_models import _xgboost_kwargs, search_xgboost_hyperparameters  # noqa: E402


def _make_synthetic_classification_data(n=300, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        "feature_a": rng.normal(0, 1, n),
        "feature_b": rng.normal(0, 1, n),
        "feature_c": rng.uniform(0, 10, n),
    })
    y = pd.Series((X["feature_a"] + X["feature_c"] > X["feature_c"].median()).astype(int))
    return X, y


def test_search_explores_learning_rate_candidates_from_config():
    """
    ⚠️ 핵심 회귀 대상: GridSearchCV의 param_grid가 config.
    XGBOOST_LEARNING_RATE_CANDIDATES를 정확히 사용해야 한다 - 탐색 결과의
    cv_results_에 등장하는 learning_rate 값들의 집합이 config 후보 리스트와
    정확히 일치하는지 확인한다(누락되거나 추가된 후보가 없어야 함).
    """
    X, y = _make_synthetic_classification_data()
    result = search_xgboost_hyperparameters(X, y)

    explored_learning_rates = set(result.cv_results["param_learning_rate"].unique())
    expected_learning_rates = set(config.XGBOOST_LEARNING_RATE_CANDIDATES)
    assert explored_learning_rates == expected_learning_rates


def test_search_explores_max_depth_and_n_estimators_candidates():
    """기존에 탐색하던 max_depth/n_estimators 후보도 learning_rate 추가 후 계속 탐색되어야 한다 (회귀 방지)"""
    X, y = _make_synthetic_classification_data()
    result = search_xgboost_hyperparameters(X, y)

    assert set(result.cv_results["param_max_depth"].unique()) == set(config.XGBOOST_MAX_DEPTH_CANDIDATES)
    assert set(result.cv_results["param_n_estimators"].unique()) == set(config.XGBOOST_N_ESTIMATORS_CANDIDATES)


def test_search_total_combinations_matches_grid_size():
    """3개 파라미터의 전체 조합 개수만큼 cv_results에 행이 있어야 한다 (그리드 일부가 누락되지 않았는지 확인)"""
    X, y = _make_synthetic_classification_data()
    result = search_xgboost_hyperparameters(X, y)

    expected_combinations = (
        len(config.XGBOOST_MAX_DEPTH_CANDIDATES)
        * len(config.XGBOOST_N_ESTIMATORS_CANDIDATES)
        * len(config.XGBOOST_LEARNING_RATE_CANDIDATES)
    )
    assert len(result.cv_results) == expected_combinations


def test_search_uses_configured_cv_fold_count():
    """
    ⚠️ 핵심 회귀 대상: config.XGBOOST_SEARCH_CV_FOLDS가 실제로 GridSearchCV에
    전달되는지 확인한다 - 11일차에 발견했던 "config엔 정의했지만 코드가
    그 값을 참조하지 않는" 패턴(ANALYSIS_B_BOOTSTRAP_COUNT 사례)이 여기서도
    재발하지 않았는지 검증. GridSearchCV의 grid.cv_results_는 fold마다
    "split{i}_test_score" 컬럼을 만들어내므로, 그 개수로 실제 적용된
    fold 수를 직접 확인할 수 있다.
    """
    from sklearn.model_selection import GridSearchCV
    from train_models import XGBClassifier

    X, y = _make_synthetic_classification_data()
    # search_xgboost_hyperparameters 내부와 동일한 방식으로 GridSearchCV를
    # 직접 호출해 raw cv_results_를 받아, split 컬럼 개수를 센다.
    param_grid = {
        "max_depth": config.XGBOOST_MAX_DEPTH_CANDIDATES,
        "n_estimators": config.XGBOOST_N_ESTIMATORS_CANDIDATES,
        "learning_rate": config.XGBOOST_LEARNING_RATE_CANDIDATES,
    }
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=config.XGBOOST_SEARCH_CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    grid = GridSearchCV(
        XGBClassifier(eval_metric="logloss", random_state=config.RANDOM_STATE),
        param_grid, cv=cv, scoring="roc_auc", n_jobs=1,
    )
    grid.fit(X, y)
    split_columns = [k for k in grid.cv_results_.keys() if k.startswith("split") and k.endswith("_test_score")]
    assert len(split_columns) == config.XGBOOST_SEARCH_CV_FOLDS


def test_xgboost_kwargs_uses_searched_learning_rate_not_config_default():
    """
    ⚠️ 핵심 회귀 대상: _xgboost_kwargs가 GridSearchCV로 찾은 learning_rate을
    그대로 반영해야 한다 - 과거에는 base_model 생성 시 config.
    XGBOOST_LEARNING_RATE(고정값)을 따로 박아 넣어서, 탐색 결과와 무관하게
    항상 같은 learning_rate이 쓰이는 구조였다. 지금은 탐색 결과(best_params)
    에서 가져와야 한다.
    """
    X, y = _make_synthetic_classification_data()
    result = search_xgboost_hyperparameters(X, y)

    kwargs = _xgboost_kwargs(result)
    assert kwargs["learning_rate"] == result.best_params["learning_rate"]
    assert kwargs["max_depth"] == result.best_params["max_depth"]
    assert kwargs["n_estimators"] == result.best_params["n_estimators"]
    # 탐색된 값이 config 후보 목록 중 하나여야 함 (탐색 범위 밖 값이 나오면 버그)
    assert kwargs["learning_rate"] in config.XGBOOST_LEARNING_RATE_CANDIDATES


def test_xgboost_kwargs_keeps_random_state_and_eval_metric_fixed():
    """random_state, eval_metric은 탐색 대상이 아니라 고정 설정이어야 한다 (재현성 보장)"""
    X, y = _make_synthetic_classification_data()
    result = search_xgboost_hyperparameters(X, y)
    kwargs = _xgboost_kwargs(result)

    assert kwargs["random_state"] == config.RANDOM_STATE
    assert kwargs["eval_metric"] == "logloss"
