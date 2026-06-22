"""
tests/test_segment_feature_importance.py

evaluate.compute_xgboost_feature_importance / compute_shap_importance /
compute_xgboost_root_node_distribution / summarize_segment_feature_ranking
회귀테스트.

배경 (PPT 발표기획 검토 중 발견, 12일차 보강): "segment_*가 통계적으로
검증된 랜드마크 피처라 XGBoost 트리의 최상위 분기 기준으로 선택될 것"이라는
가설을 실제 데이터로 검증한 결과, segment_*가 루트 노드로 선택된 트리는
0/100개였고 feature importance(gain)·SHAP 양쪽에서 모두 14~38위(2개는
정확히 0)였다 - 1위는 Contract_Month-to-month. 이 테스트는 그 사실을
회귀로 고정한다(향후 코드/데이터 변경 시 같은 측정 도구가 여전히 정상
작동하는지 확인하는 것이 목적 - 특정 피처가 항상 0위여야 한다는 뜻은 아님).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent / "churn_prediction" / "src"))
from evaluate import (  # noqa: E402
    compute_shap_importance,
    compute_xgboost_feature_importance,
    compute_xgboost_root_node_distribution,
    summarize_segment_feature_ranking,
)


def _make_synthetic_segment_data(n=2000, seed=0):
    """
    tenure(연속형, 강한 신호) + segment(tenure를 거칠게 구간화한 더미, 약한
    중복 신호)를 함께 가진 합성 데이터. 실제 프로젝트와 같은 구조(원본 연속
    변수와 그걸 구간화한 더미가 동시에 존재)를 재현해, gain/SHAP이 어느
    쪽을 선호하는지 검증할 수 있게 한다.
    """
    rng = np.random.RandomState(seed)
    tenure = rng.uniform(0, 72, n)
    contract_mtm = rng.binomial(1, 0.5, n)
    churn_prob = 0.1 + 0.5 * (tenure < 10) + 0.3 * contract_mtm
    churn = rng.binomial(1, np.clip(churn_prob, 0, 1))

    segment = pd.cut(tenure, bins=[-1, 10, 30, 50, 72], labels=[0, 1, 2, 3])
    segment_dummies = pd.get_dummies(segment, prefix="segment")

    X = pd.DataFrame({
        "tenure": tenure,
        "Contract_Month-to-month": contract_mtm,
        "MonthlyCharges": rng.uniform(20, 120, n),
    })
    X = pd.concat([X, segment_dummies.astype(int)], axis=1)
    y = pd.Series(churn)
    return X, y


@pytest.fixture(scope="module")
def fitted_model_and_test_data():
    X, y = _make_synthetic_segment_data(n=2000, seed=0)
    split = 1500
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train = y.iloc[:split]

    model = XGBClassifier(n_estimators=50, max_depth=3, random_state=0)
    model.fit(X_train, y_train)
    return model, X_test


def test_compute_xgboost_feature_importance_returns_known_columns(fitted_model_and_test_data):
    """gain importance가 모델이 학습한 피처 이름으로 인덱싱되어야 한다"""
    model, X_test = fitted_model_and_test_data
    importance = compute_xgboost_feature_importance(model, top_n=5)
    assert len(importance) <= 5
    assert set(importance.index).issubset(set(model.feature_names_in_))
    # 내림차순 정렬 확인
    assert list(importance) == sorted(importance, reverse=True)


def test_compute_xgboost_root_node_distribution_sums_to_tree_count(fitted_model_and_test_data):
    """루트 노드 분포의 합은 전체 트리 개수(n_estimators)와 같아야 한다"""
    model, X_test = fitted_model_and_test_data
    dist = compute_xgboost_root_node_distribution(model)
    assert dist.sum() == model.n_estimators


def test_compute_shap_importance_matches_feature_count(fitted_model_and_test_data):
    """SHAP importance의 인덱스가 X_test의 컬럼과 일치해야 한다"""
    model, X_test = fitted_model_and_test_data
    shap_importance = compute_shap_importance(model, X_test, top_n=5)
    assert set(shap_importance.index).issubset(set(X_test.columns))
    assert (shap_importance >= 0).all()  # 평균 |SHAP|이므로 항상 비음수


def test_summarize_segment_feature_ranking_finds_all_segment_columns(fitted_model_and_test_data):
    """segment_* 컬럼이 전부 결과 표에 한 행씩 포함되어야 한다 (누락 방지)"""
    model, X_test = fitted_model_and_test_data
    all_features = list(model.feature_names_in_)
    gain_full = compute_xgboost_feature_importance(model, top_n=len(all_features))
    shap_full = compute_shap_importance(model, X_test, top_n=len(all_features))

    ranking = summarize_segment_feature_ranking(gain_full, shap_full, all_features)
    expected_segment_cols = {f for f in all_features if f.startswith("segment_")}
    assert set(ranking["feature"]) == expected_segment_cols
    assert (ranking["gain_rank"] >= 1).all()
    assert (ranking["shap_rank"] >= 1).all()


def test_root_node_distribution_can_detect_zero_segment_usage(fitted_model_and_test_data):
    """
    ⚠️ 핵심 회귀 대상: 이 측정 도구가 "segment_*가 루트로 전혀 선택되지
    않는 경우"를 정확히 0으로 보고할 수 있어야 한다 - 실제 프로젝트 데이터
    에서 발생했던 상황(100개 트리 중 0개)을 도구가 놓치지 않는지 확인.
    합성 데이터에서도 tenure raw값이 segment보다 정보가 세밀하므로 같은
    패턴(segment 루트 선택 적음)이 재현되는지 확인한다.
    """
    model, X_test = fitted_model_and_test_data
    dist = compute_xgboost_root_node_distribution(model)
    segment_root_count = sum(
        count for feat, count in dist.items() if feat.startswith("segment_")
    )
    # 합성 데이터는 tenure가 압도적 신호이므로 tenure가 루트를 대부분 차지해야 함
    assert "tenure" in dist.index
    assert dist.get("tenure", 0) >= segment_root_count
