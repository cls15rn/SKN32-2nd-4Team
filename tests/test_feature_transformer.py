"""
tests/test_feature_transformer.py

feature_engineering.fit_feature_transformer / FeatureTransformer.transform
회귀테스트 - 코드 점검(12일차) 중 발견된 테스트 커버리지 공백을 메운다.

기존 test_feature_engineering.py는 build_feature_cols(컬럼 분류 로직)만
다뤘는데, 실제로 데이터 누수·Train/Test 일관성과 직결되는 더 중요한 함수
(fit_feature_transformer, FeatureTransformer.transform)에는 테스트가 전혀
없었다 - 점검 과정에서 "우연히 잘 맞는" 속성(컬럼 순서 일치 등)을 직접
확인했지만, 그걸 보장하는 자동 테스트가 없으면 향후 리팩터링 시 같은 속성이
깨져도 아무도 모를 수 있다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "churn_prediction" / "src"))
from feature_engineering import (  # noqa: E402
    FeatureTransformer,
    apply_segment_label,
    fit_feature_transformer,
)


def _make_synthetic_telco_df(n=200, seed=0, tenure_max=72):
    """원본 텔코 컬럼 구조를 갖춘 합성 데이터 (clean_raw_data를 거친 형태로 직접 생성)"""
    rng = np.random.RandomState(seed)
    tenure = rng.randint(0, tenure_max, n)
    churn = rng.binomial(1, 0.3, n)
    return pd.DataFrame({
        "customerID": [f"C{i:04d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.binomial(1, 0.2, n),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n),
        "tenure": tenure,
        "PhoneService": rng.choice(["Yes", "No"], n),
        "MultipleLines": rng.choice(["Yes", "No"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(["Yes", "No"], n),
        "OnlineBackup": rng.choice(["Yes", "No"], n),
        "DeviceProtection": rng.choice(["Yes", "No"], n),
        "TechSupport": rng.choice(["Yes", "No"], n),
        "StreamingTV": rng.choice(["Yes", "No"], n),
        "StreamingMovies": rng.choice(["Yes", "No"], n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)",
             "Credit card (automatic)"], n,
        ),
        "MonthlyCharges": rng.uniform(20, 120, n),
        "TotalCharges": rng.uniform(20, 8000, n),
        "Churn": np.where(churn == 1, "Yes", "No"),
    })


@pytest.fixture
def fitted_transformer_and_data():
    df_train = _make_synthetic_telco_df(n=200, seed=0)
    rules = {
        "analysis_a": {"boundaries": [10.5, 22.5, 54.5]},
        "subtrack_q": {
            "risk_attribute_values": {
                "Contract": "Month-to-month",
                "OnlineSecurity": "No",
            }
        },
    }
    transformer, df_train_with_labels = fit_feature_transformer(df_train, rules)
    return transformer, df_train, df_train_with_labels


def test_apply_segment_label_handles_tenure_beyond_training_max():
    """
    ⚠️ 핵심 회귀 대상: 추론 시점에 학습 때보다 tenure가 더 큰 신규 고객이
    들어와도 NaN 없이 마지막 세그먼트에 안전하게 분류되어야 한다.
    """
    boundaries = [10.5, 22.5, 54.5]
    df_new = pd.DataFrame({"tenure": [0, 30, 100]})  # 100은 boundaries 범위 밖
    result = apply_segment_label(df_new, boundaries)
    assert result["segment"].isna().sum() == 0
    assert result.loc[2, "segment"] == 3  # 마지막 구간(54.5+)에 분류되어야 함


def test_fitted_transformer_feature_cols_3_is_superset_of_12(fitted_transformer_and_data):
    """feature_cols_3가 feature_cols_12 + segment_*로 정확히 구성되어야 한다"""
    transformer, _, _ = fitted_transformer_and_data
    extra = set(transformer.feature_cols_3) - set(transformer.feature_cols_12)
    assert all(c.startswith("segment_") for c in extra)
    assert len(extra) > 0  # segment 컬럼이 최소 1개는 있어야 함


def test_transform_output_columns_match_fitted_feature_cols_order(fitted_transformer_and_data):
    """
    ⚠️ 핵심 회귀 대상: transform()이 반환하는 X_tree_3의 컬럼 순서가
    transformer.feature_cols_3와 정확히 일치해야 한다 - 이게 깨지면 XGBoost가
    학습 시점과 다른 컬럼 순서를 받아 에러를 내거나(이름 검증 통과 시)
    조용히 잘못된 예측을 할 위험이 있다.
    """
    transformer, df_train, _ = fitted_transformer_and_data
    transformed = transformer.transform(df_train)
    assert list(transformed["X_tree_3"].columns) == transformer.feature_cols_3
    assert list(transformed["X_tree_12"].columns) == transformer.feature_cols_12


def test_transform_handles_unseen_category_without_crashing(fitted_transformer_and_data):
    """
    ⚠️ 핵심 회귀 대상: 추론 시점에 학습 때 없던 범주값이 들어와도 (예:
    오타나 새로운 결제수단) reindex의 fill_value=0 덕분에 크래시 없이
    처리되어야 한다 - 해당 더미컬럼이 전부 0으로 채워지는 게 정상 동작.
    """
    transformer, df_train, _ = fitted_transformer_and_data
    df_new = df_train.copy().iloc[:5].reset_index(drop=True)
    df_new.loc[0, "PaymentMethod"] = "Crypto (new method)"  # 학습 때 없던 범주

    transformed = transformer.transform(df_new)
    # 크래시 없이 정상적으로 변환되어야 하고, 행 수가 보존되어야 함
    assert len(transformed["X_tree_3"]) == 5


def test_transform_does_not_refit_scaler(fitted_transformer_and_data):
    """
    ⚠️ 데이터 누수 회귀 테스트: transform()은 저장된 scaler.transform()만
    호출해야 하고 새로 fit하면 안 된다 - 추론 데이터의 분포가 학습 데이터와
    달라도 스케일링 결과가 "학습 시점 평균/표준편차" 기준으로 나와야 한다.
    """
    transformer, df_train, _ = fitted_transformer_and_data

    # 학습 데이터와 분포가 완전히 다른 새 데이터(평균이 훨씬 큼)
    df_extreme = df_train.copy()
    df_extreme["MonthlyCharges"] = df_extreme["MonthlyCharges"] + 10000

    transformed = transformer.transform(df_extreme)
    # scaler가 새로 fit됐다면 변환 후 평균이 0에 가까워야 하지만,
    # 저장된 scaler를 그대로 썼다면 평균이 크게 벗어나 있어야 한다.
    scaled_monthly_charges = transformed["X_linear_3"]["MonthlyCharges"]
    assert scaled_monthly_charges.mean() > 1.0  # 0 근처가 아니라 크게 벗어나 있어야 정상


def test_fit_feature_transformer_uses_train_data_only_for_dummy_columns():
    """
    ⚠️ 핵심 회귀 대상: dummy_columns(원-핫 인코딩 컬럼 목록)이 df_train_raw
    기준으로만 정해져야 한다 - 예를 들어 Train에 InternetService="No"가
    아예 없으면 InternetService_No 컬럼 자체가 dummy_columns에 없어야 하고,
    이후 Test/추론에서 그 값이 나타나도 reindex(fill_value=0)로 안전하게
    처리되어야 한다(test_transform_handles_unseen_category와 같은 원리).
    """
    df_train = _make_synthetic_telco_df(n=100, seed=1)
    df_train["InternetService"] = "DSL"  # Train에는 DSL만 존재(No, Fiber optic 없음)

    rules = {
        "analysis_a": {"boundaries": [10.5, 22.5, 54.5]},
        "subtrack_q": {"risk_attribute_values": {"Contract": "Month-to-month"}},
    }
    transformer, _ = fit_feature_transformer(df_train, rules)
    assert "InternetService_Fiber optic" not in transformer.dummy_columns
    assert "InternetService_No" not in transformer.dummy_columns
    assert "InternetService_DSL" in transformer.dummy_columns
