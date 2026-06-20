"""
tests/test_feature_engineering.py

⚠️ 가장 중요한 회귀 테스트: "1단계는 세그먼트 라벨을 절대 포함하지 않는다"는
원칙이 코드에서 깨지지 않는지 확인한다. feature_cols_12에 segment_* 컬럼이
하나라도 섞여 들어가면, 분석A/B의 핵심 서사(전통통계 비교군은 데이터주도
결과물을 모른다)가 코드 차원에서 무너진다.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "churn_prediction" / "src"))
from feature_engineering import build_feature_cols  # noqa: E402


@pytest.fixture
def encoded_df_with_segments():
    """get_dummies를 거친 것처럼 segment_* 컬럼이 있는 가짜 인코딩 결과"""
    return pd.DataFrame({
        "customerID": ["A", "B"],
        "Churn": ["Yes", "No"],
        "ChurnFlag": [1, 0],
        "tenure": [5, 30],
        "MonthlyCharges": [70.0, 50.0],
        "Contract_Month-to-month": [1, 0],
        "Contract_One year": [0, 1],
        "segment_0": [1, 0],
        "segment_1": [0, 1],
        "segment_2": [0, 0],
        "risk_count": [3, 1],
    })


def test_feature_cols_12_excludes_segment_columns(encoded_df_with_segments):
    """⚠️ feature_cols_12에는 segment_* 컬럼이 절대 포함되면 안 된다"""
    feature_cols_12, _ = build_feature_cols(encoded_df_with_segments)
    segment_cols_in_12 = [c for c in feature_cols_12 if c.startswith("segment_")]
    assert segment_cols_in_12 == [], (
        f"feature_cols_12에 segment 컬럼이 섞여 들어감: {segment_cols_in_12} - "
        f"'1단계는 세그먼트 라벨을 절대 포함하지 않는다'는 원칙이 깨짐"
    )


def test_feature_cols_3_includes_all_segment_columns(encoded_df_with_segments):
    """feature_cols_3에는 모든 segment_* 컬럼이 포함되어야 한다"""
    _, feature_cols_3 = build_feature_cols(encoded_df_with_segments)
    segment_cols_in_3 = [c for c in feature_cols_3 if c.startswith("segment_")]
    assert set(segment_cols_in_3) == {"segment_0", "segment_1", "segment_2"}


def test_feature_cols_3_is_superset_of_feature_cols_12(encoded_df_with_segments):
    """feature_cols_3 = feature_cols_12 + segment_* 만큼 정확히 더 커야 한다"""
    feature_cols_12, feature_cols_3 = build_feature_cols(encoded_df_with_segments)
    assert set(feature_cols_3) - set(feature_cols_12) == {
        "segment_0", "segment_1", "segment_2"
    }


def test_excluded_columns_not_in_either_feature_set(encoded_df_with_segments):
    """customerID, Churn, ChurnFlag, risk_count는 둘 다에서 빠져야 한다"""
    feature_cols_12, feature_cols_3 = build_feature_cols(encoded_df_with_segments)
    excluded = {"customerID", "Churn", "ChurnFlag", "risk_count"}
    assert excluded.isdisjoint(set(feature_cols_12))
    assert excluded.isdisjoint(set(feature_cols_3))
