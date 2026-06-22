"""
tests/test_extract_risk_attribute_values.py

analysis_b.extract_risk_attribute_values 회귀테스트.

배경 (코드 점검 중 발견, 12일차 보강): app.py가 risk_attribute_values를
사람이 직접 하드코딩하고 있었는데, 그중 PaymentMethod·TechSupport는
분석B의 top_attributes에 등장한 적이 없는 속성이었다 - 메모 4.1-B의
"분석B의 검증을 그대로 물려받음, 새로 임의 선정하지 않음" 원칙을 코드가
어기고 있던 정합성 버그. extract_risk_attribute_values는 분석B 결과에서
범주형 위험속성의 위험값만 자동으로 추출해 이 버그를 근본적으로 막는다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "segment_discovery" / "src"))
from analysis_b import extract_risk_attribute_values  # noqa: E402


def _make_segment_df_with_known_risk_pattern(n=400, seed=0):
    """
    Contract(범주형, 위험값=Month-to-month)와 MonthlyCharges(연속형)가
    둘 다 위험속성으로 나오는 합성 데이터 - 두 종류를 한 데이터에 공존시켜
    "범주형만 자동 추출, 연속형은 제외"가 동시에 검증되도록 설계.
    """
    rng = np.random.RandomState(seed)
    segment = rng.binomial(1, 0.5, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n)
    monthly_charges = rng.uniform(20, 120, n)

    # Month-to-month일 때 이탈 확률이 명확히 높도록 설계
    churn_prob = np.where(contract == "Month-to-month", 0.6, 0.1)
    churn = rng.binomial(1, churn_prob)

    return pd.DataFrame({
        "segment": segment,
        "Contract": contract,
        "MonthlyCharges": monthly_charges,
        "ChurnFlag": churn,
    })


def test_extracts_risk_value_for_categorical_attribute():
    """범주형 속성(Contract)은 이탈률이 가장 높은 값을 위험값으로 자동 추출해야 한다"""
    df = _make_segment_df_with_known_risk_pattern()
    result_b = pd.DataFrame([
        {"segment": 0, "top_attributes": ["Contract"]},
        {"segment": 1, "top_attributes": ["Contract"]},
    ])
    risk_attrs = extract_risk_attribute_values(df, result_b)
    assert risk_attrs["Contract"] == "Month-to-month"


def test_excludes_continuous_attribute_from_output():
    """
    ⚠️ 핵심 회귀 대상: 연속형 속성(MonthlyCharges)은 "위험값" 개념이 없으므로
    결과 딕셔너리에 포함되면 안 된다 - CATEGORICAL_COLS에 없는 속성은 제외.
    """
    df = _make_segment_df_with_known_risk_pattern()
    result_b = pd.DataFrame([
        {"segment": 0, "top_attributes": ["Contract", "MonthlyCharges"]},
    ])
    risk_attrs = extract_risk_attribute_values(df, result_b)
    assert "MonthlyCharges" not in risk_attrs
    assert "Contract" in risk_attrs


def test_does_not_invent_attributes_not_in_result_b():
    """
    ⚠️ 핵심 회귀 대상(정합성 버그 재발 방지): result_b의 top_attributes에
    등장하지 않는 속성은 절대로 결과에 나타나면 안 된다 - 과거 app.py가
    PaymentMethod·TechSupport를 분석B 결과와 무관하게 하드코딩했던
    정합성 버그가 재발하지 않는지 직접 검증.
    """
    df = _make_segment_df_with_known_risk_pattern()
    df["PaymentMethod"] = "Electronic check"  # 데이터에는 존재하지만
    df["TechSupport"] = "No"                   # top_attributes에는 없음

    result_b = pd.DataFrame([
        {"segment": 0, "top_attributes": ["Contract"]},  # PaymentMethod/TechSupport 미포함
    ])
    risk_attrs = extract_risk_attribute_values(df, result_b)
    assert "PaymentMethod" not in risk_attrs
    assert "TechSupport" not in risk_attrs
    assert set(risk_attrs.keys()) == {"Contract"}


def test_handles_multiple_segments_with_overlapping_attributes():
    """같은 속성이 여러 세그먼트에 걸쳐 나와도 에러 없이 처리되어야 한다"""
    df = _make_segment_df_with_known_risk_pattern()
    result_b = pd.DataFrame([
        {"segment": 0, "top_attributes": ["Contract"]},
        {"segment": 1, "top_attributes": ["Contract"]},
    ])
    risk_attrs = extract_risk_attribute_values(df, result_b)
    assert risk_attrs["Contract"] == "Month-to-month"


def test_real_data_matches_previously_hardcoded_categorical_mapping():
    """
    실제 텔코 데이터에서 자동 추출한 매핑이, 과거 사람이 손으로 적었던
    범주형 매핑(Contract=Month-to-month, InternetService=Fiber optic,
    OnlineSecurity=No)과 정확히 일치하는지 확인 - 자동화가 기존의 올바른
    부분은 그대로 재현함을 검증.
    """
    import importlib
    shared_path = str(Path(__file__).parent.parent / "shared")
    if shared_path not in sys.path:
        sys.path.insert(0, shared_path)
    from data_loader import clean_raw_data

    sys.path.insert(0, str(Path(__file__).parent.parent / "segment_discovery" / "src"))
    from preprocessing import run_preprocessing
    from analysis_a import make_segment_column

    data_path = Path(__file__).parent.parent / "data" / "WA_FnUseC_TelcoCustomerChurn.csv"
    if not data_path.exists():
        pytest.skip("실데이터 파일이 없어 스킵")

    df_train, _ = run_preprocessing(str(data_path))
    df_train = df_train.copy()
    df_train["segment"] = make_segment_column(df_train, [10.5, 22.5, 54.5])

    result_b = pd.DataFrame([
        {"segment": 0, "top_attributes": ["MonthlyCharges", "InternetService", "OnlineSecurity"]},
        {"segment": 1, "top_attributes": ["MonthlyCharges"]},
        {"segment": 2, "top_attributes": ["Contract", "MonthlyCharges"]},
        {"segment": 3, "top_attributes": ["Contract", "MonthlyCharges"]},
    ])
    risk_attrs = extract_risk_attribute_values(df_train, result_b)

    assert risk_attrs == {
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "Contract": "Month-to-month",
    }
