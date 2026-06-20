"""
tests/test_shared.py

shared/data_loader.py 의 공통 전처리 관문 테스트.

⚠️ 이 테스트는 실제로 발견됐던 버그(predict.py가 TotalCharges 결측 문자열을
처리하지 못해 에러가 났던 것, No-service 통합이 추론 경로에 빠져있던 것)를
회귀 테스트로 고정한다 - 누군가 나중에 shared/data_loader.py를 고치다가
이 처리를 빠뜨리면 이 테스트가 바로 실패해서 알려준다.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from data_loader import clean_raw_data  # noqa: E402


def test_total_charges_string_missing_becomes_zero():
    """TotalCharges가 공백 문자열(' ')이면 0.0으로 채워져야 한다 (실제 버그 회귀 테스트)"""
    df = pd.DataFrame({
        "TotalCharges": [" ", "100.5", "200"],
        "OnlineSecurity": ["No", "Yes", "No"],
        "OnlineBackup": ["No", "Yes", "No"],
        "DeviceProtection": ["No", "Yes", "No"],
        "TechSupport": ["No", "Yes", "No"],
        "StreamingTV": ["No", "Yes", "No"],
        "StreamingMovies": ["No", "Yes", "No"],
        "MultipleLines": ["No", "Yes", "No"],
    })
    result = clean_raw_data(df)
    assert result["TotalCharges"].dtype.kind == "f"
    assert result["TotalCharges"].iloc[0] == 0.0
    assert result["TotalCharges"].iloc[1] == 100.5


def test_no_internet_service_consolidated_to_no():
    """'No internet service' -> 'No' 통합이 모든 관련 컬럼에 적용되어야 한다"""
    df = pd.DataFrame({
        "TotalCharges": ["100", "200"],
        "OnlineSecurity": ["No internet service", "Yes"],
        "OnlineBackup": ["No internet service", "Yes"],
        "DeviceProtection": ["No internet service", "Yes"],
        "TechSupport": ["No internet service", "Yes"],
        "StreamingTV": ["No internet service", "Yes"],
        "StreamingMovies": ["No internet service", "Yes"],
        "MultipleLines": ["No", "Yes"],
    })
    result = clean_raw_data(df)
    assert result["OnlineSecurity"].iloc[0] == "No"
    assert result["OnlineBackup"].iloc[0] == "No"
    assert "No internet service" not in result["OnlineSecurity"].values


def test_no_phone_service_consolidated_to_no():
    """'No phone service' -> 'No' 통합이 MultipleLines에 적용되어야 한다"""
    df = pd.DataFrame({
        "TotalCharges": ["100", "200"],
        "OnlineSecurity": ["No", "Yes"],
        "OnlineBackup": ["No", "Yes"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport": ["No", "Yes"],
        "StreamingTV": ["No", "Yes"],
        "StreamingMovies": ["No", "Yes"],
        "MultipleLines": ["No phone service", "Yes"],
    })
    result = clean_raw_data(df)
    assert result["MultipleLines"].iloc[0] == "No"


def test_clean_raw_data_does_not_mutate_input():
    """원본 df는 변경되지 않아야 한다 (df.copy() 보장)"""
    df = pd.DataFrame({
        "TotalCharges": [" "],
        "OnlineSecurity": ["No internet service"],
        "OnlineBackup": ["No"], "DeviceProtection": ["No"], "TechSupport": ["No"],
        "StreamingTV": ["No"], "StreamingMovies": ["No"], "MultipleLines": ["No"],
    })
    original_value = df["TotalCharges"].iloc[0]
    clean_raw_data(df)
    assert df["TotalCharges"].iloc[0] == original_value
