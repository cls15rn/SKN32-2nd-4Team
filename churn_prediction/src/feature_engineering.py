"""
segment_rules.json (segment_discovery 의 산출물) 을 읽어서, 예측모델 학습에
바로 쓸 수 있는 X, y 를 구성한다.

⚠️ 단계별 X 구성이 다름 - 인코딩만 하고 끝내지 말 것 (기획_메모.md 7장 8단계 참조)
feature_cols_12 (1·2단계용, segment_* 컬럼 명시적 제외)
feature_cols_3  (3단계용, segment_* 포함)
하나의 X를 그대로 1·2·3단계에 다 쓰면 "1단계는 세그먼트 라벨을 절대
포함하지 않는다" 는 원칙이 코드에서 깨진다.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
BINARY_MAP_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_segment_rules(rules_path: str | Path) -> dict:
    """segment_discovery/outputs/segment_rules.json 로드"""
    with open(rules_path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_segment_label(df: pd.DataFrame, boundaries: list[float]) -> pd.DataFrame:
    """확정된 경계로 segment 컬럼 부여 (df_train/df_test 양쪽에 동일하게 적용)"""
    df = df.copy()
    bins = [-1] + list(boundaries) + [df["tenure"].max() + 1]
    labels = list(range(len(bins) - 1))
    df["segment"] = pd.cut(df["tenure"], bins=bins, labels=labels).astype(int)
    return df


def apply_risk_count(df: pd.DataFrame, risk_attribute_values: dict[str, str]) -> pd.DataFrame:
    """서브트랙Q의 risk_count 컬럼 부여 (해석/보강용, 예측모델 입력 아님)"""
    df = df.copy()
    masks = pd.DataFrame(index=df.index)
    for col, risky_value in risk_attribute_values.items():
        masks[col] = (df[col] == risky_value).astype(int)
    df["risk_count"] = masks.sum(axis=1)
    return df


def encode_features(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    - 이진변수: .map
    - 다중범주형(Contract 등) + segment: 원-핫
    - df_train 기준 더미컬럼을 df_test 에 강제 정렬
    """
    df_train = df_train.copy()
    df_test = df_test.copy()

    for col in BINARY_MAP_COLS:
        df_train[col] = df_train[col].map({"Yes": 1, "No": 0})
        df_test[col] = df_test[col].map({"Yes": 1, "No": 0})

    multi_categorical = [c for c in CATEGORICAL_COLS if c not in BINARY_MAP_COLS]
    onehot_cols = multi_categorical + ["segment"]

    df_train_enc = pd.get_dummies(df_train, columns=onehot_cols)
    df_test_enc = pd.get_dummies(df_test, columns=onehot_cols)

    # df_train 기준 더미컬럼을 df_test에 강제 정렬
    df_test_enc = df_test_enc.reindex(columns=df_train_enc.columns, fill_value=0)

    return df_train_enc, df_test_enc


def scale_numeric_for_linear(df_train_enc: pd.DataFrame, df_test_enc: pd.DataFrame):
    """선형/MLP용 표준화 (Train fit, Test transform). 트리기반은 원본 그대로 별도 유지."""
    scaler = StandardScaler()
    df_train_scaled = df_train_enc.copy()
    df_test_scaled = df_test_enc.copy()
    df_train_scaled[NUMERIC_COLS] = scaler.fit_transform(df_train_enc[NUMERIC_COLS])
    df_test_scaled[NUMERIC_COLS] = scaler.transform(df_test_enc[NUMERIC_COLS])
    return df_train_scaled, df_test_scaled


def build_feature_cols(df_encoded: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    feature_cols_12 (segment_* 제외) / feature_cols_3 (segment_* 포함) 구성.
    customerID, Churn, ChurnFlag 등 비입력 컬럼은 제외.
    """
    exclude = {"customerID", "Churn", "ChurnFlag", "risk_count"}
    all_cols = [c for c in df_encoded.columns if c not in exclude]

    segment_cols = [c for c in all_cols if c.startswith("segment_")]
    feature_cols_12 = [c for c in all_cols if c not in segment_cols]
    feature_cols_3 = feature_cols_12 + segment_cols
    return feature_cols_12, feature_cols_3


def prepare_model_inputs(
    df_train_raw: pd.DataFrame, df_test_raw: pd.DataFrame, rules: dict,
):
    """
    전체 파이프라인: segment_rules.json 적용 -> 인코딩 -> 스케일링 -> X/y 구성.

    Returns
    -------
    dict with keys:
        X_train_tree_12, X_train_tree_3, X_train_linear_12, X_train_linear_3,
        X_test_tree_12,  X_test_tree_3,  X_test_linear_12,  X_test_linear_3,
        y_train, y_test, feature_cols_12, feature_cols_3
    """
    boundaries = rules["analysis_a"]["boundaries"]
    risk_attribute_values = rules["subtrack_q"]["risk_attribute_values"]

    df_train = apply_segment_label(df_train_raw, boundaries)
    df_test = apply_segment_label(df_test_raw, boundaries)
    df_train = apply_risk_count(df_train, risk_attribute_values)
    df_test = apply_risk_count(df_test, risk_attribute_values)

    df_train_enc, df_test_enc = encode_features(df_train, df_test)
    df_train_scaled, df_test_scaled = scale_numeric_for_linear(df_train_enc, df_test_enc)

    feature_cols_12, feature_cols_3 = build_feature_cols(df_train_enc)

    y_train = (df_train["Churn"] == "Yes").astype(int)
    y_test = (df_test["Churn"] == "Yes").astype(int)

    return {
        "X_train_tree_12": df_train_enc[feature_cols_12],
        "X_train_tree_3": df_train_enc[feature_cols_3],
        "X_test_tree_12": df_test_enc[feature_cols_12],
        "X_test_tree_3": df_test_enc[feature_cols_3],
        "X_train_linear_12": df_train_scaled[feature_cols_12],
        "X_train_linear_3": df_train_scaled[feature_cols_3],
        "X_test_linear_12": df_test_scaled[feature_cols_12],
        "X_test_linear_3": df_test_scaled[feature_cols_3],
        "y_train": y_train,
        "y_test": y_test,
        "feature_cols_12": feature_cols_12,
        "feature_cols_3": feature_cols_3,
        "df_train_raw": df_train,
        "df_test_raw": df_test,
    }
