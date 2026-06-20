"""
segment_rules.json (segment_discovery 의 산출물) 을 읽어서, 예측모델 학습에
바로 쓸 수 있는 X, y 를 구성한다.

⚠️ 단계별 X 구성이 다름 - 인코딩만 하고 끝내지 말 것 (기획_메모.md 7장 8단계 참조)
feature_cols_12 (1·2단계용, segment_* 컬럼 명시적 제외)
feature_cols_3  (3단계용, segment_* 포함)
하나의 X를 그대로 1·2·3단계에 다 쓰면 "1단계는 세그먼트 라벨을 절대
포함하지 않는다" 는 원칙이 코드에서 깨진다.

추론(predict.py)이 학습(train.py)과 똑같은 방식으로 새 데이터를 변환할 수
있도록, "Train 기준으로 fit된 규칙"(더미컬럼 목록 + 스케일러)을
FeatureTransformer 객체 하나로 묶어 저장/로드한다.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
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
    """확정된 경계로 segment 컬럼 부여. 학습/추론 양쪽에서 동일하게 호출됨."""
    df = df.copy()
    upper = max(df["tenure"].max(), (boundaries[-1] if boundaries else 0)) + 1
    bins = [-1] + list(boundaries) + [upper]
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


@dataclass
class FeatureTransformer:
    """
    Train 시점에 fit된 "변환 규칙"을 통째로 담는 객체.

    - dummy_columns         : pd.get_dummies 후 df_train 기준으로 확정된 전체 컬럼 목록
    - scaler                : NUMERIC_COLS 에 대해 Train으로 fit된 StandardScaler
    - feature_cols_12 / _3  : 1·2단계용 / 3단계용 피처 목록 (Train 시점에 확정)
    - boundaries            : 분석A가 찾은 세그먼트 경계 (segment_rules.json 에서 옴)
    - risk_attribute_values : 서브트랙Q의 위험속성 정의

    train.py 가 이 객체를 만들어 outputs/feature_transformer.pkl 로 저장하면,
    predict.py 는 그걸 그대로 불러와 "학습 때와 똑같은 규칙"으로 새 데이터를 변환한다.
    재학습 없이 추론만 할 때는 이 객체 안의 규칙이 절대 바뀌지 않는다.
    """
    dummy_columns: list[str]
    scaler: StandardScaler
    feature_cols_12: list[str]
    feature_cols_3: list[str]
    boundaries: list[float]
    risk_attribute_values: dict[str, str] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "FeatureTransformer":
        return joblib.load(path)

    def transform(self, df_raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        새 고객 데이터(df_raw, 원본 컬럼 그대로)를 학습 때와 동일한 규칙으로 변환.
        추론(predict.py)에서 쓰는 진입점 - 절대 새로 fit 하지 않고 저장된 규칙만 적용.

        Returns
        -------
        dict with keys: X_tree_12, X_tree_3, X_linear_12, X_linear_3, df_with_labels
        """
        df = apply_segment_label(df_raw, self.boundaries)
        df = apply_risk_count(df, self.risk_attribute_values)

        for col in BINARY_MAP_COLS:
            df[col] = df[col].map({"Yes": 1, "No": 0})

        multi_categorical = [c for c in CATEGORICAL_COLS if c not in BINARY_MAP_COLS]
        df_enc = pd.get_dummies(df, columns=multi_categorical + ["segment"])
        # 학습 시점의 더미컬럼 목록으로 강제 정렬
        # (새 데이터에 없던 범주 -> 0, 학습때만 있던 컬럼 -> 0으로 채움)
        df_enc = df_enc.reindex(columns=self.dummy_columns, fill_value=0)

        df_scaled = df_enc.copy()
        df_scaled[NUMERIC_COLS] = self.scaler.transform(df_enc[NUMERIC_COLS])

        return {
            "X_tree_12": df_enc[self.feature_cols_12],
            "X_tree_3": df_enc[self.feature_cols_3],
            "X_linear_12": df_scaled[self.feature_cols_12],
            "X_linear_3": df_scaled[self.feature_cols_3],
            "df_with_labels": df,
        }


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


def fit_feature_transformer(
    df_train_raw: pd.DataFrame, rules: dict,
) -> tuple[FeatureTransformer, pd.DataFrame]:
    """
    학습(train.py) 전용 - df_train 으로 더미컬럼 목록과 스케일러를 새로 fit.
    추론(predict.py)에서는 절대 호출하지 않음 (FeatureTransformer.load() 만 사용).

    Returns
    -------
    transformer : 학습 직후 저장해야 할 FeatureTransformer
    df_train_with_labels : segment/risk_count가 부여된 df_train (참고/저장용)
    """
    boundaries = rules["analysis_a"]["boundaries"]
    risk_attribute_values = rules["subtrack_q"]["risk_attribute_values"]

    df_train = apply_segment_label(df_train_raw, boundaries)
    df_train = apply_risk_count(df_train, risk_attribute_values)

    df_train_for_fit = df_train.copy()
    for col in BINARY_MAP_COLS:
        df_train_for_fit[col] = df_train_for_fit[col].map({"Yes": 1, "No": 0})

    multi_categorical = [c for c in CATEGORICAL_COLS if c not in BINARY_MAP_COLS]
    df_train_enc = pd.get_dummies(df_train_for_fit, columns=multi_categorical + ["segment"])

    scaler = StandardScaler()
    scaler.fit(df_train_enc[NUMERIC_COLS])

    feature_cols_12, feature_cols_3 = build_feature_cols(df_train_enc)

    transformer = FeatureTransformer(
        dummy_columns=list(df_train_enc.columns),
        scaler=scaler,
        feature_cols_12=feature_cols_12,
        feature_cols_3=feature_cols_3,
        boundaries=boundaries,
        risk_attribute_values=risk_attribute_values,
    )
    return transformer, df_train


def prepare_model_inputs(
    df_train_raw: pd.DataFrame, df_test_raw: pd.DataFrame, rules: dict,
):
    """
    학습(train.py) 전용 전체 파이프라인: segment_rules.json 적용 -> fit ->
    Train/Test 양쪽 인코딩/스케일링 -> X/y 구성.

    Returns
    -------
    inputs : 모델 학습/평가에 바로 쓸 dict (X_train_*, X_test_*, y_train, y_test, ...)
    transformer : outputs/feature_transformer.pkl 로 저장해야 할 FeatureTransformer
    """
    transformer, df_train = fit_feature_transformer(df_train_raw, rules)

    train_transformed = transformer.transform(df_train_raw)
    test_transformed = transformer.transform(df_test_raw)

    y_train = (df_train["Churn"] == "Yes").astype(int)
    y_test = (test_transformed["df_with_labels"]["Churn"] == "Yes").astype(int)

    inputs = {
        "X_train_tree_12": train_transformed["X_tree_12"],
        "X_train_tree_3": train_transformed["X_tree_3"],
        "X_test_tree_12": test_transformed["X_tree_12"],
        "X_test_tree_3": test_transformed["X_tree_3"],
        "X_train_linear_12": train_transformed["X_linear_12"],
        "X_train_linear_3": train_transformed["X_linear_3"],
        "X_test_linear_12": test_transformed["X_linear_12"],
        "X_test_linear_3": test_transformed["X_linear_3"],
        "y_train": y_train,
        "y_test": y_test,
        "feature_cols_12": transformer.feature_cols_12,
        "feature_cols_3": transformer.feature_cols_3,
        "df_train_raw": train_transformed["df_with_labels"],
        "df_test_raw": test_transformed["df_with_labels"],
    }
    return inputs, transformer
