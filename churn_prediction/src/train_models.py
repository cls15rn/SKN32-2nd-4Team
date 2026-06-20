"""
예측모델 1·2·3단계 학습 (기획_메모.md 5장/6장 참조)

⚠️ 이 "1·2·3단계"는 분석A 내부의 ①②③ 과는 다른, 예측모델 진화서사의
번호다. 혼동 주의.

1단계 | 로지스틱 회귀 | feature_cols_12 (세그먼트 라벨 절대 미포함) | 전통통계 비교군
2단계 | XGBoost      | feature_cols_12 (세그먼트 라벨 미포함)       | 현재AI 비교군
3단계 | XGBoost      | feature_cols_3  (세그먼트 라벨 포함)         | 메인
보조  | MLP          | feature_cols_12                              | 딥러닝 비교용

왜 1단계에도 세그먼트 라벨이 없는가: 세그먼트 라벨 자체가 데이터주도(분석A)로
찾은 결과물이라, "사람이 정한 단순 형태"를 대표하는 1단계 정의와 모순되므로.
왜 2·3단계가 같은 알고리즘인가: 성능차이가 "접근법"과 "알고리즘 종류" 중
무엇 때문인지 혼동되지 않도록 통제하기 위함.
"""
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config  # noqa: E402


@dataclass
class StageResult:
    name: str
    model: object
    feature_cols: list[str]


def _xgboost_kwargs() -> dict:
    """
    ⚠️ 2단계와 3단계가 반드시 같은 하이퍼파라미터를 써야 한다(성능차이가
    "접근법" 때문인지 "알고리즘 설정" 때문인지 혼동 방지). 숫자를 두 함수에
    각각 적어두면 한쪽만 고치고 다른 쪽을 놓치는 위험이 있으므로, 이 함수
    하나로 통일해서 양쪽이 가져다 쓴다.
    """
    return dict(
        n_estimators=config.XGBOOST_N_ESTIMATORS,
        max_depth=config.XGBOOST_MAX_DEPTH,
        learning_rate=config.XGBOOST_LEARNING_RATE,
        eval_metric="logloss",
        random_state=config.RANDOM_STATE,
    )


def train_stage1_logistic(X_train_linear_12: pd.DataFrame, y_train: pd.Series) -> StageResult:
    """1단계: 로지스틱 회귀 — 전통통계 비교군, 세그먼트 라벨 절대 미포함"""
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_linear_12, y_train)
    return StageResult("1단계_로지스틱회귀", model, list(X_train_linear_12.columns))


def train_stage2_xgboost(X_train_tree_12: pd.DataFrame, y_train: pd.Series) -> StageResult:
    """2단계: XGBoost — 현재AI 비교군, 세그먼트 라벨 미포함"""
    model = XGBClassifier(**_xgboost_kwargs())
    model.fit(X_train_tree_12, y_train)
    return StageResult("2단계_XGBoost", model, list(X_train_tree_12.columns))


def train_stage3_xgboost(X_train_tree_3: pd.DataFrame, y_train: pd.Series) -> StageResult:
    """3단계: XGBoost — 메인, 세그먼트 라벨 포함 (2단계와 동일 알고리즘+동일 설정)"""
    model = XGBClassifier(**_xgboost_kwargs())
    model.fit(X_train_tree_3, y_train)
    return StageResult("3단계_XGBoost_세그먼트포함", model, list(X_train_tree_3.columns))


def train_auxiliary_mlp(X_train_linear_12: pd.DataFrame, y_train: pd.Series) -> StageResult:
    """보조: 간단한 MLP — 딥러닝 비교용, 2단계와 동일 입력(스케일링됨)"""
    model = MLPClassifier(
        hidden_layer_sizes=config.MLP_HIDDEN_LAYER_SIZES,
        max_iter=config.MLP_MAX_ITER,
        random_state=config.RANDOM_STATE,
    )
    model.fit(X_train_linear_12, y_train)
    return StageResult("보조_MLP", model, list(X_train_linear_12.columns))


def train_all_stages(inputs: dict) -> dict[str, StageResult]:
    """feature_engineering.prepare_model_inputs() 의 출력을 받아 4개 모델 전부 학습"""
    return {
        "stage1": train_stage1_logistic(inputs["X_train_linear_12"], inputs["y_train"]),
        "stage2": train_stage2_xgboost(inputs["X_train_tree_12"], inputs["y_train"]),
        "stage3": train_stage3_xgboost(inputs["X_train_tree_3"], inputs["y_train"]),
        "auxiliary_mlp": train_auxiliary_mlp(inputs["X_train_linear_12"], inputs["y_train"]),
    }
