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

⚠️ [수정] XGBoost의 max_depth/n_estimators를 예전에는 사람이 직감으로
고정값(200, 5)을 박아넣었는데, 이건 분석A에서 "가지치기 강도를 직감으로
안 정하고 교차검증으로 데이터가 직접 결정한다"는 원칙과 일관되지 않았다.
실제로 5-fold 교차검증 비교 결과 max_depth=3이 depth=5보다 AUC가 더 높고
더 안정적이었다(0.8433 vs 0.8347) - 고정값이 우리 데이터 규모(7,043건)에
과적합 방향으로 맞지 않았다는 신호.

그래서 GridSearchCV로 2단계에서 한 번만 탐색하고, 그 결과를 3단계가 그대로
물려받는다 - "2·3단계는 같은 설정을 써야 한다"는 원칙은 유지하면서, 그
설정 자체는 데이터가 정하게 한다.

⚠️ [재검토 후 정정] 학습 시점 불균형 보정(class_weight/scale_pos_weight)과
평가 시점 임계값 보정(find_recall_drop_threshold)은 둘 다 "Recall을
끌어올린다"는 같은 방향으로 작동하는 장치다. 실측 비교 결과:

  scale_pos_weight 없음 + 권장임계값: Recall=0.734, F2=0.689
  scale_pos_weight 있음 + 권장임계값: Recall=0.726, F2=0.685

거의 동일하다 - 권장 임계값 선택 하나만으로 이미 충분히 Recall을 보정하고
있어서, scale_pos_weight를 추가하면 같은 효과를 중복으로 적용하는 셈이고
"이 Recall 개선이 학습 보정 때문인지 임계값 때문인지" 해석도 불분명해진다.
(둘 다 켜고 0.5 고정임계값을 쓰면 Recall=0.806까지 과하게 치우침)

그래서 모든 단계(1·2·3·MLP)의 학습은 중립적으로 두고, "FN이 FP보다
비싸다"는 비즈니스 비용 비대칭(기획_메모.md 2.3)은 오직 평가/운영
단계의 임계값 선택(find_recall_drop_threshold)에서만 반영한다 - 1단계의
class_weight="balanced"도 같은 이유로 제거했다(전통통계 비교군이라는
역할은 "세그먼트 라벨 미포함"에서 나오는 것이지 불균형 처리 여부와는
무관하므로, 모든 단계를 동일한 중립 학습 원칙으로 통일).
"""
from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config  # noqa: E402


@dataclass
class StageResult:
    name: str
    model: object
    feature_cols: list[str]


@dataclass
class XGBoostSearchResult:
    best_params: dict
    cv_results: pd.DataFrame


def search_xgboost_hyperparameters(
    X_train_tree_12: pd.DataFrame, y_train: pd.Series,
) -> XGBoostSearchResult:
    """
    2단계 입력(feature_cols_12)으로 max_depth/n_estimators/learning_rate을
    교차검증(AUC 최대화)으로 탐색 - 분석A의 ccp_alpha 탐색과 같은 원리.

    ⚠️ learning_rate도 탐색 대상에 포함됨(이전에는 0.1로 고정) - max_depth/
    n_estimators만 데이터가 정하고 learning_rate은 사람이 고정하는 비일관성을
    해소. 세 파라미터 모두 같은 GridSearchCV 한 번으로 함께 탐색한다(개별
    탐색을 순차적으로 하면 파라미터 간 상호작용을 놓칠 수 있음).

    이 결과를 3단계도 그대로 물려받는다(_xgboost_kwargs 참조) - 탐색은
    한 번만, 두 단계가 같은 설정을 쓴다는 원칙은 유지.
    """
    param_grid = {
        "max_depth": config.XGBOOST_MAX_DEPTH_CANDIDATES,
        "n_estimators": config.XGBOOST_N_ESTIMATORS_CANDIDATES,
        "learning_rate": config.XGBOOST_LEARNING_RATE_CANDIDATES,
    }
    base_model = XGBClassifier(
        eval_metric="logloss",
        random_state=config.RANDOM_STATE,
    )
    cv = StratifiedKFold(
        n_splits=config.XGBOOST_SEARCH_CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )
    grid = GridSearchCV(base_model, param_grid, cv=cv, scoring="roc_auc", n_jobs=1)
    grid.fit(X_train_tree_12, y_train)

    cv_results = pd.DataFrame(grid.cv_results_)[
        ["param_max_depth", "param_n_estimators", "param_learning_rate", "mean_test_score", "std_test_score"]
    ].sort_values("mean_test_score", ascending=False)

    return XGBoostSearchResult(best_params=grid.best_params_, cv_results=cv_results)


def _xgboost_kwargs(search_result: XGBoostSearchResult) -> dict:
    """
    ⚠️ 2단계와 3단계가 반드시 같은 하이퍼파라미터를 써야 한다(성능차이가
    "접근법" 때문인지 "알고리즘 설정" 때문인지 혼동 방지). search_result는
    2단계에서 탐색한 결과를 3단계가 그대로 물려받기 위한 매개체 - 숫자를
    각자 따로 적어두면 한쪽만 고치고 다른 쪽을 놓치는 위험이 있으므로,
    이 함수 하나로 통일해서 양쪽이 가져다 쓴다.
    """
    return dict(
        max_depth=search_result.best_params["max_depth"],
        n_estimators=search_result.best_params["n_estimators"],
        learning_rate=search_result.best_params["learning_rate"],
        eval_metric="logloss",
        random_state=config.RANDOM_STATE,
    )


def train_stage1_logistic(X_train_linear_12: pd.DataFrame, y_train: pd.Series) -> StageResult:
    """1단계: 로지스틱 회귀 — 전통통계 비교군, 세그먼트 라벨 절대 미포함. 학습은 중립적."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_linear_12, y_train)
    return StageResult("1단계_로지스틱회귀", model, list(X_train_linear_12.columns))


def train_stage2_xgboost(
    X_train_tree_12: pd.DataFrame, y_train: pd.Series, search_result: XGBoostSearchResult,
) -> StageResult:
    """2단계: XGBoost — 현재AI 비교군, 세그먼트 라벨 미포함. 학습은 중립적."""
    model = XGBClassifier(**_xgboost_kwargs(search_result))
    model.fit(X_train_tree_12, y_train)
    return StageResult("2단계_XGBoost", model, list(X_train_tree_12.columns))


def train_stage3_xgboost(
    X_train_tree_3: pd.DataFrame, y_train: pd.Series, search_result: XGBoostSearchResult,
) -> StageResult:
    """3단계: XGBoost — 메인, 세그먼트 라벨 포함 (2단계와 동일 알고리즘+동일 설정). 학습은 중립적."""
    model = XGBClassifier(**_xgboost_kwargs(search_result))
    model.fit(X_train_tree_3, y_train)
    return StageResult("3단계_XGBoost_세그먼트포함", model, list(X_train_tree_3.columns))


def train_auxiliary_mlp(X_train_linear_12: pd.DataFrame, y_train: pd.Series) -> StageResult:
    """
    보조: 간단한 MLP — 딥러닝 비교용, 2단계와 동일 입력(스케일링됨). 학습은 중립적.

    ⚠️ MLPClassifier는 sklearn에서 class_weight 매개변수 자체를 지원하지
    않는다(다른 모델과 다른 점). 다만 이번 결정(모든 단계 학습을 중립적으로
    둔다)에 따라 불균형 보정을 sample_weight로 추가할 필요도 없어졌다.
    """
    model = MLPClassifier(
        hidden_layer_sizes=config.MLP_HIDDEN_LAYER_SIZES,
        max_iter=config.MLP_MAX_ITER,
        random_state=config.RANDOM_STATE,
    )
    model.fit(X_train_linear_12, y_train)
    return StageResult("보조_MLP", model, list(X_train_linear_12.columns))


def train_all_stages(inputs: dict) -> tuple[dict[str, StageResult], XGBoostSearchResult]:
    """
    feature_engineering.prepare_model_inputs() 의 출력을 받아 4개 모델 전부 학습.

    Returns
    -------
    stage_results : dict[str, StageResult]
    search_result : XGBoostSearchResult (2단계 탐색 결과 - 로그/메타데이터 기록용)
    """
    search_result = search_xgboost_hyperparameters(
        inputs["X_train_tree_12"], inputs["y_train"]
    )
    stage_results = {
        "stage1": train_stage1_logistic(inputs["X_train_linear_12"], inputs["y_train"]),
        "stage2": train_stage2_xgboost(
            inputs["X_train_tree_12"], inputs["y_train"], search_result
        ),
        "stage3": train_stage3_xgboost(
            inputs["X_train_tree_3"], inputs["y_train"], search_result
        ),
        "auxiliary_mlp": train_auxiliary_mlp(inputs["X_train_linear_12"], inputs["y_train"]),
    }
    return stage_results, search_result
