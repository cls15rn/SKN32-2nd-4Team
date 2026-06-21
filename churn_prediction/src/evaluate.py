"""
평가지표 및 SHAP 해석 (기획_메모.md 2.3 / 6장 참조)

⚠️ 용어 구분 - "AUC"가 두 군데서 다른 대상을 가리킴:
여기서 계산하는 ROC-AUC는 최종 예측모델(20개+ 변수 전체)의 분류 성능.
분석A의 "세그먼트단독 AUC"와는 완전히 다른 모델·다른 목적의 수치다.
코드 변수명도 model_auc / segment_only_auc 로 구분해서 절대 혼용하지 말 것.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, auc, f1_score, fbeta_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config  # noqa: E402


def compute_all_metrics(y_true: pd.Series, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """
    보고서용: Accuracy/Precision/Recall/F1/F2/ROC-AUC/PR-AUC 전부 계산.
    Accuracy는 불균형 데이터에서 참고용으로만 사용할 것.
    """
    y_pred = (y_proba >= threshold).astype(int)
    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
    # recall이 감소하는 순서로 정렬되어 있어야 auc()가 올바르게 계산됨
    pr_auc = float(auc(recalls, precisions))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "model_auc": float(roc_auc_score(y_true, y_proba)),  # "세그먼트단독 AUC"와 다른 수치
        "pr_auc": pr_auc,
        "threshold": threshold,
    }


def find_recall_drop_threshold(
    y_true: pd.Series, y_proba: np.ndarray,
    min_recall: float = config.THRESHOLD_SEARCH_MIN_RECALL,
    window: int = config.THRESHOLD_SEARCH_WINDOW,
) -> float:
    """
    분류 임계값: 0.5 고정 대신, PR곡선에서 "Recall이 급격히 꺾이기 직전 지점" 채택.
    (기획_메모.md 2.3 참조)

    ⚠️ [버그 수정] 예전에는 PR곡선 전 구간(recall이 거의 0인 영역까지 포함)에서
    "바로 다음 점과의 하락폭"만 비교했는데, 두 가지 문제가 있었다.

    1) recall이 이미 매우 낮은(이탈자를 거의 다 놓친) 구간에서는 한 점 차이의
       하락폭이 절대적으로 작아서(예: -0.00713) 노이즈와 진짜 신호를 구분하기
       어렵다 - 실제로 recall=0.74->0.73 구간(threshold=0.33, 합리적인 지점)과
       recall=0.06->0.055 구간(threshold=0.83, 이미 의미 없는 영역)이 거의
       같은 하락폭으로 동률이 나서, 부동소수점 오차로 후자가 선택되는 버그가
       실제로 발생함(2단계 XGBoost에서 threshold=0.83이 나와 Recall이
       0.06까지 붕괴).
    2) 한 점(바로 다음 점)만 보는 비교는 RF 트리개수 자동탐지에서 겪었던
       것과 같은 "단발성 노이즈에 취약한" 구조다.

    수정: ① min_recall(기본 0.5) 이상인 구간으로 후보를 한정한다 - "이미
    의미있게 많은 이탈고객을 놓친" 구간은 처음부터 탐색 대상에서 뺀다.
    ② 한 점 차이 대신 윈도우(기본 20포인트) 평균 하락률을 비교해, 단일
    포인트의 미세한 흔들림이 아니라 "추세적으로 급해지는 지점"을 찾는다.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # recalls는 thresholds보다 1개 더 길고 내림차순 정렬됨(첫 원소가 1.0).
    # recall >= min_recall 인 구간으로 후보를 한정 - "이미 너무 많은 이탈자를
    # 놓친" 구간은 비교 대상에서 제외한다.
    valid_mask = recalls[:-1] >= min_recall  # thresholds와 길이를 맞추기 위해 마지막 원소 제외
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < window + 1:
        # 유효 구간이 너무 좁으면 기준을 완화하지 않고 안전하게 0.5 반환
        return 0.5

    recalls_in_range = recalls[valid_indices[0]: valid_indices[-1] + 2]
    thresholds_in_range = thresholds[valid_indices[0]: valid_indices[-1] + 1]

    # 윈도우 평균 기반 변화율: i번째 지점에서 "앞으로 window개 동안의 평균 하락 속도"
    window = min(window, len(recalls_in_range) - 1)
    if window < 1:
        return 0.5
    windowed_diffs = (recalls_in_range[window:] - recalls_in_range[:-window]) / window

    if len(windowed_diffs) == 0:
        return 0.5
    sharpest_drop_idx = int(np.argmin(windowed_diffs))
    sharpest_drop_idx = min(sharpest_drop_idx, len(thresholds_in_range) - 1)
    return float(thresholds_in_range[sharpest_drop_idx])


def estimate_fn_cost(df_with_churn_flag: pd.DataFrame) -> float:
    """
    FN 비용 근사 추정 (기획_메모.md 2.3 참조).
    비이탈고객 평균tenure를 정상 생애주기 근사치로 보고, 이탈고객 평균tenure와의
    차이에 이탈고객 평균 MonthlyCharges를 곱해 추정.
    FP비용(리텐션비용)은 데이터에 정보 자체가 없어 추정 불가 - 이 함수로 다루지 않음.
    """
    churned = df_with_churn_flag[df_with_churn_flag["ChurnFlag"] == 1]
    survived = df_with_churn_flag[df_with_churn_flag["ChurnFlag"] == 0]
    lost_months = survived["tenure"].mean() - churned["tenure"].mean()
    return float(lost_months * churned["MonthlyCharges"].mean())


def compare_stage_metrics(stage_results: dict, inputs: dict) -> pd.DataFrame:
    """
    1·2·3단계 + 보조 MLP 의 평가지표를 한 표로 비교.

    ⚠️ [수정] 예전에는 find_recall_drop_threshold로 권장 임계값을 구해놓고도
    여기서는 항상 threshold=0.5로 평가했다(권장값은 콘솔 출력/메타데이터
    저장용으로만 쓰이고 실제 Precision/Recall/F1/F2 계산에는 반영 안 됨).
    이제 각 모델마다 자신의 PR곡선에서 권장 임계값을 직접 구해 그 값으로
    평가한다 - "0.5 고정 대신 Recall이 급격히 꺾이기 직전 지점을 채택한다"
    (기획_메모.md 2.3)는 원칙이 실제 평가지표에 반영되도록.
    """
    rows = []
    stage_to_X_test = {
        "stage1": inputs["X_test_linear_12"],
        "stage2": inputs["X_test_tree_12"],
        "stage3": inputs["X_test_tree_3"],
        "auxiliary_mlp": inputs["X_test_linear_12"],
    }
    for key, result in stage_results.items():
        X_test = stage_to_X_test[key]
        y_proba = result.model.predict_proba(X_test)[:, 1]
        threshold = find_recall_drop_threshold(inputs["y_test"], y_proba)
        metrics = compute_all_metrics(inputs["y_test"], y_proba, threshold=threshold)
        metrics["stage"] = result.name
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("stage")


def check_segment_label_incremental_contribution(
    stage2_metrics: dict, stage3_metrics: dict,
) -> dict:
    """
    ⚠️ [중요] 2·3단계 분류성능 비교의 정확한 위치 (기획_메모.md 6.2 참조)
    세그먼트 라벨 추가시 F1 증분 기여가 작아도 "세그먼트가 무의미하다"는
    뜻이 아님 - 분석A의 적절성 자체는 별도로 검증됨(세그먼트단독 AUC+순열검정).
    이 비교는 "세그먼트 발견의 증거"가 아니라 예측모델 차원의 보조적
    실용성 지표일 뿐이다.
    """
    return {
        "f1_increment": stage3_metrics["f1"] - stage2_metrics["f1"],
        "auc_increment": stage3_metrics["model_auc"] - stage2_metrics["model_auc"],
        "note": (
            "이 증분이 작아도 분석A(세그먼트단독AUC+순열검정)의 별도 검증으로 "
            "서사를 유지할 것 - 메인 증거는 분석A임을 보고서에 명시"
        ),
    }
