"""
평가지표 및 SHAP 해석 (기획_메모.md 2.3 / 6장 참조)

⚠️ 용어 구분 - "AUC"가 두 군데서 다른 대상을 가리킴:
여기서 계산하는 ROC-AUC는 최종 예측모델(20개+ 변수 전체)의 분류 성능.
분석A의 "세그먼트단독 AUC"와는 완전히 다른 모델·다른 목적의 수치다.
코드 변수명도 model_auc / segment_only_auc 로 구분해서 절대 혼용하지 말 것.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, auc, f1_score, fbeta_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)


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


def find_recall_drop_threshold(y_true: pd.Series, y_proba: np.ndarray) -> float:
    """
    분류 임계값: 0.5 고정 대신, PR곡선에서 "Recall이 급격히 꺾이기 직전 지점" 채택.
    (기획_메모.md 2.3 참조)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # recall 변화율(미분)이 가장 급격해지는 지점을 찾음 (recall 내림차순 정렬되어 있음)
    recall_diffs = np.diff(recalls)
    if len(recall_diffs) == 0:
        return 0.5
    sharpest_drop_idx = int(np.argmin(recall_diffs))  # 가장 큰 음의 변화
    sharpest_drop_idx = min(sharpest_drop_idx, len(thresholds) - 1)
    return float(thresholds[sharpest_drop_idx])


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
    """1·2·3단계 + 보조 MLP 의 평가지표를 한 표로 비교"""
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
        metrics = compute_all_metrics(inputs["y_test"], y_proba)
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
