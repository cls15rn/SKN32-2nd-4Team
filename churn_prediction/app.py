"""
churn_prediction/app.py

segment_discovery 가 만든 segment_rules.json 을 읽어 1·2·3단계 예측모델을
학습/평가하고, 예측결과를 df에 재결합해 저장한다.

이 패키지는 segment_discovery 의 내부 코드를 알 필요가 없다 - 오직
segment_rules.json 파일(인터페이스)만 의존한다.

실행 주기: 월 단위 - segment_discovery(1년 단위)보다 훨씬 자주 재실행됨.

사용법:
    cd churn_prediction
    python app.py --data ../data/WA_FnUseC_TelcoCustomerChurn.csv \
                   --rules ../segment_discovery/outputs/segment_rules.json
"""
import argparse
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "segment_discovery" / "src"))

from evaluate import (  # noqa: E402
    check_segment_label_incremental_contribution,
    compare_stage_metrics,
    estimate_fn_cost,
    find_recall_drop_threshold,
)
from feature_engineering import load_segment_rules, prepare_model_inputs  # noqa: E402
from preprocessing import run_preprocessing  # noqa: E402
from train_models import train_all_stages  # noqa: E402

OUTPUT_DIR = Path(__file__).parent / "outputs"


def main(csv_path: str, rules_path: str):
    print("[1/5] 전처리 (segment_discovery 와 동일 로직, Train/Test 재분할)")
    df_train, df_test = run_preprocessing(csv_path)

    print("[2/5] segment_rules.json 로드 및 피처 구성 (feature_cols_12 / _3 분기)")
    rules = load_segment_rules(rules_path)
    inputs = prepare_model_inputs(df_train, df_test, rules)
    print(f"      feature_cols_12: {len(inputs['feature_cols_12'])}개 "
          f"(segment_* 제외)")
    print(f"      feature_cols_3 : {len(inputs['feature_cols_3'])}개 "
          f"(segment_* 포함)")

    print("[3/5] 1·2·3단계 + 보조 MLP 학습")
    stage_results = train_all_stages(inputs)
    for key, result in stage_results.items():
        print(f"      {result.name} 학습 완료 (피처 {len(result.feature_cols)}개)")

    print("[4/5] 평가지표 비교")
    metrics_df = compare_stage_metrics(stage_results, inputs)
    print(metrics_df.to_string())

    contribution = check_segment_label_incremental_contribution(
        metrics_df.loc["2단계_XGBoost"].to_dict(),
        metrics_df.loc["3단계_XGBoost_세그먼트포함"].to_dict(),
    )
    print(f"\n      세그먼트 라벨 증분기여: F1 {contribution['f1_increment']:+.4f}, "
          f"AUC {contribution['auc_increment']:+.4f}")
    print(f"      ※ {contribution['note']}")

    threshold = find_recall_drop_threshold(
        inputs["y_test"],
        stage_results["stage3"].model.predict_proba(inputs["X_test_tree_3"])[:, 1],
    )
    print(f"\n      권장 분류 임계값(Recall 급락 직전): {threshold:.4f}")

    fn_cost = estimate_fn_cost(inputs["df_train_raw"])
    print(f"      FN 비용 근사 추정: {fn_cost:.0f} (참고용, FP비용은 추정 불가)")

    print("\n[5/5] 결과 저장")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 메인 모델(3단계) 저장
    joblib.dump(stage_results["stage3"].model, OUTPUT_DIR / "model.pkl")

    # 예측결과를 df에 재결합 (기획_메모.md 7장 9번, 손해추정용)
    df_test_result = inputs["df_test_raw"].copy()
    df_test_result["이탈확률"] = stage_results["stage3"].model.predict_proba(
        inputs["X_test_tree_3"]
    )[:, 1]
    df_test_result.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    metrics_df.to_csv(OUTPUT_DIR / "stage_metrics.csv")

    print(f"      model.pkl, predictions.csv, stage_metrics.csv -> {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="예측모델 1·2·3단계 학습/평가")
    parser.add_argument("--data", required=True, help="원본 CSV 경로")
    parser.add_argument(
        "--rules", required=True,
        help="segment_discovery/outputs/segment_rules.json 경로",
    )
    args = parser.parse_args()
    main(args.data, args.rules)
