"""
churn_prediction/train.py  (구 app.py)

segment_discovery 가 만든 segment_rules.json 을 읽어 1·2·3단계 예측모델을
학습/평가하고, 모델과 FeatureTransformer를 저장한다.

⚠️ 이 스크립트는 "재학습"용이다 - 새 고객 데이터(신규 가입/이탈 확정)가
누적되었을 때만 실행한다. 단순히 기존 고객의 최신 위험도 점수만 보고 싶다면
predict.py(추론 전용, 학습 없이 즉시 실행)를 쓸 것 - 재학습보다 훨씬 빠르다.

이 패키지는 segment_discovery 의 내부 코드를 알 필요가 없다 - 오직
segment_rules.json 파일(인터페이스)만 의존한다.

실행 주기: segment_discovery/app.py 와 같은 묶음(재학습 묶음) - 표본이 충분히
쌓였을 때(또는 패턴 변화가 의심될 때) 둘을 함께 실행하는 게 일관적이다. 표본
부족으로 일부 범주가 통째로 빠지는 위험이 있으므로, "신규 데이터만"으로
재학습하지 말고 항상 누적 전체 데이터로 학습할 것. (predict.py 는 이 묶음과
무관하게 독립적으로 자유로운 주기 - 학습이 없어 데이터 양과 무관하게 안정적)

사용법:
    cd churn_prediction
    python train.py --data ../data/WA_FnUseC_TelcoCustomerChurn.csv \
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
    print("[1/6] 전처리 (Train/Test 분할)")
    df_train, df_test = run_preprocessing(csv_path)

    print("[2/6] segment_rules.json 로드 및 피처 변환규칙 fit (feature_cols_12 / _3 분기)")
    rules = load_segment_rules(rules_path)
    inputs, transformer = prepare_model_inputs(df_train, df_test, rules)
    print(f"      feature_cols_12: {len(inputs['feature_cols_12'])}개 (segment_* 제외)")
    print(f"      feature_cols_3 : {len(inputs['feature_cols_3'])}개 (segment_* 포함)")

    print("[3/6] 1·2·3단계 + 보조 MLP 학습")
    stage_results = train_all_stages(inputs)
    for key, result in stage_results.items():
        print(f"      {result.name} 학습 완료 (피처 {len(result.feature_cols)}개)")

    print("[4/6] 평가지표 비교")
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

    print("\n[5/6] 모델 + 변환규칙(FeatureTransformer) 저장")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 메인 모델(3단계) 저장
    joblib.dump(stage_results["stage3"].model, OUTPUT_DIR / "model.pkl")

    # ⚠️ predict.py 가 재사용할 변환규칙(더미컬럼 목록 + 스케일러)을 함께 저장.
    # 이게 없으면 추론 시 새 데이터를 학습 때와 다른 방식으로 인코딩하게 되어
    # 모델이 기대하는 피처 형식과 어긋날 수 있다.
    transformer.save(OUTPUT_DIR / "feature_transformer.pkl")

    print("\n[6/6] 결과 저장")
    df_test_result = inputs["df_test_raw"].copy()
    df_test_result["이탈확률"] = stage_results["stage3"].model.predict_proba(
        inputs["X_test_tree_3"]
    )[:, 1]
    df_test_result.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    metrics_df.to_csv(OUTPUT_DIR / "stage_metrics.csv")

    print(f"      model.pkl, feature_transformer.pkl, predictions.csv, "
          f"stage_metrics.csv -> {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="예측모델 1·2·3단계 재학습")
    parser.add_argument("--data", required=True, help="원본 CSV 경로 (재학습용 데이터)")
    parser.add_argument(
        "--rules", required=True,
        help="segment_discovery/outputs/segment_rules.json 경로",
    )
    args = parser.parse_args()
    main(args.data, args.rules)
