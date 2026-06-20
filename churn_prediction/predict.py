"""
churn_prediction/predict.py

저장된 model.pkl + feature_transformer.pkl 을 불러와 새 고객 데이터에
이탈확률만 매긴다. 학습은 전혀 일어나지 않으므로 매우 빠르다.

⚠️ 이 스크립트에는 "주기"가 없다 - train.py(재학습)는 segment_discovery/app.py
(분석A/B/Q)와 같은 묶음으로, 충분한 표본이 쌓여야 의미가 있어 가끔 함께
실행하지만, 추론은 그런 제약이 없으므로 필요할 때마다(매일, 매주, 즉석
조회 등) 독립적으로 자유롭게 실행할 수 있다. "최신 데이터 반영"이라는
역할은 재학습이 아니라 바로 이 추론 단계가 담당한다.
호출하는 쪽(사람, 스케줄러, 추후 webapp)이 언제 부르든 그 순간의
입력 데이터로 즉시 답을 준다.

입력 데이터는 원본과 동일한 컬럼 구조(customerID, tenure, Contract 등)를
가진 CSV면 된다 - 신규 가입자든, 기존 고객의 갱신된 정보든 상관없다.

사용법:
    cd churn_prediction
    python predict.py --data ../data/new_customers.csv \
                       --model outputs/model.pkl \
                       --transformer outputs/feature_transformer.pkl \
                       --output outputs/latest_predictions.csv
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from feature_engineering import FeatureTransformer  # noqa: E402

DEFAULT_MODEL_PATH = Path(__file__).parent / "outputs" / "model.pkl"
DEFAULT_TRANSFORMER_PATH = Path(__file__).parent / "outputs" / "feature_transformer.pkl"
DEFAULT_OUTPUT_PATH = Path(__file__).parent / "outputs" / "latest_predictions.csv"


def predict(
    df_raw: pd.DataFrame,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    transformer_path: str | Path = DEFAULT_TRANSFORMER_PATH,
) -> pd.DataFrame:
    """
    새 고객 데이터(df_raw, 원본 컬럼 그대로)에 이탈확률을 매겨 반환.
    학습이 전혀 없으므로 호출할 때마다 즉시 끝난다 - 언제 불러도 상관없다.
    """
    model = joblib.load(model_path)
    transformer = FeatureTransformer.load(transformer_path)

    # 메인 모델(3단계)은 feature_cols_3(segment_* 포함) 기준으로 학습되었으므로
    # 추론도 반드시 같은 피처 집합을 사용해야 한다.
    transformed = transformer.transform(df_raw)
    X = transformed["X_tree_3"]

    result = transformed["df_with_labels"].copy()
    result["이탈확률"] = model.predict_proba(X)[:, 1]
    return result


def main(csv_path: str, model_path: str, transformer_path: str, output_path: str):
    print(f"[1/2] 추론 대상 데이터 로드: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    print(f"      {len(df_raw)}건")

    print("[2/2] 저장된 모델/변환규칙으로 즉시 추론 (재학습 없음)")
    result = predict(df_raw, model_path, transformer_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"      이탈확률 평균: {result['이탈확률'].mean():.4f}")
    print(f"      저장 완료: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="저장된 모델로 즉시 추론 (학습 없음)")
    parser.add_argument("--data", required=True, help="추론 대상 고객 CSV 경로")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="model.pkl 경로")
    parser.add_argument(
        "--transformer", default=str(DEFAULT_TRANSFORMER_PATH),
        help="feature_transformer.pkl 경로",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="결과 저장 경로")
    args = parser.parse_args()
    main(args.data, args.model, args.transformer, args.output)
