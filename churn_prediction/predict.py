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
전체 데이터의 일부 범위(예: tenure 0~5개월만, 또는 특정 segment만)를 미리
필터링해서 넣어도 문제없이 동작한다 - 한 행씩 독립적으로 변환/추론하므로
입력 범위와 무관하게 항상 안전하다.

⚠️ "기간(월) 단위로 범위를 나눠 본다"는 게 의미를 가지려면, 날짜/수집시점
컬럼이 아니라 tenure(가입 후 경과월)를 기준으로 나눠야 한다. 이 데이터는
절대 가입일/수집일이 없는 스냅샷 데이터라(기획_메모.md 2.2 참조), "2026년
5월에 가입한 고객" 같은 절대 시점 필터링은 할 수 없고, "tenure 0~5개월인
고객들"처럼 경과월 구간으로만 범위를 나눌 수 있다. 별도의 날짜 컬럼을
추가할 필요는 없다 - 이미 있는 tenure 컬럼으로 충분하다.

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


DEFAULT_NEW_DATA_PATH = Path(__file__).parent.parent / "data" / "new_customers.csv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="저장된 모델로 즉시 추론 (학습 없음)")
    parser.add_argument(
        "--data", default=str(DEFAULT_NEW_DATA_PATH),
        help=f"추론 대상 고객 CSV 경로 (기본값: {DEFAULT_NEW_DATA_PATH})",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="model.pkl 경로")
    parser.add_argument(
        "--transformer", default=str(DEFAULT_TRANSFORMER_PATH),
        help="feature_transformer.pkl 경로",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="결과 저장 경로")
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(
            f"[안내] 추론 대상 파일이 없습니다: {args.data}\n"
            f"       새 고객 데이터를 이 경로에 두거나, --data 로 다른 경로를 지정하세요.\n"
            f"       (VSCode 실행버튼으로 바로 돌려보려면, data/new_customers.csv 를 만들어두세요)"
        )
        raise SystemExit(1)
    print(f"[안내] 추론 대상: {args.data}\n")
    main(args.data, args.model, args.transformer, args.output)
