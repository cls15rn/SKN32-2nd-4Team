"""
분석A/B/서브트랙Q 에서 공통으로 쓰는 전처리.
원본 CSV 로드 -> shared.clean_raw_data(자료형 정리+No-service 통합) -> Train/Test 분할.

세그먼트 경계/위험속성 '탐색'에 필요한 전처리만 다룸.
(인코딩/스케일링은 segment_discovery 가 산출한 규칙을 churn_prediction 이
 가져다 쓸 때 거기서 다시 수행)

⚠️ shared.clean_raw_data 는 churn_prediction(train.py, predict.py)도 동일하게
거치는 "공통 전처리 관문" 이다. 여기서 자료형 정리를 따로 하지 않는 이유는,
같은 로직이 세 곳에 중복되면 한쪽만 고치고 다른 쪽을 놓치는 버그(실제로
predict.py 에서 한 번 발생)가 재발할 수 있기 때문.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
from data_loader import clean_raw_data  # noqa: E402


def load_raw(csv_path: str) -> pd.DataFrame:
    """원본 IBM Telco CSV 로드"""
    df = pd.read_csv(csv_path)
    return df


def add_churn_flag(df: pd.DataFrame) -> pd.DataFrame:
    """ChurnFlag: Yes/No -> 1/0 (분석A/B/Q 전용 - 예측모델 쪽은 y를 별도로 구성)"""
    df = df.copy()
    df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)
    return df


def split_train_test(df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42):
    """
    계층화 분할 (stratify=Churn).
    세그먼트 경계/위험속성 '탐색'은 df_train 만 사용 - 누수 방지.
    """
    df_train, df_test = train_test_split(
        df, test_size=test_size, stratify=df["Churn"], random_state=random_state
    )
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


def run_preprocessing(csv_path: str, test_size: float = 0.3, random_state: int = 42):
    """전체 전처리 파이프라인 (탐색용)"""
    df = load_raw(csv_path)
    df = clean_raw_data(df)  # 자료형 정리 + No-service 통합 (shared, 공통 관문)
    df = add_churn_flag(df)
    df_train, df_test = split_train_test(df, test_size=test_size, random_state=random_state)
    return df_train, df_test

