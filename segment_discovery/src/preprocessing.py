"""
분석A/B/서브트랙Q 에서 공통으로 쓰는 전처리.
원본 CSV 로드 -> 자료형 정리 -> 결측치 처리 -> 중복정보 통합 -> Train/Test 분할.

세그먼트 경계/위험속성 '탐색'에 필요한 전처리만 다룸.
(인코딩/스케일링은 segment_discovery 가 산출한 규칙을 churn_prediction 이
 가져다 쓸 때 거기서 다시 수행 - shared/data_loader.py 참조)
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw(csv_path: str) -> pd.DataFrame:
    """원본 IBM Telco CSV 로드"""
    df = pd.read_csv(csv_path)
    return df


def clean_types_and_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    - TotalCharges: 문자열 -> 숫자, 결측(11건, 전부 tenure=0) -> 0
    - ChurnFlag: Yes/No -> 1/0
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)
    return df


def consolidate_no_service_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    'No internet service' / 'No phone service' -> 'No' 로 통합.
    이유: 정보손실 방지가 아니라 InternetService_No / PhoneService_No 와
    100% 중복되어 원-핫 인코딩 시 다중공선성을 유발하기 때문 (기획_메모.md 2.11 참조)
    """
    df = df.copy()
    no_internet_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for col in no_internet_cols:
        df[col] = df[col].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
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
    df = clean_types_and_missing(df)
    df = consolidate_no_service_categories(df)
    df_train, df_test = split_train_test(df, test_size=test_size, random_state=random_state)
    return df_train, df_test
