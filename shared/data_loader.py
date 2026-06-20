"""
shared/data_loader.py

segment_discovery 와 churn_prediction 이 공통으로 쓰는 원본 데이터 경로 상수
및 "Train/Test 분할 여부와 무관하게 항상 거쳐야 하는 최소 전처리".

⚠️ 분석A/B/Q(segment_discovery), 모델 재학습(train.py), 추론(predict.py)
세 곳 모두 원본 CSV를 다루는 첫 진입점에서 반드시 clean_raw_data()를
거쳐야 한다. 이전에는 predict.py만 이 관문을 거치지 않아서, TotalCharges의
문자열 결측치(' ')로 StandardScaler.transform()에서 에러가 나는 버그가
있었다 - "predict.py에서만 한 컬럼만 따로 처리"하는 식의 부분 수정 대신,
세 곳이 전부 같은 함수를 통과하도록 통일해서 근본적으로 해결함.

그 외 "Train 기준으로만 해야 하는" 전처리(Train/Test 분할, 더미컬럼 fit,
StandardScaler fit 등)는 여전히 각 패키지의 preprocessing.py /
feature_engineering.py 에 독립적으로 남겨둔다 - 두 패키지가 서로의 내부
구현을 모른 채 재실행/교체될 수 있게.
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "WA_FnUseC_TelcoCustomerChurn.csv"

NO_SERVICE_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    원본 CSV를 읽은 직후, 항상 거쳐야 하는 전처리 관문.
    분석A/B/Q, 재학습, 추론 모두 이 함수를 통과한 데이터를 사용해야 한다.

    - TotalCharges: 문자열 -> 숫자, 결측(전부 tenure=0) -> 0
    - 'No internet service' / 'No phone service' -> 'No' 로 통합
      (InternetService_No / PhoneService_No 와 100% 중복되어 원-핫 인코딩 시
       다중공선성을 유발하기 때문 - 기획_메모.md 2.11 참조)
    """
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    for col in NO_SERVICE_COLS:
        df[col] = df[col].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

    return df

