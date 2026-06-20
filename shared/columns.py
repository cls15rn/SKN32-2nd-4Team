"""
shared/columns.py

이 데이터셋의 컬럼 분류를 정의하는 단 하나의 출처(single source of truth).

⚠️ 컬럼 목록은 segment_discovery(분석A/B/Q)와 churn_prediction(재학습/추론)
양쪽에서 똑같이 필요하다. 예전에는 각 파일(analysis_b.py,
feature_engineering.py)이 각자 CATEGORICAL_COLS 등을 따로 정의하고 있었는데,
그 결과 같은 이름(NUMERIC_COLS)이 파일마다 다른 의미(스케일링 대상 vs
분석B 모델 입력)를 가리키는 위험한 상태가 됐었다. 컬럼 목록을 바꿔야 할
일이 생기면 반드시 이 파일 하나만 고칠 것 - 다른 곳에 같은 이름의 상수를
새로 만들지 말 것.

이름이 겹치는 걸 막기 위해, "어떤 목적의 컬럼 목록인지"를 변수명에 명시한다
(예: SCALING_NUMERIC_COLS 처럼) - 그냥 NUMERIC_COLS 라고만 부르면 또 같은
혼동이 재발할 수 있다.
"""

# 원본 데이터의 다중범주형 컬럼 (성별 포함, 원-핫 인코딩 대상)
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

# CATEGORICAL_COLS 중 Yes/No 이진값이라 .map({"Yes":1,"No":0})으로 처리하는 컬럼
# (churn_prediction/feature_engineering.py 전용 - 원-핫 대상에서는 제외됨)
BINARY_MAP_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]

# StandardScaler로 스케일링하는 연속형 컬럼 (churn_prediction 의 선형모델/MLP용)
SCALING_NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# 분석B(세그먼트별 위험속성 탐지)가 모델 입력으로 함께 쓰는 비-범주형 컬럼
# ⚠️ SCALING_NUMERIC_COLS 와 다른 목적의 다른 목록이다 - 혼용하지 말 것
ANALYSIS_B_NUMERIC_COLS = ["SeniorCitizen", "MonthlyCharges"]
