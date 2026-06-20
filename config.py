"""
config.py (프로젝트 루트)

자주 조정될 가능성이 있는 값들을 한 곳에 모은다. 코드 안에 흩어진 숫자를
바꾸는 대신, 이 파일만 보고 고치면 된다.

⚠️ 이미 검증을 거쳐 "이렇게 해야 한다"고 확정된 방법론(가지치기 회귀나무,
OOB 방식 부트스트랩, feature_cols_12/_3 분리 등)은 여기서 건드리지 않는다.
여기 있는 건 "같은 방법론 안에서 조정 가능한 숫자"뿐이다.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "WA_FnUseC_TelcoCustomerChurn.csv"
DEFAULT_NEW_DATA_PATH = PROJECT_ROOT / "data" / "new_customers.csv"
SEGMENT_RULES_PATH = PROJECT_ROOT / "segment_discovery" / "outputs" / "segment_rules.json"

# ---------------------------------------------------------------------------
# 공통 랜덤시드 / 분할
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.3  # Train/Test 분할 비율

# ---------------------------------------------------------------------------
# 분석A — ①가지치기 회귀나무 + RF보조검증, ②AUC+순열검정, ③부트스트랩CI
# ---------------------------------------------------------------------------
ANALYSIS_A_CV_FOLDS = 5
ANALYSIS_A_RF_N_ESTIMATORS = 250          # RF 투표(보조검증)용 트리 개수
ANALYSIS_A_PERMUTATION_COUNT = 200        # ②순열검정 반복횟수 (운영시 기본값, 정확도용)
ANALYSIS_A_BOOTSTRAP_COUNT = 300          # ③부트스트랩(OOB) 반복횟수
ANALYSIS_A_P_VALUE_THRESHOLD = 0.05       # ② 통과 기준
ANALYSIS_A_CI_WIDTH_THRESHOLD = 0.1       # ③ 통과 기준 (신뢰구간 폭)
ANALYSIS_A_MAX_ITERATIONS = 5             # ②③ 미통과시 인접구간 통합 재시도 최대횟수

# ---------------------------------------------------------------------------
# 분석B — Ⓐ패턴탐지, Ⓑ적절성검증, Ⓒ부트스트랩CI
# ---------------------------------------------------------------------------
ANALYSIS_B_TOP_N_ATTRIBUTES = 2           # 세그먼트별로 뽑을 주요 위험속성 개수
ANALYSIS_B_PERMUTATION_COUNT = 200
ANALYSIS_B_BOOTSTRAP_COUNT = 100

# ---------------------------------------------------------------------------
# 서브트랙Q — risk_count(메인) + K-means(보조탐색)
# ---------------------------------------------------------------------------
SUBTRACK_Q_PERMUTATION_COUNT = 500
SUBTRACK_Q_BOOTSTRAP_COUNT = 300
SUBTRACK_Q_KMEANS_CLUSTERS = 6

# ---------------------------------------------------------------------------
# 예측모델 — 1·2·3단계 + 보조 MLP
# ---------------------------------------------------------------------------
XGBOOST_N_ESTIMATORS = 200
XGBOOST_MAX_DEPTH = 5
XGBOOST_LEARNING_RATE = 0.1
MLP_HIDDEN_LAYER_SIZES = (32, 16)
MLP_MAX_ITER = 500

# ---------------------------------------------------------------------------
# 개발/검증 시 빠르게 돌려보고 싶을 때 (기본값 대신 이 묶음을 넘기면 됨)
# 결과의 '정확도'는 떨어지지만 '구조 검증'에는 충분 - 실제 제출/운영 시에는
# 위 기본값(ANALYSIS_A_PERMUTATION_COUNT=200 등)을 사용할 것.
# ---------------------------------------------------------------------------
FAST_DEV_OVERRIDES = {
    "ANALYSIS_A_PERMUTATION_COUNT": 20,
    "ANALYSIS_A_BOOTSTRAP_COUNT": 20,
    "ANALYSIS_B_PERMUTATION_COUNT": 20,
    "ANALYSIS_B_BOOTSTRAP_COUNT": 20,
    "SUBTRACK_Q_PERMUTATION_COUNT": 30,
    "SUBTRACK_Q_BOOTSTRAP_COUNT": 30,
}
