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
# ⚠️ RF 보조검증의 트리 개수는 더 이상 고정값(250)이 아니다 - "득표율이
# 시드를 바꿔도 안정적인가"라는 단일 기준으로 데이터가 직접 충분한 개수를
# 찾는다(analysis_a.find_stable_rf_n_estimators). 여기 후보 목록은 그
# 탐색이 살펴볼 범위일 뿐, 최종 값이 아니다.
ANALYSIS_A_RF_N_ESTIMATORS_CANDIDATES = [50, 100, 150, 200, 250, 300, 400, 500]
ANALYSIS_A_RF_STABILITY_IMPROVEMENT_THRESHOLD = 0.15  # 이 비율 이상 개선 안 되면 그 지점에서 멈춤
ANALYSIS_A_PERMUTATION_COUNT = 200        # ②순열검정 반복횟수 (운영시 기본값, 정확도용)
ANALYSIS_A_BOOTSTRAP_COUNT = 300          # ③부트스트랩(OOB) 반복횟수
ANALYSIS_A_P_VALUE_THRESHOLD = 0.05       # ② 통과 기준
ANALYSIS_A_CI_WIDTH_THRESHOLD = 0.1       # ③ 통과 기준 (신뢰구간 폭)
ANALYSIS_A_MAX_ITERATIONS = 5             # ②③ 미통과시 인접구간 통합 재시도 최대횟수

# ---------------------------------------------------------------------------
# 분석B — Ⓐ패턴탐지, Ⓑ적절성검증, Ⓒ부트스트랩CI
# ---------------------------------------------------------------------------
# ⚠️ 세그먼트별 위험속성 개수(예전 ANALYSIS_B_TOP_N_ATTRIBUTES=2)는 더 이상
# 사람이 고정하지 않는다 - max_depth=2 결정나무의 importance>0 결과를 전부
# 쓴다(analysis_b.find_top_risk_attributes). 트리의 깊이 제약이 자연스러운
# 멈춤 기준이 되어, 세그먼트마다 1~3개로 데이터가 직접 개수를 결정한다.
ANALYSIS_B_PERMUTATION_COUNT = 200
ANALYSIS_B_BOOTSTRAP_COUNT = 300  # 분석A·서브트랙Q와 동일 - 표본부족 위험이 더 큰데 반복은 적었던 비일관성 수정

# ---------------------------------------------------------------------------
# 서브트랙Q — risk_count(메인) + K-means(보조탐색)
# ---------------------------------------------------------------------------
SUBTRACK_Q_PERMUTATION_COUNT = 500
SUBTRACK_Q_BOOTSTRAP_COUNT = 300
SUBTRACK_Q_KMEANS_CLUSTERS = 6

# ---------------------------------------------------------------------------
# 예측모델 — 1·2·3단계 + 보조 MLP
# ---------------------------------------------------------------------------
# ⚠️ max_depth/n_estimators는 더 이상 사람이 고정값으로 정하지 않는다.
# 분석A의 ccp_alpha와 같은 원리로, 2단계에서 GridSearchCV(AUC 기준)로 한 번
# 탐색하고 3단계가 그 결과를 그대로 물려받는다(train_models.py 참조).
# 실데이터 확인 결과 고정값(200,depth5)보다 depth=3 쪽이 AUC가 더 높고
# 더 안정적이었음(0.8433 vs 0.8347) - 고정값이 데이터 규모에 안 맞았다는 신호.
XGBOOST_MAX_DEPTH_CANDIDATES = [3, 4, 5, 6]
XGBOOST_N_ESTIMATORS_CANDIDATES = [100, 200, 300]
XGBOOST_LEARNING_RATE = 0.1  # 이건 탐색 대상에 포함하지 않음 - 관행값으로 고정, 필요시 후속 확장
MLP_HIDDEN_LAYER_SIZES = (32, 16)
MLP_MAX_ITER = 500

# ---------------------------------------------------------------------------
# 분류 임계값 선택 — PR곡선에서 "Recall이 급격히 꺾이기 직전 지점"
# ---------------------------------------------------------------------------
# ⚠️ recall이 이미 이 값보다 낮아진 구간은 "이미 너무 많은 이탈자를 놓친"
# 의미 없는 영역으로 보고 탐색에서 제외한다 - recall=0.06 같은 극단적으로
# 낮은 지점에서 노이즈로 인한 미세한 하락이 "급락"으로 잘못 잡히는 버그가
# 실제로 발견됨(2단계 XGBoost에서 threshold=0.83, Recall=0.06으로 붕괴).
THRESHOLD_SEARCH_MIN_RECALL = 0.5
# 한 점(바로 다음 포인트)이 아니라 이 개수만큼의 윈도우 평균 하락률로 비교해
# 단발성 노이즈에 안정적으로 만든다 (RF 트리개수 자동탐지의 patience와 같은 원리).
THRESHOLD_SEARCH_WINDOW = 20

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
