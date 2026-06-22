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
# ⚠️ 예전 ANALYSIS_A_PERMUTATION_COUNT(200)는 더 이상 쓰이지 않음 - ②의
# 반복횟수도 순차적 조기중단으로 자동화됨 (SEQUENTIAL_PERMUTATION_* 참조)
ANALYSIS_A_BOOTSTRAP_COUNT = 300          # ③부트스트랩(OOB) 반복횟수 - bootstrap_auc_confidence_interval(고정횟수 버전)의 기본값. run_analysis_a는 더 이상 이 함수를 쓰지 않음(아래 SEQUENTIAL 설정 참조)
ANALYSIS_A_P_VALUE_THRESHOLD = 0.05       # ② 통과 기준
ANALYSIS_A_CI_WIDTH_THRESHOLD = 0.1       # ③ 통과 기준 (신뢰구간 폭)
ANALYSIS_A_MAX_ITERATIONS = 5             # ②③ 미통과시 인접구간 통합 재시도 최대횟수

# ---------------------------------------------------------------------------
# ③ 부트스트랩 반복횟수 — 순차적 조기 중단 (Sequential Early Stopping, "G안")
# ---------------------------------------------------------------------------
# ⚠️ 위 ANALYSIS_A_BOOTSTRAP_COUNT(300)는 검증되지 않은 관행값이었다.
# run_analysis_a는 이제 find_stable_bootstrap_count를 써서, 표본크기·불균형도에
# 맞는 반복횟수를 매번 직접 찾는다. (기획_메모.md 4.1-C 참조)
#
# 방법: 부트스트랩을 1회씩 누적(이전 계산을 버리지 않음)하면서, 매
# SEQUENTIAL_CHECK_EVERY 회마다 CI폭을 확인한다. (1) Hanley-McNeil
# 이론값×SEQUENTIAL_STRUCTURAL_GAP 보다 좁아지거나, (2) 최근
# SEQUENTIAL_SELF_STABILITY_WINDOW 번의 측정이 서로
# SEQUENTIAL_SELF_STABILITY_THRESHOLD 이내로 거의 안 변하면, 그 상태가
# 연속 SEQUENTIAL_CEILING_PATIENCE회 유지될 때 멈춘다.
#
# 실데이터 검증(시드 5개): 전체데이터(4,930명)는 평균 40회(300회 대비
# 7.5배 절감, 표준편차 0.0037로 안정), 작은 불균형 세그먼트(749명,
# 이탈률9.8%)는 평균 242회(1.2배 절감, 표준편차 0.0050)로 더 신중하게
# 멈췄다 - "표본 특성에 맞게 반복횟수가 동적으로 달라진다"는 목표가 실증됨.
SEQUENTIAL_MAX_ITER = 500
SEQUENTIAL_CHECK_EVERY = 10
SEQUENTIAL_MIN_N_BEFORE_CHECK = 30   # 너무 적은 표본으로 CI폭을 재는 것 자체가 무의미해짐을 방지
SEQUENTIAL_STRUCTURAL_GAP = 1.3      # [콜드스타트 전용 폴백값] 관측 기록이 부족할 때만 쓰는 초기 안전망.
# 표본 크기가 다르면 실제 gap비율도 다르다는 게 실측으로 확인됨(전체데이터
# 1.065 vs 작은세그먼트 1.243) - 이 1.3을 고정해서 계속 쓰는 대신,
# find_stable_bootstrap_count가 멈출 때마다 "이미 계산해놓은 값들로 공짜로
# 계산되는" 그 실행의 실측 gap비율을 GAP_OBSERVATIONS_PATH에 누적 기록하고,
# 다음 호출부터는 비슷한 표본크기의 과거 관측치 분位수를 안전 상한으로
# 사용한다(gap_calibration.adaptive_structural_gap 참조) - 새 부트스트랩
# 실행을 추가하지 않으므로 탐색비용이 전혀 들지 않는다. 관측치가
# GAP_CALIBRATION_MIN_OBSERVATIONS 미만일 때만 이 고정값(1.3)을 그대로 쓴다.
SEQUENTIAL_CEILING_PATIENCE = 2      # 상한 도달이 단발성 우연이 아닌지 연속 확인
SEQUENTIAL_SELF_STABILITY_WINDOW = 5
SEQUENTIAL_SELF_STABILITY_THRESHOLD = 0.05  # 상한에 못 도달해도(작은 표본 등) 무한반복 방지하는 안전망

# ---------------------------------------------------------------------------
# structural_gap 적응형 보정 — 탐색비용 없이 관측치를 누적·재사용 (3번 항목)
# ---------------------------------------------------------------------------
# 배경: SEQUENTIAL_STRUCTURAL_GAP(1.3)을 "데이터가 직접 추정"하게 만들려면
# 보통 별도의 풀스케일 부트스트랩을 미리 돌려 실측 gap을 구해야 하는데,
# 이는 본 작업과 같은 무게의 작업을 또 하는 것이라 탐색비용이 운영비용보다
# 커지는 동일한 함정에 빠진다. 해법: 매 find_stable_bootstrap_count 호출이
# 끝날 때 이미 계산된 (point_auc, n_positive, n_negative, 실측 ci_width)로
# "그 실행의 실측 gap비율"을 추가 연산 없이 구해 기록만 한다. 분석A는
# 검증사이클상 여러 세그먼트·여러 재시도에 걸쳐 반복 호출되므로, 별도
# 실험 없이도 운영 중에 자연스럽게 관측치가 쌓인다.
GAP_OBSERVATIONS_PATH = PROJECT_ROOT / "segment_discovery" / "outputs" / "gap_observations.csv"
GAP_CALIBRATION_MIN_OBSERVATIONS = 8   # 이보다 적게 쌓였으면 콜드스타트로 보고 고정값(1.3) 사용
GAP_CALIBRATION_NEIGHBORS = 6          # 표본크기(n)가 가장 비슷한 과거 관측치 k개만 사용
GAP_CALIBRATION_PERCENTILE = 90        # 그 k개 중 상위 분위수(보수적 상한) 채택

# ---------------------------------------------------------------------------
# ②·Ⓑ 순열검정 반복횟수 — 순차적 조기 중단 (부트스트랩과는 다른 통계량)
# ---------------------------------------------------------------------------
# ⚠️ 위 ANALYSIS_A_PERMUTATION_COUNT(200), ANALYSIS_B_PERMUTATION_COUNT(200)도
# 검증되지 않은 관행값이었다. 다만 부트스트랩(Hanley-McNeil)과는 다른 이론적
# 근거를 쓴다 - 순열검정은 "관측값을 넘는 순열의 비율"이 이항분포를 따르므로,
# Clopper-Pearson 신뢰구간 공식으로 "이 정도 반복이면 p값에 대한 결론이
# 확실하다"는 지점을 직접 계산할 수 있다(부트스트랩의 OOB 모델재학습 같은
# 추가 구조적 변동성이 없어 이론값에 항상 안정적으로 수렴함 - 시드 3개가
# 정확히 같은 지점에서 멈춤).
#
# 실데이터 검증: 전체데이터·작은세그먼트 모두 일관되게 60회에서 멈춤(200회
# 대비 3.3배 절감) - 부트스트랩과 달리 표본크기가 아니라 "효과의 크기
# (관측 AUC가 우연분포에서 얼마나 떨어져 있는지)"가 멈춤시점을 좌우함.
SEQUENTIAL_PERMUTATION_CONFIDENCE = 0.95   # Clopper-Pearson 신뢰수준
SEQUENTIAL_PERMUTATION_CHECK_EVERY = 10
SEQUENTIAL_PERMUTATION_MAX_ITER = 500

# ---------------------------------------------------------------------------
# 분석B — Ⓐ패턴탐지, Ⓑ적절성검증, Ⓒ부트스트랩CI
# ---------------------------------------------------------------------------
# ⚠️ 세그먼트별 위험속성 개수(예전 ANALYSIS_B_TOP_N_ATTRIBUTES=2)는 더 이상
# 사람이 고정하지 않는다 - max_depth=2 결정나무의 importance>0 결과를 전부
# 쓴다(analysis_b.find_top_risk_attributes). 트리의 깊이 제약이 자연스러운
# 멈춤 기준이 되어, 세그먼트마다 1~3개로 데이터가 직접 개수를 결정한다.
# ⚠️ 예전 ANALYSIS_B_PERMUTATION_COUNT(200)도 더 이상 쓰이지 않음 - Ⓑ의
# 반복횟수도 분석A와 동일하게 순차적 조기중단으로 자동화됨
ANALYSIS_B_BOOTSTRAP_COUNT = 300  # 분석A·서브트랙Q와 동일 - 표본부족 위험이 더 큰데 반복은 적었던 비일관성 수정

# ---------------------------------------------------------------------------
# 서브트랙Q — risk_count(메인) + K-means(보조탐색)
# ---------------------------------------------------------------------------
# ⚠️ [정정] 아래 SUBTRACK_Q_PERMUTATION_COUNT(500)·SUBTRACK_Q_BOOTSTRAP_COUNT(300)는
# 분석A·B와 동일한 자동화 로직이 적용되지 않은 채 남아있던 고정값이었다.
# risk_count 검증도 분석A/B와 같은 원칙(반복횟수를 사람이 고정하지 않고
# 데이터가 직접 찾음)을 적용한다:
#
# 순열검정: risk_count의 통계량(risk_count별 이탈률의 분산)은 AUC와 다르지만,
# "관측값을 넘는 순열의 비율"은 통계량 종류와 무관하게 항상 이항분포를
# 따른다 - 분석A의 Clopper-Pearson 기반 순차적 조기중단(find_stable_
# permutation_count, analysis_a.permutation_test_for_segment에서 추출한
# 공통 로직)을 그대로 재사용한다.
#
# 부트스트랩: risk_count 최고위험구간의 부트스트랩은 AUC가 아니라 단순
# "이탈률(비율)"의 신뢰구간이므로, Hanley-McNeil(AUC 전용 공식)을 쓸 수
# 없다 - 대신 Wilson 표준오차(비율 추정량의 표준 공식, subtrack_q.wilson_se)
# 를 이론값으로 쓴다. 모델 재학습이 없는 단순 재추출이라 분석A/B의 AUC
# 기반 부트스트랩(gap비율 1.3 안팎)보다 이론값에 더 가깝게 수렴하는 것이
# 실측으로 확인됨 - SUBTRACK_Q_STRUCTURAL_GAP을 별도로 더 타이트하게 둔다.
SUBTRACK_Q_BOOTSTRAP_COUNT = 300    # ⚠️ 더 이상 쓰이지 않음(레거시 호환용) - bootstrap_top_risk_group_ci가 순차적 조기중단으로 교체됨
# ⚠️ 예전 SUBTRACK_Q_PERMUTATION_COUNT(500)는 완전히 제거됨(ANALYSIS_A/B_
# PERMUTATION_COUNT와 동일 패턴) - permutation_test_for_risk_count가
# n_permutations 파라미터 자체를 받지 않고 SEQUENTIAL_PERMUTATION_MAX_ITER
# 기반 순차적 조기중단만 쓰므로, "레거시 호환용으로 남겨둘 대상"조차 없음
# (대조: SUBTRACK_Q_BOOTSTRAP_COUNT는 레거시 함수 bootstrap_top_risk_group_ci
# 가 실제로 참조하므로 남겨둠 - 둘을 같은 패턴으로 다루면 안 됨을 점검 중 확인)
SUBTRACK_Q_KMEANS_CLUSTERS = 6
SUBTRACK_Q_STRUCTURAL_GAP = 1.1   # [콜드스타트 폴백] 모델재학습이 없는 단순 재추출이라 AUC기반(1.3)보다 타이트 - 실측 gap비율 1.02 근방 확인됨

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
# ⚠️ [정정] learning_rate도 max_depth/n_estimators와 같은 GridSearchCV 탐색
# 대상으로 편입 - 예전에는 0.1로 고정하고 "관행값, 필요시 후속 확장"이라는
# 주석만 남겨둔 채 탐색에서 제외되어 있었다. max_depth/n_estimators는 데이터가
# 정하게 하면서 learning_rate만 사람이 고정한 것은 일관성이 깨진 지점이었다.
XGBOOST_LEARNING_RATE_CANDIDATES = [0.01, 0.05, 0.1, 0.2]
# ⚠️ ANALYSIS_A_CV_FOLDS와 값이 같아도(둘 다 5) 의미가 다른 별개 설정이다 -
# 같은 이름으로 묶으면 "NUMERIC_COLS가 파일마다 다른 의미였던" 혼동이
# 재발할 수 있어 의도적으로 분리.
XGBOOST_SEARCH_CV_FOLDS = 5
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
# find_recall_drop_threshold가 탐색에 실패했을 때(유효구간 부족 등)의 안전한
# 폴백값. compute_all_metrics의 기본값도 이 상수 하나로 통일 - 같은 "0.5"라는
# 숫자가 여러 파일에 따로 적혀 있으면 한쪽만 바꾸고 다른 쪽을 놓치는 위험이 있다.
DEFAULT_CLASSIFICATION_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# [제거됨] 개발/검증 시 빠르게 돌려보고 싶을 때 쓰던 FAST_DEV_OVERRIDES
# ---------------------------------------------------------------------------
# 순열검정/부트스트랩의 반복횟수가 순차적 조기중단(Sequential Early Stopping)
# 으로 자동화되면서 이 오버라이드 자체가 불필요해졌다. 이 딕셔너리를 실제로
# 적용하는 코드도 작성된 적이 없었음(어디서도 import/참조되지 않는 죽은
# 코드였음을 점검 중 확인) - 코드 정규화 차원에서 제거.
