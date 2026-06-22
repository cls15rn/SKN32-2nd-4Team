"""
shared/stats_formulas.py

부트스트랩 안전 상한 계산에 쓰는 닫힌 형태의 표준오차 공식들.

⚠️ 파일명 안내: 표준 라이브러리 `statistics` 모듈과의 이름 충돌을 피하기
위해 `stats_formulas.py`로 명명했다 - `statistics.py`로 지으면 sys.path
순서에 따라 표준 라이브러리 모듈을 가려버릴 위험이 있다.

⚠️ 이 함수들을 옮긴 이유 (코드 점검 중 발견, 12일차 보강)
-----------------------------------------------------------
hanley_mcneil_se는 원래 analysis_a.py에, wilson_se는 subtrack_q.py에
정의되어 있었다. 그런데 gap_calibration.py가 두 함수를 모두 지연 import로
가져다 쓰고, 동시에 subtrack_q.py도 gap_calibration.py를 지연 import하는
구조라 실제로 순환 의존(gap_calibration ↔ subtrack_q)이 생겼다 - 지금은
지연 import(함수 본문 안에서 import)로 우회해서 동작하지만, 이는 "문제를
피한 것"이지 "없앤 것"이 아니다. 누군가 나중에 최상단 import로 바꾸면
즉시 ImportError가 난다.

근본 해결: 이 두 공식은 통계적으로 완전히 독립적인 순수 함수다(AUC
추정량/비율 추정량의 표준오차 - 분석A나 서브트랙Q라는 특정 분석에 속할
이유가 없음). shared/ 로 옮겨 양쪽이 같은 곳에서 가져다 쓰게 하면, 어느
모듈도 다른 분석 모듈을 가리킬 필요가 없어져 순환 자체가 사라진다.

analysis_a.hanley_mcneil_se, subtrack_q.wilson_se는 하위 호환을 위해
이 모듈의 함수를 그대로 재노출(re-export)한다 - 기존 코드/테스트가
`from analysis_a import hanley_mcneil_se`처럼 옛 경로로 import해도 깨지지
않는다.
"""
import numpy as np


def hanley_mcneil_se(auc: float, n_positive: int, n_negative: int) -> float:
    """
    Hanley & McNeil(1982)의 AUC 표준오차 근사 공식.
    표본크기·이탈률(불균형도)·AUC만으로 "이론적으로 기대되는 측정 불안정성"을
    부트스트랩 없이 즉시 계산한다 - Cohen 효과크기처럼 사전에 닫힌 형태로
    구해지는 공식이지만, 정확히는 AUC 추정량의 분산을 위한 표준 통계 공식이다.

    ⚠️ 이 공식은 "고정된 분류기"의 AUC를 가정한다. 우리는 매 부트스트랩
    반복마다 RandomForest를 새로 학습시키므로(OOB 방식), 실제 측정값은
    이 이론값보다 항상 더 넓게 나오는 구조적 격차가 있다(실측 확인됨) -
    그래서 이 값을 "목표"가 아니라 "안전 상한의 기준선"으로만 쓴다.
    """
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    return float(np.sqrt(
        (auc * (1 - auc) + (n_positive - 1) * (q1 - auc**2) + (n_negative - 1) * (q2 - auc**2))
        / (n_positive * n_negative)
    ))


def wilson_se(p: float, n: int) -> float:
    """
    Wilson 표준오차 - 비율 추정량(이탈률 등)의 표준 공식.

    Hanley-McNeil은 AUC 추정량 전용 공식이라 risk_count 최고위험구간의
    부트스트랩(단순 이탈률 신뢰구간)에는 맞지 않는다. Wilson 공식은 비율
    추정량의 표준오차를 닫힌 형태로 즉시 계산한다 - 부트스트랩 없이도
    "이론적으로 기대되는 측정 불안정성"을 구할 수 있다는 점에서
    Hanley-McNeil과 같은 역할을 한다.

    ⚠️ 이 공식도 "단순 임의추출"을 가정한다. 우리는 매 부트스트랩 반복마다
    모델을 재학습하지 않고 단순 재추출만 하므로(AUC 부트스트랩과 다름),
    구조적 격차(gap비율)가 AUC쪽(1.3)보다 작게(1.1 안팎) 나오는 것이
    실측으로 확인됨 - config.SUBTRACK_Q_STRUCTURAL_GAP 참조.
    """
    return float(np.sqrt(p * (1 - p) / n))
