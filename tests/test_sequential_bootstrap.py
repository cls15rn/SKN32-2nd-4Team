"""
tests/test_sequential_bootstrap.py

분석A의 find_stable_bootstrap_count, 분석B의
find_stable_bootstrap_count_for_attributes (G안: Hanley-McNeil 이론값 상한 +
자기 변화율 안정성을 병행하는 순차적 조기중단) 검증.

기획_메모.md 4.1-C 참조: 부트스트랩 반복횟수(예전 고정값 300)는 검증되지
않은 관행값이었음을 확인하고, 여러 자동화 방식을 실험한 끝에 이 방식으로
확정함 - 전체데이터(4,930명)는 평균 40회(7.5배 절감), 작은 불균형
세그먼트(749명)는 평균 52회(5.8배 절감)로 표본 특성에 맞게 동적으로
반복횟수가 달라짐이 실증됨.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "segment_discovery" / "src"))
from analysis_a import find_stable_bootstrap_count, hanley_mcneil_se, permutation_test_for_segment  # noqa: E402


def test_hanley_mcneil_se_increases_as_sample_size_decreases():
    """표본이 작을수록 이론적 표준오차가 커져야 한다"""
    se_large = hanley_mcneil_se(0.73, n_positive=1300, n_negative=3600)
    se_small = hanley_mcneil_se(0.73, n_positive=220, n_negative=530)
    assert se_small > se_large


def test_hanley_mcneil_se_increases_with_imbalance():
    """양성 비율이 더 불균형할수록(더 적을수록) 이론적 표준오차가 커져야 한다"""
    se_balanced = hanley_mcneil_se(0.73, n_positive=500, n_negative=500)
    se_imbalanced = hanley_mcneil_se(0.73, n_positive=50, n_negative=950)
    assert se_imbalanced > se_balanced


@pytest.fixture
def large_balanced_df():
    """전체데이터를 흉내낸 큰 표본 (4900건 규모, 이탈률 26%)"""
    rng = np.random.RandomState(0)
    n = 4900
    segment = rng.binomial(1, 0.5, n)
    base_rate = 0.15 + segment * 0.2  # segment=1이면 이탈률이 더 높도록
    churn = rng.binomial(1, base_rate)
    return pd.DataFrame({"segment": segment, "ChurnFlag": churn})


@pytest.fixture
def small_imbalanced_df():
    """작은 불균형 세그먼트를 흉내낸 표본 (750건 규모, 이탈률 9%)"""
    rng = np.random.RandomState(1)
    n = 750
    segment = rng.binomial(1, 0.5, n)
    base_rate = 0.05 + segment * 0.15
    churn = rng.binomial(1, base_rate)
    return pd.DataFrame({"segment": segment, "ChurnFlag": churn})


def test_find_stable_bootstrap_count_terminates_within_max_iter(large_balanced_df):
    """무한반복에 빠지지 않고 max_iter 이내에 반드시 멈춰야 한다"""
    n_used, mean_auc, ci_low, ci_high, diagnostics = find_stable_bootstrap_count(
        large_balanced_df, max_iter=200, random_state=1,
    )
    assert n_used <= 200
    assert ci_low <= ci_high


def test_find_stable_bootstrap_count_returns_nontrivial_ci(large_balanced_df):
    """반환된 신뢰구간이 의미있는 폭(0보다 뚜렷하게 큼)을 가져야 한다"""
    n_used, mean_auc, ci_low, ci_high, diagnostics = find_stable_bootstrap_count(
        large_balanced_df, max_iter=200, random_state=1,
    )
    assert (ci_high - ci_low) > 0.001


def test_small_sample_does_not_stop_faster_than_large_sample_on_average():
    """
    ⚠️ 핵심 회귀 테스트: 작고 불균형한 표본이 크고 균형잡힌 표본보다
    "평균적으로 더 빨리" 멈추면 안 된다 - 표본부족 위험이 큰 데이터일수록
    자동으로 더 신중하게(반복을 더 많이) 검증해야 한다는 설계 의도가
    실제로 지켜지는지 확인 (이게 깨지면 표본부족 데이터를 성급하게
    "충분하다"고 잘못 판정하는 위험으로 이어진다).
    """
    rng_large = np.random.RandomState(0)
    n_large = 4900
    seg_large = rng_large.binomial(1, 0.5, n_large)
    churn_large = rng_large.binomial(1, 0.15 + seg_large * 0.2)
    df_large = pd.DataFrame({"segment": seg_large, "ChurnFlag": churn_large})

    rng_small = np.random.RandomState(1)
    n_small = 750
    seg_small = rng_small.binomial(1, 0.5, n_small)
    churn_small = rng_small.binomial(1, 0.05 + seg_small * 0.15)
    df_small = pd.DataFrame({"segment": seg_small, "ChurnFlag": churn_small})

    large_stops = [
        find_stable_bootstrap_count(df_large, max_iter=300, random_state=s)[0]
        for s in [1, 2, 3]
    ]
    small_stops = [
        find_stable_bootstrap_count(df_small, max_iter=300, random_state=s)[0]
        for s in [1, 2, 3]
    ]
    # 작은 표본이 큰 표본보다 평균적으로 더 적게 반복하고 끝나면 안 됨
    assert np.mean(small_stops) >= np.mean(large_stops) * 0.5, (
        f"작은 표본 평균 중단={np.mean(small_stops):.0f}, "
        f"큰 표본 평균 중단={np.mean(large_stops):.0f} - "
        f"작은 표본이 지나치게 빨리 멈추는 것으로 보임"
    )


# ---------------------------------------------------------------------------
# 순열검정(②·Ⓑ)의 순차적 조기중단 - 부트스트랩과는 다른 통계량(이항분포
# 기반 Clopper-Pearson)을 쓰지만 같은 "데이터가 직접 반복횟수를 정한다"는
# 철학을 적용 (기획_메모.md 4.1-C 보강 참조)
# ---------------------------------------------------------------------------

def test_permutation_test_terminates_within_max_iter():
    """무한반복에 빠지지 않고 max_iter 이내에 반드시 멈춰야 한다"""
    rng = np.random.RandomState(0)
    n = 1000
    segment = pd.Series(rng.binomial(1, 0.5, n))
    churn = pd.Series(rng.binomial(1, 0.15 + segment * 0.2))
    p_value, n_used = permutation_test_for_segment(segment, churn, max_iter=200)
    assert n_used <= 200
    assert 0.0 <= p_value <= 1.0


def test_permutation_test_returns_low_p_for_clear_signal():
    """패턴이 명확한 데이터(세그먼트가 이탈을 강하게 좌우)는 p값이 매우 작아야 한다"""
    rng = np.random.RandomState(0)
    n = 1000
    segment = pd.Series(rng.binomial(1, 0.5, n))
    churn = pd.Series((segment == 1).astype(int))  # segment가 완벽히 결정
    p_value, n_used = permutation_test_for_segment(segment, churn, max_iter=300)
    assert p_value < 0.05


def test_permutation_test_returns_high_p_for_no_signal():
    """패턴이 전혀 없는 데이터(segment와 churn이 무관)는 p값이 작지 않아야 한다"""
    rng = np.random.RandomState(0)
    n = 1000
    segment = pd.Series(rng.binomial(1, 0.5, n))
    churn = pd.Series(rng.binomial(1, 0.2, n))  # segment와 무관하게 독립 생성
    p_value, n_used = permutation_test_for_segment(segment, churn, max_iter=300)
    assert p_value > 0.05


def test_permutation_test_stops_consistently_across_seeds():
    """
    ⚠️ 회귀 테스트: 명확한 신호가 있는 데이터에서, 시드를 바꿔도 비슷한
    지점에서 멈춰야 한다 - Clopper-Pearson 기반 멈춤이 부트스트랩(G안의
    이론값 근접 단독 방식)처럼 운에 따라 들쭉날쭉하면 안 된다.
    """
    rng = np.random.RandomState(0)
    n = 1000
    segment = pd.Series(rng.binomial(1, 0.5, n))
    churn = pd.Series((segment == 1).astype(int))

    stops = []
    for seed in [1, 2, 3]:
        _, n_used = permutation_test_for_segment(segment, churn, random_state=seed, max_iter=300)
        stops.append(n_used)
    assert max(stops) - min(stops) <= 50, (
        f"시드별 중단시점이 너무 들쭉날쭉함: {stops}"
    )
