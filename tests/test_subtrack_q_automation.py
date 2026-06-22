"""
tests/test_subtrack_q_automation.py

서브트랙Q의 순열검정·부트스트랩 자동화 검증 (기획_메모.md 4.1-E 참조, 11일차).

핵심 검증 대상:
1. wilson_se가 비율 추정량의 표준 공식과 정확히 일치한다.
2. permutation_test_for_risk_count가 analysis_a의 공통 Clopper-Pearson
   로직(find_stable_permutation_p_value)을 재사용해 명확한 신호/무신호를
   올바르게 구분한다.
3. find_stable_bootstrap_count_for_risk_group이 Wilson 기반 안전상한으로
   정상 종료하고, 신뢰구간이 합리적인 범위 안에 있다.
4. structural_gap을 명시적으로 지정하면 적응형 조회를 건너뛴다(회귀 방지).
5. record_observation=True일 때 gap_calibration에 "proportion_wilson"
   타입으로 기록된다(analysis_a/b의 "auc_hanley_mcneil"과 분리).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "segment_discovery" / "src"))
from subtrack_q import (  # noqa: E402
    wilson_se,
    permutation_test_for_risk_count,
    find_stable_bootstrap_count_for_risk_group,
)


def test_wilson_se_matches_standard_formula():
    """Wilson 표준오차가 sqrt(p(1-p)/n) 공식과 정확히 일치해야 한다"""
    p, n = 0.4, 200
    expected = np.sqrt(p * (1 - p) / n)
    assert wilson_se(p, n) == pytest.approx(expected)


def test_wilson_se_increases_as_sample_size_decreases():
    """표본이 작을수록 Wilson 표준오차가 커져야 한다"""
    se_large = wilson_se(0.5, 1000)
    se_small = wilson_se(0.5, 50)
    assert se_small > se_large


def test_wilson_se_is_maximized_at_p_half():
    """비율 추정량의 표준오차는 p=0.5에서 최대여야 한다(이항분산의 성질)"""
    se_mid = wilson_se(0.5, 100)
    se_extreme = wilson_se(0.05, 100)
    assert se_mid > se_extreme


def _make_risk_count_series(n=1000, seed=0, signal=True):
    """risk_count~이탈여부 사이에 신호가 있는/없는 합성 데이터"""
    rng = np.random.RandomState(seed)
    risk_count = pd.Series(rng.randint(0, 4, n))
    if signal:
        churn_prob = 0.1 + 0.2 * risk_count  # risk_count가 클수록 이탈 확률 증가
        churn = pd.Series(rng.binomial(1, np.clip(churn_prob, 0, 1)))
    else:
        churn = pd.Series(rng.binomial(1, 0.2, n))  # risk_count와 무관
    return risk_count, churn


def test_permutation_test_for_risk_count_detects_clear_signal():
    """risk_count가 이탈여부를 명확히 좌우하면 p값이 작아야 한다"""
    risk_count, churn = _make_risk_count_series(n=1000, seed=0, signal=True)
    p_value, n_used = permutation_test_for_risk_count(risk_count, churn, random_state=1)
    assert p_value < 0.05


def test_permutation_test_for_risk_count_detects_no_signal():
    """risk_count와 이탈여부가 무관하면 p값이 작지 않아야 한다"""
    risk_count, churn = _make_risk_count_series(n=1000, seed=0, signal=False)
    p_value, n_used = permutation_test_for_risk_count(risk_count, churn, random_state=1)
    assert p_value > 0.05


def test_permutation_test_for_risk_count_terminates_within_max_iter():
    """⚠️ 무한루프 방지: max_iter 안에서 항상 종료해야 한다"""
    risk_count, churn = _make_risk_count_series(n=500, seed=0, signal=True)
    p_value, n_used = permutation_test_for_risk_count(
        risk_count, churn, random_state=1, max_iter=500,
    )
    assert n_used <= 500


def _make_risk_group_df(n=1000, seed=0, top_churn_rate=0.6):
    """find_stable_bootstrap_count_for_risk_group 테스트용 합성 데이터"""
    rng = np.random.RandomState(seed)
    risk_count = rng.randint(0, 4, n)
    top_mask = risk_count == risk_count.max()
    churn = np.where(
        top_mask, rng.binomial(1, top_churn_rate, n), rng.binomial(1, 0.1, n)
    )
    return pd.DataFrame({"risk_count": risk_count, "ChurnFlag": churn})


def test_find_stable_bootstrap_count_for_risk_group_terminates(tmp_path, monkeypatch):
    """⚠️ 무한루프 방지: max_iter 안에서 항상 종료해야 한다"""
    import config
    monkeypatch.setattr(config, "GAP_OBSERVATIONS_PATH", tmp_path / "gap_observations.csv")

    df = _make_risk_group_df(n=1000, seed=0)
    n_used, top_value, mean_p, ci_low, ci_high, diag = find_stable_bootstrap_count_for_risk_group(
        df, random_state=1, structural_gap=1.1, max_iter=500,
    )
    assert n_used <= 500
    assert 0 <= mean_p <= 1
    assert ci_low <= mean_p <= ci_high


def test_find_stable_bootstrap_count_for_risk_group_finds_correct_top_value():
    """top_value가 실제 risk_count 최댓값과 일치해야 한다"""
    df = _make_risk_group_df(n=1000, seed=0)
    n_used, top_value, mean_p, ci_low, ci_high, diag = find_stable_bootstrap_count_for_risk_group(
        df, random_state=1, structural_gap=1.1, record_observation=False,
    )
    assert top_value == df["risk_count"].max()


def test_explicit_structural_gap_skips_adaptive_lookup(tmp_path, monkeypatch):
    """
    ⚠️ 회귀 방지: structural_gap을 명시적으로 지정하면 적응형 조회를
    건너뛰고 그 값을 그대로 써야 한다 - 같은 시드면 같은 결과가 나와야 함.
    """
    import config
    monkeypatch.setattr(config, "GAP_OBSERVATIONS_PATH", tmp_path / "gap_observations.csv")

    df = _make_risk_group_df(n=1000, seed=0)
    result_a = find_stable_bootstrap_count_for_risk_group(df, random_state=1, structural_gap=1.1)
    result_b = find_stable_bootstrap_count_for_risk_group(
        df, random_state=1, structural_gap=1.1, record_observation=False
    )
    assert result_a[0] == result_b[0]  # n_used
    assert result_a[1] == result_b[1]  # top_value
    assert result_a[2] == pytest.approx(result_b[2])  # mean_p


def test_bootstrap_records_observation_as_proportion_wilson_type(tmp_path, monkeypatch):
    """
    ⚠️ 핵심 검증: 서브트랙Q의 부트스트랩 관측치는 "proportion_wilson"
    타입으로 기록되어야 한다 - 분석A/B의 "auc_hanley_mcneil"과 절대 섞이면
    안 됨(gap_calibration.adaptive_structural_gap의 statistic_type 필터링).
    """
    import config
    obs_path = tmp_path / "gap_observations.csv"
    monkeypatch.setattr(config, "GAP_OBSERVATIONS_PATH", obs_path)

    df = _make_risk_group_df(n=1000, seed=0)
    find_stable_bootstrap_count_for_risk_group(df, random_state=1, structural_gap=1.1)

    assert obs_path.exists()
    obs = pd.read_csv(obs_path)
    assert len(obs) == 1
    assert obs.loc[0, "statistic_type"] == "proportion_wilson"


def test_record_observation_false_does_not_write_file(tmp_path, monkeypatch):
    """record_observation=False면 파일이 생성되지 않아야 한다"""
    import config
    obs_path = tmp_path / "gap_observations.csv"
    monkeypatch.setattr(config, "GAP_OBSERVATIONS_PATH", obs_path)

    df = _make_risk_group_df(n=1000, seed=0)
    find_stable_bootstrap_count_for_risk_group(
        df, random_state=1, structural_gap=1.1, record_observation=False
    )
    assert not obs_path.exists()
