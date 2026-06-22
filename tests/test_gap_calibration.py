"""
tests/test_gap_calibration.py

structural_gap 적응형 보정(gap_calibration.py, 기획_메모.md 4.1-F 참조,
3번 항목) 검증.

핵심 검증 대상:
1. 관측치가 부족하면(콜드스타트) 항상 기존 고정값(fallback)을 그대로 쓴다 -
   안전성 손실이 없어야 함.
2. find_stable_bootstrap_count가 멈출 때, 이미 계산된 값들로부터 관측치를
   "추가 부트스트랩 없이" 정확히 기록한다.
3. 관측치가 충분히 쌓이면 적응형 값으로 전환되고, 그 값이 실제 관측 분포의
   분位수와 일치한다(임의의 숫자가 아님).
4. structural_gap을 명시적으로 지정하면 적응형 조회를 건너뛰어 기존
   동작(고정값 사용)을 그대로 보존한다 - 회귀 방지.
5. statistic_type이 다른 관측치(AUC vs 비율)는 서로 섞이지 않는다 -
   같은 파일을 공유해도 조회 시 분리되어야 함.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "segment_discovery" / "src"))
import gap_calibration  # noqa: E402
from analysis_a import find_stable_bootstrap_count  # noqa: E402


@pytest.fixture
def tmp_obs_path(tmp_path):
    return tmp_path / "gap_observations.csv"


def _make_segment_df(n=1000, seed=0, churn_rate_segment0=0.1, churn_rate_segment1=0.4):
    """세그먼트~이탈여부 사이에 적당한 신호가 있는 합성 데이터"""
    rng = np.random.RandomState(seed)
    segment = rng.binomial(1, 0.5, n)
    churn = np.where(
        segment == 1,
        rng.binomial(1, churn_rate_segment1, n),
        rng.binomial(1, churn_rate_segment0, n),
    )
    return pd.DataFrame({"segment": segment, "ChurnFlag": churn})


def _write_observations(path, n_list, observed_gap_list, statistic_type="auc_hanley_mcneil"):
    """테스트용 관측치 파일을 직접 작성하는 헬퍼 (새 스키마 기준)"""
    rows = pd.DataFrame({
        "n": n_list,
        "point_estimate": [0.72] * len(n_list),
        "ci_width": [0.03] * len(n_list),
        "theory_width": [0.03] * len(n_list),
        "observed_gap": observed_gap_list,
        "statistic_type": [statistic_type] * len(n_list),
    })
    rows.to_csv(path, index=False)


def test_adaptive_gap_uses_fallback_when_no_observations(tmp_obs_path):
    """관측 파일이 아예 없으면 콜드스타트로 보고 fallback을 그대로 쓴다"""
    gap, source = gap_calibration.adaptive_structural_gap(
        n=1000, statistic_type="auc_hanley_mcneil", fallback=1.3, path=tmp_obs_path
    )
    assert source == "fallback"
    assert gap == 1.3


def test_adaptive_gap_uses_fallback_when_observations_insufficient(tmp_obs_path):
    """관측치가 min_observations 미만이면 관측값이 있어도 fallback을 쓴다"""
    _write_observations(tmp_obs_path, n_list=[1000] * 3, observed_gap_list=[0.9, 1.0, 1.1])

    gap, source = gap_calibration.adaptive_structural_gap(
        n=1000, statistic_type="auc_hanley_mcneil", fallback=1.3,
        min_observations=8, path=tmp_obs_path,
    )
    assert source == "fallback"
    assert gap == 1.3


def test_record_auc_gap_observation_uses_only_already_computed_values(tmp_obs_path):
    """
    기록 시 추가 부트스트랩을 실행하지 않고, 이미 계산된 값들(point_auc,
    n_positive, n_negative, ci_width)만으로 gap비율을 구해야 한다 -
    이론값(Hanley-McNeil)을 직접 재계산해 나눗셈으로 일치하는지 확인.
    """
    from analysis_a import hanley_mcneil_se

    point_auc, n_pos, n_neg, ci_width = 0.73, 300, 700, 0.05
    observed_gap = gap_calibration.record_auc_gap_observation(
        n=1000, point_auc=point_auc, n_positive=n_pos, n_negative=n_neg,
        ci_width=ci_width, path=tmp_obs_path,
    )

    expected_theory_width = 2 * 1.96 * hanley_mcneil_se(point_auc, n_pos, n_neg)
    expected_gap = ci_width / expected_theory_width
    assert observed_gap == pytest.approx(expected_gap)

    obs = pd.read_csv(tmp_obs_path)
    assert len(obs) == 1
    assert obs.loc[0, "observed_gap"] == pytest.approx(expected_gap)
    assert obs.loc[0, "statistic_type"] == "auc_hanley_mcneil"


def test_record_proportion_gap_observation_uses_wilson_se(tmp_obs_path):
    """
    서브트랙Q(비율 부트스트랩)는 Hanley-McNeil이 아니라 Wilson 표준오차를
    이론값으로 써야 한다 - 두 공식이 다른 수치를 내는 케이스로 구분 확인.
    """
    from subtrack_q import wilson_se

    point_p, n, ci_width = 0.4, 200, 0.08
    observed_gap = gap_calibration.record_proportion_gap_observation(
        n=n, point_proportion=point_p, ci_width=ci_width, path=tmp_obs_path,
    )

    expected_theory_width = 2 * 1.96 * wilson_se(point_p, n)
    expected_gap = ci_width / expected_theory_width
    assert observed_gap == pytest.approx(expected_gap)

    obs = pd.read_csv(tmp_obs_path)
    assert obs.loc[0, "statistic_type"] == "proportion_wilson"


def test_adaptive_gap_switches_to_observed_once_enough_accumulated(tmp_obs_path):
    """
    관측치가 min_observations 이상 쌓이면 fallback이 아니라 실제 관측
    분포의 분位수를 반환해야 한다 - 그 값이 임의의 숫자가 아니라
    np.percentile(관측치, percentile)과 정확히 일치하는지 확인.
    """
    gaps = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25]  # 8개, 모두 1.3 미만
    _write_observations(tmp_obs_path, n_list=[1000] * len(gaps), observed_gap_list=gaps)

    gap, source = gap_calibration.adaptive_structural_gap(
        n=1000, statistic_type="auc_hanley_mcneil", fallback=1.3,
        min_observations=8, n_neighbors=8, percentile=90, path=tmp_obs_path,
    )
    assert source == "observed"
    assert gap == pytest.approx(np.percentile(gaps, 90))
    # 실제 데이터에서 표본이 큰 경우 gap이 1.3보다 작게 나온다는 발견을
    # 반영할 수 있어야 한다 - 무조건 fallback 이상으로 강제하지 않음.
    assert gap < 1.3


def test_adaptive_gap_only_uses_nearest_neighbors_by_sample_size(tmp_obs_path):
    """표본크기(n)가 가까운 관측치만 골라써야 한다 - 동떨어진 n의 관측치가 섞이면 안 됨"""
    _write_observations(
        tmp_obs_path,
        n_list=[100] * 8 + [50000] * 8,
        observed_gap_list=[0.5] * 8 + [2.5] * 8,  # 극단적으로 다른 두 그룹
    )

    gap_near_small, _ = gap_calibration.adaptive_structural_gap(
        n=100, statistic_type="auc_hanley_mcneil",
        min_observations=8, n_neighbors=8, path=tmp_obs_path,
    )
    assert gap_near_small == pytest.approx(0.5)

    gap_near_large, _ = gap_calibration.adaptive_structural_gap(
        n=50000, statistic_type="auc_hanley_mcneil",
        min_observations=8, n_neighbors=8, path=tmp_obs_path,
    )
    assert gap_near_large == pytest.approx(2.5)


def test_adaptive_gap_does_not_mix_statistic_types(tmp_obs_path):
    """
    ⚠️ 핵심 검증: AUC(Hanley-McNeil) 관측치와 비율(Wilson) 관측치가 같은
    파일에 있어도, 조회 시 statistic_type으로 분리되어 서로의 분포에
    영향을 주지 않아야 한다.
    """
    rows_auc = pd.DataFrame({
        "n": [1000] * 8, "point_estimate": [0.72] * 8,
        "ci_width": [0.03] * 8, "theory_width": [0.03] * 8,
        "observed_gap": [1.2] * 8, "statistic_type": ["auc_hanley_mcneil"] * 8,
    })
    rows_prop = pd.DataFrame({
        "n": [1000] * 8, "point_estimate": [0.3] * 8,
        "ci_width": [0.05] * 8, "theory_width": [0.05] * 8,
        "observed_gap": [0.5] * 8, "statistic_type": ["proportion_wilson"] * 8,
    })
    pd.concat([rows_auc, rows_prop]).to_csv(tmp_obs_path, index=False)

    gap_auc, source_auc = gap_calibration.adaptive_structural_gap(
        n=1000, statistic_type="auc_hanley_mcneil", min_observations=8,
        n_neighbors=8, path=tmp_obs_path,
    )
    gap_prop, source_prop = gap_calibration.adaptive_structural_gap(
        n=1000, statistic_type="proportion_wilson", min_observations=8,
        n_neighbors=8, path=tmp_obs_path,
    )

    assert source_auc == "observed" and gap_auc == pytest.approx(1.2)
    assert source_prop == "observed" and gap_prop == pytest.approx(0.5)
    assert gap_auc != gap_prop  # 섞였다면 두 값이 같아져야 함 - 분리됐는지 확인


def test_proportion_statistic_uses_different_fallback_than_auc(tmp_obs_path):
    """
    콜드스타트일 때 statistic_type별 기본 fallback이 달라야 한다
    (AUC=SEQUENTIAL_STRUCTURAL_GAP=1.3, 비율=SUBTRACK_Q_STRUCTURAL_GAP=1.1) -
    모델 재학습이 없는 비율 기반 부트스트랩이 이론값에 더 가깝다는 실측을 반영.
    """
    import config

    gap_auc, _ = gap_calibration.adaptive_structural_gap(
        n=1000, statistic_type="auc_hanley_mcneil", path=tmp_obs_path,
    )
    gap_prop, _ = gap_calibration.adaptive_structural_gap(
        n=1000, statistic_type="proportion_wilson", path=tmp_obs_path,
    )
    assert gap_auc == config.SEQUENTIAL_STRUCTURAL_GAP
    assert gap_prop == config.SUBTRACK_Q_STRUCTURAL_GAP
    assert gap_prop < gap_auc


def test_adaptive_gap_falls_back_safely_on_legacy_schema_file(tmp_obs_path):
    """
    ⚠️ 스키마 마이그레이션 방어: statistic_type 컬럼이 없는 옛 형식의
    관측치 파일(3번 항목 도입 이전, point_auc/n_positive/n_negative 컬럼만
    있던 버전)을 만나면 크래시하지 않고 콜드스타트로 안전하게 폴백해야 한다.
    """
    legacy_rows = pd.DataFrame({
        "n": [4900], "point_auc": [0.63], "n_positive": [1182],
        "n_negative": [3718], "ci_width": [0.039], "observed_gap": [1.04],
    })
    legacy_rows.to_csv(tmp_obs_path, index=False)

    gap, source = gap_calibration.adaptive_structural_gap(
        n=4900, statistic_type="auc_hanley_mcneil", fallback=1.3,
        min_observations=1, path=tmp_obs_path,
    )
    assert source == "fallback"
    assert gap == 1.3


def test_find_stable_bootstrap_count_records_observation_on_completion(tmp_path, monkeypatch):
    """
    find_stable_bootstrap_count가 정상적으로 멈추면, 그 실행의 관측치가
    config.GAP_OBSERVATIONS_PATH에 자동으로 기록되어야 한다.
    """
    import config
    obs_path = tmp_path / "gap_observations.csv"
    monkeypatch.setattr(config, "GAP_OBSERVATIONS_PATH", obs_path)

    df = _make_segment_df(n=1000, seed=0)
    n_used, mean_auc, ci_low, ci_high, diag = find_stable_bootstrap_count(
        df, random_state=1, structural_gap=1.3,  # 명시적 지정 -> 적응형 조회는 건너뜀
    )
    assert obs_path.exists()
    obs = pd.read_csv(obs_path)
    assert len(obs) == 1
    assert obs.loc[0, "ci_width"] == pytest.approx(ci_high - ci_low)
    assert obs.loc[0, "statistic_type"] == "auc_hanley_mcneil"


def test_explicit_structural_gap_skips_adaptive_lookup_for_backward_compatibility(tmp_path, monkeypatch):
    """
    ⚠️ 회귀 방지: structural_gap을 명시적으로 지정하면(기존 호출 방식과
    호환) 적응형 조회를 건너뛰고 그 값을 그대로 써야 한다. 즉 같은 시드,
    같은 데이터, 같은 structural_gap이면 적응형 기능 도입 전후로 동일한
    결과(n_used, ci_low, ci_high)가 나와야 한다.
    """
    import config
    obs_path = tmp_path / "gap_observations.csv"
    monkeypatch.setattr(config, "GAP_OBSERVATIONS_PATH", obs_path)

    df = _make_segment_df(n=1000, seed=0)
    result_a = find_stable_bootstrap_count(df, random_state=1, structural_gap=1.3)
    result_b = find_stable_bootstrap_count(
        df, random_state=1, structural_gap=1.3, record_observation=False
    )
    assert result_a[0] == result_b[0]
    assert result_a[1] == pytest.approx(result_b[1])
    assert result_a[2] == pytest.approx(result_b[2])
    assert result_a[3] == pytest.approx(result_b[3])


def test_record_observation_false_does_not_write_file(tmp_path, monkeypatch):
    """record_observation=False면 파일이 생성되지 않아야 한다 (테스트 환경 오염 방지용 옵션)"""
    import config
    obs_path = tmp_path / "gap_observations.csv"
    monkeypatch.setattr(config, "GAP_OBSERVATIONS_PATH", obs_path)

    df = _make_segment_df(n=1000, seed=0)
    find_stable_bootstrap_count(
        df, random_state=1, structural_gap=1.3, record_observation=False
    )
    assert not obs_path.exists()


def _make_attribute_df(n=500, seed=0):
    """analysis_b.find_stable_bootstrap_count_for_attributes 테스트용 합성 데이터.
    숫자형 속성(attr1, attr2)으로 신호를 만들고, _encode_categoricals가 요구하는
    CATEGORICAL_COLS(실제 텔코 컬럼들)는 분석 결과에 영향 없는 더미값으로 채운다."""
    import analysis_b  # noqa: F401  (이 import가 sys.path에 shared/ 를 등록시킴)
    from columns import CATEGORICAL_COLS

    rng = np.random.RandomState(seed)
    attr1 = rng.normal(0, 1, n)
    churn = (attr1 > 0).astype(int)
    df = pd.DataFrame({
        "attr1": attr1,
        "attr2": rng.normal(0, 1, n),
        "ChurnFlag": churn,
    })
    for col in CATEGORICAL_COLS:
        df[col] = "dummy"  # attr1/attr2만 모델 입력으로 쓰이므로 분석 결과엔 영향 없음
    return df


def test_analysis_b_bootstrap_records_observation_to_shared_file(tmp_path, monkeypatch):
    """
    분석B의 find_stable_bootstrap_count_for_attributes도 분석A와 같은
    gap_observations.csv에 관측치를 기록해야 한다 - 두 분석이 관측치를
    공유해 더 빨리 충분한 표본을 확보하는 게 설계 의도.
    """
    import config
    from analysis_b import find_stable_bootstrap_count_for_attributes

    obs_path = tmp_path / "gap_observations.csv"
    monkeypatch.setattr(config, "GAP_OBSERVATIONS_PATH", obs_path)

    df = _make_attribute_df(n=500, seed=0)
    find_stable_bootstrap_count_for_attributes(
        df, ["attr1", "attr2"], random_state=1, structural_gap=1.3,
    )
    assert obs_path.exists()
    obs = pd.read_csv(obs_path)
    assert len(obs) == 1
    assert obs.loc[0, "statistic_type"] == "auc_hanley_mcneil"


def test_analysis_a_and_b_share_accumulated_observations(tmp_path, monkeypatch):
    """
    분석A 실행으로 쌓인 관측치를 분석B 호출 시에도 그대로 조회할 수 있어야
    한다(같은 파일을 공유, 같은 statistic_type) - 두 모듈이 각자 따로
    관측치를 쌓지 않는지 확인.
    """
    import config
    from analysis_b import find_stable_bootstrap_count_for_attributes

    obs_path = tmp_path / "gap_observations.csv"
    monkeypatch.setattr(config, "GAP_OBSERVATIONS_PATH", obs_path)

    df_a = _make_segment_df(n=1000, seed=0)
    for seed in range(1, 9):
        find_stable_bootstrap_count(df_a, random_state=seed, structural_gap=1.3)

    obs_after_a = pd.read_csv(obs_path)
    assert len(obs_after_a) == 8

    df_b = _make_attribute_df(n=500, seed=0)
    find_stable_bootstrap_count_for_attributes(df_b, ["attr1", "attr2"], random_state=1)

    obs_after_b = pd.read_csv(obs_path)
    assert len(obs_after_b) == 9
