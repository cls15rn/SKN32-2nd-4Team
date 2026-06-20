"""
tests/test_analysis_a.py

분석A의 ②③ 검증. 특히 ③(부트스트랩 신뢰구간)은 실제로 데이터 누수 버그가
발견됐던 지점이라(복원추출 표본 안에서 재교차검증 -> 점추정값이 신뢰구간
하한보다 낮게 나옴), 그 버그가 재발하지 않는지 자동으로 확인한다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "segment_discovery" / "src"))
from analysis_a import (  # noqa: E402
    bootstrap_auc_confidence_interval,
    make_segment_column,
    segment_only_auc,
)


@pytest.fixture
def synthetic_segmented_df():
    """세그먼트별로 이탈률이 뚜렷하게 다른 합성 데이터 (실제 패턴과 유사한 비율)"""
    rng = np.random.RandomState(0)
    n_per_segment = 300
    rows = []
    segment_churn_rates = {0: 0.5, 1: 0.3, 2: 0.15, 3: 0.05}
    for segment, rate in segment_churn_rates.items():
        churn = rng.binomial(1, rate, n_per_segment)
        rows.append(pd.DataFrame({"segment": segment, "ChurnFlag": churn}))
    return pd.concat(rows, ignore_index=True)


def test_bootstrap_ci_contains_point_estimate(synthetic_segmented_df):
    """
    ⚠️ 회귀 테스트: 점추정 AUC는 부트스트랩 신뢰구간 안에 있어야 한다.
    (실제 버그: OOB 적용 전에는 점추정이 신뢰구간 하한보다 낮게 나왔음 -
    복원추출 표본 안에서 재교차검증을 하면 데이터 누수로 신뢰구간이
    체계적으로 부풀려졌기 때문)
    """
    df = synthetic_segmented_df
    point_auc = segment_only_auc(df["segment"], df["ChurnFlag"], random_state=0)
    mean_auc, ci_low, ci_high = bootstrap_auc_confidence_interval(
        df, n_bootstrap=60, random_state=0
    )

    assert ci_low <= point_auc <= ci_high, (
        f"점추정({point_auc:.4f})이 신뢰구간 [{ci_low:.4f}, {ci_high:.4f}] "
        f"밖에 있음 - OOB 데이터 누수 버그가 재발했을 가능성"
    )


def test_bootstrap_ci_is_reasonably_tight(synthetic_segmented_df):
    """표본이 충분하면(세그먼트당 300건) 신뢰구간 폭이 과도하게 넓지 않아야 한다"""
    df = synthetic_segmented_df
    _, ci_low, ci_high = bootstrap_auc_confidence_interval(
        df, n_bootstrap=60, random_state=0
    )
    assert (ci_high - ci_low) < 0.3


def test_make_segment_column_produces_expected_bins():
    """경계값으로 만든 segment 컬럼이 0부터 시작하는 정수이며 개수가 맞아야 한다"""
    df = pd.DataFrame({"tenure": [0, 3, 10, 20, 50, 70]})
    result = make_segment_column(df, boundaries=[5.5, 17.5, 43.5])
    assert result.min() == 0
    assert result.max() == 3  # 경계 3개 -> 세그먼트 4개(0~3)
    assert result.dtype.kind == "i"
