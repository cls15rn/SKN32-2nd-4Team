"""
tests/test_config_consistency.py

⚠️ 발견된 패턴: config.py에 설정값을 정의해두고도, 실제 함수 시그니처가
그 값을 참조하지 않고 별도의 숫자를 직접 박아두는 경우가 있었다(예:
ANALYSIS_B_BOOTSTRAP_COUNT=300으로 정의했지만 analysis_b.py의
bootstrap_attribute_auc_ci 기본값은 여전히 100). "계산은 했지만 연결을
안 한" 버그(③ 임계값 미적용)와 같은 패턴이 config 차원에서도 발생할 수
있음을 보여준 사례. 이 테스트는 주요 함수의 기본값이 실제로 config.py의
값과 일치하는지 확인해 같은 문제가 재발하지 않도록 한다.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "segment_discovery" / "src"))
import analysis_a  # noqa: E402
import analysis_b  # noqa: E402
import subtrack_q  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "churn_prediction" / "src"))
import evaluate  # noqa: E402


def _default(func, param_name):
    import inspect
    return inspect.signature(func).parameters[param_name].default


@pytest.mark.parametrize("func,param_name,config_name", [
    (analysis_a.find_boundaries_pruned_tree, "cv_folds", "ANALYSIS_A_CV_FOLDS"),
    (analysis_a.find_boundaries_pruned_tree, "random_state", "RANDOM_STATE"),
    (analysis_a.segment_only_auc, "cv_folds", "ANALYSIS_A_CV_FOLDS"),
    (analysis_b.attribute_based_auc, "cv_folds", "ANALYSIS_A_CV_FOLDS"),
    (analysis_b.permutation_test_for_attributes, "n_permutations", "ANALYSIS_B_PERMUTATION_COUNT"),
    (analysis_b.bootstrap_attribute_auc_ci, "n_bootstrap", "ANALYSIS_B_BOOTSTRAP_COUNT"),
    (subtrack_q.permutation_test_for_risk_count, "n_permutations", "SUBTRACK_Q_PERMUTATION_COUNT"),
    (subtrack_q.bootstrap_top_risk_group_ci, "n_bootstrap", "SUBTRACK_Q_BOOTSTRAP_COUNT"),
    (subtrack_q.run_kmeans_exploration, "n_clusters", "SUBTRACK_Q_KMEANS_CLUSTERS"),
    (evaluate.find_recall_drop_threshold, "min_recall", "THRESHOLD_SEARCH_MIN_RECALL"),
    (evaluate.find_recall_drop_threshold, "window", "THRESHOLD_SEARCH_WINDOW"),
    (evaluate.compute_all_metrics, "threshold", "DEFAULT_CLASSIFICATION_THRESHOLD"),
])
def test_function_default_matches_config(func, param_name, config_name):
    """함수 기본값이 config.py에 정의된 값과 실제로 일치하는지 확인"""
    expected = getattr(config, config_name)
    actual = _default(func, param_name)
    assert actual == expected, (
        f"{func.__name__}의 {param_name} 기본값({actual})이 "
        f"config.{config_name}({expected})과 다름 - config를 고쳐도 "
        f"실제 함수에는 반영되지 않는 '연결 누락' 버그"
    )
