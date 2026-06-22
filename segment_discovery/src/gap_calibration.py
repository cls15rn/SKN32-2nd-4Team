"""
structural_gap 적응형 보정 (기획_메모.md 4.1-F 참조, 3번 항목)

배경 — 왜 새 부트스트랩을 도입하지 않는가
-----------------------------------------
SEQUENTIAL_STRUCTURAL_GAP(1.3)은 OOB 방식의 구조적 변동성(모델 재학습)을
감안한 안전 상한 배율이다. 실측 결과 이 비율이 표본 특성에 따라 다르다는
것이 확인됐다(전체데이터 1.065, 작은 세그먼트 1.243) - 1.3 고정값은
작은 표본엔 거의 맞지만 큰 표본엔 과도하게 여유롭다.

이걸 "데이터가 직접 결정"하게 하려는 자연스러운 방법은 매번 gap비율을
실측해서 쓰는 것이지만, 그 실측 자체가 본 작업(부트스트랩)과 같은 무게의
작업을 또 요구한다 - 우리가 부트스트랩 반복횟수 자체를 자동화했던 이유
(탐색비용 > 운영비용)와 완전히 같은 함정이다.

해법 — 운영 중 공짜로 쌓이는 관측치를 재사용
-----------------------------------------
find_stable_bootstrap_count가 멈추는 순간, 그 실행은 이미
(point_estimate, n_positive, n_negative, 실측 ci_width)를 갖고 있다. 여기서
"이론값 대비 실측값의 비율"을 구하는 데는 나눗셈 한 번 외에 추가 계산이
필요 없다. 분석A/B/서브트랙Q는 검증 사이클상 여러 세그먼트·여러 재시도에
걸쳐 같은 함수를 반복 호출하므로, 별도 실험 없이도 운영 과정에서 자연스럽게
관측치가 누적된다.

다음 호출부터는 "표본크기(n)가 가장 비슷한 과거 관측치들의 상위 분위수"를
안전 상한 배율로 쓴다 - 관측치가 충분히 쌓이기 전(콜드스타트)에는 기존
고정값(폴백)을 그대로 사용해 안전성을 잃지 않는다.

⚠️ statistic_type 구분 — AUC와 비율(proportion)을 섞지 않음
-----------------------------------------------------------
분석A/B는 AUC의 부트스트랩 CI(이론값: Hanley-McNeil)이고, 서브트랙Q의
최고위험구간 부트스트랩은 단순 이탈률(비율)의 CI(이론값: Wilson)이다 -
통계량의 종류가 다르면 "이론값 대비 실측값의 비율"이 가리키는 의미도
다르다(모델 재학습 여부도 다름: AUC쪽은 매번 RF를 재학습하고, 비율쪽은
단순 재추출뿐). 두 종류를 같은 분位수 계산에 섞으면 왜곡된 결과가 나오므로,
모든 관측치에 statistic_type(예: "auc_hanley_mcneil", "proportion_wilson")
태그를 붙이고, 조회 시 같은 타입만 필터링한다 - 파일은 공유하지만 통계량
종류별로는 분리되는 구조.

⚠️ 주의: 이 모듈은 "이미 끝난 실행의 부산물을 기록/조회"만 한다.
gap비율을 추정하기 위해 별도로 부트스트랩을 실행하는 함수는 의도적으로
포함하지 않음 - 그러면 처음 문제(탐색비용 재발)로 되돌아간다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
from stats_formulas import hanley_mcneil_se, wilson_se  # noqa: E402  (analysis_a/subtrack_q 의존 제거 - 12일차 보강)


# ---------------------------------------------------------------------------
# 기록: 부트스트랩 실행 1회의 부산물을 누적 (통계량 종류 무관하게 공통 진입점)
# ---------------------------------------------------------------------------

def record_gap_observation(
    n: int,
    point_estimate: float,
    theory_width: float,
    ci_width: float,
    statistic_type: str,
    path: Path | None = None,
) -> float:
    """
    이미 계산이 끝난 값들로부터 "이번 실행의 실측 gap비율"을 구해 기록한다.
    추가되는 연산은 나눗셈 한 번뿐 - 새 부트스트랩을 실행하지 않는다.

    Parameters
    ----------
    theory_width : 이론적으로 기대되는 CI폭(2*1.96*표준오차). 통계량 종류에
        따라 표준오차 공식이 다르므로(AUC=Hanley-McNeil, 비율=Wilson) 호출
        측에서 미리 계산해 넘긴다 - 이 함수는 공식 자체를 모른다.
    statistic_type : "auc_hanley_mcneil" 또는 "proportion_wilson" 등.
        같은 파일에 기록되지만 조회 시 이 값으로 필터링되어 서로 섞이지 않음.

    Returns
    -------
    observed_gap : 이번 실행에서 관측된 gap비율 (실측폭 / 이론폭)
    """
    if path is None:
        path = config.GAP_OBSERVATIONS_PATH  # 호출 시점에 조회 (테스트 monkeypatch 반영)

    observed_gap = ci_width / theory_width if theory_width > 0 else np.nan

    row = pd.DataFrame([{
        "n": n,
        "point_estimate": point_estimate,
        "ci_width": ci_width,
        "theory_width": theory_width,
        "observed_gap": observed_gap,
        "statistic_type": statistic_type,
    }])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        row.to_csv(path, mode="a", header=False, index=False)
    else:
        row.to_csv(path, mode="w", header=True, index=False)

    return float(observed_gap)


def record_auc_gap_observation(
    n: int, point_auc: float, n_positive: int, n_negative: int, ci_width: float,
    path: Path | None = None,
) -> float:
    """분석A/B(AUC 부트스트랩, Hanley-McNeil)용 편의 래퍼"""
    se_hm = hanley_mcneil_se(point_auc, n_positive, n_negative)
    theory_width = 2 * 1.96 * se_hm
    return record_gap_observation(
        n=n, point_estimate=point_auc, theory_width=theory_width,
        ci_width=ci_width, statistic_type="auc_hanley_mcneil", path=path,
    )


def record_proportion_gap_observation(
    n: int, point_proportion: float, ci_width: float,
    path: Path | None = None,
) -> float:
    """서브트랙Q(비율 부트스트랩, Wilson)용 편의 래퍼"""
    se_w = wilson_se(point_proportion, n)
    theory_width = 2 * 1.96 * se_w
    return record_gap_observation(
        n=n, point_estimate=point_proportion, theory_width=theory_width,
        ci_width=ci_width, statistic_type="proportion_wilson", path=path,
    )


# ---------------------------------------------------------------------------
# 조회: 비슷한 표본크기·같은 통계량종류의 과거 관측치로 안전 상한 배율 결정
# ---------------------------------------------------------------------------

def adaptive_structural_gap(
    n: int,
    statistic_type: str = "auc_hanley_mcneil",
    fallback: float | None = None,
    min_observations: int | None = None,
    n_neighbors: int | None = None,
    percentile: float | None = None,
    path: Path | None = None,
) -> tuple[float, str]:
    """
    표본크기 n, 통계량 종류 statistic_type에 대해 쓸 structural_gap을 결정한다.

    관측치가 min_observations 미만이면 콜드스타트로 보고 fallback을 그대로
    사용한다 - 이 단계에서는 안전성을 잃지 않는 게 우선이다. fallback의
    기본값은 statistic_type에 따라 다르다(AUC=1.3, 비율=1.1) - 비율 기반
    부트스트랩은 모델 재학습이 없어 이론값에 더 가깝게 수렴하는 것이
    실측으로 확인됐기 때문.

    같은 statistic_type의 관측치만 걸러낸 뒤, n이 가장 비슷한 과거 관측치
    n_neighbors개를 골라 그 중 percentile(기본 90)을 안전 상한 배율로
    채택한다. 단순 평균이 아니라 분位수를 쓰는 이유: gap비율이 "이 정도면
    충분히 넓다"는 안전 마진이므로, 평균보다 보수적인 상위 분位수가 안전망의
    취지에 맞는다.

    Returns
    -------
    gap : 이번 호출에 사용할 structural_gap
    source : "fallback" 또는 "observed" (보고서/로그용 — 어느 값을 썼는지 추적)
    """
    if fallback is None:
        fallback = (
            config.SEQUENTIAL_STRUCTURAL_GAP if statistic_type == "auc_hanley_mcneil"
            else config.SUBTRACK_Q_STRUCTURAL_GAP
        )
    if min_observations is None:
        min_observations = config.GAP_CALIBRATION_MIN_OBSERVATIONS
    if n_neighbors is None:
        n_neighbors = config.GAP_CALIBRATION_NEIGHBORS
    if percentile is None:
        percentile = config.GAP_CALIBRATION_PERCENTILE
    if path is None:
        path = config.GAP_OBSERVATIONS_PATH

    path = Path(path)
    if not path.exists():
        return fallback, "fallback"

    obs = pd.read_csv(path)
    required_cols = {"n", "observed_gap", "statistic_type"}
    if not required_cols.issubset(obs.columns):
        # ⚠️ 스키마가 바뀐 적이 있으면(예: 3번 항목 도입 이전의 옛 형식)
        # 운영 환경에 구 버전 관측치 파일이 남아있을 수 있다. 이 경우
        # 크래시 대신 콜드스타트로 안전하게 폴백한다 - 관측치 재사용은
        # 최적화일 뿐이므로, 호환 안 되는 데이터를 만나면 그냥 안 쓴다.
        return fallback, "fallback"

    obs = obs.dropna(subset=["observed_gap"])
    obs = obs[obs["statistic_type"] == statistic_type]  # ⚠️ 통계량 종류가 다른 관측치는 섞지 않음
    if len(obs) < min_observations:
        return fallback, "fallback"

    obs["n_distance"] = (obs["n"] - n).abs()
    nearest = obs.nsmallest(n_neighbors, "n_distance")
    gap = float(np.percentile(nearest["observed_gap"], percentile))
    return gap, "observed"


# ---------------------------------------------------------------------------
# 부트스트랩 순차적 조기중단 — 멈춤 판단 공통 로직 (코드 점검 중 발견, 12일차 보강)
# ---------------------------------------------------------------------------
# ⚠️ 발견된 문제: analysis_a.find_stable_bootstrap_count,
# analysis_b.find_stable_bootstrap_count_for_attributes,
# subtrack_q.find_stable_bootstrap_count_for_risk_group 세 함수가 "매
# check_every회마다 CI폭을 계산하고, 안전 상한(ceiling_streak)·자기 안정성
# (self_stable) 두 조건을 확인해 멈출지 판단하는" 로직을 글자 그대로
# 복붙해서 갖고 있었다 - 순열검정 쪽은 find_stable_permutation_p_value로
# 이미 공통화했는데 부트스트랩 쪽은 누락된 비일관성이었음(분석A/B의 실제
# diff 결과: model 학습 줄의 max_depth 파라미터 하나만 다르고 나머지
# 멈춤판단 로직은 완전히 동일). 한쪽만 고치고 다른 쪽을 놓치는 위험이
# 실제로 존재했던 지점 - 여기로 추출해 세 호출부가 공유하게 한다.
#
# 설계: 호출 측은 "한 번의 재추출에서 측정값(AUC 또는 비율) 하나를 만드는
# 방법"(measurement_fn)만 제공한다. 그 측정값을 누적하고, 멈출지 판단하는
# 절차 자체는 이 함수가 전담한다 - 모델을 재학습하는지(분석A/B) 단순
# 재추출만 하는지(서브트랙Q)는 measurement_fn 내부의 문제이므로 이 함수는
# 그 차이를 몰라도 된다.

def run_sequential_bootstrap(
    measurement_fn,
    safe_ceiling: float,
    max_iter: int,
    check_every: int,
    min_n_before_check: int,
    ceiling_patience: int,
    self_stability_window: int,
    self_stability_threshold: float,
) -> tuple[int, list[float], float, float, list[dict]]:
    """
    부트스트랩을 1회씩 누적하면서, 매 check_every회마다 안전 상한 비교와
    자기 안정성 확인으로 멈출지 판단하는 공통 루프.

    Parameters
    ----------
    measurement_fn : () -> float | None
        한 번의 재추출에서 측정값 하나를 만들어 반환. 표본 부족 등으로
        이번 반복을 건너뛰어야 하면 None을 반환(예: OOB 표본이 너무 적거나
        클래스가 한쪽만 있는 경우) - 호출 횟수(i)는 그래도 소비되지만
        measurements 리스트에는 쌓이지 않는다(기존 analysis_a/b의
        `continue` 동작과 동일).

    Returns
    -------
    stop_n : 멈춘 시점의 누적 반복 횟수
    measurements : 누적된 유효 측정값 리스트
    last_ci_low, last_ci_high : 마지막으로 계산된 95% 신뢰구간
    diagnostics_rows : 체크포인트별 (n, ci_width) 기록 (보고서/로그용)
    """
    measurements: list[float] = []
    width_history: list[float] = []
    rows: list[dict] = []
    ceiling_streak = 0
    last_ci_low, last_ci_high = float("nan"), float("nan")

    for i in range(1, max_iter + 1):
        value = measurement_fn()
        if value is None:
            continue
        measurements.append(value)

        if i % check_every == 0 and len(measurements) >= min_n_before_check:
            last_ci_low, last_ci_high = np.percentile(measurements, [2.5, 97.5])
            width = last_ci_high - last_ci_low
            width_history.append(width)
            rows.append({"n": i, "ci_width": width})

            # 조건 ①: 안전 상한보다 좁은가
            if width <= safe_ceiling:
                ceiling_streak += 1
            else:
                ceiling_streak = 0

            # 조건 ②: 최근 변화가 충분히 작은가 (자기 안정성, 상한 도달 못해도 안전망)
            self_stable = False
            if len(width_history) >= self_stability_window + 1:
                recent = width_history[-(self_stability_window + 1):]
                rel_changes = [
                    abs(recent[j + 1] - recent[j]) / recent[j]
                    for j in range(len(recent) - 1) if recent[j] > 0
                ]
                self_stable = bool(rel_changes) and all(
                    rc < self_stability_threshold for rc in rel_changes
                )

            if ceiling_streak >= ceiling_patience or self_stable:
                return i, measurements, float(last_ci_low), float(last_ci_high), rows

    return max_iter, measurements, float(last_ci_low), float(last_ci_high), rows
