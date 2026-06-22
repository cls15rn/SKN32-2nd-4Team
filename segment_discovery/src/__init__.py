"""
segment_discovery/src/ - 분석A/B/서브트랙Q의 실제 구현.

analysis_a.py      : ①경계탐지 ②적절성검증(AUC+순열검정) ③표본충분성(부트스트랩)
analysis_b.py      : 세그먼트별 위험속성 탐지 (분석A와 동일한 2단 구조 재사용)
subtrack_q.py      : risk_count(메인) + K-means(보조탐색)
gap_calibration.py : structural_gap 적응형 보정 + 부트스트랩 순차적 조기중단 공통 루프
preprocessing.py   : Train/Test 분할 + 분석A/B/Q 전용 전처리
"""
