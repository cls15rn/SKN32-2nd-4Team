"""
shared/ - segment_discovery와 churn_prediction 양쪽이 공유하는 단일 출처 모듈.

columns.py        : 컬럼 분류 정의 (단일 출처)
data_loader.py     : 원본 CSV 공통 전처리 관문 (clean_raw_data)
logging_setup.py   : 공통 로거 설정
stats_formulas.py  : 부트스트랩 안전 상한 계산용 표준오차 공식 (Hanley-McNeil, Wilson)

⚠️ 이 디렉토리의 함수/상수를 다른 곳에 중복 정의하지 말 것 - 한쪽만 고치고
다른 쪽을 놓치는 비일관성이 실제로 여러 차례 발견된 적이 있다.
"""
