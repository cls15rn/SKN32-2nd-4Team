"""
shared/data_loader.py

segment_discovery 와 churn_prediction 이 공통으로 쓰는 원본 데이터 경로 상수.

(현재는 두 패키지가 각자 preprocessing.py 에서 동일한 로직을 독립적으로
 수행하도록 설계되어 있다 - 의도적으로 코드를 약간 중복시켜, 두 패키지가
 서로의 내부 구현을 import 하지 않고도 완전히 독립적으로 재실행/교체될 수
 있게 함. 만약 전처리 로직이 자주 바뀌어 중복 관리가 부담스러워지면,
 이 파일에 공통 함수를 옮기는 것을 고려할 것.)
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "WA_FnUseC_TelcoCustomerChurn.csv"
