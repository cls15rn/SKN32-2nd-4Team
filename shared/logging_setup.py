"""
shared/logging_setup.py

간단한 로깅 헬퍼. 콘솔에는 그대로 보이고, logs/{스크립트명}.log 파일에도
같은 내용이 누적되어 "어제 새벽 재학습이 왜 실패했는지" 같은 걸 나중에
들여다볼 수 있게 한다.

⚠️ 기존 print() 호출을 전부 logging으로 바꾸는 대규모 작업은 하지 않는다 -
이미 검증된 출력 로직을 건드리는 위험이 더 크다. 대신 "실패를 추적해야
하는 지점"(train.py 등 재학습 스크립트)에만 추가로 파일 로그를 남기는
가벼운 방식을 쓴다.
"""
import logging
from pathlib import Path

LOGS_DIR = Path(__file__).parent.parent / "logs"


def get_logger(name: str) -> logging.Logger:
    """
    name(보통 스크립트 파일명)별로 logs/{name}.log 파일에 로그를 남기는
    로거를 반환한다. 같은 이름으로 여러 번 호출해도 핸들러가 중복 추가되지
    않는다.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # 이미 설정됨

    logger.setLevel(logging.INFO)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(LOGS_DIR / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(file_handler)
    return logger
