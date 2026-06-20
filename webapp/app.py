"""
webapp/app.py

Streamlit 대시보드 진입점 (추후 구현 예정).

segment_discovery/outputs/segment_rules.json 과
churn_prediction/outputs/{model.pkl, predictions.csv, stage_metrics.csv}
를 읽어다 시각화하는 용도로 채워질 자리.

예정 페이지 (pages/ 폴더):
- 01_분석A_세그먼트.py   : 경계, 세그먼트단독 AUC, 순열검정/부트스트랩 결과
- 02_분석B_위험속성.py   : 세그먼트별 위험속성, AUC, p값
- 03_서브트랙Q.py        : risk_count 분포, K-means 보조탐색 결과
- 04_예측모델_비교.py    : 1·2·3단계 성능지표 비교, SHAP
- 05_고객조회.py         : customerID로 개별 고객 이탈확률/위험요인 조회

실행 방법(추후): streamlit run webapp/app.py
"""
import streamlit as st

st.set_page_config(page_title="고객 이탈 예측 대시보드", layout="wide")

st.title("가입 고객 이탈 예측 대시보드")
st.info(
    "이 페이지는 추후 구현될 자리입니다. "
    "segment_discovery 와 churn_prediction 의 산출물을 시각화할 예정입니다."
)
