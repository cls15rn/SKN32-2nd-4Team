"""
churn_prediction/src/ - 예측모델 학습/평가/피처 변환의 실제 구현.

feature_engineering.py : segment_rules.json 적용 + FeatureTransformer (Train-only fit)
train_models.py         : 1·2·3단계 + 보조 MLP 학습
evaluate.py             : 평가지표, SHAP/feature importance, 비용 추정
"""
