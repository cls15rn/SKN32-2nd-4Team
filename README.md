# SKN32-2nd-4Team — 가입 고객 이탈 예측

## 폴더 구조

```
SKN32-2nd-4Team/
├── segment_discovery/   # 분석A·분석B·서브트랙Q (재학습 묶음 — train.py와 같은 주기로 가끔 실행)
│   ├── src/
│   │   ├── preprocessing.py   # 탐색용 전처리 (원본 로드~Train/Test 분할)
│   │   ├── analysis_a.py      # ①가지치기회귀나무+RF보조검증 ②AUC+순열검정 ③부트스트랩CI
│   │   ├── analysis_b.py      # Ⓐ패턴탐지 Ⓑ적절성검증(AUC+순열검정) Ⓒ부트스트랩CI
│   │   └── subtrack_q.py      # risk_count(메인) + K-means(보조탐색)
│   ├── outputs/
│   │   └── segment_rules.json # ← churn_prediction이 읽는 유일한 인터페이스
│   └── app.py                 # 진입점
│
├── churn_prediction/     # 예측모델
│   ├── src/
│   │   ├── feature_engineering.py  # segment_rules.json 적용, FeatureTransformer(저장/로드 가능)
│   │   ├── train_models.py         # 1단계 로지스틱회귀 / 2·3단계 XGBoost / 보조 MLP
│   │   └── evaluate.py             # 평가지표, 임계값, FN비용 추정
│   ├── outputs/
│   │   └── model.pkl, feature_transformer.pkl, predictions.csv, stage_metrics.csv
│   ├── train.py        # 재학습 묶음 — segment_discovery와 같은 주기 (가끔, 표본 충분히 쌓였을 때)
│   └── predict.py      # 추론 — 학습 없음, 독립적으로 자유로운 주기 (언제든 즉시)
│
├── webapp/               # Streamlit 대시보드 (추후 구현)
│   ├── app.py
│   └── pages/
│
├── shared/                # 두 패키지 공통 상수/경로
├── data/                  # 원본 데이터 (WA_FnUseC_TelcoCustomerChurn.csv)
└── requirements.txt
```

## 핵심 설계 원칙

- **이원화의 진짜 경계선은 "재학습 vs 추론"이다 — "분석 vs 예측"이 아니다**
  - **재학습 묶음(가끔, 표본이 충분히 쌓였을 때만 함께 실행)**: `segment_discovery`(세그먼트 경계·위험속성 발견)와 `churn_prediction/train.py`(모델 학습)는 같은 주기를 따른다. `train.py`가 쓰는 `segment_rules.json` 자체가 `segment_discovery`의 산출물이므로, 분석A/B/Q가 갱신될 때 모델도 같은 누적 데이터로 함께 재학습하는 게 일관적이다. 표본부족 위험(범주가 통째로 빠지는 등) 때문에 "신규 데이터만"으로 재학습하지 않고 항상 누적 전체 데이터로 학습한다.
  - **추론(`churn_prediction/predict.py`, 자유 주기)**: 학습 과정이 전혀 없으므로(`model.pkl`을 그대로 불러와 `predict_proba`만 호출) 데이터 양·시점과 무관하게 항상 안정적으로 동작한다. 정해진 주기 없이 필요할 때마다(매일, 즉석 조회, 월별 범위 선택 등) 자유롭게 실행 가능 — "최신성을 반영한다"는 역할은 재학습이 아니라 이 추론 단계가 담당한다.
- **인터페이스는 코드가 아니라 결과 파일**: `churn_prediction`은 `segment_discovery`의 내부 함수를 import하지 않고, `segment_rules.json`(경계, 위험속성, risk_count 계산식)만 읽는다. `predict.py`도 `train.py`의 코드를 다시 실행하지 않고, `train.py`가 저장해둔 `model.pkl` + `feature_transformer.pkl`만 읽는다. 세 작업 모두 서로의 내부 구현을 모른 채 독립적으로 교체·재실행 가능하다.
- **FeatureTransformer로 "학습 때 fit한 규칙"을 고정**: 더미컬럼 목록과 StandardScaler를 `train.py`가 fit해서 `feature_transformer.pkl`로 저장해두면, `predict.py`는 그 규칙을 그대로 재사용해 새 데이터를 변환한다. 추론 시 새로 fit하지 않으므로, 학습 때와 다른 인코딩이 생기는 일이 없다.
- **feature_cols_12 / feature_cols_3 분리**: "1단계는 세그먼트 라벨을 절대 포함하지 않는다"는 원칙이 코드에서 깨지지 않도록, 인코딩 직후 두 가지 피처 목록을 명시적으로 따로 구성한다.
- **분석A·B 모두 ②③(또는 Ⓑ Ⓒ) 검증을 통과해야 확정**: 가지치기/결정나무로 패턴을 찾는 것과, 그 패턴이 우연이 아님을 순열검정·부트스트랩으로 검증하는 것은 완전히 별개 절차다. ②③ 미통과 시 인접 구간을 통합하고 ①부터 재실행한다.

## 실행 순서

> ⚠️ 분석A의 ②③(순열검정·부트스트랩)은 반복 계산이 많아 환경에 따라 수 분 정도 걸릴 수 있습니다. (결과값에는 영향 없음, 속도만 환경마다 다름)

```bash
pip install -r requirements.txt

# 1. 세그먼트/위험속성/risk_count 발견 (표본이 충분히 쌓였을 때)
cd segment_discovery
python app.py --data ../data/WA_FnUseC_TelcoCustomerChurn.csv

# 2. 예측모델 재학습 (새 데이터가 쌓였을 때, segment_rules.json 필요)
cd ../churn_prediction
python train.py --data ../data/WA_FnUseC_TelcoCustomerChurn.csv \
                --rules ../segment_discovery/outputs/segment_rules.json

# 3. 추론 (학습 없음, 언제든 원하는 만큼 자주 실행 가능)
python predict.py --data ../data/new_customers.csv

# 4. (추후) 대시보드
streamlit run ../webapp/app.py
```
