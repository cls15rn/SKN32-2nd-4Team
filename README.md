# SKN32-2nd-4Team — 가입 고객 이탈 예측

상세 설계/검증 근거는 `docs/기획_메모.md`, 구현 가이드는 `docs/기획구현.md` 참조.

## 폴더 구조

```
SKN32-2nd-4Team/
├── segment_discovery/   # 분석A·분석B·서브트랙Q (1년 단위 재실행)
│   ├── src/
│   │   ├── preprocessing.py   # 탐색용 전처리 (원본 로드~Train/Test 분할)
│   │   ├── analysis_a.py      # ①가지치기회귀나무+RF보조검증 ②AUC+순열검정 ③부트스트랩CI
│   │   ├── analysis_b.py      # Ⓐ패턴탐지 Ⓑ적절성검증(AUC+순열검정) Ⓒ부트스트랩CI
│   │   └── subtrack_q.py      # risk_count(메인) + K-means(보조탐색)
│   ├── outputs/
│   │   └── segment_rules.json # ← churn_prediction이 읽는 유일한 인터페이스
│   └── app.py                 # 진입점
│
├── churn_prediction/     # 예측모델 1·2·3단계 (월 단위 재실행)
│   ├── src/
│   │   ├── feature_engineering.py  # segment_rules.json 적용, feature_cols_12/_3 분기
│   │   ├── train_models.py         # 1단계 로지스틱회귀 / 2·3단계 XGBoost / 보조 MLP
│   │   └── evaluate.py             # 평가지표, 임계값, FN비용 추정
│   ├── outputs/
│   │   └── model.pkl, predictions.csv, stage_metrics.csv
│   └── app.py
│
├── webapp/               # Streamlit 대시보드 (추후 구현)
│   ├── app.py
│   └── pages/
│
├── shared/                # 두 패키지 공통 상수/경로
├── data/                  # 원본 데이터 (WA_FnUseC_TelcoCustomerChurn.csv)
├── docs/                  # 기획_메모.md, 기획구현.md
└── requirements.txt
```

## 핵심 설계 원칙

- **두 트랙은 운영 주기가 다르다**: `segment_discovery`(세그먼트 경계·위험속성처럼 천천히 변하는 구조)는 1년 단위, `churn_prediction`(최신 고객 데이터를 반영하는 위험도 점수)은 월 단위로 재실행한다. 이 차이를 폴더 단위로 분리해, 예측모델을 매달 재학습할 때마다 분석A/B/Q를 다시 돌릴 필요가 없게 했다.
- **인터페이스는 코드가 아니라 결과 파일**: `churn_prediction`은 `segment_discovery`의 내부 함수를 import하지 않고, `segment_rules.json`(경계, 위험속성, risk_count 계산식)만 읽는다. 두 패키지는 서로의 구현을 모른 채 독립적으로 교체·재실행 가능하다.
- **feature_cols_12 / feature_cols_3 분리**: "1단계는 세그먼트 라벨을 절대 포함하지 않는다"는 원칙이 코드에서 깨지지 않도록, 인코딩 직후 두 가지 피처 목록을 명시적으로 따로 구성한다.
- **분석A·B 모두 ②③(또는 Ⓑ Ⓒ) 검증을 통과해야 확정**: 가지치기/결정나무로 패턴을 찾는 것과, 그 패턴이 우연이 아님을 순열검정·부트스트랩으로 검증하는 것은 완전히 별개 절차다. ②③ 미통과 시 인접 구간을 통합하고 ①부터 재실행한다.

## 실행 순서

> ⚠️ 분석A의 ②③(순열검정·부트스트랩)은 반복 계산이 많아 환경에 따라 수 분 정도 걸릴 수 있습니다. (결과값에는 영향 없음, 속도만 환경마다 다름)

```bash
pip install -r requirements.txt

# 1. 세그먼트/위험속성/risk_count 발견 (1년 단위)
cd segment_discovery
python app.py --data ../data/WA_FnUseC_TelcoCustomerChurn.csv

# 2. 예측모델 학습 (월 단위, segment_rules.json 필요)
cd ../churn_prediction
python app.py --data ../data/WA_FnUseC_TelcoCustomerChurn.csv \
              --rules ../segment_discovery/outputs/segment_rules.json

# 3. (추후) 대시보드
streamlit run ../webapp/app.py
```
