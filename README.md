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
│   │   ├── versions/{timestamp}/   # 재학습할 때마다 새 폴더 (model.pkl, feature_transformer.pkl, metadata.json 등)
│   │   ├── latest/                  # versions/ 중 가장 최근 버전의 복사본 — predict.py가 항상 여기만 봄
│   │   └── run_history.csv          # 실행마다 한 줄씩 누적 (언제, 어떤 데이터로, 성능이 어땠는지)
│   ├── train.py        # 재학습 묶음 — segment_discovery와 같은 주기 (가끔, 표본 충분히 쌓였을 때)
│   └── predict.py      # 추론 — 학습 없음, 독립적으로 자유로운 주기 (언제든 즉시)
│
├── webapp/               # Streamlit 대시보드 (추후 구현)
│   ├── app.py
│   └── pages/
│
├── shared/                # 두 패키지 공통 모듈 (data_loader.py: 경로 상수 + 최소 자료형 정리)
├── data/                  # 원본 데이터 (WA_FnUseC_TelcoCustomerChurn.csv), 신규고객은 new_customers.csv
└── requirements.txt
```

## 핵심 설계 원칙

- **이원화의 진짜 경계선은 "재학습 vs 추론"이다 — "분석 vs 예측"이 아니다**
  - **재학습 묶음(가끔, 표본이 충분히 쌓였을 때만 함께 실행)**: `segment_discovery`(세그먼트 경계·위험속성 발견)와 `churn_prediction/train.py`(모델 학습)는 같은 주기를 따른다. `train.py`가 쓰는 `segment_rules.json` 자체가 `segment_discovery`의 산출물이므로, 분석A/B/Q가 갱신될 때 모델도 같은 누적 데이터로 함께 재학습하는 게 일관적이다. 표본부족 위험(범주가 통째로 빠지는 등) 때문에 "신규 데이터만"으로 재학습하지 않고 항상 누적 전체 데이터로 학습한다.
  - **추론(`churn_prediction/predict.py`, 자유 주기)**: 학습 과정이 전혀 없으므로(`outputs/latest/model.pkl`을 그대로 불러와 `predict_proba`만 호출) 데이터 양·시점과 무관하게 항상 안정적으로 동작한다. 정해진 주기 없이 필요할 때마다(매일, 즉석 조회, 특정 구간만 골라보기 등) 자유롭게 실행 가능 — "최신성을 반영한다"는 역할은 재학습이 아니라 이 추론 단계가 담당한다.
- **재학습 결과는 덮어쓰지 않고 버전별로 보관한다**: `train.py`를 실행할 때마다 `outputs/versions/{실행시각}/`에 그 버전의 모델·변환규칙·평가결과·메타데이터(데이터 경로, 행 수, 성능)가 통째로 보관되고, 그중 최신 버전만 `outputs/latest/`로 복사된다. `predict.py`는 `outputs/latest/`만 보므로 항상 가장 최근 모델을 쓰며, 과거 버전이 필요하면 `outputs/versions/`에서 직접 꺼내 비교할 수 있다. `outputs/run_history.csv`에는 실행마다 한 줄씩(시각·데이터 경로·성능 요약) 누적되어 "언제 누가 재학습했는지" 추적 가능하다.
- **인터페이스는 코드가 아니라 결과 파일**: `churn_prediction`은 `segment_discovery`의 내부 함수를 import하지 않고, `segment_rules.json`(경계, 위험속성, risk_count 계산식)만 읽는다. `predict.py`도 `train.py`의 코드를 다시 실행하지 않고, `train.py`가 저장해둔 결과(`outputs/latest/`)만 읽는다. 세 작업 모두 서로의 내부 구현을 모른 채 독립적으로 교체·재실행 가능하다.
- **FeatureTransformer로 "학습 때 fit한 규칙"을 고정**: 더미컬럼 목록과 StandardScaler를 `train.py`가 fit해서 `feature_transformer.pkl`로 저장해두면, `predict.py`는 그 규칙을 그대로 재사용해 새 데이터를 변환한다. 추론 시 새로 fit하지 않으므로, 학습 때와 다른 인코딩이 생기는 일이 없다. (원본 CSV 형태의 자료형 정리는 `shared/data_loader.py`의 `clean_raw_types`로 양쪽 패키지가 공통으로 재사용한다.)
- **"범위를 나눠 본다"는 건 날짜가 아니라 tenure(경과월) 기준이다**: 이 데이터는 절대 가입일/수집일이 없는 스냅샷 데이터다(기획_메모.md 2.2). 그래서 "2026년 5월 가입자만" 같은 절대 시점 필터링은 할 수 없고, "tenure 0~5개월 구간만", "특정 segment만"처럼 경과월 기준으로만 범위를 나눌 수 있다. `predict.py`는 입력을 한 행씩 독립적으로 변환·추론하므로, 전체든 부분 범위든 신규 데이터든 항상 안전하게 동작한다 — 별도의 날짜 컬럼을 추가할 필요는 없다.
- **feature_cols_12 / feature_cols_3 분리**: "1단계는 세그먼트 라벨을 절대 포함하지 않는다"는 원칙이 코드에서 깨지지 않도록, 인코딩 직후 두 가지 피처 목록을 명시적으로 따로 구성한다.
- **분석A·B 모두 ②③(또는 Ⓑ Ⓒ) 검증을 통과해야 확정**: 가지치기/결정나무로 패턴을 찾는 것과, 그 패턴이 우연이 아님을 순열검정·부트스트랩으로 검증하는 것은 완전히 별개 절차다. ②③ 미통과 시 인접 구간을 통합하고 ①부터 재실행한다. 부트스트랩은 Out-of-Bag(OOB) 방식으로 구현되어 있다 — 복원추출 표본 안에서 또 교차검증을 나누면 데이터 누수가 생기는 버그가 실제로 발견되어 수정됨.

## 실행 순서

> ⚠️ 분석A의 ②③(순열검정·부트스트랩)은 반복 계산이 많아 환경에 따라 수 분 정도 걸릴 수 있습니다. (결과값에는 영향 없음, 속도만 환경마다 다름)

**VSCode에서 버튼으로 실행**: `app.py`, `train.py`, `predict.py` 모두 기본 경로가 코드에 들어가 있어, 명령줄 인자 없이 파일을 열고 우측 상단 ▶(Run Python File) 버튼만 눌러도 그대로 실행됩니다.
- `predict.py`는 `data/new_customers.csv`가 없으면, 원본 데이터로 자동 대체해 "데모 모드"로 동작 확인만 시켜줍니다(콘솔에 `[데모 모드]`라고 명시됨). 진짜 신규 데이터가 준비되면 그 경로에 파일을 두고 다시 실행하세요.
- `predict.py`는 학습된 모델(`churn_prediction/outputs/latest/`)이 없으면 안내만 하고 종료합니다 — 먼저 `train.py`를 한 번 실행해두어야 합니다.

**터미널에서 인자를 직접 지정하고 싶을 때**:

```bash
pip install -r requirements.txt

# 1. 세그먼트/위험속성/risk_count 발견 (표본이 충분히 쌓였을 때)
cd segment_discovery
python app.py --data ../data/WA_FnUseC_TelcoCustomerChurn.csv

# 2. 예측모델 재학습 (segment_rules.json 필요) — 실행마다 outputs/versions/{시각}/에 새로 쌓이고 outputs/latest/가 갱신됨
cd ../churn_prediction
python train.py --data ../data/WA_FnUseC_TelcoCustomerChurn.csv \
                --rules ../segment_discovery/outputs/segment_rules.json

# 3. 추론 (학습 없음, 언제든 원하는 만큼 자주 실행 가능) — outputs/latest/의 모델을 사용
python predict.py --data ../data/new_customers.csv

# 4. (추후) 대시보드
streamlit run ../webapp/app.py
```

