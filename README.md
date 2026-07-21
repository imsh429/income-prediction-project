# 📊 Adult Census Income 분석 및 머신러닝 프로젝트

UCI Adult Census Income 데이터셋을 활용하여 데이터 로딩 성능 비교(Pandas vs Polars), 기술통계량 및 t-검정, Streamlit/Plotly/Seaborn 대시보드 시각화, Scikit-Learn 머신러닝 파이프라인 구축 및 모델 저장까지 모듈별로 구현한 통합 프로젝트입니다.

---

## 👥 팀원 및 역할 분담

* **김서영**:
* **정준용**:
* **정웅기**:
* **우상민**: 
* **신서현**: Streamlit 대시보드 레이아웃 설계 및 Plotly 인터랙티브 시각화 개발 (`app.py`)
* **문영진**: 

---

## 🛠️ 개발 환경 및 필수 라이브러리

- **Python Version**: `3.11.X`
- **의존성 라이브러리** (`requirements.txt`):

```text
pandas
polars
pyarrow
streamlit
plotly
matplotlib
seaborn
scikit-learn
scipy
joblib
```

---

## 📂 파일 구조

```text
├── adult_census_processed.parquet   # 전처리 완료 데이터셋
├── adult_census_model.joblib        # 학습 및 저장된 ML 파이프라인 모델 (joblib)
├── load_comparison.py               # Pandas vs Polars 로딩 및 성능 비교 모듈
├── ipp_stat.py                  # 기술통계량, 상관계수 및 t-검정 모듈
├── app.py                           # Streamlit/Plotly/Seaborn 대시보드 모듈
├── ipp_pipeline.py                  # ML 파이프라인 구축, 학습/평가 및 저장 모듈
└── requirements.txt                 # 프로젝트 의존성 라이브러리 목록
```

---

## 💡 모듈별 주요 기능 및 구현 상세

### 1. 데이터 로딩 비교 (`load_comparison.py`)
- UCI Machine Learning Repository의 원본 데이터(`adult.data`)를 **Pandas**와 **Polars**로 각각 불러와 구동 성능을 비교합니다.
- `perf_counter()`를 사용하여 각 라이브러리의 **행/열 크기, 결측치 수, 중복 행, 메모리 점유량(MB), 로딩 시간(초)**을 측정하고 정밀 비교합니다.

### 2. 통계 분석 및 가설 검정 (`ipp_통계분석.py`)
- **기술통계량**: `describe()` 기본 지표와 더불어 `10%`, `90%` 분위수(Quantile)를 산출합니다.
- **상관관계 분석**: Pearson 상관계수 행렬을 계산하고, 절댓값 기준 상위 상관 변수 쌍을 추출합니다.
- **t-검정 (Independent Samples t-test)**: `scipy.stats.ttest_ind(equal_var=False)`를 활용하여 소득 그룹(`income`) 간 연령(`age`) 및 주요 수치형 변수의 평균 차이에 대한 통계적 유의성(p-value)을 검정합니다.

### 3. 인터랙티브 시각화 대시보드 (`app.py`)
- **상단 접이식 필터 (`st.expander`)**: 소득(`income`), 성별(`sex`), 인종(`race`) 선택에 따른 동적 데이터 필터링을 제공합니다.
- **핵심 요약 지표 (KPI Cards)**: 총 대상 수, 평균 연령, 평균 주당 근무 시간, 고소득(>50K) 비율을 실시간 계산하여 상단에 출력합니다.
- **3개 탭 구성**:
  1. **그룹 비교**: 직업군 및 소득별 평균 근무시간(Plotly), 성별 소득 분포 / 소득별 근무시간 / 학력별 평균 나이(Seaborn).
  2. **상관관계**: 연령 vs 주당 근무시간 산점도(Plotly), 수치형 변수 상관관계 히트맵 및 산점도(Seaborn).
  3. **분포 분석**: 학력 및 소득별 연령 분포 박스플롯(Plotly), 연령 및 주당 근무시간 히스토그램(Seaborn).

### 4. 머신러닝 파이프라인 (`ipp_pipeline.py`)
- **전처리 파이프라인 (`ColumnTransformer`)**:
  - 수치형 변수: `StandardScaler` 적용
  - 범주형 변수: `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` 적용
- **모델 구축 및 가중치 반영**:
  - `HistGradientBoostingClassifier(random_state=42)` 기반 분류기 적용.
  - 모델 학습 시 샘플 가중치(`fnlwgt`)를 `classifier__sample_weight` 매개변수로 반영하여 학습 편향을 최소화합니다.
- **성능 평가 및 모델 저장**:
  - Accuracy, F1-Score, ROC-AUC 평가지표를 출력합니다.
  - 전처리 및 모델 학습이 완료된 전체 파이프라인 객체를 `adult_census_model.joblib`으로 저장합니다.

---

## 🚀 실행 방법

### 1. 의존성 패키지 설치
`requirements.txt`에 명시된 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

### 2. 데이터 로딩 성능 비교 (Pandas vs Polars)
```bash
python load_comparison.py
```

### 3. 통계 분석 및 t-검정 실행
```bash
python ipp_stat.py
```

### 4. 머신러닝 파이프라인 학습 및 모델 저장
```bash
python ipp_pipeline.py
```

### 5. Streamlit 시각화 대시보드 실행
```bash
streamlit run app.py
```
