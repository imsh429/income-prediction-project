# income-prediction-project

# 📊 Adult Census Income 분석 및 머신러닝 파이프라인

UCI Adult Census Income 데이터셋을 활용해 데이터 준비부터 통계 분석, 머신러닝 파이프라인 구축, 그리고 보고서 자동화까지 직접 구현해본 실습 프로젝트입니다. 연령, 교육 수준, 주당 근무 시간 및 자본 이익 등 개인의 특성을 분석해, 이를 바탕으로 연간 소득이 5만 달러를 초과하는지 예측하는 머신러닝 모델을 구축하는 것을 목적으로 합니다.

## 실행 Python 버전 3.11.X

## 실행방법
1. `adult_census_processed.parquet` 파일을 `종합실습.py`와 동일한 경로에 위치시킵니다.
2. 아래 명령어를 통해 전체 통합 실행 코드를 구동합니다.

```bash
python 종합실습.py
```
## 📂 파일 구조

```text
├── adult_census_processed.parquet # 전처리 완료 데이터
├── adult_model_pipeline.pkl       # 학습된 모델 파이프라인 (joblib)
├── report.md                      # 자동 생성된 분석 리포트
└── 종합실습.py                    # 전체 통합 실행 코드
```

## 주요 기능

### 1. 데이터 준비 및 전처리
본 데이터: url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
- **Pandas & Polars 비교**: 데이터를 Pandas와 Polars 양쪽으로 모두 불러와 구조를 비교·확인
- **데이터 정제**: 결측치와 중복 데이터를 정제. workclass 결측치는 Private(최빈값)으로 교체, 그 외 결측치(occupation, native-country)는 제거

### 2. 시각화
- **Seaborn 정적 차트**: 나이, 주당 근무시간 분포(Histogram), 성별에 따른 소득 분포(Countplot), 교육 수준별 평균 나이(Barplot), 상관관계 히트맵(Heatmap) 등을 시각화
- **Plotly 인터랙티브 차트**: 마우스 호버 등 동적 조작이 가능한 인터랙티브 차트를 구현

### 3. 통계 분석
- **기술통계**: 주요 수치형 변수들의 평균, 표준편차, 사분위수 등 기초 통계를 산출
- **상관관계**: 변수 간 상관계수를 계산하여 관계성을 파악
- **T-test**: `scipy.stats.ttest_ind`를 사용해 그룹 간 평균 차이를 검정하고 p-value를 해석

### 4. 머신러닝 파이프라인 (ML Pipeline)
- **Scikit-Learn Pipeline**: 전처리 과정과 모델 학습을 하나의 파이프라인으로 묶어 일관성 있게 구성
- **성능 평가 및 저장**: 정확도(Accuracy) 및 F1-Score 등의 평가지표를 출력하고, 학습된 모델은 `joblib`을 이용해 파일로 저장

### 5. 자동화 보고서
- **보고서 자동 생성**: 분석 및 모델 평가 결과를 정리하여 `report.md` 파일로 자동 출력되도록 구현

