"""SKALA 4기 광주캠퍼스 2반
작성자: 우상민, woosm901@gmail.com
작성일: 2026/07/20
과제-시스템명: ipp_pipeline.py - https://github.com/imsh429/income-prediction-project
[프로그램 목적 및 개요]: ML Pipelinesklearn.pipeline.Pipeline을 이용하여
1. 전처리 + 모델 파이프라인 구성
2. 모델학습 구성
3. 평가지표 출력
4. joblib으로 모델 저장
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

def build_pipeline(numeric_features, categorical_features):
    # 조건 1) 전처리 + 모델 파이프라인 구성
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", HistGradientBoostingClassifier(random_state=42))
    ])
    return pipeline

def train_and_evaluate(df, model_path="adult_census_model.joblib"):
    #모델 학습 구성
    try:
        # Feature / Target 분리
        X = df.drop(columns=["income"])
        y = df["income"]
        sample_weight = df["fnlwgt"]

        # 수치형 / 범주형 구분
        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

        # 파이프라인 생성
        pipeline = build_pipeline(numeric_features, categorical_features)

        # Train/Test Split
        X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
            X, y, sample_weight, test_size=0.2, random_state=42
        )

        # 학습 (가중치 반영)
        pipeline.fit(X_train, y_train, classifier__sample_weight=sw_train)

        # 예측
        y_pred = pipeline.predict(X_test)

        # 평가 지표 출력
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("F1 Score:", f1_score(y_test, y_pred))

        try:
            print("ROC-AUC:", roc_auc_score(y_test, pipeline.predict_proba(X_test)[:,1]))
        except ValueError:
            print("ROC-AUC 계산 불가: 클래스가 하나만 존재합니다.")

        # joblib으로 모델 저장
        try:
            joblib.dump(pipeline, model_path)
            print(f"모델이 저장되었습니다: {model_path}")
        except OSError as e:
            print(f"모델 저장 실패: {e}")

    except Exception as e:
        print(f"학습/평가 중 오류 발생: {e}")

def main():
    try:
        df = pd.read_parquet("adult_census_processed.parquet")
        train_and_evaluate(df)
    except FileNotFoundError:
        print("데이터 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
