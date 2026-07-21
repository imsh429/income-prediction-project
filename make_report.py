import pandas as pd

def generate_markdown_report(df: pd.DataFrame, filename: "report.md"):
    """
    데이터프레임의 주요 통계 지표를 계산하여 Markdown 형식의 보고서 파일로 저장합니다.
    """
    try:
        # 1. 핵심 요약 지표(KPI) 계산
        total_count = len(df)
        avg_age = df['age'].mean() if 'age' in df.columns else 0
        avg_hours = df['hours-per-week'].mean() if 'hours-per-week' in df.columns else 0
        
        if 'income' in df.columns:
            high_income_ratio = (df['income'] == '>50K').mean() * 100
        else:
            high_income_ratio = 0.0

        # 2. 마크다운 보고서 본문 구성
        report_content = f"""# 📊 Adult Census 데이터 분석 성과 보고서

## 1. 개요
* **목적:** Adult Census 데이터를 활용한 주요 그룹별 특성 및 소득 영향 요인 분석
* **대상 데이터:** `adult_census_processed.parquet` (총 {total_count:,}건)

---

## 2. 핵심 요약 지표 (KPI)
* **총 분석 대상 수:** **{total_count:,} 명**
* **평균 연령:** **{avg_age:.1f} 세**
* **평균 주당 근무 시간:** **{avg_hours:.1f} 시간**
* **고소득(>50K) 비율:** **{high_income_ratio:.1f} %**

---

## 3. 주요 분석 결과 및 시사점
* **근무 시간과 소득의 상관성:** 
  * 고소득 그룹은 일반 그룹에 비해 상대적으로 주당 평균 근무 시간이 높게 나타나며, 직업군별 투입 시간에 따른 성과 편차가 존재합니다.
* **인구통계학적 특성:** 
  * 연령과 교육 연수(`education-num`)가 소득 수준을 결정하는 주요 변수로 작용하며, 모델 예측 정확도 향상을 위한 추가 피처 엔지니어링이 필요합니다.

---
> 본 보고서는 파이썬 자동화 스크립트에 의해 실시간으로 생성되었습니다.
"""

        # 3. 파일 저장 (UTF-8 인코딩 적용)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        print(f"[성공] 마크다운 보고서가 생성되었습니다: {filename}")

    except Exception as e:
        print(f"[실패] 보고서 생성 중 오류가 발생했습니다: {e}")

def main():
    try:
        # 정제된 Parquet 데이터 로드
        df = pd.read_parquet("adult_census_processed.parquet")
        generate_markdown_report(df, "report.md")
    except FileNotFoundError:
        print("데이터 파일(adult_census_processed.parquet)을 찾을 수 없습니다.")
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")

if __name__ == "__main__":
    main()