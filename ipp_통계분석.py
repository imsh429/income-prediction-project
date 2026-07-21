# ==============================================================================
# Adult Census 데이터 통계 분석
# 1) 기술통계량 산출 (평균, 표준편차, 분위수)
# 2) 변수 간 상관계수 계산
# 3) scipy.stats.ttest_ind를 이용한 t-검정 및 p-value 해석
# ==============================================================================

import pandas as pd
import numpy as np
from scipy import stats

# ------------------------------------------------------------------------
# 0. 데이터 불러오기
# ------------------------------------------------------------------------
df = pd.read_parquet("adult_census_processed.parquet")

print("=" * 70)
print("데이터 기본 정보")
print("=" * 70)
print(f"행(row) 개수: {df.shape[0]}, 열(column) 개수: {df.shape[1]}")
print(df.dtypes)
print()

# 분석에 사용할 '수치형' 컬럼만 따로 뽑기
# (income은 True/False 값을 갖는 불리언 컬럼으로, '고소득 여부'를 나타내는
#  범주형(그룹 구분용) 변수로 사용)
numeric_cols = ["age", "fnlwgt", "education-num",
                 "capital-gain", "capital-loss", "hours-per-week"]


# ------------------------------------------------------------------------
# 1. 기술통계량 (평균 / 표준편차 / 분위수 등)
# ------------------------------------------------------------------------
print("=" * 70)
print("1. 기술통계량 (Descriptive Statistics)")
print("=" * 70)

# pandas의 describe()는 count, mean, std, min, 25%, 50%(중앙값), 75%, max를 한 번에 계산-> '기술통계'의 핵심 지표
desc = df[numeric_cols].describe().T  # .T로 전치해서 변수를 행으로 보기 쉽게 함

desc["10%"] = df[numeric_cols].quantile(0.10)
desc["90%"] = df[numeric_cols].quantile(0.90)

print(desc.round(2))
print()



# ------------------------------------------------------------------------
# 2. 변수 간 상관계수 (Correlation)
# ------------------------------------------------------------------------
print("=" * 70)
print("2. 변수 간 상관계수 (Pearson Correlation)")
print("=" * 70)

# corr()->피어슨(Pearson) 상관계수를 계산

corr_matrix = df[numeric_cols].corr(method="pearson")
print(corr_matrix.round(3))
print()

# 상관계수 절댓값이 큰 순서로 상위 쌍(pair)을 뽑아 보기 쉽게 정리
corr_pairs = (
    corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    .stack()
    .rename("correlation")
    .reset_index()
)
corr_pairs.columns = ["변수1", "변수2", "상관계수"]
corr_pairs = corr_pairs.reindex(
    corr_pairs["상관계수"].abs().sort_values(ascending=False).index
)

print("[상관계수 절댓값 기준 상위 변수 쌍]")
print(corr_pairs.round(3).to_string(index=False))
print()



# ------------------------------------------------------------------------
# 3. t-검정 (Independent Samples t-test)
# ------------------------------------------------------------------------
print("=" * 70)
print("3. t-검정 (income 그룹 간 age 평균 비교)")
print("=" * 70)


group_high = df.loc[df["income"] == False, "age"]  # 고소득 그룹 (>50K)
group_low = df.loc[df["income"] == True, "age"]     # 저소득 그룹 (<=50K)


t_stat, p_value = stats.ttest_ind(group_high, group_low, equal_var=False)

print(f"고소득(>50K) 그룹: n={len(group_high)}, "
      f"평균={group_high.mean():.2f}, 표준편차={group_high.std():.2f}")
print(f"저소득(<=50K) 그룹: n={len(group_low)}, "
      f"평균={group_low.mean():.2f}, 표준편차={group_low.std():.2f}")
print(f"\nt-statistic = {t_stat:.4f}")
print(f"p-value가 0.05보다 작다")
print()

# p-value 해석: 유의수준(alpha)은 관례적으로 0.05를 사용
alpha = 0.05
print("[p-value 해석]")
if p_value < alpha:
    print(f"p-value가 0.05보다 작다")
    print("  → 귀무가설을 기각한다. 즉, 고소득 여부에 따라 "
          "평균 나이에 통계적으로 유의미한 차이가 있다고 볼 수 있다.")
else:
    print(f"- p-value({p_value:.6f}) >= 유의수준({alpha})")
    print("  → 귀무가설을 기각할 수 없다. 즉, 두 그룹의 평균 나이 차이가 "
          "통계적으로 유의미하다고 보기 어렵다.")
print()

# ------------------------------------------------------------------------
# 3-1. 다른 수치형 변수들에 대해서도 동일한 t-test를 반복 수행
#      (여러 변수를 한 번에 비교하고 싶을 때 활용)
# ------------------------------------------------------------------------
print("=" * 70)
print("3-1. income 그룹 간 여러 변수 t-검정 결과 요약")
print("=" * 70)

results = []
for col in numeric_cols:
    g1 = df.loc[df["income"] == False, col]  # 고소득(>50K)
    g2 = df.loc[df["income"] == True, col]   # 저소득(<=50K)
    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
    results.append({
        "변수": col,
        "고소득 평균": g1.mean(),
        "저소득 평균": g2.mean(),
        "t-statistic": t_stat,
        "p-value": p_val,
        "유의함(p<0.05)": "Yes" if p_val < 0.05 else "No"
    })

results_df = pd.DataFrame(results)
print(results_df.round(4).to_string(index=False))
print()
