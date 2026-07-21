# =============================================================================
# [프로젝트 모듈 설명]
# 모듈명: Adult Census 데이터 시각화 대시보드 (Visualization Module)
# 
# [작성자 및 역할 분담]
#   - Streamlit 대시보드 설계 및 Plotly 인터랙티브 시각화: 신서현
#   - Seaborn 정적 시각화 코드 개발: 문영진
# 
# [사용 라이브러리]
#   - Streamlit, Pandas, Plotly Express, Seaborn, Matplotlib
# 
# [주요 기능]
# 1. 상단 접이식 필터바를 통한 인종, 성별, 소득별 동적 데이터 필터링
# 2. 필터 연동 핵심 요약 지표(KPI) 제공
# 3. Plotly(인터랙티브 차트)와 Seaborn(정적 탐색 차트)의 탭별/좌우 병합 배치
#    - TAB 1: 그룹 비교 (직업군, 성별, 학력, 소득)
#    - TAB 2: 상관관계 (연령 vs 근무시간, 수치형 변수 상관계수 히트맵)
#    - TAB 3: 분포 분석 (학력별 연령 분포 박스플롯, 연령/근무시간 히스토그램)
# =============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn 정적 차트 스타일 글로벌 설정
sns.set_theme(style="whitegrid")

# -----------------------------------------------------------------------------
# 1. 페이지 기본 설정 (Page Config)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Adult Census Visualization Dashboard",
    page_icon="📊",
    layout="wide",                       # 넓은 화면 레이아웃 적용
    initial_sidebar_state="collapsed"    # 사이드바 숨김 (상단 필터 사용)
)

st.title("📊 Adult Census 데이터 분석 대시보드")
st.markdown("본 모듈은 **Plotly**(인터랙티브)와 **Seaborn**(정적 시각화)을 결합하여 데이터의 **그룹 비교, 상관관계, 분포**를 탐색하는 **시각화 전용 모듈**입니다.")

# -----------------------------------------------------------------------------
# 2. 데이터 로드 및 정제 (Data Loading & Preprocessing)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(file_path: str = "adult_census_processed.parquet") -> pd.DataFrame:
    """
    Parquet 데이터를 로드하고, 시각화 범주(Legend) 표현을 위해
    Boolean/숫자 형태의 income 컬럼을 명확한 문자열('>50K', '<=50K')로 자동 통일합니다.
    """
    try:
        df = pd.read_parquet(file_path)
        
        # income 컬럼 데이터 타입 및 라벨 정제 (True/False, 1/0 버그 방지)
        if 'income' in df.columns:
            df['income'] = df['income'].astype(str).str.strip().replace({
                'True': '>50K', 'False': '<=50K',
                '1': '>50K', '0': '<=50K',
                '1.0': '>50K', '0.0': '<=50K'
            })
        return df
    except Exception as e:
        st.error(f"데이터 파일({file_path})을 불러오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

df_raw = load_data()

# 데이터 로드 실패 시 앱 중단
if df_raw.empty:
    st.stop()

# -----------------------------------------------------------------------------
# 3. 상단 접이식 인터랙티브 필터 (Top Collapsible Filter Bar)
# -----------------------------------------------------------------------------
with st.expander("🛠️ **데이터 필터링 설정 (클릭하여 필터 열기/접기)**", expanded=False):
    f_col1, f_col2, f_col3 = st.columns(3)
    
    with f_col1:
        income_options = list(df_raw['income'].unique()) if 'income' in df_raw.columns else []
        selected_income = st.multiselect("소득 수준 (Income) 선택:", options=income_options, default=income_options)
        
    with f_col2:
        sex_options = list(df_raw['sex'].unique()) if 'sex' in df_raw.columns else []
        selected_sex = st.multiselect("성별 (Sex) 선택:", options=sex_options, default=sex_options)
        
    with f_col3:
        race_options = list(df_raw['race'].unique()) if 'race' in df_raw.columns else []
        selected_race = st.multiselect("인종 (Race) 선택:", options=race_options, default=race_options)

# 선택된 필터 조건 데이터 프레임에 적용
filtered_df = df_raw.copy()
if selected_income:
    filtered_df = filtered_df[filtered_df['income'].isin(selected_income)]
if selected_sex:
    filtered_df = filtered_df[filtered_df['sex'].isin(selected_sex)]
if selected_race:
    filtered_df = filtered_df[filtered_df['race'].isin(selected_race)]

# -----------------------------------------------------------------------------
# 4. 상단 주요 요약 지표 (KPI Metrics)
# -----------------------------------------------------------------------------
st.markdown("### 📌 핵심 요약 지표")
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

# KPI 항목 계산
avg_age = filtered_df['age'].mean() if 'age' in filtered_df.columns else 0
avg_hours = filtered_df['hours-per-week'].mean() if 'hours-per-week' in filtered_df.columns else 0

if 'income' in filtered_df.columns and len(filtered_df) > 0:
    high_income_ratio = (filtered_df['income'] == '>50K').mean() * 100
else:
    high_income_ratio = 0.0

# KPI 카드 출력
with col_kpi1:
    st.metric("총 분석 대상 수", f"{len(filtered_df):,} 명")
with col_kpi2:
    st.metric("평균 연령", f"{avg_age:.1f} 세")
with col_kpi3:
    st.metric("평균 주당 근무 시간", f"{avg_hours:.1f} 시간")
with col_kpi4:
    st.metric("고소득(>50K) 비율", f"{high_income_ratio:.1f} %")

st.markdown("---")

# -----------------------------------------------------------------------------
# 5. 시각화 영역 (Plotly 인터랙티브 & Seaborn 정적 차트 병합)
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 1. 그룹 비교 (Group Comparison)", 
    "📈 2. 상관관계 (Correlation)", 
    "🔔 3. 분포 분석 (Distribution)"
])

# =============================================================================
# TAB 1: 그룹 비교 (Group Comparison)
# =============================================================================
with tab1:
    col_t1_left, col_t1_right = st.columns(2)
    
    # -------------------------------------------------------------------------
    # [Left] Plotly - 직업군 및 소득별 평균 근무시간
    # -------------------------------------------------------------------------
    with col_t1_left:
        st.subheader("🔵 [Plotly] 직업군 및 소득별 평균 근무시간")
        
        occ_summary = (
            filtered_df.groupby(['occupation', 'income'], as_index=False)['hours-per-week']
            .mean()
            .sort_values(by='hours-per-week', ascending=False)
        )
        
        fig_group_bar = px.bar(
            occ_summary,
            x='occupation',
            y='hours-per-week',
            color='income',
            barmode='group',
            color_discrete_map={'<=50K': '#636EFA', '>50K': '#EF553B'},
            title="<b>주요 직업군 및 소득 구간별 평균 주당 근무시간</b>",
            labels={
                'occupation': '직업군 (Occupation)',
                'hours-per-week': '평균 주당 근무 시간 (Hours)',
                'income': '소득 수준 (Income)'
            },
            text_auto='.1f'
        )
        fig_group_bar.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig_group_bar, use_container_width=True)

    # -------------------------------------------------------------------------
    # [Right] Seaborn (개발: 신서현, 문영진) - 성별 소득 분포, 소득별 근무시간, 학력별 평균 나이
    # -------------------------------------------------------------------------
    with col_t1_right:
        st.subheader("🟠 [Seaborn] 그룹별 분석 (성별 / 소득 / 교육)")
        
        # 1-1. 성별에 따른 소득 분포 (countplot)
        fig_sns3, ax3 = plt.subplots(figsize=(7, 3.5))
        sns.countplot(data=filtered_df, x="sex", hue="income", ax=ax3)
        ax3.set_title("Income by Sex")
        ax3.set_xlabel("Sex")
        ax3.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig_sns3)
        plt.close(fig_sns3)

        # 1-2. 소득에 따른 근무시간 분포 (boxplot)
        fig_sns5, ax5 = plt.subplots(figsize=(7, 3.5))
        sns.boxplot(data=filtered_df, x="income", y="hours-per-week", ax=ax5)
        ax5.set_title("Hours per Week by Income")
        ax5.set_xlabel("Income")
        ax5.set_ylabel("Hours per Week")
        plt.tight_layout()
        st.pyplot(fig_sns5)
        plt.close(fig_sns5)

        # 1-3. 교육 수준별 평균 나이 (barplot)
        if 'education' in filtered_df.columns and not filtered_df.empty:
            fig_sns4, ax4 = plt.subplots(figsize=(7, 4))
            edu_order = filtered_df.groupby("education")["age"].mean().sort_values().index
            sns.barplot(data=filtered_df, x="education", y="age", order=edu_order, estimator="mean", ax=ax4)
            plt.xticks(rotation=45)
            ax4.set_title("Average Age by Education")
            ax4.set_xlabel("Education")
            ax4.set_ylabel("Average Age")
            plt.tight_layout()
            st.pyplot(fig_sns4)
            plt.close(fig_sns4)

# =============================================================================
# TAB 2: 상관관계 (Correlation)
# =============================================================================
with tab2:
    col_t2_left, col_t2_right = st.columns(2)
    
    # -------------------------------------------------------------------------
    # [Left] Plotly - 연령 vs 주당 근무시간 상관 산점도
    # -------------------------------------------------------------------------
    with col_t2_left:
        st.subheader("🔵 [Plotly] 연령 vs 주당 근무시간 산점도")
        
        # 렌더링 성능 최적화를 위한 샘플링 (최대 3,000건)
        scatter_df = filtered_df.sample(n=min(3000, len(filtered_df)), random_state=42)
        
        fig_scatter = px.scatter(
            scatter_df,
            x='age',
            y='hours-per-week',
            color='income',
            size='education-num',
            hover_data=['occupation', 'workclass'],
            color_discrete_map={'<=50K': '#19D3F3', '>50K': '#FF6692'},
            opacity=0.65,
            title="<b>연령과 주당 근무시간의 관계 (Plotly)</b>",
            labels={
                'age': '연령 (Age)',
                'hours-per-week': '주당 근무 시간 (Hours)',
                'income': '소득 수준 (Income)',
                'education-num': '교육 연수'
            }
        )
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # -------------------------------------------------------------------------
    # [Right] Seaborn (개발: 신서현, 문영진) - 수치형 변수 상관관계 히트맵 및 산점도
    # -------------------------------------------------------------------------
    with col_t2_right:
        st.subheader("🟠 [Seaborn] 상관관계 분석")
        
        # 2-1. 수치형 변수 상관관계 Heatmap
        numeric_cols = ["age", "education-num", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week"]
        avail_cols = [c for c in numeric_cols if c in filtered_df.columns]
        
        if len(avail_cols) > 1 and not filtered_df.empty:
            fig_sns6, ax6 = plt.subplots(figsize=(7, 4.5))
            corr = filtered_df[avail_cols].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax6)
            ax6.set_title("Correlation Heatmap (Seaborn)")
            plt.tight_layout()
            st.pyplot(fig_sns6)
            plt.close(fig_sns6)

        # 2-2. 나이와 근무시간 관계 Scatterplot
        fig_sns7, ax7 = plt.subplots(figsize=(7, 4))
        sns.scatterplot(data=filtered_df, x="age", y="hours-per-week", alpha=0.3, ax=ax7)
        ax7.set_title("Age vs Hours per Week (Seaborn)")
        ax7.set_xlabel("Age")
        ax7.set_ylabel("Hours per Week")
        plt.tight_layout()
        st.pyplot(fig_sns7)
        plt.close(fig_sns7)

# =============================================================================
# TAB 3: 분포 분석 (Distribution)
# =============================================================================
with tab3:
    col_t3_left, col_t3_right = st.columns(2)
    
    # -------------------------------------------------------------------------
    # [Left] Plotly - 학력 수준별 연령 분포 박스플롯
    # -------------------------------------------------------------------------
    with col_t3_left:
        st.subheader("🔵 [Plotly] 학력 수준별 연령 분포 Box Plot")
        
        fig_box = px.box(
            filtered_df,
            x='education',
            y='age',
            color='income',
            color_discrete_map={'<=50K': '#00CC96', '>50K': '#FF5733'},
            title="<b>학력 수준 및 소득에 따른 연령 박스플롯</b>",
            labels={
                'education': '학력 (Education)',
                'age': '연령 (Age)',
                'income': '소득 수준 (Income)'
            }
        )
        fig_box.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig_box, use_container_width=True)
        
    # -------------------------------------------------------------------------
    # [Right] Seaborn (개발: 신서현, 문영진) - 나이 및 주당 근무시간 단일 분포 히스토그램
    # -------------------------------------------------------------------------
    with col_t3_right:
        st.subheader("🟠 [Seaborn] 주요 변수 단일 분포")
        
        # 3-1. 나이 분포 Histogram
        fig_sns1, ax1 = plt.subplots(figsize=(7, 4))
        sns.histplot(filtered_df["age"], bins=30, kde=True, ax=ax1)
        ax1.set_title("Age Distribution (Seaborn)")
        ax1.set_xlabel("Age")
        ax1.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig_sns1)
        plt.close(fig_sns1)

        # 3-2. 주당 근무시간 분포 Histogram
        fig_sns2, ax2 = plt.subplots(figsize=(7, 4))
        sns.histplot(filtered_df["hours-per-week"], bins=30, kde=True, ax=ax2)
        ax2.set_title("Hours per Week Distribution (Seaborn)")
        ax2.set_xlabel("Hours per Week")
        ax2.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig_sns2)
        plt.close(fig_sns2)