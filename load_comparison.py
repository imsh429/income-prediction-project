"""Adult Census 데이터를 Pandas와 Polars로 로딩·정제하고 결과를 비교한다."""

from pathlib import Path
from time import perf_counter

import pandas as pd
import polars as pl


DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
OUTPUT_PATH = Path("adult_census_processed.parquet")
COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]
NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]


def load_with_pandas():
    """Adult Census 원본을 Pandas DataFrame으로 불러온다."""

    start = perf_counter()
    try:
        dataframe = pd.read_csv(
            DATA_URL,
            header=None,
            names=COLUMNS,
            na_values="?",
            skipinitialspace=True,
        )
    except (OSError, pd.errors.ParserError) as error:
        print(f"Pandas 데이터 로딩 오류: {error}")
        return None

    return dataframe, perf_counter() - start


def load_with_polars():
    """Adult Census 원본을 Polars DataFrame으로 불러온다."""

    start = perf_counter()
    try:
        dataframe = pl.read_csv(
            DATA_URL,
            has_header=False,
            new_columns=COLUMNS,
            null_values=" ?",
        )

        # 마지막 빈 행을 제거하고 문자 앞뒤의 공백과 숫자 타입을 정리한다.
        dataframe = (
            dataframe.filter(pl.col("age").is_not_null())
            .with_columns(pl.col(pl.String).str.strip_chars())
            .with_columns(pl.col(NUMERIC_COLUMNS).cast(pl.Int64))
        )
    except (OSError, pl.exceptions.PolarsError) as error:
        print(f"Polars 데이터 로딩 오류: {error}")
        return None

    return dataframe, perf_counter() - start


def compare_results(pandas_result, polars_result):
    """두 DataFrame의 크기, 결측치, 중복, 메모리와 시간을 비교한다."""

    pandas_data, pandas_time = pandas_result
    polars_data, polars_time = polars_result

    comparison = pd.DataFrame(
        {
            "구분": ["Pandas", "Polars"],
            "행 수": [len(pandas_data), polars_data.height],
            "열 수": [len(pandas_data.columns), polars_data.width],
            "결측치": [
                int(pandas_data.isnull().sum().sum()),
                polars_data.null_count().sum_horizontal().item(),
            ],
            "중복 행": [
                int(pandas_data.duplicated().sum()),
                polars_data.height - polars_data.unique().height,
            ],
            "메모리(MB)": [
                pandas_data.memory_usage(deep=True).sum() / 1024**2,
                polars_data.estimated_size("mb"),
            ],
            "로딩 시간(초)": [pandas_time, polars_time],
        }
    )

    print("Pandas와 Polars 로딩 결과 비교")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    if pandas_data.shape == polars_data.shape:
        print("\n두 라이브러리의 행과 열 수가 같습니다.")
    else:
        print("\n두 라이브러리의 행과 열 수가 다릅니다.")


def clean_with_pandas(dataframe):
    """Pandas에서 결측치 행과 완전 중복 행을 순서대로 제거한다."""

    missing_rows = int(dataframe.isnull().any(axis=1).sum())
    without_missing = dataframe.dropna()
    duplicate_rows = int(without_missing.duplicated().sum())
    cleaned = without_missing.drop_duplicates().reset_index(drop=True)

    summary = {
        "구분": "Pandas",
        "원본 행": len(dataframe),
        "결측치 행 제거": missing_rows,
        "중복 행 제거": duplicate_rows,
        "정제 후 행": len(cleaned),
        "남은 결측치": int(cleaned.isnull().sum().sum()),
        "남은 중복 행": int(cleaned.duplicated().sum()),
    }
    return cleaned, summary


def clean_with_polars(dataframe):
    """Polars에서 결측치 행과 완전 중복 행을 순서대로 제거한다."""

    missing_rows = dataframe.filter(pl.any_horizontal(pl.all().is_null())).height
    without_missing = dataframe.drop_nulls()
    duplicate_rows = without_missing.height - without_missing.unique().height
    cleaned = without_missing.unique(maintain_order=True)

    summary = {
        "구분": "Polars",
        "원본 행": dataframe.height,
        "결측치 행 제거": missing_rows,
        "중복 행 제거": duplicate_rows,
        "정제 후 행": cleaned.height,
        "남은 결측치": cleaned.null_count().sum_horizontal().item(),
        "남은 중복 행": cleaned.height - cleaned.unique().height,
    }
    return cleaned, summary


def save_processed_data(dataframe, output_path=OUTPUT_PATH):
    """정제한 Pandas 데이터를 후속 모듈용 Parquet 파일로 저장한다."""

    processed = dataframe.copy()
    processed["income"] = processed["income"].map({"<=50K": True, ">50K": False})
    if processed["income"].isnull().any():
        raise ValueError("income 컬럼에 알 수 없는 값이 있습니다.")

    processed.to_parquet(output_path, index=False)
    print(f"\n정제 데이터 저장: {output_path} ({len(processed):,}행)")


def main():
    """Pandas와 Polars 로딩 비교, 결측치·중복 처리, Parquet 저장을 수행한다."""

    pandas_result = load_with_pandas()
    polars_result = load_with_polars()
    if pandas_result is None or polars_result is None:
        return

    compare_results(pandas_result, polars_result)

    pandas_cleaned, pandas_summary = clean_with_pandas(pandas_result[0])
    polars_cleaned, polars_summary = clean_with_polars(polars_result[0])
    cleaning_summary = pd.DataFrame([pandas_summary, polars_summary])

    print("\n결측치·중복 처리 결과")
    print(cleaning_summary.to_string(index=False))

    if pandas_cleaned.shape != polars_cleaned.shape:
        raise ValueError("Pandas와 Polars의 정제 후 데이터 크기가 다릅니다.")

    save_processed_data(pandas_cleaned)


if __name__ == "__main__":
    main()
