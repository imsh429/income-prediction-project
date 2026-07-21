"""Adult Census 데이터를 Pandas와 Polars로 불러와 결과를 비교한다."""

from time import perf_counter

import pandas as pd
import polars as pl


DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
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


def main():
    """Pandas와 Polars로 데이터를 로딩한 뒤 결과를 비교한다."""

    pandas_result = load_with_pandas()
    polars_result = load_with_polars()
    if pandas_result is None or polars_result is None:
        return

    compare_results(pandas_result, polars_result)


if __name__ == "__main__":
    main()
