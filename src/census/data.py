from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

EXPECTED_COLUMNS = [
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
]

TARGET_COLUMN = "salary"


def clean_census_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    cleaned = dataframe.copy()
    object_columns = cleaned.select_dtypes(include="object").columns

    for column in object_columns:
        cleaned[column] = cleaned[column].str.strip()

    return cleaned.replace("?", pd.NA)


def process_data(
    dataframe: pd.DataFrame,
    categorical_features: Iterable[str],
) -> tuple[pd.DataFrame, pd.Series | None]:
    cleaned = clean_census_data(dataframe)
    features = cleaned.drop(columns=[TARGET_COLUMN], errors="ignore")
    target = cleaned.get(TARGET_COLUMN, None)

    ordered_columns = [
        column for column in EXPECTED_COLUMNS if column in features.columns
    ]
    remaining_columns = [
        column for column in features.columns if column not in ordered_columns
    ]
    features = features[ordered_columns + remaining_columns]

    _ = list(categorical_features)
    return features, target
