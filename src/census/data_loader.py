from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from loguru import logger

# ── Paths ─────────────────────────────────────────────────────────────────────
# Resolved from this file's location so the module works regardless of CWD.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs" / "data_log"
DATA_CLEANED_DIR = PROJECT_ROOT / "data" / "data_cleaned"

# ── Constants ─────────────────────────────────────────────────────────────────

# Legacy schema — kept as-is for backward compatibility with model.py and
# existing tests.  NOTE: the CSV column is 'fnlgt'; 'fnlwgt' is the canonical
# UCI name used by the earlier version of this codebase.
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

# Post-clean feature schema (used by the sklearn Pipeline defined in
# pipeline_architecture.md).  Column names reflect the state *after*
# clean_raw_input() runs: hyphens renamed to underscores, 'education' dropped.
# 'fnlgt' matches the actual CSV header (not the legacy 'fnlwgt').
NUMERIC_FEATURES: list[str] = [
    "age",
    "fnlgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_FEATURES: list[str] = [
    "workclass",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


# ── Legacy functions — kept for backward compatibility ─────────────────────────


def clean_census_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and replace '?' with pd.NA.

    Used by the existing model.py / process_data pipeline.
    Prefer clean_raw_input() for new code.
    """
    cleaned = dataframe.copy()
    object_columns = cleaned.select_dtypes(include="object").columns

    for column in object_columns:
        cleaned[column] = cleaned[column].str.strip()

    return cleaned.replace("?", pd.NA)


def process_data(
    dataframe: pd.DataFrame,
    categorical_features: Iterable[str],
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Clean and split a dataframe into features and target.

    Used by the existing training script and model.py.
    Prefer the sklearn Pipeline approach for new code.
    """
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


# ── New canonical preprocessing function ──────────────────────────────────────


def clean_raw_input(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic cleaning step — runs identically in train and inference.

    Does NOT fit or learn from the data, so it is safe to call before the
    train/test split and safe to call on single-row API payloads in production.

    Transformations applied (in order):
        1. Strip leading/trailing whitespace from column names.
           census.csv uses ', ' as separator, which adds a leading space to
           every header after the first.
        2. Strip leading/trailing whitespace from string values.
           Same root cause as (1): all categorical values arrive with a
           leading space when loaded with pd.read_csv.
        3. Replace the string '?' with pd.NA.
           Missing values in census.csv are encoded as '?' not NaN, so
           df.isnull() returns 0 for all columns — a silent false-negative.
           Affected columns: workclass (1836), occupation (1843),
           native-country (583).
        4. Rename hyphens to underscores in column names.
           Six columns use hyphens (e.g. 'education-num'), which cause a
           SyntaxError when accessed as df.education-num.
        5. Drop the 'education' column.
           It is ordinal-encoded in 'education-num'; keeping both is redundant
           and wastes model capacity.  'education' is dropped here rather than
           in the sklearn Pipeline so that the same function works for both
           full CSV files and single-row payloads that may not include it.

    Args:
        df: Raw DataFrame — either from pd.read_csv or from an API payload.

    Returns:
        Cleaned DataFrame with Python-safe column names and pd.NA for missings.
    """
    logger.info("clean_raw_input started | input_shape={}", df.shape)
    result = df.copy()

    # ── Step 1: strip column names ───────────────────────────────────────────
    original_cols = list(result.columns)
    result.columns = result.columns.str.strip()
    n_header_stripped = sum(
        b != a for b, a in zip(original_cols, result.columns, strict=True)
    )
    logger.debug(
        "Step 1 — headers stripped | columns_with_spaces={}",
        n_header_stripped,
    )

    # ── Step 2 + 3: strip values and replace '?' → pd.NA ────────────────────
    object_cols = result.select_dtypes(include="object").columns.tolist()
    for col in object_cols:
        result[col] = result[col].str.strip().replace("?", pd.NA)

    missing_counts = result[object_cols].isna().sum()
    missing_found = missing_counts[missing_counts > 0].to_dict()
    if missing_found:
        logger.warning(
            "Step 2/3 — '?' missings detected after replacement | {}",
            missing_found,
        )
    else:
        logger.debug(
            "Step 2/3 — values stripped | object_cols={} | no '?' missings found",
            len(object_cols),
        )

    # ── Step 4: hyphens → underscores ───────────────────────────────────────
    hyphen_cols = [c for c in result.columns if "-" in c]
    result.columns = result.columns.str.replace("-", "_", regex=False)
    logger.debug(
        "Step 4 — hyphens renamed to underscores | affected_columns={}",
        hyphen_cols,
    )

    # ── Step 5: drop 'education' (redundant with 'education_num') ────────────
    if "education" in result.columns:
        result = result.drop(columns=["education"])
        logger.debug(
            "Step 5 — 'education' dropped | reason=redundant with education_num"
        )
    else:
        logger.debug("Step 5 — 'education' not present, skipped")

    logger.info(
        "clean_raw_input complete | output_shape={} | columns_dropped=education",
        result.shape,
    )
    return result


def save_cleaned_data(
    df: pd.DataFrame,
    output_path: Path | str | None = None,
) -> Path:
    """Persist a cleaned DataFrame to CSV.

    Args:
        df:          Cleaned DataFrame (output of clean_raw_input).
        output_path: Destination path.  Defaults to
                     data/data_cleaned/census_cleaned.csv relative to the
                     project root.

    Returns:
        Resolved Path where the file was written.
    """
    dest = Path(output_path) if output_path else DATA_CLEANED_DIR / "census_cleaned.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "save_cleaned_data started | dest={} | shape={}",
        dest,
        df.shape,
    )
    df.to_csv(dest, index=False)
    logger.info(
        "save_cleaned_data complete | rows={} | columns={} | path={}",
        df.shape[0],
        df.shape[1],
        dest,
    )
    return dest.resolve()


if __name__ == "__main__":
    from census.configure_logging import configure_logging

    configure_logging(LOG_DIR, "data")

    raw_path = PROJECT_ROOT / "data" / "data_raw" / "census.csv"
    logger.info("Loading raw data | path={}", raw_path)
    df_raw = pd.read_csv(raw_path)
    logger.info("Raw data loaded | shape={}", df_raw.shape)

    df_clean = clean_raw_input(df_raw)
    output_path = save_cleaned_data(df_clean)
    logger.info("Pipeline finished | output={}", output_path)
