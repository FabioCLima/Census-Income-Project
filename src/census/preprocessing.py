"""Census Income — preprocessing pipeline.

Implements Phases 3.5, 4, and 5 from docs/pipeline_architecture.md:

    Phase 3.5  split_features_target()  — separate X / y, encode target as 0/1
    Phase 4    split_train_test()        — stratified 80/20 split
    Phase 5    build_preprocessor()      — ColumnTransformer (numeric + categorical)
               fit_preprocessor()        — fit on X_train only, then transform both splits
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import ValidationError
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from census.data_loader import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN
from census.schemas import PreprocessingFeatureRow

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CLEANED_PATH = _PROJECT_ROOT / "data" / "data_cleaned" / "census_cleaned.csv"
LOG_DIR = _PROJECT_ROOT / "logs" / "preprocessing_log"

# ── Split constants ────────────────────────────────────────────────────────────
TEST_SIZE: float = 0.20
RANDOM_STATE: int = 42
POSITIVE_CLASS: str = ">50K"


# ── Phase 3.5 ─────────────────────────────────────────────────────────────────


def load_cleaned_data(path: Path | str = DATA_CLEANED_PATH) -> pd.DataFrame:
    """Load the cleaned CSV produced by clean_raw_input() + save_cleaned_data().

    Args:
        path: Path to the cleaned CSV file. Defaults to
              data/data_cleaned/census_cleaned.csv relative to project root.

    Returns:
        DataFrame with 14 columns (no 'education', no hyphens in names).

    Raises:
        FileNotFoundError: if the cleaned CSV does not exist at *path*.
    """
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at '{resolved}'. "
            "Run data_loader.__main__ (or save_cleaned_data) first."
        )
    logger.info("load_cleaned_data | path={}", resolved)
    df = pd.read_csv(resolved)
    logger.info("load_cleaned_data complete | shape={}", df.shape)
    return df


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix and target vector; encode target as binary int.

    Corresponds to Phase 3.5 in docs/pipeline_architecture.md.

    Encoding:
        >50K  → 1  (positive class — the minority we care about)
        <=50K → 0  (negative class)

    Args:
        df: Cleaned DataFrame that includes the 'salary' column.

    Returns:
        features: Feature matrix — all columns except 'salary'.
        target:   Binary integer Series (0 / 1) named 'salary'.

    Raises:
        KeyError: if 'salary' is not present in *df*.
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    features = df.drop(columns=[TARGET_COLUMN])
    target = (df[TARGET_COLUMN] == POSITIVE_CLASS).astype(int).rename(TARGET_COLUMN)

    positive_rate = target.mean()
    logger.info(
        "split_features_target | features={} | target={} | positive_rate={:.1%}",
        features.shape,
        target.shape,
        positive_rate,
    )
    return features, target


# ── Phase 4 ───────────────────────────────────────────────────────────────────


def split_train_test(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train / test split that preserves the class ratio.

    Corresponds to Phase 4 in docs/pipeline_architecture.md.

    The census dataset is imbalanced (≈3.2:1). Without stratification a random
    split can produce slightly different positive-class ratios in train and test,
    making evaluation metrics less comparable. ``stratify=target`` guarantees
    both sets reflect the original 24.1% / 75.9% distribution.

    IMPORTANT: the preprocessor (ColumnTransformer) must be fitted ONLY on
    features_train and then applied to features_test. Fitting on the full
    dataset before the split constitutes data leakage.

    Args:
        features:     Feature matrix (output of split_features_target).
        target:       Binary target Series (output of split_features_target).
        test_size:    Fraction of data to allocate to the test set. Default 0.20.
        random_state: Seed for reproducibility. Default 42.

    Returns:
        features_train, features_test, target_train, target_test
    """
    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    logger.info(
        "split_train_test | train={} | test={} | "
        "target_train_positive_rate={:.1%} | target_test_positive_rate={:.1%}",
        features_train.shape,
        features_test.shape,
        target_train.mean(),
        target_test.mean(),
    )
    return features_train, features_test, target_train, target_test


# ── Phase 5 ───────────────────────────────────────────────────────────────────


def validate_feature_schema(features: pd.DataFrame, dataset_name: str) -> None:
    """Validate cleaned feature schema before preprocessing.

    Checks:
        1) required feature columns are present;
        2) numeric features have numeric dtype;
        3) categorical features are string/object/category dtypes;
        4) a small sample of rows validates against the Pydantic schema.
    """
    expected_columns = set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    actual_columns = set(features.columns)

    missing_columns = sorted(expected_columns - actual_columns)
    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required feature columns: {missing_columns}"
        )

    extra_columns = sorted(actual_columns - expected_columns)
    if extra_columns:
        logger.warning(
            "validate_feature_schema | dataset={} | extra_columns_ignored={}",
            dataset_name,
            extra_columns,
        )

    non_numeric_columns = [
        col
        for col in NUMERIC_FEATURES
        if not pd.api.types.is_numeric_dtype(features[col])
    ]
    if non_numeric_columns:
        found_dtypes = {
            col: str(features[col].dtype) for col in non_numeric_columns
        }
        raise TypeError(
            f"{dataset_name} has non-numeric dtypes in numeric features: {found_dtypes}"
        )

    invalid_categorical_columns = [
        col
        for col in CATEGORICAL_FEATURES
        if not (
            pd.api.types.is_object_dtype(features[col])
            or pd.api.types.is_string_dtype(features[col])
            or isinstance(features[col].dtype, pd.CategoricalDtype)
        )
    ]
    if invalid_categorical_columns:
        found_dtypes = {
            col: str(features[col].dtype) for col in invalid_categorical_columns
        }
        raise TypeError(
            f"{dataset_name} has invalid categorical dtypes: {found_dtypes}"
        )

    sample_size = min(5, len(features))
    for idx, row in features.head(sample_size).iterrows():
        # pandas encodes missings as np.nan (a float), but PreprocessingFeatureRow
        # uses `str | None` / `int | float | None`. Pydantic rejects np.nan for
        # str fields, so we normalise all NaN-like values to None first.
        record = {
            k: (None if pd.isna(v) else v)
            for k, v in row[NUMERIC_FEATURES + CATEGORICAL_FEATURES].to_dict().items()
        }
        try:
            PreprocessingFeatureRow.model_validate(record)
        except ValidationError as exc:
            raise ValueError(
                f"{dataset_name} failed schema validation at row index {idx}: {exc}"
            ) from exc

    logger.info(
        "validate_feature_schema | dataset={} | rows={} | cols={} | status=ok",
        dataset_name,
        features.shape[0],
        features.shape[1],
    )


def build_preprocessor() -> ColumnTransformer:
    """Build the sklearn ColumnTransformer for the census feature schema.

    Corresponds to Phase 5 in docs/pipeline_architecture.md.

    Sub-pipelines
    -------------
    Numeric (age, fnlgt, education_num, capital_gain, capital_loss, hours_per_week):
        1. SimpleImputer(strategy="median")
           Median is robust to the extreme outliers in capital_gain / capital_loss.
           No numeric column has observed nulls in the cleaned CSV, but the imputer
           makes the pipeline safe against unseen production payloads.
        2. StandardScaler()
           Required by linear models; neutral for tree-based models.

    Categorical (workclass, marital_status, occupation, relationship, race, sex,
                 native_country):
        1. SimpleImputer(strategy="most_frequent")
           workclass and occupation each have ≈5.6% nulls; native_country has
           ≈1.8%. "most_frequent" avoids introducing an "Unknown" category that
           the OrdinalEncoder may not have seen during training.
        2. OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
           Chosen over OneHotEncoder to keep dimensionality manageable —
           native_country alone has 41 unique values. Tree-based models handle
           ordinal encodings well. unknown_value=-1 ensures unseen categories at
           inference time do not raise an exception.

    remainder="drop" — any column not listed in NUMERIC_FEATURES or
    CATEGORICAL_FEATURES is silently excluded, making the pipeline robust to
    extra columns that may arrive in production payloads.

    Returns:
        Unfitted ColumnTransformer ready to be included in a sklearn Pipeline
        or used directly via fit() / transform().
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    logger.debug(
        "build_preprocessor | numeric_features={} | categorical_features={}",
        len(NUMERIC_FEATURES),
        len(CATEGORICAL_FEATURES),
    )
    return preprocessor


def fit_preprocessor(
    preprocessor: ColumnTransformer,
    features_train: pd.DataFrame,
    features_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the preprocessor on features_train only; transform both splits.

    This separation is the key anti-leakage guard: imputer statistics (medians,
    most-frequent values) and scaler parameters (mean, std) are learned
    exclusively from the training set. features_test is transformed using those
    parameters — exactly what happens at inference time.

    Args:
        preprocessor:   Unfitted ColumnTransformer (from build_preprocessor).
        features_train: Training feature matrix.
        features_test:  Test feature matrix.

    Returns:
        train_array: numpy array with shape (n_train, n_features_out).
        test_array:  numpy array with shape (n_test,  n_features_out).
    """
    logger.info(
        "fit_preprocessor | fitting on features_train={} | "
        "will transform features_test={}",
        features_train.shape,
        features_test.shape,
    )
    validate_feature_schema(features_train, "features_train")
    validate_feature_schema(features_test, "features_test")

    # ColumnTransformer.fit_transform / transform return ndarray | spmatrix.
    # All sub-transformers here (StandardScaler, OrdinalEncoder) produce dense
    # output, so the result is always ndarray. cast() communicates this to the
    # type-checker without a runtime overhead.
    train_array = cast(np.ndarray, preprocessor.fit_transform(features_train))
    test_array = cast(np.ndarray, preprocessor.transform(features_test))

    logger.info(
        "fit_preprocessor complete | n_features_out={} | "
        "train_array={} | test_array={}",
        train_array.shape[1],
        train_array.shape,
        test_array.shape,
    )
    return train_array, test_array


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from census.configure_logging import configure_logging

    configure_logging(LOG_DIR, "preprocessing")

    df = load_cleaned_data()
    features, target = split_features_target(df)
    features_train, features_test, target_train, target_test = split_train_test(
        features, target
    )
    preprocessor = build_preprocessor()
    train_array, test_array = fit_preprocessor(
        preprocessor, features_train, features_test
    )

    logger.info(
        "Pipeline smoke-test complete | "
        "train_array={} | test_array={} | features_out={}",
        train_array.shape,
        test_array.shape,
        train_array.shape[1],
    )
