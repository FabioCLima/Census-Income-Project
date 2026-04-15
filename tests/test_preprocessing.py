"""Tests for census.preprocessing.

Test design principles:
    - Unit tests use minimal synthetic DataFrames that mirror the schema of
      census_cleaned.csv — no file I/O, no side-effects, fast.
    - Integration tests load the real census_cleaned.csv; they are marked with
      @pytest.mark.integration and skipped automatically when the file is absent
      (CI environments without data artefacts).
    - Each test class targets exactly one function.
    - Numeric assertions use explicit tolerances where floating-point is involved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from census.data_loader import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from census.preprocessing import (
    DATA_CLEANED_PATH,
    POSITIVE_CLASS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    build_preprocessor,
    fit_preprocessor,
    load_cleaned_data,
    split_features_target,
    split_train_test,
    validate_feature_schema,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

_CLEANED_CSV_EXISTS = DATA_CLEANED_PATH.exists()

integration = pytest.mark.skipif(
    not _CLEANED_CSV_EXISTS,
    reason=f"Census cleaned CSV not found at {DATA_CLEANED_PATH}",
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

_N_ROWS = 20


@pytest.fixture
def cleaned_df() -> pd.DataFrame:
    """Minimal synthetic DataFrame that mirrors the schema of census_cleaned.csv.

    Contains all 13 feature columns + 'salary', plus one null in each of the
    three columns that have real missings (workclass, occupation, native_country).
    Uses the same value types and categories as the real dataset.
    """
    rng = np.random.default_rng(0)
    n = _N_ROWS

    return pd.DataFrame(
        {
            "age": rng.integers(18, 90, size=n),
            "workclass": (
                ["Private"] * (n - 1) + [None]  # 1 null
            ),
            "fnlgt": rng.integers(10_000, 500_000, size=n),
            "education_num": rng.integers(1, 16, size=n),
            "marital_status": ["Never-married", "Married-civ-spouse"] * (n // 2),
            "occupation": (
                [None] + ["Adm-clerical"] * (n - 1)  # 1 null
            ),
            "relationship": ["Husband", "Not-in-family"] * (n // 2),
            "race": ["White"] * n,
            "sex": ["Male", "Female"] * (n // 2),
            "capital_gain": [0] * (n - 2) + [5000, 15000],
            "capital_loss": [0] * n,
            "hours_per_week": rng.integers(10, 80, size=n),
            "native_country": (
                ["United-States"] * (n - 1) + [None]  # 1 null
            ),
            "salary": (
                ["<=50K"] * (n - 5) + [">50K"] * 5  # 25% positive
            ),
        }
    )


@pytest.fixture
def X_y(cleaned_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return split_features_target(cleaned_df)


@pytest.fixture
def features_only(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix only (salary column removed)."""
    features, _ = split_features_target(cleaned_df)
    return features


@pytest.fixture
def train_test_splits(
    X_y: tuple[pd.DataFrame, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X, y = X_y
    return split_train_test(X, y)


# ── TestLoadCleanedData ───────────────────────────────────────────────────────


class TestLoadCleanedData:
    def test_raises_when_file_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Cleaned data not found"):
            load_cleaned_data(tmp_path / "nonexistent.csv")

    @integration
    def test_returns_dataframe(self) -> None:
        df = load_cleaned_data()
        assert isinstance(df, pd.DataFrame)

    @integration
    def test_shape_matches_census(self) -> None:
        df = load_cleaned_data()
        # 32 561 rows, 14 columns (13 features + salary)
        assert df.shape == (32_561, 14)

    @integration
    def test_expected_columns_present(self) -> None:
        df = load_cleaned_data()
        expected = set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES) | {TARGET_COLUMN}
        assert expected.issubset(set(df.columns))

    @integration
    def test_no_hyphenated_column_names(self) -> None:
        df = load_cleaned_data()
        assert all("-" not in col for col in df.columns)

    @integration
    def test_education_column_absent(self) -> None:
        """'education' was dropped by clean_raw_input — must not reappear."""
        df = load_cleaned_data()
        assert "education" not in df.columns


# ── TestSplitFeaturesTarget ───────────────────────────────────────────────────


class TestSplitFeaturesTarget:
    def test_target_not_in_X(self, cleaned_df: pd.DataFrame) -> None:
        X, _ = split_features_target(cleaned_df)
        assert TARGET_COLUMN not in X.columns

    def test_y_name_is_salary(self, cleaned_df: pd.DataFrame) -> None:
        _, y = split_features_target(cleaned_df)
        assert y.name == TARGET_COLUMN

    def test_y_contains_only_zero_and_one(self, cleaned_df: pd.DataFrame) -> None:
        _, y = split_features_target(cleaned_df)
        assert set(y.unique()).issubset({0, 1})

    def test_positive_class_encoded_as_one(self, cleaned_df: pd.DataFrame) -> None:
        """Rows where salary == '>50K' must map to 1."""
        _, y = split_features_target(cleaned_df)
        positive_mask = cleaned_df[TARGET_COLUMN] == POSITIVE_CLASS
        assert (y[positive_mask] == 1).all()

    def test_negative_class_encoded_as_zero(self, cleaned_df: pd.DataFrame) -> None:
        """Rows where salary == '<=50K' must map to 0."""
        _, y = split_features_target(cleaned_df)
        negative_mask = cleaned_df[TARGET_COLUMN] != POSITIVE_CLASS
        assert (y[negative_mask] == 0).all()

    def test_row_count_preserved(self, cleaned_df: pd.DataFrame) -> None:
        X, y = split_features_target(cleaned_df)
        assert len(X) == len(cleaned_df)
        assert len(y) == len(cleaned_df)

    def test_X_column_count(self, cleaned_df: pd.DataFrame) -> None:
        X, _ = split_features_target(cleaned_df)
        assert X.shape[1] == cleaned_df.shape[1] - 1

    def test_raises_when_salary_missing(self) -> None:
        df = pd.DataFrame({"age": [39], "workclass": ["Private"]})
        with pytest.raises(KeyError, match="salary"):
            split_features_target(df)

    def test_does_not_mutate_input(self, cleaned_df: pd.DataFrame) -> None:
        original_columns = list(cleaned_df.columns)
        split_features_target(cleaned_df)
        assert list(cleaned_df.columns) == original_columns

    @integration
    def test_positive_rate_on_real_data(self) -> None:
        """Real dataset has ~24.1% positive class."""
        df = load_cleaned_data()
        _, y = split_features_target(df)
        assert abs(y.mean() - 0.241) < 0.005


# ── TestSplitTrainTest ────────────────────────────────────────────────────────


class TestSplitTrainTest:
    def test_test_size_proportion(
        self,
        X_y: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = X_y
        X_train, X_test, _, _ = split_train_test(X, y)
        actual_test_ratio = len(X_test) / (len(X_train) + len(X_test))
        assert abs(actual_test_ratio - TEST_SIZE) < 0.05

    def test_no_row_loss(
        self,
        X_y: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = X_y
        X_train, X_test, y_train, y_test = split_train_test(X, y)
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)

    def test_X_and_y_shapes_aligned(
        self,
        X_y: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = X_y
        X_train, X_test, y_train, y_test = split_train_test(X, y)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

    def test_no_overlap_between_train_and_test(
        self,
        X_y: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = X_y
        X_train, X_test, _, _ = split_train_test(X, y)
        assert len(set(X_train.index) & set(X_test.index)) == 0

    def test_reproducibility(
        self,
        X_y: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = X_y
        split_1 = split_train_test(X, y, random_state=RANDOM_STATE)
        split_2 = split_train_test(X, y, random_state=RANDOM_STATE)
        pd.testing.assert_frame_equal(split_1[0], split_2[0])
        pd.testing.assert_frame_equal(split_1[1], split_2[1])

    def test_different_seeds_produce_different_splits(
        self,
        X_y: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = X_y
        X_train_a, *_ = split_train_test(X, y, random_state=0)
        X_train_b, *_ = split_train_test(X, y, random_state=99)
        assert not X_train_a.index.equals(X_train_b.index)

    @integration
    def test_stratification_on_real_data(self) -> None:
        """Train and test positive rates must be within 0.5 p.p. of each other."""
        df = load_cleaned_data()
        X, y = split_features_target(df)
        _, _, y_train, y_test = split_train_test(X, y)
        assert abs(y_train.mean() - y_test.mean()) < 0.005

    @integration
    def test_train_test_sizes_on_real_data(self) -> None:
        df = load_cleaned_data()
        X, y = split_features_target(df)
        X_train, X_test, _, _ = split_train_test(X, y)
        assert X_train.shape == (26_048, 13)
        assert X_test.shape == (6_513, 13)


# ── TestBuildPreprocessor ─────────────────────────────────────────────────────


class TestBuildPreprocessor:
    def test_returns_column_transformer(self) -> None:
        assert isinstance(build_preprocessor(), ColumnTransformer)

    def test_has_two_transformers(self) -> None:
        preprocessor = build_preprocessor()
        assert len(preprocessor.transformers) == 2

    def test_numeric_transformer_name(self) -> None:
        preprocessor = build_preprocessor()
        names = [name for name, _, _ in preprocessor.transformers]
        assert "num" in names

    def test_categorical_transformer_name(self) -> None:
        preprocessor = build_preprocessor()
        names = [name for name, _, _ in preprocessor.transformers]
        assert "cat" in names

    def test_numeric_pipeline_contains_imputer(self) -> None:
        preprocessor = build_preprocessor()
        num_pipe = dict((name, pipe) for name, pipe, _ in preprocessor.transformers)[
            "num"
        ]
        assert isinstance(num_pipe, Pipeline)
        step_types = [type(step) for _, step in num_pipe.steps]
        assert SimpleImputer in step_types

    def test_numeric_pipeline_contains_scaler(self) -> None:
        preprocessor = build_preprocessor()
        num_pipe = dict((name, pipe) for name, pipe, _ in preprocessor.transformers)[
            "num"
        ]
        step_types = [type(step) for _, step in num_pipe.steps]
        assert StandardScaler in step_types

    def test_numeric_imputer_uses_median(self) -> None:
        preprocessor = build_preprocessor()
        num_pipe = dict((name, pipe) for name, pipe, _ in preprocessor.transformers)[
            "num"
        ]
        imputer = dict(num_pipe.steps)["imputer"]
        assert imputer.strategy == "median"

    def test_categorical_pipeline_contains_imputer(self) -> None:
        preprocessor = build_preprocessor()
        cat_pipe = dict((name, pipe) for name, pipe, _ in preprocessor.transformers)[
            "cat"
        ]
        step_types = [type(step) for _, step in cat_pipe.steps]
        assert SimpleImputer in step_types

    def test_categorical_pipeline_contains_encoder(self) -> None:
        preprocessor = build_preprocessor()
        cat_pipe = dict((name, pipe) for name, pipe, _ in preprocessor.transformers)[
            "cat"
        ]
        step_types = [type(step) for _, step in cat_pipe.steps]
        assert OrdinalEncoder in step_types

    def test_categorical_imputer_uses_most_frequent(self) -> None:
        preprocessor = build_preprocessor()
        cat_pipe = dict((name, pipe) for name, pipe, _ in preprocessor.transformers)[
            "cat"
        ]
        imputer = dict(cat_pipe.steps)["imputer"]
        assert imputer.strategy == "most_frequent"

    def test_ordinal_encoder_handles_unknown(self) -> None:
        preprocessor = build_preprocessor()
        cat_pipe = dict((name, pipe) for name, pipe, _ in preprocessor.transformers)[
            "cat"
        ]
        encoder: OrdinalEncoder = dict(cat_pipe.steps)["encoder"]
        assert encoder.handle_unknown == "use_encoded_value"
        assert encoder.unknown_value == -1

    def test_remainder_is_drop(self) -> None:
        preprocessor = build_preprocessor()
        assert preprocessor.remainder == "drop"

    def test_numeric_features_assigned(self) -> None:
        preprocessor = build_preprocessor()
        features_map = {name: cols for name, _, cols in preprocessor.transformers}
        assert features_map["num"] == NUMERIC_FEATURES

    def test_categorical_features_assigned(self) -> None:
        preprocessor = build_preprocessor()
        features_map = {name: cols for name, _, cols in preprocessor.transformers}
        assert features_map["cat"] == CATEGORICAL_FEATURES

    def test_each_call_returns_unfitted_instance(self) -> None:
        """build_preprocessor() must return a fresh unfitted object each call."""
        p1 = build_preprocessor()
        p2 = build_preprocessor()
        assert p1 is not p2


# ── TestFitPreprocessor ───────────────────────────────────────────────────────


class TestFitPreprocessor:
    def test_returns_two_numpy_arrays(
        self,
        train_test_splits: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    ) -> None:
        X_train, X_test, _, _ = train_test_splits
        result = fit_preprocessor(build_preprocessor(), X_train, X_test)
        assert len(result) == 2
        assert all(isinstance(arr, np.ndarray) for arr in result)

    def test_output_column_count_equals_total_features(
        self,
        train_test_splits: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    ) -> None:
        X_train, X_test, _, _ = train_test_splits
        X_train_t, X_test_t = fit_preprocessor(build_preprocessor(), X_train, X_test)
        expected_cols = len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
        assert X_train_t.shape[1] == expected_cols
        assert X_test_t.shape[1] == expected_cols

    def test_train_row_count_preserved(
        self,
        train_test_splits: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    ) -> None:
        X_train, X_test, _, _ = train_test_splits
        X_train_t, _ = fit_preprocessor(build_preprocessor(), X_train, X_test)
        assert X_train_t.shape[0] == len(X_train)

    def test_test_row_count_preserved(
        self,
        train_test_splits: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    ) -> None:
        X_train, X_test, _, _ = train_test_splits
        _, X_test_t = fit_preprocessor(build_preprocessor(), X_train, X_test)
        assert X_test_t.shape[0] == len(X_test)

    def test_no_nan_in_train_output(
        self,
        train_test_splits: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    ) -> None:
        X_train, X_test, _, _ = train_test_splits
        X_train_t, _ = fit_preprocessor(build_preprocessor(), X_train, X_test)
        assert not np.isnan(X_train_t).any()

    def test_no_nan_in_test_output(
        self,
        train_test_splits: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    ) -> None:
        X_train, X_test, _, _ = train_test_splits
        _, X_test_t = fit_preprocessor(build_preprocessor(), X_train, X_test)
        assert not np.isnan(X_test_t).any()

    def test_imputer_fitted_on_train_only(
        self,
        train_test_splits: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    ) -> None:
        """Numeric imputer statistics must be derived solely from X_train.

        Inject a sentinel outlier (999_999) in X_test capital_gain that does
        NOT appear in X_train. The scaler mean and std must reflect X_train only,
        i.e. the standardised sentinel value in X_test must be far from 0.
        """
        X_train, X_test, _, _ = train_test_splits
        X_test_modified = X_test.copy()
        X_test_modified["capital_gain"] = 999_999

        preprocessor = build_preprocessor()
        X_train_t, X_test_t = fit_preprocessor(preprocessor, X_train, X_test_modified)
        # capital_gain is the 4th numeric feature (index 3)
        capital_gain_idx = NUMERIC_FEATURES.index("capital_gain")
        assert (X_test_t[:, capital_gain_idx] > 1).all()

    def test_unknown_category_encoded_as_minus_one(
        self,
        train_test_splits: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    ) -> None:
        """OrdinalEncoder must encode unseen categories as -1, not raise."""
        X_train, X_test, _, _ = train_test_splits
        X_test_unseen = X_test.copy()
        X_test_unseen["workclass"] = "UNSEEN_CATEGORY"

        preprocessor = build_preprocessor()
        _, X_test_t = fit_preprocessor(preprocessor, X_train, X_test_unseen)

        # workclass is the 1st categorical feature (index = len(NUMERIC_FEATURES))
        workclass_idx = len(NUMERIC_FEATURES) + CATEGORICAL_FEATURES.index("workclass")
        assert (X_test_t[:, workclass_idx] == -1).all()

    @integration
    def test_output_shape_on_real_data(self) -> None:
        df = load_cleaned_data()
        X, y = split_features_target(df)
        X_train, X_test, _, _ = split_train_test(X, y)
        X_train_t, X_test_t = fit_preprocessor(build_preprocessor(), X_train, X_test)
        expected_cols = len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
        assert X_train_t.shape == (26_048, expected_cols)
        assert X_test_t.shape == (6_513, expected_cols)

    @integration
    def test_no_nan_on_real_data(self) -> None:
        df = load_cleaned_data()
        X, y = split_features_target(df)
        X_train, X_test, _, _ = split_train_test(X, y)
        X_train_t, X_test_t = fit_preprocessor(build_preprocessor(), X_train, X_test)
        assert not np.isnan(X_train_t).any()
        assert not np.isnan(X_test_t).any()

    @integration
    def test_numeric_columns_are_standardised(self) -> None:
        """After StandardScaler the train columns must have mean ≈ 0 and std ≈ 1."""
        df = load_cleaned_data()
        X, y = split_features_target(df)
        X_train, X_test, _, _ = split_train_test(X, y)
        X_train_t, _ = fit_preprocessor(build_preprocessor(), X_train, X_test)
        numeric_block = X_train_t[:, : len(NUMERIC_FEATURES)]
        assert np.abs(numeric_block.mean(axis=0)).max() < 1e-9
        assert np.abs(numeric_block.std(axis=0) - 1.0).max() < 1e-6


# ── TestValidateFeatureSchema ─────────────────────────────────────────────────


class TestValidateFeatureSchema:
    """Tests for validate_feature_schema(), which guards fit_preprocessor().

    The function enforces four contracts (in order):
        1. All required feature columns are present        → ValueError
        2. Numeric columns have a numeric dtype            → TypeError
        3. Categorical columns have object/string/category → TypeError
        4. A 5-row Pydantic sample matches the schema      → ValueError
    """

    def test_passes_on_valid_features(self, features_only: pd.DataFrame) -> None:
        """Valid feature DataFrame must not raise."""
        validate_feature_schema(features_only, "test_dataset")

    # ── Contract 1: required columns ─────────────────────────────────────────

    def test_raises_value_error_when_column_missing(
        self, features_only: pd.DataFrame
    ) -> None:
        df = features_only.drop(columns=["age"])
        with pytest.raises(ValueError, match="missing required feature columns"):
            validate_feature_schema(df, "test_dataset")

    def test_error_lists_all_missing_columns(self, features_only: pd.DataFrame) -> None:
        df = features_only.drop(columns=["age", "capital_gain"])
        with pytest.raises(ValueError) as exc_info:
            validate_feature_schema(df, "test_dataset")
        msg = str(exc_info.value)
        assert "age" in msg
        assert "capital_gain" in msg

    def test_dataset_name_appears_in_missing_column_error(
        self, features_only: pd.DataFrame
    ) -> None:
        df = features_only.drop(columns=["fnlgt"])
        with pytest.raises(ValueError, match="sentinel_name"):
            validate_feature_schema(df, "sentinel_name")

    # ── Contract 2: numeric dtypes ────────────────────────────────────────────

    def test_raises_type_error_when_numeric_column_is_string(
        self, features_only: pd.DataFrame
    ) -> None:
        df = features_only.copy()
        df["age"] = df["age"].astype(str)
        with pytest.raises(TypeError, match="non-numeric dtypes"):
            validate_feature_schema(df, "test_dataset")

    def test_error_names_the_offending_numeric_column(
        self, features_only: pd.DataFrame
    ) -> None:
        df = features_only.copy()
        df["education_num"] = df["education_num"].astype(str)
        with pytest.raises(TypeError) as exc_info:
            validate_feature_schema(df, "test_dataset")
        assert "education_num" in str(exc_info.value)

    # ── Contract 3: categorical dtypes ───────────────────────────────────────

    def test_raises_type_error_when_categorical_column_has_numeric_dtype(
        self, features_only: pd.DataFrame
    ) -> None:
        """Replacing a categorical column with a numeric dtype must fail."""
        df = features_only.copy()
        # 'sex' has no nulls — safe to cast to int
        df["sex"] = range(len(features_only))  # int64 dtype, not object/string
        with pytest.raises(TypeError, match="invalid categorical dtypes"):
            validate_feature_schema(df, "test_dataset")

    def test_categorical_dtype_is_accepted(self, features_only: pd.DataFrame) -> None:
        """pd.CategoricalDtype is a valid dtype for categorical columns."""
        df = features_only.copy()
        df["sex"] = pd.Categorical(df["sex"])
        validate_feature_schema(df, "test_dataset")  # must not raise

    def test_error_names_the_offending_categorical_column(
        self, features_only: pd.DataFrame
    ) -> None:
        df = features_only.copy()
        df["race"] = range(len(features_only))
        with pytest.raises(TypeError) as exc_info:
            validate_feature_schema(df, "test_dataset")
        assert "race" in str(exc_info.value)

    # ── Contract 4: Pydantic row-level schema ─────────────────────────────────

    def test_raises_value_error_on_pydantic_schema_violation(
        self, features_only: pd.DataFrame
    ) -> None:
        """Object-dtype column with int values passes dtype check but fails Pydantic.

        workclass: str | None in PreprocessingFeatureRow does not accept int.
        """
        df = features_only.copy()
        # pd.array(..., dtype=object) keeps dtype=object (passes is_object_dtype)
        # but the values are Python ints — Pydantic V2 rejects int for str field.
        df["workclass"] = pd.array([1] * len(features_only), dtype=object)
        with pytest.raises(ValueError, match="failed schema validation"):
            validate_feature_schema(df, "test_dataset")

    def test_pydantic_error_includes_row_index(
        self, features_only: pd.DataFrame
    ) -> None:
        df = features_only.copy()
        df["workclass"] = pd.array([1] * len(features_only), dtype=object)
        with pytest.raises(ValueError, match="row index"):
            validate_feature_schema(df, "test_dataset")

    def test_nulls_in_categorical_columns_pass_pydantic(
        self, features_only: pd.DataFrame
    ) -> None:
        """None values in nullable categorical columns are explicitly allowed."""
        df = features_only.copy()
        df.loc[df.index[0], "workclass"] = None
        validate_feature_schema(df, "test_dataset")  # must not raise

    # ── Extra columns (warning, no exception) ────────────────────────────────

    def test_does_not_raise_on_extra_columns(self, features_only: pd.DataFrame) -> None:
        df = features_only.copy()
        df["unexpected_column"] = 42
        validate_feature_schema(df, "test_dataset")  # must not raise

    # ── Integration ───────────────────────────────────────────────────────────

    @integration
    def test_passes_on_real_train_split(self) -> None:
        df = load_cleaned_data()
        features, target = split_features_target(df)
        features_train, features_test, _, _ = split_train_test(features, target)
        validate_feature_schema(features_train, "features_train")
        validate_feature_schema(features_test, "features_test")
