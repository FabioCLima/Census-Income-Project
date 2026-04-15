"""Unit tests for census.data — clean_raw_input() and save_cleaned_data().

Test design principles:
    - Each test targets exactly one behaviour (single assertion of intent).
    - Fixtures create the minimum DataFrame needed — no full 15-column tables
      unless the test is explicitly about whole-dataset behaviour.
    - File-output tests use pytest's tmp_path fixture so the real
      data/data_cleaned/ directory is never touched during CI.
    - The existing tests for clean_census_data() and process_data() live in
      tests/test_model.py and are not duplicated here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from census.data_loader import clean_raw_input, save_cleaned_data

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def raw_csv_df() -> pd.DataFrame:
    """Mimics the exact output of pd.read_csv('data/census.csv').

    census.csv uses ', ' (comma + space) as its separator.  This means every
    column after the first arrives with a leading space in both the header and
    the values.  Missing values are encoded as ' ?' (space + question mark).
    """
    return pd.DataFrame(
        {
            "age": [39, 50, 38],
            " workclass": [" State-gov", " Private", " ?"],
            " fnlgt": [77516, 83311, 215646],
            " education": [" Bachelors", " Bachelors", " HS-grad"],
            " education-num": [13, 13, 9],
            " marital-status": [
                " Never-married",
                " Married-civ-spouse",
                " Divorced",
            ],
            " occupation": [" Adm-clerical", " Exec-managerial", " ?"],
            " relationship": [" Not-in-family", " Husband", " Not-in-family"],
            " race": [" White", " White", " White"],
            " sex": [" Male", " Male", " Male"],
            " capital-gain": [2174, 0, 0],
            " capital-loss": [0, 0, 0],
            " hours-per-week": [40, 13, 40],
            " native-country": [
                " United-States",
                " United-States",
                " United-States",
            ],
            " salary": [" <=50K", " <=50K", " <=50K"],
        }
    )


@pytest.fixture
def api_payload_df() -> pd.DataFrame:
    """Mimics a single-row payload received by the FastAPI /predict endpoint.

    API payloads arrive already clean: no leading spaces in keys or values,
    no '?' — clean_raw_input() must be idempotent on this input.
    """
    return pd.DataFrame(
        {
            "age": [39],
            "workclass": ["State-gov"],
            "fnlgt": [77516],
            "education": ["Bachelors"],
            "education-num": [13],
            "marital-status": ["Never-married"],
            "occupation": ["Adm-clerical"],
            "relationship": ["Not-in-family"],
            "race": ["White"],
            "sex": ["Male"],
            "capital-gain": [2174],
            "capital-loss": [0],
            "hours-per-week": [40],
            "native-country": ["United-States"],
        }
    )


# ── Tests: clean_raw_input — Step 1: strip column names ───────────────────────


class TestStripColumnNames:
    def test_removes_leading_space_from_headers(self) -> None:
        df = pd.DataFrame({" workclass": ["Private"], " age": [39]})
        result = clean_raw_input(df)
        assert "workclass" in result.columns
        assert "age" in result.columns

    def test_original_spaced_headers_no_longer_present(self) -> None:
        df = pd.DataFrame({" workclass": ["Private"]})
        result = clean_raw_input(df)
        assert " workclass" not in result.columns

    def test_headers_already_clean_are_unchanged(self) -> None:
        df = pd.DataFrame({"workclass": ["Private"], "age": [39]})
        result = clean_raw_input(df)
        assert list(result.columns) == ["workclass", "age"]


# ── Tests: clean_raw_input — Step 2+3: strip values and replace '?' → NaN ─────


class TestStripValuesAndReplaceMissing:
    def test_strips_leading_space_from_string_values(self) -> None:
        df = pd.DataFrame({"workclass": [" State-gov", " Private"]})
        result = clean_raw_input(df)
        assert result["workclass"].tolist() == ["State-gov", "Private"]

    def test_question_mark_becomes_nan(self) -> None:
        df = pd.DataFrame({"workclass": ["?", "Private"]})
        result = clean_raw_input(df)
        assert pd.isna(result["workclass"].iloc[0])
        assert result["workclass"].iloc[1] == "Private"

    def test_spaced_question_mark_becomes_nan(self) -> None:
        """' ?' is how '?' arrives after pd.read_csv with the census CSV format."""
        df = pd.DataFrame({" workclass": [" ?", " Private"]})
        result = clean_raw_input(df)
        assert pd.isna(result["workclass"].iloc[0])

    def test_non_missing_values_are_not_nullified(self) -> None:
        df = pd.DataFrame({"workclass": ["Private", "State-gov"]})
        result = clean_raw_input(df)
        assert result["workclass"].notna().all()

    def test_numeric_columns_are_not_affected(self) -> None:
        df = pd.DataFrame({"age": [39, 50], "capital-gain": [2174, 0]})
        result = clean_raw_input(df)
        assert result["age"].tolist() == [39, 50]

    def test_workclass_missing_count(self, raw_csv_df: pd.DataFrame) -> None:
        result = clean_raw_input(raw_csv_df)
        assert result["workclass"].isna().sum() == 1

    def test_occupation_missing_count(self, raw_csv_df: pd.DataFrame) -> None:
        result = clean_raw_input(raw_csv_df)
        assert result["occupation"].isna().sum() == 1


# ── Tests: clean_raw_input — Step 4: rename hyphens to underscores ────────────


class TestRenameHyphens:
    def test_education_num_renamed(self) -> None:
        df = pd.DataFrame({"education-num": [13]})
        result = clean_raw_input(df)
        assert "education_num" in result.columns
        assert "education-num" not in result.columns

    def test_marital_status_renamed(self) -> None:
        df = pd.DataFrame({"marital-status": ["Never-married"]})
        result = clean_raw_input(df)
        assert "marital_status" in result.columns

    def test_all_six_hyphen_columns_renamed(self) -> None:
        hyphen_cols = {
            "education-num": [13],
            "marital-status": ["Never-married"],
            "capital-gain": [0],
            "capital-loss": [0],
            "hours-per-week": [40],
            "native-country": ["United-States"],
        }
        df = pd.DataFrame(hyphen_cols)
        result = clean_raw_input(df)
        for original in hyphen_cols:
            assert original not in result.columns
            assert original.replace("-", "_") in result.columns

    def test_hyphens_in_values_are_not_renamed(self) -> None:
        """'Never-married' is a value, not a column name — must be preserved."""
        df = pd.DataFrame({"marital-status": ["Never-married", "Married-civ-spouse"]})
        result = clean_raw_input(df)
        assert "Never-married" in result["marital_status"].values
        assert "Married-civ-spouse" in result["marital_status"].values


# ── Tests: clean_raw_input — Step 5: drop 'education' ─────────────────────────


class TestDropEducation:
    def test_education_column_is_dropped(self) -> None:
        df = pd.DataFrame({"education": ["Bachelors"], "education-num": [13]})
        result = clean_raw_input(df)
        assert "education" not in result.columns

    def test_education_num_is_kept(self) -> None:
        df = pd.DataFrame({"education": ["Bachelors"], "education-num": [13]})
        result = clean_raw_input(df)
        assert "education_num" in result.columns

    def test_no_error_when_education_absent(self) -> None:
        """Inference payloads may not include 'education' — must not raise."""
        df = pd.DataFrame({"workclass": ["Private"], "education-num": [13]})
        result = clean_raw_input(df)
        assert "education_num" in result.columns


# ── Tests: clean_raw_input — immutability and shape ───────────────────────────


class TestImmutabilityAndShape:
    def test_does_not_mutate_input_dataframe(self) -> None:
        df = pd.DataFrame({" workclass": [" Private"], " education": [" Bachelors"]})
        original_columns = list(df.columns)
        original_values = df[" workclass"].tolist()
        clean_raw_input(df)
        assert list(df.columns) == original_columns
        assert df[" workclass"].tolist() == original_values

    def test_output_has_one_fewer_column_than_input_with_education(
        self, raw_csv_df: pd.DataFrame
    ) -> None:
        """15 columns in (including education) → 14 columns out."""
        result = clean_raw_input(raw_csv_df)
        assert result.shape[1] == raw_csv_df.shape[1] - 1

    def test_output_row_count_unchanged(self, raw_csv_df: pd.DataFrame) -> None:
        result = clean_raw_input(raw_csv_df)
        assert result.shape[0] == raw_csv_df.shape[0]

    def test_no_leading_spaces_in_any_string_value(
        self, raw_csv_df: pd.DataFrame
    ) -> None:
        result = clean_raw_input(raw_csv_df)
        for col in result.select_dtypes(include="object").columns:
            non_null = result[col].dropna()
            has_leading_space = non_null.str.startswith(" ").any()
            assert not has_leading_space, (
                f"Column '{col}' still contains values with a leading space"
            )


# ── Tests: clean_raw_input — inference / API payload ──────────────────────────


class TestInferencePayload:
    def test_api_payload_values_are_not_altered(
        self, api_payload_df: pd.DataFrame
    ) -> None:
        result = clean_raw_input(api_payload_df)
        assert result["workclass"].iloc[0] == "State-gov"
        assert result["age"].iloc[0] == 39
        assert result["capital_gain"].iloc[0] == 2174

    def test_api_payload_education_is_dropped(
        self, api_payload_df: pd.DataFrame
    ) -> None:
        result = clean_raw_input(api_payload_df)
        assert "education" not in result.columns

    def test_api_payload_columns_have_underscores_not_hyphens(
        self, api_payload_df: pd.DataFrame
    ) -> None:
        result = clean_raw_input(api_payload_df)
        assert all("-" not in col for col in result.columns)


# ── Tests: save_cleaned_data ───────────────────────────────────────────────────


class TestSaveCleanedData:
    def test_creates_output_file(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"age": [39], "workclass": ["Private"]})
        dest = tmp_path / "census_cleaned.csv"
        save_cleaned_data(df, output_path=dest)
        assert dest.exists()

    def test_returns_resolved_path(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"age": [39]})
        dest = tmp_path / "out.csv"
        result = save_cleaned_data(df, output_path=dest)
        assert result == dest.resolve()

    def test_saved_csv_row_count_matches_input(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"age": [39, 50, 38], "workclass": ["A", "B", "C"]})
        dest = tmp_path / "census_cleaned.csv"
        save_cleaned_data(df, output_path=dest)
        loaded = pd.read_csv(dest)
        assert len(loaded) == len(df)

    def test_saved_csv_column_names_match_input(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"age": [39], "education_num": [13], "capital_gain": [0]})
        dest = tmp_path / "census_cleaned.csv"
        save_cleaned_data(df, output_path=dest)
        loaded = pd.read_csv(dest)
        assert list(loaded.columns) == list(df.columns)

    def test_saved_csv_values_match_input(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"age": [39, 50], "workclass": ["Private", "Gov"]})
        dest = tmp_path / "census_cleaned.csv"
        save_cleaned_data(df, output_path=dest)
        loaded = pd.read_csv(dest)
        assert loaded["age"].tolist() == [39, 50]
        assert loaded["workclass"].tolist() == ["Private", "Gov"]

    def test_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"age": [39]})
        dest = tmp_path / "nested" / "subdir" / "census_cleaned.csv"
        save_cleaned_data(df, output_path=dest)
        assert dest.exists()

    def test_full_pipeline_round_trip(
        self, raw_csv_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """clean_raw_input → save_cleaned_data → pd.read_csv produces valid data."""
        cleaned = clean_raw_input(raw_csv_df)
        dest = tmp_path / "census_cleaned.csv"
        save_cleaned_data(cleaned, output_path=dest)
        reloaded = pd.read_csv(dest)
        assert reloaded.shape == cleaned.shape
        assert list(reloaded.columns) == list(cleaned.columns)
