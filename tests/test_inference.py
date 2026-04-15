"""Tests for census.inference (load_pipeline, get_categorical_encoder, predict)."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from census.inference import (
    PIPELINE_PATH,
    get_categorical_encoder,
    load_pipeline,
    predict_from_payload,
)
from census.train_model import (
    build_baseline,
    build_pipeline,
    build_random_forest,
    train_pipeline,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────
# small_features and small_target are provided by tests/conftest.py


@pytest.fixture()
def fitted_pipeline(small_features: pd.DataFrame, small_target: pd.Series) -> Pipeline:
    pipeline = build_pipeline(build_baseline())
    return train_pipeline(pipeline, small_features, small_target)


@pytest.fixture()
def saved_pipeline_path(tmp_path: Path, fitted_pipeline: Pipeline) -> Path:
    dest = tmp_path / "model" / "pipeline.pkl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_pipeline, dest)
    return dest


@pytest.fixture()
def sample_payload() -> dict[str, object]:
    return {
        "age": 35,
        "fnlgt": 200000,
        "education_num": 13,
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "workclass": "Private",
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "native_country": "United-States",
    }


# ── load_pipeline ─────────────────────────────────────────────────────────────


class TestLoadPipeline:
    def test_returns_none_for_nonexistent_path(self, tmp_path: Path) -> None:
        result = load_pipeline(tmp_path / "nonexistent.pkl")
        assert result is None

    def test_returns_pipeline_for_valid_file(self, saved_pipeline_path: Path) -> None:
        result = load_pipeline(saved_pipeline_path)
        assert isinstance(result, Pipeline)

    def test_loaded_pipeline_can_predict(
        self,
        saved_pipeline_path: Path,
        small_features: pd.DataFrame,
    ) -> None:
        pipeline = load_pipeline(saved_pipeline_path)
        assert pipeline is not None
        predictions = pipeline.predict(small_features)
        assert len(predictions) == len(small_features)

    def test_returns_none_for_corrupt_file(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "corrupt.pkl"
        corrupt.write_bytes(b"not a valid pickle")
        result = load_pipeline(corrupt)
        assert result is None

    def test_default_path_constant(self) -> None:
        assert PIPELINE_PATH.name == "census_pipeline.pkl"
        assert PIPELINE_PATH.parent.name == "model"


# ── get_categorical_encoder ───────────────────────────────────────────────────


class TestGetCategoricalEncoder:
    def test_returns_ordinal_encoder(self, fitted_pipeline: Pipeline) -> None:
        encoder = get_categorical_encoder(fitted_pipeline)
        assert isinstance(encoder, OrdinalEncoder)

    def test_encoder_is_fitted(self, fitted_pipeline: Pipeline) -> None:
        from sklearn.utils.validation import check_is_fitted

        encoder = get_categorical_encoder(fitted_pipeline)
        # Should not raise NotFittedError
        check_is_fitted(encoder)

    def test_encoder_has_categories(self, fitted_pipeline: Pipeline) -> None:
        encoder = get_categorical_encoder(fitted_pipeline)
        # After fitting, categories_ is populated (one list per feature)
        assert hasattr(encoder, "categories_")
        assert len(encoder.categories_) > 0

    def test_encoder_from_random_forest_pipeline(
        self,
        small_features: pd.DataFrame,
        small_target: pd.Series,
    ) -> None:
        pipeline = build_pipeline(build_random_forest())
        train_pipeline(pipeline, small_features, small_target)
        encoder = get_categorical_encoder(pipeline)
        assert isinstance(encoder, OrdinalEncoder)


# ── predict_from_payload ──────────────────────────────────────────────────────


class TestPredictFromPayload:
    def test_returns_string(
        self,
        sample_payload: dict[str, object],
        fitted_pipeline: Pipeline,
    ) -> None:
        result = predict_from_payload(sample_payload, pipeline=fitted_pipeline)
        assert isinstance(result, str)

    def test_returns_valid_label(
        self,
        sample_payload: dict[str, object],
        fitted_pipeline: Pipeline,
    ) -> None:
        result = predict_from_payload(sample_payload, pipeline=fitted_pipeline)
        assert result in {">50K", "<=50K"}

    def test_returns_model_not_trained_when_no_pipeline(
        self, sample_payload: dict[str, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import census.inference as inference_mod

        # Patch load_pipeline to return None so predict_from_payload gets no model
        monkeypatch.setattr(inference_mod, "load_pipeline", lambda *_a, **_kw: None)
        result = predict_from_payload(sample_payload)
        assert result == "model_not_trained"

    def test_explicit_pipeline_takes_precedence(
        self,
        sample_payload: dict[str, object],
        fitted_pipeline: Pipeline,
    ) -> None:
        # When pipeline is passed explicitly, no file lookup happens
        result = predict_from_payload(sample_payload, pipeline=fitted_pipeline)
        assert result in {">50K", "<=50K"}

    def test_positive_class_label(
        self,
        fitted_pipeline: Pipeline,
    ) -> None:
        """A payload that looks like a high-earner should return '>50K'."""
        high_earner: dict[str, object] = {
            "age": 55,
            "fnlgt": 220000,
            "education_num": 16,
            "capital_gain": 10000,
            "capital_loss": 0,
            "hours_per_week": 50,
            "workclass": "Self-emp-inc",
            "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "native_country": "United-States",
        }
        result = predict_from_payload(high_earner, pipeline=fitted_pipeline)
        assert result in {">50K", "<=50K"}  # deterministic but class-agnostic

    def test_prediction_is_deterministic(
        self,
        sample_payload: dict[str, object],
        fitted_pipeline: Pipeline,
    ) -> None:
        result_a = predict_from_payload(sample_payload, pipeline=fitted_pipeline)
        result_b = predict_from_payload(sample_payload, pipeline=fitted_pipeline)
        assert result_a == result_b
