"""Tests for census.train_model (SRP: build + train + save responsibilities)."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from census.train_model import (
    PIPELINE_PATH,
    build_baseline,
    build_pipeline,
    build_random_forest,
    save_pipeline,
    train_pipeline,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────
# small_features and small_target are provided by tests/conftest.py


@pytest.fixture()
def fitted_pipeline(small_features: pd.DataFrame, small_target: pd.Series) -> Pipeline:
    pipeline = build_pipeline(build_baseline())
    return train_pipeline(pipeline, small_features, small_target)


# ── build_baseline ────────────────────────────────────────────────────────────


class TestBuildBaseline:
    def test_returns_decision_tree(self) -> None:
        model = build_baseline()
        assert isinstance(model, DecisionTreeClassifier)

    def test_max_depth(self) -> None:
        model = build_baseline()
        assert model.max_depth == 5

    def test_min_samples_leaf(self) -> None:
        model = build_baseline()
        assert model.min_samples_leaf == 10

    def test_class_weight_balanced(self) -> None:
        model = build_baseline()
        assert model.class_weight == "balanced"

    def test_random_state(self) -> None:
        model = build_baseline()
        assert model.random_state == 42


# ── build_random_forest ───────────────────────────────────────────────────────


class TestBuildRandomForest:
    def test_returns_random_forest(self) -> None:
        model = build_random_forest()
        assert isinstance(model, RandomForestClassifier)

    def test_n_estimators(self) -> None:
        model = build_random_forest()
        assert model.n_estimators == 100

    def test_class_weight_balanced(self) -> None:
        model = build_random_forest()
        assert model.class_weight == "balanced"

    def test_n_jobs_parallel(self) -> None:
        model = build_random_forest()
        assert model.n_jobs == -1

    def test_random_state(self) -> None:
        model = build_random_forest()
        assert model.random_state == 42


# ── build_pipeline ────────────────────────────────────────────────────────────


class TestBuildPipeline:
    def test_returns_pipeline(self) -> None:
        pipeline = build_pipeline(build_baseline())
        assert isinstance(pipeline, Pipeline)

    def test_has_preprocessor_step(self) -> None:
        pipeline = build_pipeline(build_baseline())
        assert "preprocessor" in pipeline.named_steps

    def test_has_model_step(self) -> None:
        pipeline = build_pipeline(build_baseline())
        assert "model" in pipeline.named_steps

    def test_model_step_matches_baseline(self) -> None:
        pipeline = build_pipeline(build_baseline())
        assert isinstance(pipeline.named_steps["model"], DecisionTreeClassifier)

    def test_model_step_matches_random_forest(self) -> None:
        pipeline = build_pipeline(build_random_forest())
        assert isinstance(pipeline.named_steps["model"], RandomForestClassifier)

    def test_step_order(self) -> None:
        pipeline = build_pipeline(build_baseline())
        assert list(pipeline.named_steps.keys()) == ["preprocessor", "model"]


# ── train_pipeline ────────────────────────────────────────────────────────────


class TestTrainPipeline:
    def test_returns_same_pipeline_object(
        self, small_features: pd.DataFrame, small_target: pd.Series
    ) -> None:
        pipeline = build_pipeline(build_baseline())
        result = train_pipeline(pipeline, small_features, small_target)
        assert result is pipeline

    def test_pipeline_is_fitted(
        self, small_features: pd.DataFrame, small_target: pd.Series
    ) -> None:
        pipeline = build_pipeline(build_baseline())
        train_pipeline(pipeline, small_features, small_target)
        # A fitted pipeline can predict without raising NotFittedError
        predictions = pipeline.predict(small_features)
        assert len(predictions) == len(small_features)

    def test_predictions_are_binary(
        self, small_features: pd.DataFrame, small_target: pd.Series
    ) -> None:
        pipeline = build_pipeline(build_baseline())
        train_pipeline(pipeline, small_features, small_target)
        predictions = pipeline.predict(small_features)
        assert set(predictions).issubset({0, 1})


# ── save_pipeline ─────────────────────────────────────────────────────────────


class TestSavePipeline:
    def test_creates_file(self, tmp_path: Path, fitted_pipeline: Pipeline) -> None:
        dest = save_pipeline(fitted_pipeline, tmp_path / "model.pkl")
        assert dest.exists()

    def test_returns_resolved_path(
        self, tmp_path: Path, fitted_pipeline: Pipeline
    ) -> None:
        dest = save_pipeline(fitted_pipeline, tmp_path / "model.pkl")
        assert dest.is_absolute()

    def test_creates_parent_directory(
        self, tmp_path: Path, fitted_pipeline: Pipeline
    ) -> None:
        nested = tmp_path / "subdir" / "model.pkl"
        save_pipeline(fitted_pipeline, nested)
        assert nested.exists()

    def test_file_is_loadable(self, tmp_path: Path, fitted_pipeline: Pipeline) -> None:
        dest = save_pipeline(fitted_pipeline, tmp_path / "model.pkl")
        loaded = joblib.load(dest)
        assert isinstance(loaded, Pipeline)

    def test_default_path_constant(self) -> None:
        assert PIPELINE_PATH.name == "census_pipeline.pkl"
        assert PIPELINE_PATH.parent.name == "model"
