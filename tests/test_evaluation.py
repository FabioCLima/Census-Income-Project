"""Tests for census.evaluation (ModelMetrics, compute_metrics, CV, results table)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from census.evaluation import (
    POSITIVE_LABEL,
    ModelMetrics,
    build_results_table,
    compute_metrics,
    cross_validate_pipeline,
    save_results,
)
from census.train_model import build_baseline, build_pipeline

# small_features and small_target are provided by tests/conftest.py

# ── ModelMetrics ──────────────────────────────────────────────────────────────


class TestModelMetrics:
    def test_is_frozen(self) -> None:
        m = ModelMetrics(
            model_name="Test",
            dataset="train",
            precision=0.8,
            recall=0.7,
            f1=0.75,
            fbeta=0.75,
        )
        with pytest.raises((AttributeError, TypeError)):
            m.precision = 0.9  # type: ignore[misc]

    def test_str_contains_model_name(self) -> None:
        m = ModelMetrics("MyModel", "test", 0.8, 0.7, 0.75, 0.75)
        assert "MyModel" in str(m)

    def test_str_contains_dataset(self) -> None:
        m = ModelMetrics("MyModel", "cv_fold_3", 0.8, 0.7, 0.75, 0.75)
        assert "cv_fold_3" in str(m)

    def test_str_contains_precision_and_recall(self) -> None:
        m = ModelMetrics("M", "train", 0.8000, 0.7000, 0.7467, 0.7467)
        s = str(m)
        assert "precision=" in s
        assert "recall=" in s
        assert "f1=" in s


# ── compute_metrics ───────────────────────────────────────────────────────────


class TestComputeMetrics:
    def test_perfect_predictions(self) -> None:
        target = pd.Series([0, 1, 0, 1])
        preds = [0, 1, 0, 1]
        m = compute_metrics(target, preds, "Model", "test")
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)
        assert m.f1 == pytest.approx(1.0)

    def test_all_negative_predictions(self) -> None:
        target = pd.Series([0, 1, 0, 1])
        preds = [0, 0, 0, 0]
        m = compute_metrics(target, preds, "Model", "test")
        # zero_division=0: precision and recall are 0 when no positives predicted
        assert m.precision == pytest.approx(0.0)
        assert m.recall == pytest.approx(0.0)
        assert m.f1 == pytest.approx(0.0)

    def test_returns_model_metrics_instance(self) -> None:
        target = pd.Series([0, 1])
        preds = [0, 1]
        m = compute_metrics(target, preds, "Model", "train")
        assert isinstance(m, ModelMetrics)

    def test_model_name_preserved(self) -> None:
        target = pd.Series([0, 1])
        preds = [0, 1]
        m = compute_metrics(target, preds, "RandomForest", "train")
        assert m.model_name == "RandomForest"

    def test_dataset_preserved(self) -> None:
        target = pd.Series([0, 1])
        preds = [0, 1]
        m = compute_metrics(target, preds, "Model", "cv_fold_2")
        assert m.dataset == "cv_fold_2"

    def test_f1_equals_fbeta_with_beta_one(self) -> None:
        target = pd.Series([0, 1, 0, 1, 1])
        preds = [0, 1, 1, 0, 1]
        m = compute_metrics(target, preds, "Model", "test")
        assert m.f1 == pytest.approx(m.fbeta)

    def test_scores_are_python_floats(self) -> None:
        target = pd.Series([0, 1])
        preds = [0, 1]
        m = compute_metrics(target, preds, "Model", "test")
        assert isinstance(m.precision, float)
        assert isinstance(m.recall, float)
        assert isinstance(m.f1, float)
        assert isinstance(m.fbeta, float)

    def test_scores_in_unit_interval(self) -> None:
        target = pd.Series([0, 1, 0, 1, 1])
        preds = [0, 1, 1, 0, 1]
        m = compute_metrics(target, preds, "Model", "test")
        for score in (m.precision, m.recall, m.f1, m.fbeta):
            assert 0.0 <= score <= 1.0

    def test_positive_label_is_one(self) -> None:
        assert POSITIVE_LABEL == 1


# ── cross_validate_pipeline ───────────────────────────────────────────────────


class TestCrossValidatePipeline:
    def test_returns_correct_number_of_folds(
        self, small_features: pd.DataFrame, small_target: pd.Series
    ) -> None:
        pipeline = build_pipeline(build_baseline())
        results = cross_validate_pipeline(
            pipeline, small_features, small_target, n_splits=3
        )
        assert len(results) == 3

    def test_default_five_folds(
        self, small_features: pd.DataFrame, small_target: pd.Series
    ) -> None:
        pipeline = build_pipeline(build_baseline())
        results = cross_validate_pipeline(pipeline, small_features, small_target)
        assert len(results) == 5

    def test_fold_datasets_are_labelled(
        self, small_features: pd.DataFrame, small_target: pd.Series
    ) -> None:
        pipeline = build_pipeline(build_baseline())
        results = cross_validate_pipeline(
            pipeline, small_features, small_target, n_splits=3
        )
        datasets = [m.dataset for m in results]
        assert datasets == ["cv_fold_1", "cv_fold_2", "cv_fold_3"]

    def test_all_results_are_model_metrics(
        self, small_features: pd.DataFrame, small_target: pd.Series
    ) -> None:
        pipeline = build_pipeline(build_baseline())
        results = cross_validate_pipeline(
            pipeline, small_features, small_target, n_splits=3
        )
        assert all(isinstance(m, ModelMetrics) for m in results)

    def test_model_name_from_pipeline(
        self, small_features: pd.DataFrame, small_target: pd.Series
    ) -> None:
        pipeline = build_pipeline(build_baseline())
        results = cross_validate_pipeline(
            pipeline, small_features, small_target, n_splits=3
        )
        assert all(m.model_name == "DecisionTreeClassifier" for m in results)

    def test_original_pipeline_not_modified(
        self, small_features: pd.DataFrame, small_target: pd.Series
    ) -> None:
        """cross_validate_pipeline clones per fold; original remains unfitted."""
        from sklearn.exceptions import NotFittedError
        from sklearn.utils.validation import check_is_fitted

        pipeline = build_pipeline(build_baseline())
        cross_validate_pipeline(
            pipeline, small_features, small_target, n_splits=3
        )
        with pytest.raises(NotFittedError):
            check_is_fitted(pipeline)


# ── build_results_table ───────────────────────────────────────────────────────


class TestBuildResultsTable:
    def _make_fold_metrics(
        self, model_name: str, n_folds: int = 3
    ) -> list[ModelMetrics]:
        return [
            ModelMetrics(
                model_name=model_name,
                dataset=f"cv_fold_{i}",
                precision=0.7 + i * 0.01,
                recall=0.65 + i * 0.01,
                f1=0.67 + i * 0.01,
                fbeta=0.67 + i * 0.01,
            )
            for i in range(1, n_folds + 1)
        ]

    def test_returns_dataframe(self) -> None:
        results = build_results_table(
            {"DecisionTree": self._make_fold_metrics("DecisionTree")}
        )
        assert isinstance(results, pd.DataFrame)

    def test_has_required_columns(self) -> None:
        df = build_results_table(
            {"Model": self._make_fold_metrics("Model")}
        )
        expected = {"model_name", "fold", "precision", "recall", "f1", "fbeta"}
        assert expected.issubset(set(df.columns))

    def test_contains_fold_rows(self) -> None:
        df = build_results_table(
            {"Model": self._make_fold_metrics("Model", n_folds=3)}
        )
        fold_rows = df[df["fold"].str.startswith("cv_fold")]
        assert len(fold_rows) == 3

    def test_contains_mean_and_std_rows(self) -> None:
        df = build_results_table(
            {"Model": self._make_fold_metrics("Model", n_folds=3)}
        )
        assert "mean" in df["fold"].values
        assert "std" in df["fold"].values

    def test_two_models_produce_separate_rows(self) -> None:
        df = build_results_table(
            {
                "ModelA": self._make_fold_metrics("ModelA", n_folds=3),
                "ModelB": self._make_fold_metrics("ModelB", n_folds=3),
            }
        )
        assert set(df["model_name"].unique()) == {"ModelA", "ModelB"}

    def test_total_row_count_with_two_models(self) -> None:
        # 3 fold rows + 2 summary (mean/std) per model x 2 models = 10
        df = build_results_table(
            {
                "ModelA": self._make_fold_metrics("ModelA", n_folds=3),
                "ModelB": self._make_fold_metrics("ModelB", n_folds=3),
            }
        )
        assert len(df) == 10

    def test_mean_row_is_average_of_folds(self) -> None:
        metrics = self._make_fold_metrics("Model", n_folds=3)
        df = build_results_table({"Model": metrics})
        mean_row = df[(df["model_name"] == "Model") & (df["fold"] == "mean")]
        expected_f1 = float(np.mean([m.f1 for m in metrics]))
        assert mean_row["f1"].iloc[0] == pytest.approx(expected_f1)

    def test_std_row_is_std_of_folds(self) -> None:
        metrics = self._make_fold_metrics("Model", n_folds=3)
        df = build_results_table({"Model": metrics})
        std_row = df[(df["model_name"] == "Model") & (df["fold"] == "std")]
        expected_std = float(np.std([m.f1 for m in metrics]))
        assert std_row["f1"].iloc[0] == pytest.approx(expected_std)


# ── save_results ──────────────────────────────────────────────────────────────


class TestSaveResults:
    def test_creates_file(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"model_name": ["M"], "fold": ["cv_fold_1"], "f1": [0.8]})
        dest = save_results(df, tmp_path / "results.csv")
        assert dest.exists()

    def test_returns_resolved_path(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"model_name": ["M"], "fold": ["cv_fold_1"], "f1": [0.8]})
        dest = save_results(df, tmp_path / "results.csv")
        assert dest.is_absolute()

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"model_name": ["M"], "fold": ["cv_fold_1"], "f1": [0.8]})
        nested = tmp_path / "subdir" / "results.csv"
        save_results(df, nested)
        assert nested.exists()

    def test_csv_content_matches_dataframe(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "model_name": ["Model"],
                "fold": ["cv_fold_1"],
                "precision": [0.8],
                "recall": [0.7],
                "f1": [0.75],
                "fbeta": [0.75],
            }
        )
        dest = save_results(df, tmp_path / "results.csv")
        loaded = pd.read_csv(dest)
        assert list(loaded.columns) == list(df.columns)
        assert loaded["f1"].iloc[0] == pytest.approx(0.75)
