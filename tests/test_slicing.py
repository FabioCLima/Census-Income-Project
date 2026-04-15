"""Tests for census.slicing (slice metrics computation and persistence)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from census.slicing import (
    compute_slice_metrics,
    format_slice_metrics,
    run_slice_evaluation,
    save_slice_metrics,
)
from census.train_model import build_baseline, build_pipeline, train_pipeline


@pytest.fixture()
def fitted_pipeline(small_features: pd.DataFrame, small_target: pd.Series) -> Pipeline:
    pipeline = build_pipeline(build_baseline())
    return train_pipeline(pipeline, small_features, small_target)


class TestComputeSliceMetrics:
    def test_returns_metrics_for_available_slices(
        self,
        small_features: pd.DataFrame,
        small_target: pd.Series,
        fitted_pipeline: Pipeline,
    ) -> None:
        results = compute_slice_metrics(
            small_features, small_target, fitted_pipeline, slice_feature="sex"
        )
        assert len(results) >= 1
        assert all(m.dataset.startswith("sex=") for m in results)


class TestFormatSliceMetrics:
    def test_formats_results_as_strings(
        self,
        small_features: pd.DataFrame,
        small_target: pd.Series,
        fitted_pipeline: Pipeline,
    ) -> None:
        results = compute_slice_metrics(
            small_features, small_target, fitted_pipeline, slice_feature="race"
        )
        lines = format_slice_metrics(results)
        assert all(isinstance(line, str) for line in lines)


class TestRunAndSaveSliceEvaluation:
    def test_run_slice_evaluation_includes_feature_header(
        self,
        small_features: pd.DataFrame,
        small_target: pd.Series,
        fitted_pipeline: Pipeline,
    ) -> None:
        lines = run_slice_evaluation(
            fitted_pipeline,
            small_features,
            small_target,
            slice_features=("sex",),
        )
        assert lines
        assert lines[0] == "[slice_feature=sex]"

    def test_save_slice_metrics_writes_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import census.slicing as slicing_mod

        out_path = tmp_path / "slice_output.txt"
        monkeypatch.setattr(slicing_mod, "SLICE_OUTPUT_PATH", out_path)

        save_slice_metrics(["line 1", "line 2"])
        assert out_path.exists()
        assert out_path.read_text(encoding="utf-8").splitlines() == ["line 1", "line 2"]
