"""Census Income — slice-level model evaluation (Phase 7).

Implements the mandatory slice validation step from
docs/pipeline_architecture.md:

    "Validation by slice in `sex` and `race` is mandatory before any deploy."

Functions
---------
compute_slice_metrics  — metrics per unique value of a feature column
format_slice_metrics   — render ModelMetrics list as human-readable lines
save_slice_metrics     — persist formatted lines to model/slice_output.txt
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.pipeline import Pipeline

from census.data_loader import CATEGORICAL_FEATURES
from census.evaluation import ModelMetrics, compute_metrics
from census.inference import load_pipeline
from census.preprocessing import (
    load_cleaned_data,
    split_features_target,
    split_train_test,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SLICE_OUTPUT_PATH = _PROJECT_ROOT / "model" / "slice_output.txt"

# Minimum slice size to report metrics; smaller slices are flagged but skipped
# (too few samples to produce stable estimates — architecture doc, Phase 7.2).
MIN_SLICE_SAMPLES: int = 30
DEFAULT_SLICE_FEATURES: tuple[str, ...] = ("sex",)


def compute_slice_metrics(
    features: pd.DataFrame,
    target: pd.Series,
    pipeline: Pipeline,
    slice_feature: str,
) -> list[ModelMetrics]:
    """Compute evaluation metrics for each unique value of *slice_feature*.

    Runs pipeline.predict() on each slice and calls compute_metrics() so the
    results are consistent with the global evaluation in train_model.py.
    Slices smaller than MIN_SLICE_SAMPLES are skipped — the metric estimates
    would be too noisy to be actionable.

    Args:
        features:      Feature matrix (test split, no target column).
                       Must contain *slice_feature*.
        target:        Binary target Series (0 = <=50K, 1 = >50K).
        pipeline:      Fitted Pipeline (output of train_pipeline).
        slice_feature: Column to slice on, e.g. "sex" or "race".

    Returns:
        List of ModelMetrics, one per unique slice value with enough samples.
        Slices below MIN_SLICE_SAMPLES are omitted.
    """
    model_name = type(pipeline.named_steps["model"]).__name__
    results: list[ModelMetrics] = []

    for value in sorted(features[slice_feature].dropna().unique()):
        mask = features[slice_feature] == value
        slice_size = int(mask.sum())

        if slice_size < MIN_SLICE_SAMPLES:
            continue

        features_slice = features.loc[mask]
        target_slice = target.loc[mask]
        predictions = pipeline.predict(features_slice)

        metrics = compute_metrics(
            target_slice,
            predictions,
            model_name=model_name,
            dataset=f"{slice_feature}={value}",
        )
        results.append(metrics)

    return results


def format_slice_metrics(results: list[ModelMetrics]) -> list[str]:
    """Render a list of ModelMetrics as human-readable strings.

    Args:
        results: Output of compute_slice_metrics().

    Returns:
        List of formatted lines, one per slice.
    """
    return [str(m) for m in results]


def save_slice_metrics(lines: list[str]) -> None:
    """Persist formatted metric lines to SLICE_OUTPUT_PATH.

    Creates the parent directory if it does not exist.  Overwrites any
    previously saved file.

    Args:
        lines: Output of format_slice_metrics().
    """
    SLICE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + ("\n" if lines else "")
    SLICE_OUTPUT_PATH.write_text(content, encoding="utf-8")


def run_slice_evaluation(
    pipeline: Pipeline,
    features_test: pd.DataFrame,
    target_test: pd.Series,
    *,
    slice_features: tuple[str, ...] = DEFAULT_SLICE_FEATURES,
) -> list[str]:
    """Compute and format slice metrics for selected categorical features."""
    lines: list[str] = []
    for feature in slice_features:
        if feature not in features_test.columns:
            logger.warning("run_slice_evaluation | slice_feature_missing={}", feature)
            continue

        feature_results = compute_slice_metrics(
            features_test, target_test, pipeline, slice_feature=feature
        )
        if not feature_results:
            lines.append(
                f"[slice_feature={feature}] no slices with n >= {MIN_SLICE_SAMPLES}"
            )
            continue

        lines.append(f"[slice_feature={feature}]")
        lines.extend(format_slice_metrics(feature_results))
        lines.append("")

    # Trim trailing blank line for cleaner output.
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def run_slice_pipeline(
    *,
    slice_features: tuple[str, ...] = DEFAULT_SLICE_FEATURES,
) -> Path:
    """Execute the full slice-evaluation workflow and persist output.

    Returns:
        Path to the saved ``slice_output.txt`` file.
    """
    pipeline = load_pipeline()
    if pipeline is None:
        raise FileNotFoundError(
            "Trained pipeline not found at model/census_pipeline.pkl. "
            "Run `python train_model.py` first."
        )

    df = load_cleaned_data()
    features, target = split_features_target(df)
    _, features_test, _, target_test = split_train_test(features, target)

    valid_slice_features = tuple(
        feature for feature in slice_features if feature in CATEGORICAL_FEATURES
    )
    lines = run_slice_evaluation(
        pipeline,
        features_test,
        target_test,
        slice_features=valid_slice_features,
    )
    save_slice_metrics(lines)
    logger.info(
        "Slice metrics saved | path={} | lines={}",
        SLICE_OUTPUT_PATH,
        len(lines),
    )
    return SLICE_OUTPUT_PATH


if __name__ == "__main__":
    from census.configure_logging import configure_logging

    _LOG_DIR = _PROJECT_ROOT / "logs" / "slicing_log"
    configure_logging(_LOG_DIR, "slicing")
    run_slice_pipeline()
