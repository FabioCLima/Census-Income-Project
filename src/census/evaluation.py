"""Census Income — model evaluation (Phase 6 — SRP: evaluation module).

Responsibilities
----------------
ModelMetrics      — frozen dataclass for classification metrics.
compute_metrics   — compute precision / recall / F1 / F-beta for one split.
cross_validate_pipeline — stratified K-fold CV on training data *only*;
                          no test-set leakage into model selection.
build_results_table     — aggregate fold metrics into a tidy DataFrame
                          with mean / std summary rows.
save_results            — persist the results DataFrame to CSV.

Design notes
------------
* Uses StratifiedKFold to preserve the 3.2:1 class ratio in each fold.
* sklearn.base.clone() is called per fold so each fold gets a fresh,
  unfitted copy of the pipeline — avoids state leaking between folds.
* All metrics use pos_label=1 (>50K is the positive / minority class).
* zero_division=0 prevents noisy warnings on small slices.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import clone
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = _PROJECT_ROOT / "model"
RESULTS_PATH = MODEL_DIR / "cv_results.csv"

# ── Constants ─────────────────────────────────────────────────────────────────
POSITIVE_LABEL: int = 1   # >50K encoded as 1 by split_features_target()
N_SPLITS: int = 5
RANDOM_STATE: int = 42

# Metric columns used in the results table
_METRIC_COLS: list[str] = ["precision", "recall", "f1", "fbeta"]


# ── Metrics dataclass ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelMetrics:
    """Evaluation metrics for the positive class (>50K).

    All scores are computed with pos_label=1 and zero_division=0.

    Attributes:
        model_name:  Human-readable identifier, e.g. "RandomForestClassifier".
        dataset:     Split name — "train", "test", or "cv_fold_N".
        precision:   Precision for the >50K class.
        recall:      Recall for the >50K class.
        f1:          F1-score (beta=1) for the >50K class.
        fbeta:       F-beta score (beta=1) — identical to f1; kept for
                     interface compatibility with the course rubric.
    """

    model_name: str
    dataset: str
    precision: float
    recall: float
    f1: float
    fbeta: float

    def __str__(self) -> str:
        return (
            f"[{self.model_name} | {self.dataset}] "
            f"precision={self.precision:.4f}  "
            f"recall={self.recall:.4f}  "
            f"f1={self.f1:.4f}"
        )


# ── Metrics computation ───────────────────────────────────────────────────────


def compute_metrics(
    target: pd.Series,
    predictions: Any,
    model_name: str,
    dataset: str,
) -> ModelMetrics:
    """Compute precision, recall, F1, and F-beta for the positive class.

    All metrics target pos_label=1 (>50K).  zero_division=0 prevents
    ill-defined metric warnings on slices with no predicted positives.

    Args:
        target:      True binary labels (0 / 1).
        predictions: Predicted binary labels (0 / 1).
        model_name:  Identifier for the model (used in ModelMetrics).
        dataset:     Split name ("train", "test", or "cv_fold_N").

    Returns:
        Frozen ModelMetrics dataclass.
    """
    precision = float(
        precision_score(target, predictions, pos_label=POSITIVE_LABEL, zero_division=0)
    )
    recall = float(
        recall_score(target, predictions, pos_label=POSITIVE_LABEL, zero_division=0)
    )
    f1 = float(
        f1_score(target, predictions, pos_label=POSITIVE_LABEL, zero_division=0)
    )
    fbeta = float(
        fbeta_score(
            target, predictions, beta=1, pos_label=POSITIVE_LABEL, zero_division=0
        )
    )
    return ModelMetrics(
        model_name=model_name,
        dataset=dataset,
        precision=precision,
        recall=recall,
        f1=f1,
        fbeta=fbeta,
    )


# ── Cross-validation ──────────────────────────────────────────────────────────


def cross_validate_pipeline(
    pipeline: Pipeline,
    features_train: pd.DataFrame,
    target_train: pd.Series,
    n_splits: int = N_SPLITS,
) -> list[ModelMetrics]:
    """Stratified K-fold CV on the training split only — no test-set exposure.

    A fresh clone of *pipeline* is fitted on each fold's training sub-split so
    the preprocessor statistics (imputer medians, scaler parameters, encoder
    categories) are learned only from that fold's training rows.  This mirrors
    the exact anti-leakage contract used during the final model training.

    Args:
        pipeline:       Unfitted Pipeline (output of build_pipeline).
        features_train: Training feature matrix — test data must NOT be passed.
        target_train:   Binary target Series (0 / 1) for the training split.
        n_splits:       Number of CV folds. Defaults to 5.

    Returns:
        List of ModelMetrics, one per fold (dataset field = "cv_fold_N").
    """
    model_name = type(pipeline.named_steps["model"]).__name__
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    results: list[ModelMetrics] = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        cv.split(features_train, target_train), start=1
    ):
        fold_pipeline = clone(pipeline)

        x_fold_train = features_train.iloc[train_idx]
        y_fold_train = target_train.iloc[train_idx]
        x_fold_val = features_train.iloc[val_idx]
        y_fold_val = target_train.iloc[val_idx]

        fold_pipeline.fit(x_fold_train, y_fold_train)
        predictions = fold_pipeline.predict(x_fold_val)

        metrics = compute_metrics(
            y_fold_val,
            predictions,
            model_name=model_name,
            dataset=f"cv_fold_{fold_idx}",
        )
        logger.info(
            "cross_validate | model={} | fold={}/{} | {}",
            model_name,
            fold_idx,
            n_splits,
            metrics,
        )
        results.append(metrics)

    return results


# ── Results table ─────────────────────────────────────────────────────────────


def build_results_table(
    cv_results_per_model: dict[str, list[ModelMetrics]],
) -> pd.DataFrame:
    """Build a tidy DataFrame of fold-level metrics with mean / std summary rows.

    Structure:
        model_name | fold       | precision | recall | f1 | fbeta
        -----------|------------|-----------|--------|----|------
        ModelA     | cv_fold_1  | ...       | ...    | ...| ...
        ModelA     | cv_fold_2  | ...       | ...    | ...| ...
        ...
        ModelA     | mean       | ...       | ...    | ...| ...
        ModelA     | std        | ...       | ...    | ...| ...
        ModelB     | cv_fold_1  | ...       | ...    | ...| ...
        ...

    Args:
        cv_results_per_model: Mapping of model_name → list[ModelMetrics]
                              (one entry per fold, from cross_validate_pipeline).

    Returns:
        DataFrame with columns [model_name, fold, precision, recall, f1, fbeta].
    """
    rows: list[dict[str, Any]] = []

    for model_name, fold_metrics in cv_results_per_model.items():
        fold_rows: list[dict[str, Any]] = []

        for m in fold_metrics:
            row: dict[str, Any] = {
                "model_name": model_name,
                "fold": m.dataset,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "fbeta": m.fbeta,
            }
            rows.append(row)
            fold_rows.append(row)

        # Summary rows: mean and std across folds
        mean_row: dict[str, Any] = {"model_name": model_name, "fold": "mean"}
        std_row: dict[str, Any] = {"model_name": model_name, "fold": "std"}
        for col in _METRIC_COLS:
            values = [r[col] for r in fold_rows]
            mean_row[col] = float(np.mean(values))
            std_row[col] = float(np.std(values))
        rows.append(mean_row)
        rows.append(std_row)

    return pd.DataFrame(rows, columns=["model_name", "fold", *_METRIC_COLS])


# ── Persistence ───────────────────────────────────────────────────────────────


def save_results(
    df: pd.DataFrame,
    path: Path | str = RESULTS_PATH,
) -> Path:
    """Persist the results DataFrame to a CSV file.

    Args:
        df:   Output of build_results_table().
        path: Destination path. Defaults to model/cv_results.csv.

    Returns:
        Resolved Path where the file was written.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    logger.info("save_results | path={} | rows={}", dest.resolve(), len(df))
    return dest.resolve()
