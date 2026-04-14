"""Census Income — model training pipeline (Phase 6).

Implements Phase 6 of docs/pipeline_architecture.md.

Architecture decision
---------------------
The full sklearn Pipeline (preprocessor + model) is serialised as a single
artefact.  This guarantees that every inference call — whether in tests, the
FastAPI endpoint, or batch scoring — runs the exact same transformation path
that was used during training.

Models
------
DecisionTreeClassifier  — baseline: interpretable, fast, no hyperparameter
                          tuning required.  Sets the F1 floor that every
                          subsequent model must beat.

RandomForestClassifier  — main candidate: ensemble of decision trees that
                          captures feature interactions (occupation x hours,
                          H4) and threshold effects (capital_gain, H2) that
                          a single tree misses.

Both models use class_weight="balanced" to compensate for the 3.2:1
imbalance without resampling the training data.

Positive class
--------------
After split_features_target() the target is binary int: >50K → 1, <=50K → 0.
All metrics are computed for pos_label=1 (the minority class that matters).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from census.preprocessing import build_preprocessor

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = _PROJECT_ROOT / "model"
PIPELINE_PATH = MODEL_DIR / "census_pipeline.pkl"

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE: int = 42
POSITIVE_LABEL: int = 1   # >50K encoded as 1 by split_features_target()


# ── Metrics dataclass ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelMetrics:
    """Evaluation metrics for the positive class (>50K).

    All scores are computed with pos_label=1 and zero_division=0.

    Attributes:
        model_name:  Human-readable identifier, e.g. "RandomForest".
        dataset:     Split name, e.g. "train" or "test".
        precision:   Precision for the >50K class.
        recall:      Recall for the >50K class.
        f1:          F1-score (beta=1) for the >50K class.
        fbeta:       F-beta score (beta=1) — identical to f1 here; kept for
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


# ── Model builders ────────────────────────────────────────────────────────────


def build_baseline() -> DecisionTreeClassifier:
    """Return an unfitted DecisionTreeClassifier baseline.

    Serves as the performance floor: every candidate model must achieve a
    higher F1 on the test set to justify the added complexity.

    Design choices:
        max_depth=None   — fully grown tree; deliberately overfit to show the
                           upper-bound capacity of a single tree.
        class_weight="balanced" — compensates for the 3.2:1 imbalance.
        random_state=42  — reproducible splits.

    Returns:
        Unfitted DecisionTreeClassifier.
    """
    return DecisionTreeClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def build_random_forest() -> RandomForestClassifier:
    """Return an unfitted RandomForestClassifier (main candidate).

    Design choices:
        n_estimators=100      — standard starting point; enough trees to
                                stabilise variance without excessive runtime.
        class_weight="balanced" — each tree rebalances class weights; avoids
                                  resampling the training data.
        n_jobs=-1             — parallelise across all CPU cores.
        random_state=42       — reproducible ensemble.

    Returns:
        Unfitted RandomForestClassifier.
    """
    return RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


# ── Pipeline builder ──────────────────────────────────────────────────────────


def build_pipeline(model: DecisionTreeClassifier | RandomForestClassifier) -> Pipeline:
    """Wrap *model* inside a full sklearn Pipeline with the census preprocessor.

    The Pipeline serialises as a single artefact: loading it at inference
    time gives a ready-to-use object that accepts raw feature DataFrames
    (after clean_raw_input) with no additional setup.

    Args:
        model: An unfitted classifier (build_baseline() or build_random_forest()).

    Returns:
        Unfitted Pipeline with steps [("preprocessor", ...), ("model", model)].
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


# ── Training ──────────────────────────────────────────────────────────────────


def train_pipeline(
    pipeline: Pipeline,
    features_train: pd.DataFrame,
    target_train: pd.Series,
) -> Pipeline:
    """Fit *pipeline* on the training split.

    The preprocessor learns imputer statistics and scaler parameters from
    features_train only — no leakage from the test set.

    Args:
        pipeline:       Unfitted Pipeline (output of build_pipeline).
        features_train: Training feature matrix.
        target_train:   Binary target Series (0 / 1).

    Returns:
        The same pipeline object, now fitted.
    """
    model_name = type(pipeline.named_steps["model"]).__name__
    logger.info(
        "train_pipeline | model={} | train_shape={}",
        model_name,
        features_train.shape,
    )
    pipeline.fit(features_train, target_train)
    logger.info("train_pipeline complete | model={}", model_name)
    return pipeline


# ── Evaluation ────────────────────────────────────────────────────────────────


def compute_metrics(
    target: pd.Series,
    predictions: Any,
    model_name: str,
    dataset: str,
) -> ModelMetrics:
    """Compute precision, recall, F1 and F-beta for the positive class.

    All metrics target pos_label=1 (>50K).  zero_division=0 prevents
    ill-defined metric warnings on slices with no predicted positives.

    Args:
        target:      True binary labels (0 / 1).
        predictions: Predicted binary labels (0 / 1).
        model_name:  Identifier for the model (used in ModelMetrics).
        dataset:     Split name ("train" or "test").

    Returns:
        Frozen ModelMetrics dataclass.
    """
    precision = float(precision_score(
        target, predictions, pos_label=POSITIVE_LABEL, zero_division=0
    ))
    recall = float(recall_score(
        target, predictions, pos_label=POSITIVE_LABEL, zero_division=0
    ))
    f1 = float(f1_score(
        target, predictions, pos_label=POSITIVE_LABEL, zero_division=0
    ))
    fbeta = float(fbeta_score(
        target, predictions, beta=1, pos_label=POSITIVE_LABEL, zero_division=0
    ))
    return ModelMetrics(
        model_name=model_name,
        dataset=dataset,
        precision=precision,
        recall=recall,
        f1=f1,
        fbeta=fbeta,
    )


def evaluate_pipeline(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    dataset: str,
) -> ModelMetrics:
    """Run inference on *features* and return metrics against *target*.

    Args:
        pipeline: Fitted Pipeline (output of train_pipeline).
        features: Feature matrix (train or test split).
        target:   True binary labels for the split.
        dataset:  Split name ("train" or "test") for labelling.

    Returns:
        ModelMetrics for the given split.
    """
    model_name = type(pipeline.named_steps["model"]).__name__
    predictions = pipeline.predict(features)
    metrics = compute_metrics(target, predictions, model_name, dataset)
    logger.info("{}", metrics)
    return metrics


# ── Serialisation ─────────────────────────────────────────────────────────────


def save_pipeline(
    pipeline: Pipeline,
    path: Path | str = PIPELINE_PATH,
) -> Path:
    """Serialise a fitted pipeline to disk with joblib.

    Args:
        pipeline: Fitted Pipeline to persist.
        path:     Destination path. Defaults to model/census_pipeline.pkl.

    Returns:
        Resolved Path where the file was written.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, dest)
    logger.info("save_pipeline | path={}", dest.resolve())
    return dest.resolve()


def load_pipeline(path: Path | str = PIPELINE_PATH) -> Pipeline | None:
    """Load a serialised pipeline from disk.

    Returns None (instead of raising) so callers can handle the
    model-not-trained case gracefully — e.g. the FastAPI startup event.

    Args:
        path: Path to the .pkl file. Defaults to model/census_pipeline.pkl.

    Returns:
        Fitted Pipeline, or None if the file does not exist or is corrupt.
    """
    dest = Path(path)
    if not dest.exists():
        logger.warning("load_pipeline | file not found | path={}", dest)
        return None
    try:
        pipeline: Pipeline = joblib.load(dest)
        logger.info("load_pipeline | loaded | path={}", dest)
        return pipeline
    except Exception as exc:
        logger.error("load_pipeline | failed to load | path={} | error={}", dest, exc)
        return None


# ── API inference ─────────────────────────────────────────────────────────────


def predict_from_payload(
    payload: dict[str, object],
    pipeline: Pipeline | None = None,
) -> str:
    """Run inference on a single raw API payload dict.

    Loads the persisted pipeline if *pipeline* is not provided.  The payload
    is converted to a single-row DataFrame; clean_raw_input() must have been
    applied upstream (the FastAPI endpoint handles this via PredictRequest).

    Args:
        payload:  Dict of feature name → value, already cleaned (no hyphens,
                  no leading spaces, 'education' already dropped).
        pipeline: Pre-loaded Pipeline. If None, loads from PIPELINE_PATH.

    Returns:
        ">50K" or "<=50K" as a string.  Returns "model_not_trained" if no
        serialised pipeline is found.
    """
    if pipeline is None:
        pipeline = load_pipeline()

    if pipeline is None:
        return "model_not_trained"

    features = pd.DataFrame([payload])
    prediction: int = int(pipeline.predict(features)[0])
    return ">50K" if prediction == POSITIVE_LABEL else "<=50K"


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from census.configure_logging import configure_logging
    from census.preprocessing import (
        load_cleaned_data,
        split_features_target,
        split_train_test,
    )

    _LOG_DIR = _PROJECT_ROOT / "logs" / "training_log"
    configure_logging(_LOG_DIR, "training")

    # ── Load and split ─────────────────────────────────────────────────────
    logger.info("Loading cleaned data…")
    df = load_cleaned_data()
    features, target = split_features_target(df)
    features_train, features_test, target_train, target_test = split_train_test(
        features, target
    )

    # ── Baseline: DecisionTree ─────────────────────────────────────────────
    logger.info("Training baseline (DecisionTreeClassifier)…")
    baseline_pipeline = build_pipeline(build_baseline())
    train_pipeline(baseline_pipeline, features_train, target_train)
    baseline_train = evaluate_pipeline(
        baseline_pipeline, features_train, target_train, "train"
    )
    baseline_test = evaluate_pipeline(
        baseline_pipeline, features_test, target_test, "test"
    )

    # ── Candidate: RandomForest ────────────────────────────────────────────
    logger.info("Training main model (RandomForestClassifier)…")
    rf_pipeline = build_pipeline(build_random_forest())
    train_pipeline(rf_pipeline, features_train, target_train)
    rf_train = evaluate_pipeline(rf_pipeline, features_train, target_train, "train")
    rf_test = evaluate_pipeline(rf_pipeline, features_test, target_test, "test")

    # ── Model selection ────────────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info("Model comparison (test F1 — positive class >50K):")
    logger.info("  Baseline  (DecisionTree) : {:.4f}", baseline_test.f1)
    logger.info("  Candidate (RandomForest) : {:.4f}", rf_test.f1)

    best_pipeline = rf_pipeline if rf_test.f1 >= baseline_test.f1 else baseline_pipeline
    best_name = type(best_pipeline.named_steps["model"]).__name__
    logger.info("Selected: {}", best_name)

    # ── Persist ────────────────────────────────────────────────────────────
    saved_path = save_pipeline(best_pipeline)
    logger.info("Pipeline saved → {}", saved_path)
