"""Census Income — model training pipeline (Phase 6 — SRP: training module).

Responsibilities (Single Responsibility Principle)
---------------------------------------------------
This module is concerned *only* with constructing and persisting models.
Evaluation logic lives in census.evaluation; inference in census.inference.

build_baseline      — DecisionTreeClassifier baseline (max_depth=5,
                      min_samples_leaf=10) — sets the F1 floor.
build_random_forest — RandomForestClassifier main candidate.
build_pipeline      — wrap a classifier in a full sklearn Pipeline.
train_pipeline      — fit the Pipeline on training data.
save_pipeline       — serialise a fitted Pipeline to disk with joblib.

Models
------
DecisionTreeClassifier  — baseline: interpretable, constrained depth avoids
                          fully-grown overfitting.  max_depth=5 keeps the
                          tree readable; min_samples_leaf=10 regularises leaf
                          purity and reduces variance.

RandomForestClassifier  — main candidate: ensemble that captures feature
                          interactions (occupation x hours, H4) and threshold
                          effects (capital_gain, H2) that a single tree misses.

Both models use class_weight="balanced" to compensate for the 3.2:1 imbalance
without resampling the training data.

Positive class
--------------
The target is binary int: >50K → 1, <=50K → 0.
All classification metrics are evaluated for pos_label=1 (minority class).
Metric computation is delegated to census.evaluation.compute_metrics.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from census.preprocessing import build_preprocessor

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = _PROJECT_ROOT / "model"
PIPELINE_PATH = MODEL_DIR / "census_pipeline.pkl"

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE: int = 42


# ── Model builders ────────────────────────────────────────────────────────────


def build_baseline() -> DecisionTreeClassifier:
    """Return an unfitted DecisionTreeClassifier baseline.

    Hyperparameters
    ---------------
    max_depth=5         — limits tree depth to prevent full overfitting;
                          keeps the model interpretable and reduces variance.
    min_samples_leaf=10 — each leaf must contain at least 10 samples, which
                          regularises splits on rare feature combinations.
    class_weight="balanced" — compensates for the 3.2:1 class imbalance.
    random_state=42     — reproducible splits.

    Returns:
        Unfitted DecisionTreeClassifier.
    """
    return DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def build_random_forest() -> RandomForestClassifier:
    """Return an unfitted RandomForestClassifier (main candidate).

    Hyperparameters
    ---------------
    n_estimators=100        — standard starting point; stabilises variance
                              without excessive runtime.
    class_weight="balanced" — each tree rebalances class weights; avoids
                              resampling the training data.
    n_jobs=-1               — parallelise across all CPU cores.
    random_state=42         — reproducible ensemble.

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

    The Pipeline serialises as a single artefact: loading it at inference time
    gives a ready-to-use object that accepts raw feature DataFrames (after
    clean_raw_input) with no additional setup.

    Args:
        model: An unfitted classifier (build_baseline or build_random_forest).

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


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from census.configure_logging import configure_logging
    from census.evaluation import (
        build_results_table,
        cross_validate_pipeline,
        save_results,
    )
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
    # split_train_test → (features_train, features_test, target_train, target_test).
    # features_test / target_test are reserved for slice evaluation in slicing.py;
    # model selection here is based on CV on the training set only.
    features_train, _, target_train, _ = split_train_test(features, target)

    # ── Cross-validation — training data only ─────────────────────────────
    logger.info("Running cross-validation (StratifiedKFold, n_splits=5)…")

    baseline_cv = cross_validate_pipeline(
        build_pipeline(build_baseline()), features_train, target_train
    )
    rf_cv = cross_validate_pipeline(
        build_pipeline(build_random_forest()), features_train, target_train
    )

    # ── Results table ──────────────────────────────────────────────────────
    results_df = build_results_table(
        {
            "DecisionTreeClassifier": baseline_cv,
            "RandomForestClassifier": rf_cv,
        }
    )
    saved_csv = save_results(results_df)
    logger.info("CV results saved → {}", saved_csv)

    # ── Model selection (CV mean F1 on training data) ──────────────────────
    baseline_mean_f1 = float(
        results_df.loc[
            (results_df["model_name"] == "DecisionTreeClassifier")
            & (results_df["fold"] == "mean"),
            "f1",
        ].iloc[0]
    )
    rf_mean_f1 = float(
        results_df.loc[
            (results_df["model_name"] == "RandomForestClassifier")
            & (results_df["fold"] == "mean"),
            "f1",
        ].iloc[0]
    )

    logger.info("─" * 60)
    logger.info("CV mean F1 — positive class >50K (training data):")
    logger.info("  Baseline  (DecisionTree) : {:.4f}", baseline_mean_f1)
    logger.info("  Candidate (RandomForest) : {:.4f}", rf_mean_f1)

    # ── Retrain best model on full training set ────────────────────────────
    if rf_mean_f1 >= baseline_mean_f1:
        best_pipeline = build_pipeline(build_random_forest())
        best_name = "RandomForestClassifier"
    else:
        best_pipeline = build_pipeline(build_baseline())
        best_name = "DecisionTreeClassifier"

    logger.info("Selected: {} — retraining on full training set…", best_name)
    train_pipeline(best_pipeline, features_train, target_train)

    # ── Persist ────────────────────────────────────────────────────────────
    saved_path = save_pipeline(best_pipeline)
    logger.info("Pipeline saved → {}", saved_path)
