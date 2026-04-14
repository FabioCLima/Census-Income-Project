"""Census Income — model inference (Phase 6 — SRP: inference module).

Responsibilities
----------------
load_pipeline          — deserialise a fitted Pipeline from disk; returns None
                         on missing file so callers handle the untrained case.
get_categorical_encoder — extract the fitted OrdinalEncoder from the Pipeline;
                          useful for mapping category indices back to labels.
predict_from_payload   — run end-to-end inference on a single API payload dict.

Design notes
------------
* The census Pipeline structure is:
    Pipeline([
        ("preprocessor", ColumnTransformer([
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ])),
        ("model", classifier),
    ])
  The OrdinalEncoder lives at:
    pipeline["preprocessor"].named_transformers_["cat"]["encoder"]

* predict_from_payload wraps a raw dict → single-row DataFrame → predict()
  so each API request gets the identical transformation path used in training.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from loguru import logger
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = _PROJECT_ROOT / "model"
PIPELINE_PATH = MODEL_DIR / "census_pipeline.pkl"

# ── Constants ─────────────────────────────────────────────────────────────────
POSITIVE_LABEL: int = 1   # >50K encoded as 1 by split_features_target()


# ── Pipeline loading ──────────────────────────────────────────────────────────


def load_pipeline(path: Path | str = PIPELINE_PATH) -> Pipeline | None:
    """Load a serialised Pipeline from disk.

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
        logger.error(
            "load_pipeline | failed to load | path={} | error={}", dest, exc
        )
        return None


# ── Encoder extraction ────────────────────────────────────────────────────────


def get_categorical_encoder(pipeline: Pipeline) -> OrdinalEncoder:
    """Extract the fitted OrdinalEncoder from the census Pipeline.

    The encoder lives at:
        pipeline["preprocessor"].named_transformers_["cat"]["encoder"]

    This is useful for mapping ordinal indices back to their original
    category labels (e.g. for slice-level interpretation).

    Args:
        pipeline: Fitted Pipeline produced by train_pipeline().

    Returns:
        The fitted OrdinalEncoder instance.

    Raises:
        KeyError:      if the expected pipeline structure is not found.
        AttributeError: if the pipeline has not been fitted yet.
    """
    encoder: OrdinalEncoder = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["encoder"]
    )
    return encoder


# ── Prediction ────────────────────────────────────────────────────────────────


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
