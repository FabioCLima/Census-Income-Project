from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from census.data_loader import EXPECTED_COLUMNS

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = _PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"


def train_model(features: pd.DataFrame, target: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(random_state=42)
    model.fit(features, target)
    return model


def inference(model: Any, features: pd.DataFrame) -> Any:
    return model.predict(features)


def compute_metrics(target: pd.Series, predictions: Any) -> tuple[float, float, float]:
    precision = precision_score(target, predictions, pos_label=">50K")
    recall = recall_score(target, predictions, pos_label=">50K")
    fbeta = fbeta_score(target, predictions, beta=1, pos_label=">50K")
    return precision, recall, fbeta


def load_model() -> Any | None:
    if not MODEL_PATH.exists():
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def predict_from_payload(payload: dict[str, object]) -> str:
    model = load_model()
    if model is None:
        return "model_not_trained"

    ordered_payload = {column: payload[column] for column in EXPECTED_COLUMNS}
    features = pd.DataFrame([ordered_payload])
    prediction = inference(model, features)
    return str(prediction[0])
