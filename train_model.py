"""Entry-point script to train and persist the Census Income model.

Usage:
    uv run python train_model.py

Expects the cleaned dataset at data/data_cleaned/census_cleaned.csv.
Run the EDA/cleaning notebook first if the file does not exist.
"""

from pathlib import Path

import joblib
import pandas as pd

from census.data_loader import TARGET_COLUMN, process_data
from census.model import MODEL_PATH, train_model

DATA_PATH = Path("data") / "data_cleaned" / "census_cleaned.csv"

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found at '{DATA_PATH}'. "
            "Run the data cleaning notebook to generate it first."
        )

    dataframe = pd.read_csv(DATA_PATH)
    features, target = process_data(
        dataframe,
        categorical_features=CATEGORICAL_FEATURES,
    )
    if target is None:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in training data.")

    model = train_model(features, target)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
