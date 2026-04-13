import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from census.data import TARGET_COLUMN, process_data
from census.model import MODEL_PATH, train_model

DATA_PATH = Path("data") / "clean_census.csv"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Expected 'data/clean_census.csv' before training the model."
        )

    dataframe = pd.read_csv(DATA_PATH)
    features, target = process_data(
        dataframe,
        categorical_features=[
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ],
    )
    if target is None:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in training data.")

    model = train_model(features, target)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    import joblib

    joblib.dump(model, MODEL_PATH)


if __name__ == "__main__":
    main()
