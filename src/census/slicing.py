from __future__ import annotations

from pathlib import Path

import pandas as pd

from census.model import compute_metrics, inference

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SLICE_OUTPUT_PATH = _PROJECT_ROOT / "model" / "slice_output.txt"


def compute_slice_metrics(
    dataframe: pd.DataFrame,
    feature: str,
    model: object,
    target_column: str = "salary",
) -> list[str]:
    lines: list[str] = []

    for value in sorted(dataframe[feature].dropna().unique()):
        sliced = dataframe[dataframe[feature] == value]
        if sliced.empty:
            continue

        features = sliced.drop(columns=[target_column])
        target = sliced[target_column]
        predictions = inference(model, features)
        precision, recall, fbeta = compute_metrics(target, predictions)
        lines.append(
            f"{feature}={value}: precision={precision:.3f}, "
            f"recall={recall:.3f}, fbeta={fbeta:.3f}"
        )

    return lines


def save_slice_metrics(lines: list[str]) -> None:
    SLICE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + ("\n" if lines else "")
    SLICE_OUTPUT_PATH.write_text(content, encoding="utf-8")
