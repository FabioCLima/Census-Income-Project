"""Shared pytest fixtures for the Census Income test suite."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture()
def small_features() -> pd.DataFrame:
    """Minimal synthetic feature DataFrame with all required columns.

    60 rows with both target classes present — enough for StratifiedKFold.
    Values are deliberately simple; tests care about schema, not realism.
    """
    n = 60
    workclass = [
        "Private", "Self-emp-not-inc", "Private",
        "Self-emp-inc", "Private", "Federal-gov",
    ] * (n // 6)
    marital_status = [
        "Never-married", "Married-civ-spouse", "Never-married",
        "Married-civ-spouse", "Divorced", "Married-civ-spouse",
    ] * (n // 6)
    occupation = [
        "Adm-clerical", "Exec-managerial", "Handlers-cleaners",
        "Prof-specialty", "Sales", "Craft-repair",
    ] * (n // 6)
    relationship = [
        "Not-in-family", "Husband", "Own-child",
        "Husband", "Unmarried", "Husband",
    ] * (n // 6)
    race = [
        "White", "White", "Black",
        "Asian-Pac-Islander", "White", "White",
    ] * (n // 6)

    return pd.DataFrame(
        {
            "age": [30, 45, 25, 55, 35, 50] * (n // 6),
            "fnlgt": [200_000, 150_000, 180_000, 220_000, 160_000, 210_000] * (n // 6),
            "education_num": [13, 9, 10, 16, 12, 14] * (n // 6),
            "capital_gain": [0, 5000, 0, 10_000, 0, 0] * (n // 6),
            "capital_loss": [0, 0, 0, 0, 0, 2000] * (n // 6),
            "hours_per_week": [40, 40, 35, 50, 40, 45] * (n // 6),
            "workclass": workclass,
            "marital_status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "race": race,
            "sex": ["Male", "Male", "Male", "Male", "Female", "Male"] * (n // 6),
            "native_country": ["United-States"] * n,
        }
    )


@pytest.fixture()
def small_target() -> pd.Series:
    """Binary target Series (0 / 1) with both classes present."""
    n = 60
    labels = [0, 1, 0, 1, 0, 0] * (n // 6)
    return pd.Series(labels, name="salary")
