import pandas as pd

from census.data_loader import clean_census_data, process_data


def test_clean_census_data_strips_spaces() -> None:
    dataframe = pd.DataFrame({"workclass": [" Private "], "salary": [" <=50K "]})

    cleaned = clean_census_data(dataframe)

    assert cleaned.loc[0, "workclass"] == "Private"
    assert cleaned.loc[0, "salary"] == "<=50K"


def test_process_data_splits_features_and_target() -> None:
    dataframe = pd.DataFrame(
        {
            "age": [39],
            "workclass": ["Private"],
            "education": ["Bachelors"],
            "education-num": [13],
            "salary": [">50K"],
        }
    )

    features, target = process_data(
        dataframe,
        categorical_features=["workclass", "education"],
    )

    assert "salary" not in features.columns
    assert target is not None
    assert target.iloc[0] == ">50K"
