from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    """Validated input payload for the /predict endpoint.

    Accepts hyphenated JSON keys (e.g. "education-num") via field aliases,
    which match the original UCI column names. Python-safe underscore names
    (e.g. "education_num") are also accepted when populate_by_name is True.
    """

    model_config = ConfigDict(populate_by_name=True)

    age: int = Field(..., examples=[39])
    workclass: str = Field(..., examples=["State-gov"])
    fnlwgt: int = Field(..., examples=[77516])
    education: str = Field(..., examples=["Bachelors"])
    education_num: int = Field(..., alias="education-num", examples=[13])
    marital_status: str = Field(..., alias="marital-status", examples=["Never-married"])
    occupation: str = Field(..., examples=["Adm-clerical"])
    relationship: str = Field(..., examples=["Not-in-family"])
    race: str = Field(..., examples=["White"])
    sex: str = Field(..., examples=["Male"])
    capital_gain: int = Field(..., alias="capital-gain", examples=[2174])
    capital_loss: int = Field(..., alias="capital-loss", examples=[0])
    hours_per_week: int = Field(..., alias="hours-per-week", examples=[40])
    native_country: str = Field(..., alias="native-country", examples=["United-States"])

    def to_feature_dict(self) -> dict[str, object]:
        """Return a cleaned feature dict matching the trained pipeline schema.

        Applies the same transformations as clean_raw_input():
        - Renames fnlwgt → fnlgt (the actual CSV header used during training).
        - Converts hyphenated names to underscores.
        - Drops 'education' (redundant with education_num; excluded at training).

        The returned dict can be passed directly to predict_from_payload() as a
        single-row payload.
        """
        return {
            "age": self.age,
            "fnlgt": self.fnlwgt,  # renamed: fnlwgt → fnlgt
            "education_num": self.education_num,
            "capital_gain": self.capital_gain,
            "capital_loss": self.capital_loss,
            "hours_per_week": self.hours_per_week,
            "workclass": self.workclass,
            "marital_status": self.marital_status,
            "occupation": self.occupation,
            "relationship": self.relationship,
            "race": self.race,
            "sex": self.sex,
            "native_country": self.native_country,
            # 'education' is intentionally excluded — dropped during training
        }


class PredictResponse(BaseModel):
    prediction: str


class PreprocessingFeatureRow(BaseModel):
    """Schema for one row of cleaned features used by preprocessing.py."""

    model_config = ConfigDict(extra="forbid")

    age: int | float | None
    workclass: str | None
    fnlgt: int | float | None
    education_num: int | float | None
    marital_status: str | None
    occupation: str | None
    relationship: str | None
    race: str | None
    sex: str | None
    capital_gain: int | float | None
    capital_loss: int | float | None
    hours_per_week: int | float | None
    native_country: str | None
