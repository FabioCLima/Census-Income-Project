from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    """Validated input payload for the /predict endpoint.

    Accepts hyphenated JSON keys (e.g. "education-num") via field aliases,
    which match the original UCI column names. Python-safe underscore names
    (e.g. "education_num") are also accepted when populate_by_name is True.
    """

    model_config = ConfigDict(populate_by_name=True)

    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    def to_feature_dict(self) -> dict[str, object]:
        """Return a dict keyed by the original hyphenated column names.

        The underlying model was trained on features with hyphenated column
        names (e.g. "education-num"), so we must reconstruct that format
        before calling inference.
        """
        return {
            "age": self.age,
            "workclass": self.workclass,
            "fnlwgt": self.fnlwgt,
            "education": self.education,
            "education-num": self.education_num,
            "marital-status": self.marital_status,
            "occupation": self.occupation,
            "relationship": self.relationship,
            "race": self.race,
            "sex": self.sex,
            "capital-gain": self.capital_gain,
            "capital-loss": self.capital_loss,
            "hours-per-week": self.hours_per_week,
            "native-country": self.native_country,
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
