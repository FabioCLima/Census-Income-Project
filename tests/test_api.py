from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_root_returns_ok() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_placeholder_when_model_is_missing() -> None:
    payload = {
        "age": 39,
        "workclass": "State-gov",
        "fnlwgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {"prediction": "model_not_trained"}

