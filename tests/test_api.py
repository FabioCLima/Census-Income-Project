from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

# ── Shared fixture data ────────────────────────────────────────────────────────

VALID_PAYLOAD = {
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


# ── Healthcheck ────────────────────────────────────────────────────────────────


def test_root_returns_ok() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ── /predict — happy path ──────────────────────────────────────────────────────


def test_predict_returns_placeholder_when_model_is_missing() -> None:
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    assert response.json() == {"prediction": "model_not_trained"}


def test_predict_response_has_prediction_key() -> None:
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert "prediction" in response.json()


# ── /predict — validation ──────────────────────────────────────────────────────


def test_predict_returns_422_when_required_field_is_missing() -> None:
    incomplete = {k: v for k, v in VALID_PAYLOAD.items() if k != "age"}

    response = client.post("/predict", json=incomplete)

    assert response.status_code == 422


def test_predict_returns_422_when_payload_is_empty() -> None:
    response = client.post("/predict", json={})

    assert response.status_code == 422


def test_predict_returns_422_when_field_has_wrong_type() -> None:
    bad_payload = {**VALID_PAYLOAD, "age": "not-a-number"}

    response = client.post("/predict", json=bad_payload)

    assert response.status_code == 422
