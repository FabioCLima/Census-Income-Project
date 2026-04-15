"""Unit tests for the Census Income Classification API.

Test coverage
-------------
- GET /          : status code + welcome message content
- GET /health    : health-check contract for deploy probes
- POST /predict  : one test per possible model output (">50K" and "<=50K")
- POST /predict  : Pydantic validation (422 on bad payloads)
"""

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

# ── Shared payloads ────────────────────────────────────────────────────────────

# First row of the UCI Adult dataset — ground-truth label: <=50K
LOW_INCOME_PAYLOAD = {
    "age": 25,
    "workclass": "Private",
    "fnlwgt": 226956,
    "education": "11th",
    "education-num": 7,
    "marital-status": "Never-married",
    "occupation": "Machine-op-inspct",
    "relationship": "Own-child",
    "race": "Black",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# High-income profile — strong signals for >50K:
#   married executive with a master's degree and significant capital gain
HIGH_INCOME_PAYLOAD = {
    "age": 50,
    "workclass": "Self-emp-not-inc",
    "fnlwgt": 209642,
    "education": "Masters",
    "education-num": 14,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 15024,
    "capital-loss": 0,
    "hours-per-week": 60,
    "native-country": "United-States",
}


# ── GET / ──────────────────────────────────────────────────────────────────────


def test_root_returns_welcome_message() -> None:
    """GET / should return 200 with a welcome message in the body."""
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert "message" in body
    assert "Welcome" in body["message"]


def test_health_returns_ok_status() -> None:
    """GET /health should return 200 with a simple readiness payload."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ── POST /predict — one test per model output ──────────────────────────────────


def test_predict_returns_low_income() -> None:
    """POST /predict should return '<=50K' for a low-income profile."""
    response = client.post("/predict", json=LOW_INCOME_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["prediction"] == "<=50K"


def test_predict_returns_high_income() -> None:
    """POST /predict should return '>50K' for a high-income profile."""
    response = client.post("/predict", json=HIGH_INCOME_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["prediction"] == ">50K"


# ── POST /predict — Pydantic validation ───────────────────────────────────────


def test_predict_returns_422_when_required_field_is_missing() -> None:
    incomplete = {k: v for k, v in LOW_INCOME_PAYLOAD.items() if k != "age"}

    response = client.post("/predict", json=incomplete)

    assert response.status_code == 422


def test_predict_returns_422_when_payload_is_empty() -> None:
    response = client.post("/predict", json={})

    assert response.status_code == 422


def test_predict_returns_422_when_field_has_wrong_type() -> None:
    bad_payload = {**LOW_INCOME_PAYLOAD, "age": "not-a-number"}

    response = client.post("/predict", json=bad_payload)

    assert response.status_code == 422
