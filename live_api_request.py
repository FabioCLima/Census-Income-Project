"""Script to test the live Census Income Classification API on Heroku.

Usage
-----
    python live_api_request.py

Set the APP_URL variable below to your deployed Heroku app URL before running.
"""

import json

import requests

# ── Configuration ──────────────────────────────────────────────────────────────
APP_URL = "https://census-income-api-7cfe90f1b0a4.herokuapp.com"

SAMPLE_PAYLOAD = {
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


def check_root(base_url: str) -> None:
    """GET / — verify the API is live and read the welcome message."""
    url = f"{base_url}/"
    print(f"GET {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    print(f"  Status : {response.status_code}")
    print(f"  Body   : {response.json()}\n")


def run_inference(base_url: str, payload: dict) -> str:
    """POST /predict — send a census record and return the income prediction."""
    url = f"{base_url}/predict"
    print(f"POST {url}")
    print(f"  Payload: {json.dumps(payload, indent=4)}")
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    prediction: str = response.json()["prediction"]
    print(f"  Status     : {response.status_code}")
    print(f"  Prediction : {prediction}\n")
    return prediction


if __name__ == "__main__":
    check_root(APP_URL)
    run_inference(APP_URL, SAMPLE_PAYLOAD)
