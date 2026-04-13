import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi import FastAPI, HTTPException

from census.data import EXPECTED_COLUMNS
from census.model import predict_from_payload

app = FastAPI(
    title="Census Income Classification API",
    description="FastAPI service for Census Income binary classification.",
    version="0.1.0",
)


@app.get("/")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: dict[str, object]) -> dict[str, object]:
    missing_fields = [column for column in EXPECTED_COLUMNS if column not in payload]
    if missing_fields:
        msg = f"Missing required fields: {', '.join(missing_fields)}"
        raise HTTPException(status_code=422, detail=msg)

    prediction = predict_from_payload(payload)
    return {"prediction": prediction}
