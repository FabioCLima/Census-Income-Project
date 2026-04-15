import os

from fastapi import FastAPI

from census.inference import predict_from_payload
from census.schemas import PredictRequest, PredictResponse

# ── DVC pull on Heroku startup ─────────────────────────────────────────────────
# When running on a Heroku dyno, pull any DVC-tracked artifacts (data, model)
# from the configured remote store before the API starts serving requests.
# The os.path.isdir(".dvc") guard makes this a no-op if DVC is not set up.
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI(
    title="Census Income Classification API",
    description="Predicts whether a person earns >50K/year based on census features.",
    version="0.1.0",
)


@app.get("/")
def root() -> dict[str, str]:
    """Welcome message for the Census Income Classification API."""
    return {
        "message": (
            "Welcome to the Census Income Classification API. "
            "Send a POST request to /predict with census features "
            "to predict whether income exceeds $50K/year."
        )
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Lightweight health-check endpoint for uptime probes and deploy checks."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    prediction = predict_from_payload(request.to_feature_dict())
    return PredictResponse(prediction=prediction)
