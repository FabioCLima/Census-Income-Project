from fastapi import FastAPI

from census.model import predict_from_payload
from census.schemas import PredictRequest, PredictResponse

app = FastAPI(
    title="Census Income Classification API",
    description="Predicts whether a person earns >50K/year based on census features.",
    version="0.1.0",
)


@app.get("/")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    prediction = predict_from_payload(request.to_feature_dict())
    return PredictResponse(prediction=prediction)
