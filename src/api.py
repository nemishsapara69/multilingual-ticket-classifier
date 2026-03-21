import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data_preprocessing import clean_text
from src.inference import InferenceEngine


class TicketRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Customer support message")


class PredictionResponse(BaseModel):
    category: str
    confidence: float
    cleaned_text: str


app = FastAPI(title="Multilingual Ticket Classifier API", version="1.0.0")


@app.on_event("startup")
def startup_event() -> None:
    app.state.engine = None
    app.state.startup_error = None

    if os.getenv("SKIP_MODEL_LOAD", "0") == "1":
        return

    try:
        app.state.engine = InferenceEngine.from_local_or_hub(
            local_model_dir=os.getenv("MODEL_DIR", "models/best"),
            hub_model_id=os.getenv("MODEL_ID"),
        )
    except Exception as exc:
        app.state.startup_error = str(exc)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": app.state.engine is not None,
        "startup_error": app.state.startup_error,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TicketRequest) -> PredictionResponse:
    if app.state.engine is None:
        raise HTTPException(
            status_code=503,
            detail=app.state.startup_error or "Model is not loaded yet.",
        )

    cleaned = clean_text(payload.text)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Text is empty after preprocessing.")

    pred = app.state.engine.predict(cleaned)
    return PredictionResponse(
        category=pred["category"],
        confidence=pred["confidence"],
        cleaned_text=cleaned,
    )
