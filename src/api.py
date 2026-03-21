import os
from collections import deque
from datetime import datetime, timezone
from time import perf_counter
from typing import Deque

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data_preprocessing import clean_text
from src.inference import InferenceEngine


class TicketRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Customer support message")
    priority: str = Field(default="Normal", description="Ticket priority")
    channel: str = Field(default="Web Portal", description="Ticket source channel")


class PredictionResponse(BaseModel):
    category: str
    confidence: float
    cleaned_text: str
    processing_timeline: list["TimelineStep"]
    total_processing_ms: int


class TimelineStep(BaseModel):
    label: str
    elapsed_ms: int


class TicketFeedItem(BaseModel):
    id: str
    channel: str
    language: str
    category: str
    status: str
    age: str
    priority: str


class TicketDetailResponse(BaseModel):
    id: str
    channel: str
    language: str
    category: str
    status: str
    age: str
    priority: str
    cleaned_text: str
    confidence: float
    processing_timeline: list[TimelineStep]
    total_processing_ms: int


class QueueCounts(BaseModel):
    total_open: int
    technical_support: int
    billing_inquiry: int
    order_tracking: int
    manual_review: int


class DashboardResponse(BaseModel):
    queue: QueueCounts
    recent_tickets: list[TicketFeedItem]
    updated_at: str


def detect_language(text: str) -> str:
    lower = text.lower()
    es_markers = ["hola", "pedido", "factura", "pago", "cobrado", "cuenta", "no funciona"]
    de_markers = ["hallo", "rechnung", "paket", "bestellung", "nicht", "funktioniert", "konto"]

    if any(token in lower for token in es_markers):
        return "ES"
    if any(token in lower for token in de_markers):
        return "DE"
    return "EN"


def format_age(created_at: datetime) -> str:
    delta_seconds = int((datetime.now(timezone.utc) - created_at).total_seconds())
    minutes = max(delta_seconds // 60, 0)
    return f"{minutes:02d}m"


def to_feed_item(ticket: dict) -> TicketFeedItem:
    return TicketFeedItem(
        id=ticket["id"],
        channel=ticket["channel"],
        language=ticket["language"],
        category=ticket["category"],
        status=ticket["status"],
        age=format_age(ticket["created_at"]),
        priority=ticket["priority"],
    )


def to_ticket_detail(ticket: dict) -> TicketDetailResponse:
    return TicketDetailResponse(
        id=ticket["id"],
        channel=ticket["channel"],
        language=ticket["language"],
        category=ticket["category"],
        status=ticket["status"],
        age=format_age(ticket["created_at"]),
        priority=ticket["priority"],
        cleaned_text=ticket["cleaned_text"],
        confidence=ticket["confidence"],
        processing_timeline=ticket["processing_timeline"],
        total_processing_ms=ticket["total_processing_ms"],
    )


def calculate_queue_counts(tickets: list[dict]) -> QueueCounts:
    return QueueCounts(
        total_open=len(tickets),
        technical_support=sum(1 for t in tickets if t["category"] == "Technical Support"),
        billing_inquiry=sum(1 for t in tickets if t["category"] == "Billing Inquiry"),
        order_tracking=sum(1 for t in tickets if t["category"] == "Order Tracking"),
        manual_review=sum(1 for t in tickets if t["status"] == "Escalated"),
    )


app = FastAPI(title="Multilingual Ticket Classifier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    app.state.engine = None
    app.state.startup_error = None
    app.state.ticket_counter = 1200
    app.state.tickets: Deque[dict] = deque(maxlen=150)

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


@app.get("/dashboard", response_model=DashboardResponse)
def dashboard() -> DashboardResponse:
    tickets = list(app.state.tickets)
    tickets.sort(key=lambda item: item["created_at"], reverse=True)

    return DashboardResponse(
        queue=calculate_queue_counts(tickets),
        recent_tickets=[to_feed_item(ticket) for ticket in tickets[:12]],
        updated_at=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/tickets/{ticket_id}", response_model=TicketDetailResponse)
def ticket_details(ticket_id: str) -> TicketDetailResponse:
    for ticket in app.state.tickets:
        if ticket["id"] == ticket_id:
            return to_ticket_detail(ticket)

    raise HTTPException(status_code=404, detail="Ticket not found")


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TicketRequest) -> PredictionResponse:
    t0 = perf_counter()

    if app.state.engine is None:
        raise HTTPException(
            status_code=503,
            detail=app.state.startup_error or "Model is not loaded yet.",
        )

    cleaned = clean_text(payload.text)
    t1 = perf_counter()

    if not cleaned:
        raise HTTPException(status_code=400, detail="Text is empty after preprocessing.")

    pred = app.state.engine.predict(cleaned)
    t2 = perf_counter()

    status = "Escalated" if pred["confidence"] < 0.5 else "Assigned"
    app.state.ticket_counter += 1
    t3 = perf_counter()

    timeline = [
        TimelineStep(label=f"Received from {payload.channel}", elapsed_ms=0),
        TimelineStep(label="Text normalized and cleaned", elapsed_ms=int(round((t1 - t0) * 1000))),
        TimelineStep(label="Model inference complete", elapsed_ms=int(round((t2 - t0) * 1000))),
        TimelineStep(label="Queue recommendation generated", elapsed_ms=int(round((t3 - t0) * 1000))),
    ]
    total_processing_ms = int(round((t3 - t0) * 1000))

    app.state.tickets.append(
        {
            "id": f"GT-{app.state.ticket_counter}",
            "channel": payload.channel,
            "language": detect_language(cleaned),
            "category": pred["category"],
            "status": status,
            "priority": payload.priority,
            "cleaned_text": cleaned,
            "confidence": pred["confidence"],
            "processing_timeline": timeline,
            "total_processing_ms": total_processing_ms,
            "created_at": datetime.now(timezone.utc),
        }
    )

    return PredictionResponse(
        category=pred["category"],
        confidence=pred["confidence"],
        cleaned_text=cleaned,
        processing_timeline=timeline,
        total_processing_ms=total_processing_ms,
    )
