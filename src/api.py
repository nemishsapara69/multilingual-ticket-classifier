import os
from datetime import datetime, timezone
from time import perf_counter
from typing import Callable

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from src.auth_store import AuthStore
from src.data_preprocessing import clean_text
from src.inference import InferenceEngine
from src.ticket_store import TicketStore


class KeywordFallbackEngine:
    def __init__(self) -> None:
        self.rules = {
            "Technical Support": [
                "not working",
                "error",
                "issue",
                "broken",
                "no funciona",
                "problema",
                "kaputt",
                "funktioniert nicht",
                "login",
                "iniciar sesion",
            ],
            "Billing Inquiry": [
                "charged",
                "billing",
                "invoice",
                "payment",
                "refund",
                "cobrado",
                "factura",
                "pago",
                "rechnung",
            ],
            "Order Tracking": [
                "where is my package",
                "tracking",
                "delivery",
                "shipment",
                "pedido",
                "entrega",
                "paket",
                "lieferung",
            ],
        }

    def predict(self, text: str) -> dict:
        lower = text.lower()
        best_label = "Technical Support"
        best_score = 0

        for label, terms in self.rules.items():
            score = sum(1 for term in terms if term in lower)
            if score > best_score:
                best_score = score
                best_label = label

        confidence = 0.82 if best_score > 0 else 0.55
        return {"category": best_label, "confidence": confidence}


class TicketRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Customer support message")
    priority: str = Field(default="Normal", description="Ticket priority")
    channel: str = Field(default="Web Portal", description="Ticket source channel")


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str


class AuthConfigResponse(BaseModel):
    enabled: bool


class UserProfile(BaseModel):
    username: str
    role: str


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


class TicketMessageRequest(BaseModel):
    message: str = Field(..., min_length=1)


class TicketMessageItem(BaseModel):
    id: int
    ticket_id: str
    sender: str
    role: str
    message: str
    created_at: str


class TicketMessagesResponse(BaseModel):
    ticket_id: str
    messages: list[TicketMessageItem]


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
security = HTTPBearer(auto_error=False)

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

    app.state.auth_enabled = os.getenv("ENABLE_AUTH", "1") == "1"
    auth_db_path = os.getenv("AUTH_DB_PATH", "data/processed/auth.db")
    app.state.auth_store = AuthStore(db_path=auth_db_path)
    app.state.auth_store.initialize()
    app.state.auth_store.ensure_user(
        username=os.getenv("AUTH_ADMIN_USERNAME", "admin"),
        password=os.getenv("AUTH_ADMIN_PASSWORD", "change-admin-pass"),
        role="admin",
    )
    app.state.auth_store.ensure_user(
        username=os.getenv("AUTH_AGENT_USERNAME", "agent"),
        password=os.getenv("AUTH_AGENT_PASSWORD", "change-agent-pass"),
        role="agent",
    )
    app.state.auth_store.ensure_user(
        username=os.getenv("AUTH_VIEWER_USERNAME", "viewer"),
        password=os.getenv("AUTH_VIEWER_PASSWORD", "change-viewer-pass"),
        role="viewer",
    )

    db_path = os.getenv("TICKET_DB_PATH", "data/processed/tickets.db")
    app.state.ticket_store = TicketStore(db_path=db_path, max_rows=150)
    app.state.ticket_store.initialize()
    app.state.ticket_counter = app.state.ticket_store.next_counter(default_counter=1200)

    if os.getenv("SKIP_MODEL_LOAD", "0") == "1":
        return

    try:
        app.state.engine = InferenceEngine.from_local_or_hub(
            local_model_dir=os.getenv("MODEL_DIR", "models/best"),
            hub_model_id=os.getenv("MODEL_ID"),
        )
    except Exception as exc:
        app.state.startup_error = str(exc)
        if os.getenv("ENABLE_RULE_FALLBACK", "1") == "1":
            app.state.engine = KeywordFallbackEngine()


def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> dict:
    if not app.state.auth_enabled:
        return {"username": "local-dev", "role": "admin"}

    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    session = app.state.auth_store.get_session(credentials.credentials)
    if not session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    return {"username": session["username"], "role": session["role"]}


def require_roles(*roles: str) -> Callable:
    def _checker(user: dict = Depends(get_current_user)) -> dict:
        if user["role"] not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return user

    return _checker


@app.get("/auth/config", response_model=AuthConfigResponse)
def auth_config() -> AuthConfigResponse:
    return AuthConfigResponse(enabled=bool(app.state.auth_enabled))


@app.post("/auth/login", response_model=LoginResponse)
def auth_login(payload: LoginRequest) -> LoginResponse:
    if not app.state.auth_enabled:
        token = app.state.auth_store.create_session(username="local-dev", role="admin", ttl_hours=48)
        return LoginResponse(access_token=token, username="local-dev", role="admin")

    user = app.state.auth_store.verify_credentials(payload.username.strip(), payload.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")

    token = app.state.auth_store.create_session(username=user["username"], role=user["role"], ttl_hours=24)
    return LoginResponse(access_token=token, username=user["username"], role=user["role"])


@app.get("/auth/me", response_model=UserProfile)
def auth_me(user: dict = Depends(get_current_user)) -> UserProfile:
    return UserProfile(username=user["username"], role=user["role"])


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": app.state.engine is not None,
        "startup_error": app.state.startup_error,
    }


@app.get("/dashboard", response_model=DashboardResponse)
def dashboard(_: dict = Depends(require_roles("admin", "agent", "viewer"))) -> DashboardResponse:
    tickets = app.state.ticket_store.list_tickets()

    return DashboardResponse(
        queue=calculate_queue_counts(tickets),
        recent_tickets=[to_feed_item(ticket) for ticket in tickets[:12]],
        updated_at=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/tickets/{ticket_id}", response_model=TicketDetailResponse)
def ticket_details(ticket_id: str, _: dict = Depends(require_roles("admin", "agent", "viewer"))) -> TicketDetailResponse:
    ticket = app.state.ticket_store.get_ticket(ticket_id)
    if ticket:
        return to_ticket_detail(ticket)

    raise HTTPException(status_code=404, detail="Ticket not found")


@app.get("/tickets/{ticket_id}/messages", response_model=TicketMessagesResponse)
def ticket_messages(ticket_id: str, _: dict = Depends(require_roles("admin", "agent", "viewer"))) -> TicketMessagesResponse:
    ticket = app.state.ticket_store.get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    messages = app.state.ticket_store.list_messages(ticket_id)
    return TicketMessagesResponse(ticket_id=ticket_id, messages=[TicketMessageItem(**msg) for msg in messages])


@app.post("/tickets/{ticket_id}/messages", response_model=TicketMessageItem)
def add_ticket_message(
    ticket_id: str,
    payload: TicketMessageRequest,
    user: dict = Depends(require_roles("admin", "agent")),
) -> TicketMessageItem:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        created = app.state.ticket_store.add_message(
            ticket_id=ticket_id,
            sender=user["username"],
            role=user["role"],
            message=message,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Ticket not found")

    return TicketMessageItem(**created)


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TicketRequest, _: dict = Depends(require_roles("admin", "agent"))) -> PredictionResponse:
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

    app.state.ticket_store.insert_ticket(
        {
            "id": f"GT-{app.state.ticket_counter}",
            "channel": payload.channel,
            "language": detect_language(cleaned),
            "category": pred["category"],
            "status": status,
            "priority": payload.priority,
            "cleaned_text": cleaned,
            "confidence": pred["confidence"],
            "processing_timeline": [step.model_dump() for step in timeline],
            "total_processing_ms": total_processing_ms,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    app.state.ticket_store.add_message(
        ticket_id=f"GT-{app.state.ticket_counter}",
        sender="customer",
        role="customer",
        message=payload.text,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    return PredictionResponse(
        category=pred["category"],
        confidence=pred["confidence"],
        cleaned_text=cleaned,
        processing_timeline=timeline,
        total_processing_ms=total_processing_ms,
    )
