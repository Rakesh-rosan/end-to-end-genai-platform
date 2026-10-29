"""FastAPI entrypoint.

Run:  uvicorn app.main:create_app --factory --reload
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from app.config import ROOT, Settings, get_settings
from app.gateway.auth import User, authenticate, issue_token, require_role
from app.gateway.rate_limit import rate_limited_user
from app.observability import log_event, metrics, request_id_var, setup_logging
from app.services import build_services

logger = logging.getLogger("app")


class TokenRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(default="default", max_length=64)


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    intent: str
    sources: list[str]
    cached: bool
    escalated: bool
    ticket_id: int | None
    blocked: bool
    trace: list[dict]


class FeedbackRequest(BaseModel):
    request_id: str
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1000)


class ResolveRequest(BaseModel):
    resolution: str = Field(min_length=1, max_length=2000)


class ProductUpdate(BaseModel):
    price: float | None = Field(default=None, gt=0)
    stock: int | None = Field(default=None, ge=0)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging()
    services = build_services(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(services.cdc.run()) if settings.cdc_enabled else None
        log_event(logger, "startup", llm_provider=settings.llm_provider, chunks=len(services.store))
        yield
        if task:
            task.cancel()

    app = FastAPI(title="ShopAssist GenAI Platform", lifespan=lifespan)
    app.state.settings = settings
    app.state.services = services

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        ms = round((time.perf_counter() - start) * 1000, 2)
        route = request.scope.get("route")
        path = route.path if route else "unmatched"
        response.headers["x-request-id"] = rid
        metrics.inc("http_requests_total", path=path, status=response.status_code)
        metrics.observe("http_latency_ms", ms, path=path)
        log_event(logger, "request", request_id=rid, method=request.method, path=path,
                  status=response.status_code, ms=ms)
        return response

    # --- public ----------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(ROOT / "static" / "index.html")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        ok = services.db.ping() and len(services.store) > 0
        if not ok:
            raise HTTPException(503, "not ready")
        return {"status": "ready", "chunks": len(services.store)}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics():
        return metrics.render()

    @app.post("/auth/token")
    async def login(body: TokenRequest):
        user = authenticate(body.username, body.password)
        if user is None:
            raise HTTPException(401, "invalid credentials")
        return {"access_token": issue_token(user, settings), "token_type": "bearer", "roles": user.roles}

    # --- shopper -----------------------------------------------------------------
    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(body: ChatRequest, user: User = Depends(rate_limited_user)):
        result = await services.pipeline.run(body.message, user, body.session_id)
        return ChatResponse(request_id=request_id_var.get(), **result.__dict__)

    @app.post("/v1/feedback")
    async def feedback(body: FeedbackRequest, user: User = Depends(rate_limited_user)):
        services.db.add_feedback(body.request_id, user.sub, body.rating, body.comment)
        metrics.inc("feedback_total", rating=body.rating)
        return {"status": "recorded"}

    # --- human agents / operators -------------------------------------------------
    agent_only = Depends(require_role("agent"))

    @app.get("/admin/tickets", dependencies=[agent_only])
    async def list_tickets(status: str | None = None):
        return services.db.list_tickets(status)

    @app.post("/admin/tickets/{ticket_id}/resolve", dependencies=[agent_only])
    async def resolve_ticket(ticket_id: int, body: ResolveRequest):
        if not services.db.resolve_ticket(ticket_id, body.resolution):
            raise HTTPException(404, "open ticket not found")
        return {"status": "resolved"}

    @app.post("/admin/products/{product_id}", dependencies=[agent_only])
    async def update_product(product_id: str, body: ProductUpdate):
        """Changing a product fires a DB trigger -> change_log -> CDC -> cache invalidation."""
        if not services.db.update_product(product_id, body.price, body.stock):
            raise HTTPException(404, "product not found")
        return {"status": "updated"}

    @app.post("/admin/cdc/poll", dependencies=[agent_only])
    async def cdc_poll():
        return await asyncio.to_thread(services.cdc.poll_once)

    @app.get("/admin/cache", dependencies=[agent_only])
    async def cache_stats():
        return services.cache.stats()

    @app.delete("/admin/cache", dependencies=[agent_only])
    async def cache_clear():
        services.cache.clear()
        return {"status": "cleared"}

    return app
