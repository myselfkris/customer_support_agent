"""FastAPI application shell for the Customer Support Agent.

Run:
    uvicorn app.main:app --reload

Endpoints:
    GET  /health            — service + configuration status
    POST /chat              — one agent turn (requires customer API key)
    POST /admin/login       — admin password -> JWT
    GET  /admin/approvals   — list pending approvals (requires admin JWT)

Mock mode: set ``MOCK_MODE=true`` in ``.env`` to run fully offline (no Gemini calls).
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException

from app.auth import (
    authenticate_admin,
    create_access_token,
    require_admin,
    require_customer,
)
from app.config import settings
from app.schemas import AdminLoginRequest, ChatRequest, ChatResponse, TokenResponse
from agents.reliable_agent import ReliableAgent
from agents.tool_calling_agent import list_pending_approvals
from db.base import init_db

# Single shared agent instance. ReliableAgent holds in-memory per-user state
# (memory + rate limiting), so a lock serializes access until those move to
# shared storage (Phase 4/5).
_agent: Optional[ReliableAgent] = None
_agent_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    init_db()  # create any missing tables (incl. admin_users, customer_api_keys)
    # Built lazily; ReliableAgent imports the LangGraph graph on demand
    # (or uses MockApp when MOCK_MODE=true).
    _agent = ReliableAgent()
    yield
    _agent = None


app = FastAPI(
    title="Customer Support Agent",
    description="Agentic AI customer support API (Gemini + LangGraph + pgvector).",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    """Lightweight liveness/configuration check (no heavy imports)."""
    from agents.mock import is_mock_mode

    return {
        "status": "ok",
        "llm_configured": settings.llm_configured,
        "mock_mode": is_mock_mode(),
        "generation_model": settings.generation_model,
        "embedding_model": settings.embedding_model,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, customer_id: int = Depends(require_customer)) -> ChatResponse:
    """Run one agent turn. Identity comes from the API key, not the request body."""
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    def _run() -> str:
        with _agent_lock:
            return _agent.chat(str(customer_id), request.message)

    response = await asyncio.to_thread(_run)
    return ChatResponse(user_id=str(customer_id), response=response)


@app.post("/admin/login", response_model=TokenResponse)
async def admin_login(body: AdminLoginRequest) -> TokenResponse:
    """Password -> short-lived JWT. The vault's front door."""
    admin = authenticate_admin(body.username, body.password)
    if admin is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return TokenResponse(access_token=create_access_token(admin["id"]))


@app.get("/admin/approvals")
async def list_approvals(admin: dict = Depends(require_admin)):
    """List every approval waiting for a human decision (admin only)."""
    return list_pending_approvals()
