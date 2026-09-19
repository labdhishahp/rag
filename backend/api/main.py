"""
FastAPI application entry point.

Run locally from backend/:
    uvicorn api.main:app --reload --port 8000

The RAG core (embedding model, LLM client) is built once at startup and held
on app.state, because they are expensive to build and cheap to hold. Sessions
(one indexed document + its conversation) are addressed by a server-issued
session_id instead of a browser's session_state.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .rag_bridge import AppState
from .routers import chat, documents, health

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_api")


def _check_deployment_config() -> None:
    """
    Refuse to start misconfigured on a deployment.

    Both of these are safe to omit locally and dangerous to omit in public:
    an unauthenticated API fronting two paid quotas is an open invitation, and
    a wildcard CORS origin hands it to any website. Failing at boot makes the
    mistake obvious instead of quietly expensive.
    """
    if not os.getenv("VERCEL"):
        return
    problems = []
    if not settings.api_key:
        problems.append("API_KEY is not set — the API would be open to anyone.")
    if "*" in settings.allowed_origins:
        problems.append("ALLOWED_ORIGINS contains '*' — any site could call this API.")
    if problems:
        raise RuntimeError("Refusing to start: " + " ".join(problems))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    _check_deployment_config()
    state = AppState.build(database_url=settings.database_url)
    app.state.rag_state = state
    logger.info(
        "Startup complete. llm_providers=%s storage=%s",
        state.available_llms or "none",
        type(state.storage).__name__,
    )
    yield
    # Hand the database connection back rather than leaving the pooler to time
    # it out. Vercel allows ~500ms for shutdown, and closing a socket is fast.
    close = getattr(state.storage, "close", None)
    if close is not None:
        try:
            close()
        except Exception:  # noqa: BLE001 - shutdown must not raise
            logger.warning("Ignoring error while closing storage", exc_info=True)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Knowledge Assistant API",
        description="HTTP API in front of the document ingestion / RAG / conversation core in src/.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(chat.router)
    return app


app = create_app()
