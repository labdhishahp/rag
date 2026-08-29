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
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .rag_bridge import AppState
from .routers import chat, documents, health

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading embedding model and LLM client...")
    app.state.rag_state = AppState.build(
        database_url=settings.database_url,
        llm_provider=settings.llm_provider,
    )
    logger.info(
        "Startup complete. llm_configured=%s storage=%s",
        app.state.rag_state.llm is not None,
        type(app.state.rag_state.storage).__name__,
    )
    yield


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
