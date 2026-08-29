"""Liveness/readiness for deployment monitoring."""

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request):
    state = request.app.state.rag_state
    return {
        "status": "ok",
        "embedding_model_loaded": state.embedding_model is not None,
        "embedding_model_name": getattr(state.embedding_model, "model_name", None),
        "embedding_dimension": getattr(state.embedding_model, "dimension", None),
        "llm_configured": state.llm is not None,
        "llm_error": state.llm_init_error,
        "active_sessions": state.sessions.count(),
    }
