"""Liveness/readiness for deployment monitoring."""

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request):
    state = request.app.state.rag_state
    storage_ok, storage_error = state.storage.healthy()
    return {
        # "ok" only when the dependencies a request actually needs are reachable.
        # A process that answers HTTP but cannot reach its database is not ready,
        # and a monitor should be able to tell the difference.
        "status": "ok" if storage_ok else "degraded",
        "embedding_backend": getattr(state.embedding_model, "backend_name", None),
        "embedding_model": getattr(state.embedding_model, "model_name", None),
        "embedding_dimension": getattr(state.embedding_model, "dimension", None),
        "llm_configured": state.llm is not None,
        "llm_error": state.llm_init_error,
        "storage": type(state.storage).__name__,
        "storage_ok": storage_ok,
        "storage_error": storage_error,
    }
