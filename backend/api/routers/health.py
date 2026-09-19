"""Liveness/readiness for deployment monitoring."""

from fastapi import APIRouter, Request

from ..config import settings
from ..rag_bridge import EmbeddingError

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request):
    state = request.app.state.rag_state
    storage_ok, storage_error = state.storage.healthy()

    # Report each provider separately: the primary being down is survivable
    # (new documents fall back to the secondary), but a provider being down
    # makes documents ALREADY indexed by it unanswerable, so an operator needs
    # to see them individually rather than as one aggregate flag.
    providers = {}
    for name in ("huggingface", "gemini"):
        try:
            provider = state.provider(name)
            providers[name] = {
                "available": True,
                "model": provider.model_name,
                "dimension": provider.dimension,
            }
        except EmbeddingError as exc:
            providers[name] = {"available": False, "error": str(exc)}

    embedding_ok = any(p["available"] for p in providers.values())
    return {
        # "ok" only when the dependencies a request actually needs are
        # reachable. A process that answers HTTP but cannot embed or reach its
        # database is not ready, and a monitor should see the difference.
        "status": "ok" if (storage_ok and embedding_ok) else "degraded",
        "embedding_primary": "huggingface",
        "embedding_providers": providers,
        # Per provider, so the UI can offer only what will actually answer.
        "llm_providers": {
            name: {"available": err is None, **({} if err is None else {"error": err})}
            for name, err in state.llm_status.items()
        },
        "llm_default": settings.llm_provider,
        "llm_configured": bool(state.available_llms),
        "storage": type(state.storage).__name__,
        "storage_ok": storage_ok,
        "storage_error": storage_error,
    }
