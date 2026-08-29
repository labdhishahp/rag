"""
The one file that knows both the API and the RAG core.

Every other module in this package (routers, schemas) talks to session
objects and plain JSON-safe dicts. This module is where that boundary is
built: it puts src/ on sys.path exactly the way app.py and tests/conftest.py
already do (no package restructuring of the RAG core), holds the process-wide
singletons (embedding model, LLM client) the same way Streamlit's
st.cache_resource did, and turns rag.py's dataclass-shaped results into JSON.

Session store: an in-memory dict keyed by a server-issued session_id, because
this API replaces Streamlit's per-browser-tab st.session_state with an
explicit handle a stateless HTTP client can pass back on every request. One
process, one dict — the same "one document at a time" model the Streamlit
app used, just addressable over HTTP instead of implicit in a Streamlit
session. No database: a TTL + max-session cap bounds memory instead.
"""

import dataclasses
import logging
import sys
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np

API_DIR = Path(__file__).resolve().parent
BACKEND_DIR = API_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from conversation import Conversation  # noqa: E402
from embeddings import EmbeddingModel  # noqa: E402
from llm import LLMClient, LLMError, create_llm  # noqa: E402
from pipeline import DocumentProcessingError, index_document_from_upload  # noqa: E402
from rag import RAGSystem  # noqa: E402
from retriever import Retriever  # noqa: E402

logger = logging.getLogger("rag_api")

__all__ = [
    "Conversation",
    "LLMClient",
    "LLMError",
    "DocumentProcessingError",
    "index_document_from_upload",
    "RAGSystem",
    "Session",
    "SessionStore",
    "AppState",
    "serialize_answer_result",
]


class Session:
    """One indexed document plus the conversation held against it."""

    def __init__(self, session_id: str, retriever: Retriever, metadata: dict):
        self.session_id = session_id
        self.retriever = retriever
        self.metadata = metadata
        self.conversation = Conversation()
        self.created_at = time.time()
        self.last_used_at = self.created_at


class SessionStore:
    def __init__(self, max_sessions: int = 200, ttl_seconds: int = 6 * 3600):
        self._sessions: dict[str, Session] = {}
        self._lock = Lock()
        self._max_sessions = max_sessions
        self._ttl_seconds = ttl_seconds

    def create(self, retriever: Retriever, metadata: dict) -> Session:
        with self._lock:
            self._evict_expired_locked()
            if len(self._sessions) >= self._max_sessions:
                oldest_id = min(self._sessions, key=lambda k: self._sessions[k].last_used_at)
                del self._sessions[oldest_id]
            session_id = uuid.uuid4().hex
            session = Session(session_id, retriever, metadata)
            self._sessions[session_id] = session
            return session

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if time.time() - session.last_used_at > self._ttl_seconds:
                del self._sessions[session_id]
                return None
            session.last_used_at = time.time()
            return session

    def reset_conversation(self, session_id: str) -> bool:
        session = self.get(session_id)
        if session is None:
            return False
        session.conversation = Conversation()
        return True

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _evict_expired_locked(self) -> None:
        now = time.time()
        expired = [sid for sid, s in self._sessions.items() if now - s.last_used_at > self._ttl_seconds]
        for sid in expired:
            del self._sessions[sid]


class AppState:
    """Process-wide singletons, built once at startup."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        llm: Optional[LLMClient],
        llm_init_error: Optional[str],
        sessions: SessionStore,
    ):
        self.embedding_model = embedding_model
        self.llm = llm
        self.llm_init_error = llm_init_error
        self.sessions = sessions

    @classmethod
    def build(cls, *, max_sessions: int, ttl_seconds: int, llm_provider: str = "gemini") -> "AppState":
        embedding_model = EmbeddingModel()

        llm: Optional[LLMClient] = None
        llm_init_error: Optional[str] = None
        try:
            llm = create_llm(llm_provider)
        except Exception as exc:  # noqa: BLE001 - missing/invalid key, not a crash
            llm_init_error = str(exc)
            logger.warning("LLM not configured at startup: %s", llm_init_error)

        sessions = SessionStore(max_sessions=max_sessions, ttl_seconds=ttl_seconds)
        return cls(embedding_model, llm, llm_init_error, sessions)


def _sanitize(obj):
    """Recursively turn dataclasses / numpy scalars into plain JSON-safe values."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _sanitize(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def serialize_answer_result(result: dict) -> dict:
    """
    RAGSystem.answer()'s result, as JSON the frontend can render directly.

    Mirrors exactly what app.py's `_render_result_details` showed (sources,
    matched vs. expanded chunks, evidence level, the evidence block verbatim)
    so the new frontend can reproduce the same "what did retrieval do"
    inspector without re-deriving anything the RAG core already computed.

    Deliberately omits `result["chunks"]` (the raw pre-expansion similarity
    hits) — the Streamlit UI never rendered them either; `passages` plus
    entry/expanded/dropped chunk ids already cover the inspectable surface.
    """
    context = result["context"]
    return _sanitize(
        {
            "question": result["question"],
            "retrieval_query": result["retrieval_query"],
            "was_follow_up": result["was_follow_up"],
            "answer": result["answer"],
            "depth": result["depth"],
            "wants_example": result["understanding"].wants_example,
            "low_confidence": result["low_confidence"],
            "refused": result["refused"],
            "llm_called": result["llm_called"],
            "llm_model": result["llm_model"],
            "evidence_level": context.evidence_level,
            "best_similarity": result["best_similarity"],
            "sources": result["sources"],
            "source_citations": result["source_citations"],
            "cited_labels": result["cited_labels"],
            "cited_sources": result["cited_sources"],
            "entry_chunk_ids": context.entry_chunk_ids,
            "expanded_chunk_ids": context.expanded_chunk_ids,
            "dropped_chunk_ids": context.dropped_chunk_ids,
            "context_chars": context.total_chars,
            "duplicate_chars_removed": context.duplicate_chars_removed,
            "context_formatted": context.formatted,
            "passages": [
                {
                    "text": p.text,
                    "document_name": p.document_name,
                    "page_number": p.page_number,
                    "page_label": p.page_label,
                    "section": p.section,
                    "chunk_ids": p.chunk_ids,
                    "entry_chunk_ids": p.entry_chunk_ids,
                    "is_pure_expansion": p.is_pure_expansion,
                    "best_similarity": p.best_similarity,
                }
                for p in context.passages
            ],
            "top_k": result["top_k"],
        }
    )


def public_document_metadata(metadata: dict) -> dict:
    """Document metadata safe to store on a session and return over the API.

    Strips `chunks` and `embeddings` — pipeline.py attaches the full chunk
    list and the raw embedding matrix (a numpy array) for callers that build
    on top of it in-process; neither belongs in an HTTP response or in
    long-lived session memory.
    """
    return {k: v for k, v in metadata.items() if k not in ("chunks", "embeddings")}
