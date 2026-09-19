"""
The one file that knows both the API and the RAG core.

Every other module in this package (routers, storage) talks to session objects
and plain JSON-safe dicts. This module is where that boundary is built. It:

  * puts src/ on sys.path, so the RAG core is importable without restructuring
    it into a package,
  * holds the singletons worth building once per process (embedding model, LLM
    client),
  * assembles a session per request out of storage (see storage.py — nothing
    about a session is kept in memory between requests),
  * and turns rag.py's dataclass-shaped results into JSON the frontend renders.
"""

import dataclasses
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

API_DIR = Path(__file__).resolve().parent
BACKEND_DIR = API_DIR.parent
# src/ lives inside backend/ so that this directory is a self-contained Vercel
# service root — a service cannot read files above its own root.
SRC_DIR = BACKEND_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from conversation import Conversation  # noqa: E402
from document_loader import SUPPORTED_EXTENSIONS  # noqa: E402
from embeddings import (  # noqa: E402
    EmbeddingError,
    EmbeddingModel,
    create_provider,
    provider_for_indexing,
)
from llm import LLMClient, LLMError, create_llm  # noqa: E402
from pipeline import DocumentProcessingError, index_document_from_upload  # noqa: E402
from rag import RAGSystem  # noqa: E402
from retriever import Retriever  # noqa: E402
from vector_store import VectorStore  # noqa: E402

logger = logging.getLogger("rag_api")

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "Conversation",
    "EmbeddingError",
    "EmbeddingModel",
    "LLMClient",
    "LLMError",
    "DocumentProcessingError",
    "index_document_from_upload",
    "RAGSystem",
    "Retriever",
    "VectorStore",
    "AppState",
    "serialize_answer_result",
    "public_document_metadata",
]


class LoadedSession:
    """
    One request's view of a session: the document, its retriever, its history.

    Assembled per request rather than held between them. On a serverless
    platform there is no "between them" to hold it in — see storage.py.
    """

    def __init__(self, session_id: str, document_id: str, metadata: dict,
                 retriever: Retriever, conversation: Conversation):
        self.session_id = session_id
        self.document_id = document_id
        self.metadata = metadata
        self.retriever = retriever
        self.conversation = conversation
        # How many turns were already persisted when this session was loaded,
        # so a later append writes only what this request actually added.
        self.persisted_turns = len(conversation.turns)


class AppState:
    """
    Everything a request needs that is worth building once per process.

    The embedding model and LLM client are cheap to hold and expensive to
    rebuild; sessions and documents deliberately are NOT held here, because a
    serverless instance cannot be trusted to still exist on the next request.
    """

    def __init__(
        self,
        llm: Optional[LLMClient],
        llm_init_error: Optional[str],
        storage,
        retrievers,
    ):
        self.llm = llm
        self.llm_init_error = llm_init_error
        self.storage = storage
        self.retrievers = retrievers
        # Built on demand and kept, because constructing one is cheap but not
        # free. Keyed by provider name — a process may hold several at once when
        # documents in the store were indexed by different providers.
        self._providers: dict[str, object] = {}

    @classmethod
    def build(cls, *, database_url: Optional[str] = None, llm_provider: str = "gemini") -> "AppState":
        from .storage import RetrieverCache, create_storage

        llm: Optional[LLMClient] = None
        llm_init_error: Optional[str] = None
        try:
            llm = create_llm(llm_provider)
        except Exception as exc:  # noqa: BLE001 - missing/invalid key, not a crash
            llm_init_error = str(exc)
            logger.warning("LLM not configured at startup: %s", llm_init_error)

        storage = create_storage(database_url)
        return cls(llm, llm_init_error, storage, RetrieverCache(storage))

    # ---- embedding providers ---------------------------------------------
    def provider(self, name: str):
        """One named provider, built once per process."""
        if name not in self._providers:
            self._providers[name] = create_provider(name)
        return self._providers[name]

    def indexing_provider(self):
        """
        The provider for a NEW document: primary, or the fallback if the
        primary is unavailable. The only place a fallback is allowed.
        """
        provider = provider_for_indexing()
        self._providers.setdefault(provider.name, provider)
        return provider

    def document_provider(self, metadata: dict):
        """
        The provider a stored document was indexed with — never a fallback.
        Answering with a different provider would compare vectors from two
        different embedding spaces, which is meaningless.
        """
        # A document stored before providers were recorded predates the switch
        # to Hugging Face, so its vectors are Gemini's.
        return self.provider(metadata.get("embedding_provider") or "gemini")

    # ---- session assembly -------------------------------------------------
    def load_session(self, session_id: str) -> Optional[LoadedSession]:
        """Rebuild a session from storage, or None if it is unknown/expired."""
        document_id = self.storage.touch_session(session_id)
        if document_id is None:
            return None
        metadata = self.storage.get_document(document_id)
        if metadata is None:
            return None
        retriever = self.retrievers.get(document_id, self.document_provider(metadata))
        if retriever is None:
            return None
        conversation = self.storage.load_conversation(session_id)
        return LoadedSession(session_id, document_id, metadata, retriever, conversation)

    def save_turn(self, session: "LoadedSession") -> None:
        self.storage.append_turns(
            session.session_id, session.conversation, session.persisted_turns
        )


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

    Carries everything the "what did retrieval actually do" inspector shows:
    sources, matched vs. expanded chunks, evidence level, and the evidence
    block verbatim — so the frontend renders it without re-deriving anything
    the RAG core already computed.

    Deliberately omits `result["chunks"]` (the raw pre-expansion similarity
    hits): `passages` plus the entry/expanded/dropped chunk ids already cover
    the inspectable surface.
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
