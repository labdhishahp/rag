"""
Integration tests for the HTTP API boundary in front of the RAG core.

Covers the full path browser -> API -> RAG -> (fake) LLM -> response, plus the
failure cases the migration explicitly needs to preserve: backend/LLM
unavailable, invalid upload, empty document, empty query, insufficient
evidence, and upstream model failure.
"""

import pytest

from tests.conftest import FailingLLM


# ---- health ---------------------------------------------------------------

def test_health_reports_readiness(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["storage_ok"] is True
    assert "llm_configured" in body
    # Providers are reported individually: the primary being down is survivable
    # (new documents fall back), but a provider being down makes documents
    # already indexed by it unanswerable, so they must not be one aggregate flag.
    assert body["embedding_primary"] == "huggingface"
    providers = body["embedding_providers"]
    assert set(providers) == {"huggingface", "gemini"}
    assert providers["huggingface"]["available"] is True
    assert providers["huggingface"]["dimension"] == 384
    assert "bge-small" in providers["huggingface"]["model"]


def test_health_is_degraded_when_storage_is_unreachable(client, app):
    """A process that answers HTTP but cannot reach its store is not ready."""
    original = app.state.rag_state.storage.healthy
    app.state.rag_state.storage.healthy = lambda: (False, "connection refused")
    try:
        body = client.get("/health").json()
        assert body["status"] == "degraded"
        assert body["storage_ok"] is False
        assert body["storage_error"] == "connection refused"
    finally:
        app.state.rag_state.storage.healthy = original


# ---- document upload --------------------------------------------------------

def test_upload_document_indexes_and_returns_metadata(client, formula_pdf_bytes):
    response = client.post(
        "/api/documents",
        files={"file": ("formula_sample.pdf", formula_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["session_id"]
    doc = body["document"]
    assert doc["filename"] == "formula_sample.pdf"
    assert doc["chunk_count"] > 0
    assert doc["embedding_dimension"] > 0
    assert doc["status"] == "ready"
    assert "chunks" not in doc and "embeddings" not in doc


def test_upload_rejects_unsupported_file_type(client):
    response = client.post(
        "/api/documents",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_rejects_empty_file(client):
    response = client.post(
        "/api/documents",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_rejects_document_with_no_extractable_text(client):
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    doc.new_page()  # a blank page: valid PDF, zero extractable text
    blank_pdf_bytes = doc.tobytes()
    doc.close()

    response = client.post(
        "/api/documents",
        files={"file": ("blank.pdf", blank_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 422
    assert response.json()["detail"]


def test_get_document_unknown_session_returns_404(client):
    response = client.get("/api/documents/does-not-exist")
    assert response.status_code == 404


# ---- chat: happy path -------------------------------------------------------

def test_chat_answers_from_evidence_and_records_citations(client, indexed_session):
    response = client.post(
        "/api/chat",
        json={"session_id": indexed_session, "question": "What is the compound interest formula?"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["llm_called"] is True
    assert body["evidence_level"] in ("weak", "ok")
    assert body["refused"] is False
    assert body["source_citations"]
    assert body["passages"]
    assert "[S1]" in body["context_formatted"]


def test_chat_follow_up_uses_conversation_memory(client, indexed_session):
    first = client.post(
        "/api/chat",
        json={"session_id": indexed_session, "question": "What is the compound interest formula?"},
    )
    assert first.status_code == 200
    second = client.post(
        "/api/chat",
        json={"session_id": indexed_session, "question": "Explain it in more detail."},
    )
    assert second.status_code == 200, second.text
    assert second.json()["was_follow_up"] is True


# ---- chat: failure cases -----------------------------------------------------

def test_chat_rejects_empty_question(client, indexed_session):
    response = client.post("/api/chat", json={"session_id": indexed_session, "question": "   "})
    assert response.status_code == 400


def test_chat_unknown_session_returns_404(client):
    response = client.post(
        "/api/chat", json={"session_id": "does-not-exist", "question": "What is the formula?"}
    )
    assert response.status_code == 404


def test_chat_declines_without_llm_call_on_insufficient_evidence(client, indexed_session, app):
    response = client.post(
        "/api/chat",
        json={"session_id": indexed_session, "question": "What is the parental leave policy?"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["evidence_level"] == "none"
    assert body["llm_called"] is False
    assert body["refused"] is True
    assert app.state.rag_state.llm.calls == 0


def test_chat_returns_503_when_llm_not_configured(client, indexed_session, app):
    app.state.rag_state.llm = None
    app.state.rag_state.llm_init_error = "GEMINI_API_KEY not found."
    response = client.post(
        "/api/chat",
        json={"session_id": indexed_session, "question": "What is the compound interest formula?"},
    )
    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]


def test_chat_returns_502_on_upstream_llm_failure(client, indexed_session, app):
    app.state.rag_state.llm = FailingLLM()
    response = client.post(
        "/api/chat",
        json={"session_id": indexed_session, "question": "What is the compound interest formula?"},
    )
    assert response.status_code == 502
    assert "temporarily unavailable" in response.json()["detail"]


# ---- session lifecycle -------------------------------------------------------

def test_reset_clears_conversation_but_keeps_document(client, indexed_session):
    client.post(
        "/api/chat",
        json={"session_id": indexed_session, "question": "What is the compound interest formula?"},
    )
    reset = client.post(f"/api/sessions/{indexed_session}/reset")
    assert reset.status_code == 200

    follow_up = client.post(
        "/api/chat",
        json={"session_id": indexed_session, "question": "Explain it in more detail."},
    )
    assert follow_up.status_code == 200
    # No prior turn left after reset, so this is read as standalone, not a follow-up.
    assert follow_up.json()["was_follow_up"] is False

    still_indexed = client.get(f"/api/documents/{indexed_session}")
    assert still_indexed.status_code == 200


def test_delete_session_removes_it(client, indexed_session):
    response = client.delete(f"/api/sessions/{indexed_session}")
    assert response.status_code == 200

    after = client.get(f"/api/documents/{indexed_session}")
    assert after.status_code == 404


# ---- persistence (the serverless requirement) --------------------------------

def test_conversation_survives_losing_the_in_process_cache(client, indexed_session, app):
    """
    The serverless case: the next request may hit an instance that has never
    seen this session. Dropping the hydrated-retriever cache simulates that;
    the answer must still resolve the follow-up, which is only possible if the
    document AND the conversation came back from storage rather than memory.
    """
    first = client.post(
        "/api/chat",
        json={"session_id": indexed_session, "question": "What is the compound interest formula?"},
    )
    assert first.status_code == 200

    app.state.rag_state.retrievers._cache.clear()

    follow_up = client.post(
        "/api/chat",
        json={"session_id": indexed_session, "question": "Explain it in more detail."},
    )
    assert follow_up.status_code == 200, follow_up.text
    assert follow_up.json()["was_follow_up"] is True


def test_stored_vectors_round_trip_exactly(client, formula_pdf_bytes, app):
    """
    Embeddings are persisted as raw float32 bytes precisely so that what comes
    back is bit-identical. If it were not, every similarity score — and so
    every calibrated threshold — would shift after a restart.
    """
    import numpy as np

    upload = client.post(
        "/api/documents",
        files={"file": ("formula_sample.pdf", formula_pdf_bytes, "application/pdf")},
    ).json()
    storage = app.state.rag_state.storage
    chunks, vectors = storage.load_chunks(upload["document"]["document_id"])

    assert len(chunks) == upload["document"]["chunk_count"]
    assert vectors.dtype == np.float32
    assert vectors.shape[1] == upload["document"]["embedding_dimension"]
    # Unit-normalized on the way in, so still unit-normalized on the way out.
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    # Chunk metadata must survive the JSON round trip intact — expansion depends
    # on prev/next links and section labels being exactly what the chunker set.
    assert {"chunk_id", "text", "section", "prev_chunk_id", "next_chunk_id"} <= set(chunks[0])


# ---- connection-string handling (no database required) ----------------------

def test_supabase_pooler_query_parameters_are_stripped():
    """
    Supabase's transaction-pooler URI carries `?pgbouncer=true`, a hint meant
    for Prisma. libpq rejects any query parameter it does not recognise, so
    connecting with the string Supabase hands you fails outright with
    `invalid URI query parameter: "pgbouncer"`. Copying that string should just
    work, so the parameter is dropped rather than pushed onto the user.
    """
    from api.storage import _clean_dsn

    dsn, dropped = _clean_dsn(
        "postgresql://u:p@aws-0-x.pooler.supabase.com:6543/postgres?pgbouncer=true"
    )
    assert dropped == ["pgbouncer"]
    assert "pgbouncer" not in dsn
    assert dsn.endswith("/postgres")

    # Real libpq parameters must survive untouched.
    dsn, dropped = _clean_dsn("postgresql://u:p@h:6543/postgres?sslmode=require")
    assert dropped == [] and "sslmode=require" in dsn

    # A plain URI is returned unchanged.
    assert _clean_dsn("postgresql://u:p@h:5432/postgres") == (
        "postgresql://u:p@h:5432/postgres",
        [],
    )
