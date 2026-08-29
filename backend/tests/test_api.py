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
    assert body["embedding_model_loaded"] is True
    assert "llm_configured" in body


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
