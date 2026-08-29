"""Ask a question of an indexed document — the API equivalent of app.py's chat loop."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..rag_bridge import LLMError, RAGSystem, serialize_answer_result

logger = logging.getLogger("rag_api")
router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    session_id: str
    question: str
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


@router.post("/api/chat")
def chat(payload: ChatRequest, request: Request):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    state = request.app.state.rag_state
    session = state.load_session(payload.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired. Upload a document to start a new one.",
        )

    if state.llm is None:
        raise HTTPException(
            status_code=503,
            detail=state.llm_init_error or "The LLM is not configured on the server.",
        )

    rag = RAGSystem(
        retriever=session.retriever,
        llm=state.llm,
        top_k=payload.top_k,
        embedding_dimension=session.metadata.get("embedding_dimension"),
        debug=False,
    )

    try:
        result = rag.answer(question, session.conversation)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception:
        logger.exception("RAG answer failed for session %s", payload.session_id)
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while generating the answer. See server logs for details.",
        ) from None

    RAGSystem.record_turn(session.conversation, result)
    # Persist only after a successful answer, so a failed turn never leaves a
    # half-written history for the next request to read back.
    state.save_turn(session)
    return serialize_answer_result(result)


@router.post("/api/sessions/{session_id}/reset")
def reset_session(session_id: str, request: Request):
    state = request.app.state.rag_state
    if not state.storage.clear_conversation(session_id):
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    return {"status": "ok"}


@router.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, request: Request):
    state = request.app.state.rag_state
    if not state.storage.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"status": "ok"}
