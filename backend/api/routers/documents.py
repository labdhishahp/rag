"""Document upload and indexing — the API equivalent of app.py's sidebar uploader."""

import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from ..config import settings
from ..rag_bridge import DocumentProcessingError, index_document_from_upload, public_document_metadata

logger = logging.getLogger("rag_api")
router = APIRouter(tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".docx"}


@router.post("/api/documents")
async def upload_document(request: Request, file: UploadFile = File(...)):
    filename = file.filename or "uploaded"
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension or '(none)'}'. Upload a PDF or DOCX document.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_upload_bytes // (1024 * 1024)}MB upload limit.",
        )

    state = request.app.state.rag_state
    try:
        retriever, metadata = index_document_from_upload(
            data, filename=filename, embedding_model=state.embedding_model,
        )
    except DocumentProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected document processing error for %s", filename)
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while processing the document. See server logs for details.",
        ) from None

    metadata = public_document_metadata(metadata)
    session = state.sessions.create(retriever, metadata)
    return {"session_id": session.session_id, "document": metadata}


@router.get("/api/documents/{session_id}")
def get_document(session_id: str, request: Request):
    state = request.app.state.rag_state
    session = state.sessions.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired. Upload a document to start a new one.",
        )
    return {"session_id": session.session_id, "document": session.metadata}
