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
        _, metadata = index_document_from_upload(
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

    # Persist the chunks and their vectors, then open a session against the
    # document. The retriever built during indexing is deliberately discarded:
    # the next request may land on a different instance, so the only copy that
    # matters is the stored one, and it is re-hydrated on demand (storage.py).
    chunks = metadata["chunks"]
    embeddings = metadata["embeddings"]
    document = public_document_metadata(metadata)
    try:
        state.storage.save_document(document, chunks, embeddings)
        state.retrievers.invalidate(document["document_id"])
        session_id = state.storage.create_session(document["document_id"])
    except Exception:
        logger.exception("Failed to persist document %s", filename)
        raise HTTPException(
            status_code=500,
            detail="The document was processed but could not be saved. See server logs for details.",
        ) from None

    return {"session_id": session_id, "document": document}


@router.get("/api/documents/{session_id}")
def get_document(session_id: str, request: Request):
    state = request.app.state.rag_state
    session = state.load_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired. Upload a document to start a new one.",
        )
    return {"session_id": session.session_id, "document": session.metadata}
