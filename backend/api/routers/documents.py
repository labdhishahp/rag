"""Upload a document and index it: the entry point of the ingestion pipeline."""

import logging
from pathlib import Path

import numpy as np

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from ..config import settings
from ..security import AUTH, UPLOAD_LIMIT
from ..rag_bridge import (
    DocumentProcessingError,
    EmbeddingError,
    index_document_from_upload,
    public_document_metadata,
)

logger = logging.getLogger("rag_api")
router = APIRouter(tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".docx"}


@router.post("/api/documents", dependencies=[AUTH, UPLOAD_LIMIT])
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
        # Fallback to the secondary embedding provider is allowed HERE and only
        # here: the provider that wins is recorded on the document, and every
        # later query against it uses that same provider (see embeddings.py).
        embedder = state.indexing_provider()
    except EmbeddingError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"No embedding provider is available: {exc}",
        ) from exc

    try:
        _, metadata = index_document_from_upload(
            data, filename=filename, embedding_model=embedder,
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


@router.get("/api/documents/{session_id}", dependencies=[AUTH])
def get_document(session_id: str, request: Request):
    state = request.app.state.rag_state
    session = state.load_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired. Upload a document to start a new one.",
        )
    return {"session_id": session.session_id, "document": session.metadata}


# How many values of each vector the list view carries. Enough to see that a
# vector is real and that two chunks differ, without shipping the whole matrix:
# 400 chunks x 384 floats is roughly 3MB of JSON, past Vercel's 4.5MB response
# limit. The full vector is a separate, per-chunk request below.
EMBEDDING_PREVIEW_VALUES = 8


def _load_stored(request: Request, session_id: str):
    """The chunks and vectors as they are stored — never re-derived."""
    state = request.app.state.rag_state
    document_id = state.storage.touch_session(session_id)
    if document_id is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired. Upload a document to start a new one.",
        )
    metadata = state.storage.get_document(document_id)
    chunks, vectors = state.storage.load_chunks(document_id)
    if metadata is None or not chunks:
        raise HTTPException(status_code=404, detail="No stored chunks for this document.")
    return metadata, chunks, vectors


@router.get("/api/documents/{session_id}/chunks", dependencies=[AUTH])
def list_chunks(session_id: str, request: Request):
    """
    Every chunk of the document with a preview of its embedding.

    Read straight out of storage rather than re-chunking or re-embedding, so
    what this shows is what retrieval actually searches.
    """
    metadata, chunks, vectors = _load_stored(request, session_id)
    rows = []
    for index, chunk in enumerate(chunks):
        vector = vectors[index] if vectors is not None and index < len(vectors) else None
        rows.append(
            {
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk.get("page_number"),
                "page_label": chunk.get("page_label", "page"),
                "section": chunk.get("section"),
                "text": chunk.get("text", ""),
                "char_start": chunk.get("char_start"),
                "char_end": chunk.get("char_end"),
                "prev_chunk_id": chunk.get("prev_chunk_id"),
                "next_chunk_id": chunk.get("next_chunk_id"),
                "embedding_preview": (
                    [float(v) for v in vector[:EMBEDDING_PREVIEW_VALUES]] if vector is not None else []
                ),
                # Unit length is what makes a dot product a cosine similarity,
                # so it is worth being able to see it.
                "embedding_norm": float(np.linalg.norm(vector)) if vector is not None else None,
            }
        )
    return {
        "document": public_document_metadata(metadata),
        "preview_values": EMBEDDING_PREVIEW_VALUES,
        "chunks": rows,
    }


@router.get("/api/documents/{session_id}/chunks/{chunk_id}/embedding", dependencies=[AUTH])
def get_chunk_embedding(session_id: str, chunk_id: int, request: Request):
    """The full vector for one chunk — fetched only when a reader expands it."""
    metadata, chunks, vectors = _load_stored(request, session_id)
    for index, chunk in enumerate(chunks):
        if chunk["chunk_id"] == chunk_id:
            if vectors is None or index >= len(vectors):
                raise HTTPException(status_code=404, detail="No embedding stored for that chunk.")
            return {
                "chunk_id": chunk_id,
                "dimension": int(vectors.shape[1]),
                "model": metadata.get("embedding_model"),
                "provider": metadata.get("embedding_provider"),
                "values": [float(v) for v in vectors[index]],
            }
    raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found in this document.")
