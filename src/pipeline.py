"""
Build a retriever from an uploaded document — shared by CLI and Streamlit app.

Input:  file bytes + filename (PDF or DOCX)
Output: Retriever ready to search (+ document metadata for the UI)
"""

import hashlib
from pathlib import Path
from typing import Callable, Optional, Union

import fitz

from chunker import chunk_pages
from document_loader import load_document_from_bytes, load_pdf
from embeddings import EmbeddingModel
from retriever import Retriever
from vector_store import VectorStore

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


class DocumentProcessingError(Exception):
    """Raised when document ingestion or indexing fails."""


def make_document_id(file_bytes: bytes) -> str:
    """
    A short, stable identifier for a document, derived from its CONTENT.

    Why content and not the filename:
      The same file renamed is still the same document and should keep its ID.
      Two different files that happen to share a name must not collide. Hashing
      the bytes gives us both properties for free.

    12 hex characters is 48 bits — ample for the handful of documents this app
    holds, and short enough to read in debug output.
    """
    return hashlib.sha256(file_bytes).hexdigest()[:12]


def build_retriever_from_pages(
    pages: list[dict],
    embedding_model: Optional[EmbeddingModel] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Retriever:
    """Turn extracted pages into a searchable Retriever."""
    chunks = chunk_pages(
        pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if embedding_model is None:
        embedding_model = EmbeddingModel()

    texts = [chunk["text"] for chunk in chunks]

    try:
        embeddings = embedding_model.embed_texts(texts)
    except Exception as exc:
        raise DocumentProcessingError(
            "Failed to generate embeddings. See terminal logs for details."
        ) from exc

    store = VectorStore(dimension=embedding_model.dimension)
    store.add(embeddings, chunks)

    return Retriever(embedding_model, store)


def _document_metadata(
    pages: list[dict],
    retriever: Retriever,
    embedding_model: EmbeddingModel,
    filename: str,
    document_id: str | None = None,
) -> dict:
    return {
        "filename": filename,
        "document_id": document_id,
        "page_label": pages[0].get("page_label", "page") if pages else "page",
        "page_count": len(pages),
        "chunk_count": retriever.vector_store.index.ntotal,
        "embedding_dimension": embedding_model.dimension,
        "status": "ready",
    }


def index_document_from_upload(
    file_bytes: bytes,
    filename: str,
    embedding_model: Optional[EmbeddingModel] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    on_step: Optional[Callable[[str], None]] = None,
) -> tuple[Retriever, dict]:
    """
    Process an uploaded document once: extract → chunk → embed → FAISS.

    on_step: optional callback for UI progress messages.
    Returns (retriever, metadata) for session storage.
    """

    def step(message: str) -> None:
        if on_step:
            on_step(message)

    if not file_bytes:
        raise DocumentProcessingError("The uploaded file is empty.")

    # ---------------------------------------------------------
    # STEP 1: EXTRACT TEXT
    # ---------------------------------------------------------
    step("Extracting text...")

    try:
        pages = load_document_from_bytes(file_bytes, filename)

    except ValueError as exc:
        raise DocumentProcessingError(str(exc)) from exc

    except fitz.FileDataError as exc:
        raise DocumentProcessingError(
            "Invalid PDF file. Please upload a valid PDF document."
        ) from exc

    except Exception as exc:
        raise DocumentProcessingError(
            "Failed to read the document. See terminal logs for details."
        ) from exc

    if embedding_model is None:
        embedding_model = EmbeddingModel()

    # ---------------------------------------------------------
    # STEP 2: CREATE CHUNKS
    # ---------------------------------------------------------
    step("Creating chunks...")

    document_id = make_document_id(file_bytes)

    try:
        chunks = chunk_pages(
            pages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            document_name=filename,
            document_id=document_id,
        )
    except ValueError as exc:
        raise DocumentProcessingError(str(exc)) from exc

    # Compact pipeline visibility: one line per chunk, not the full text.
    # For the full text of every chunk, use scripts/baseline_retrieval.py
    # or the "Chunks" viewer in the Streamlit UI.
    print("\n=== CHUNKING ===")
    print(f"document={filename} id={document_id}")
    print(f"pages={len(pages)} chunks={len(chunks)} "
          f"size={chunk_size} overlap={chunk_overlap}")
    for chunk in chunks:
        preview = chunk["text"][:60].replace("\n", " ")
        print(f"  chunk {chunk['chunk_id']:>3} | {chunk['page_label']} "
              f"{chunk['page_number']:>2} | {len(chunk['text']):>4} chars | {preview}...")

    # ---------------------------------------------------------
    # STEP 3: GENERATE EMBEDDINGS
    # ---------------------------------------------------------
    step("Generating embeddings...")

    texts = [chunk["text"] for chunk in chunks]

    try:
        embeddings = embedding_model.embed_texts(texts)

    except Exception as exc:
        raise DocumentProcessingError(
            "Failed to generate embeddings. See terminal logs for details."
        ) from exc

    # ---------------------------------------------------------
    # STEP 4: BUILD VECTOR SEARCH INDEX
    # ---------------------------------------------------------
    step("Building search index...")

    try:
        store = VectorStore(
            dimension=embedding_model.dimension
        )

        store.add(
            embeddings,
            chunks,
        )

        retriever = Retriever(
            embedding_model,
            store,
        )

    except Exception as exc:
        raise DocumentProcessingError(
            "Failed to build the search index. See terminal logs for details."
        ) from exc

    # ---------------------------------------------------------
    # STEP 5: CREATE DOCUMENT METADATA
    # ---------------------------------------------------------
    metadata = _document_metadata(
        pages,
        retriever,
        embedding_model,
        filename,
        document_id=document_id,
    )
    metadata["chunks"] = chunks
    metadata["embeddings"] = embeddings

    return retriever, metadata


def build_retriever_from_pdf(
    pdf_path: Union[str, Path],
    embedding_model: Optional[EmbeddingModel] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Retriever:
    """Load a PDF from disk and build a Retriever (Phase 1 CLI)."""

    pages = load_pdf(pdf_path)

    return build_retriever_from_pages(
        pages,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def index_document_from_bytes(
    file_bytes: bytes,
    filename: str = "uploaded.pdf",
    embedding_model: Optional[EmbeddingModel] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[Retriever, dict]:
    """Backward-compatible alias for index_document_from_upload."""

    return index_document_from_upload(
        file_bytes,
        filename=filename,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def build_retriever_from_bytes(
    file_bytes: bytes,
    filename: str = "uploaded.pdf",
    embedding_model: Optional[EmbeddingModel] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Retriever:
    """Load an uploaded file and build a Retriever (Phase 1 compatible)."""

    retriever, _ = index_document_from_upload(
        file_bytes,
        filename=filename,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return retriever