"""
Build a retriever from a PDF — shared by the CLI (Phase 1) and Streamlit app (Phase 2).

Input:  PDF file path or raw bytes
Output: Retriever ready to search (+ document metadata for the UI)
"""

from pathlib import Path
from typing import Optional, Union

import fitz

from chunker import chunk_pages
from document_loader import load_pdf, load_pdf_from_bytes
from embeddings import EmbeddingModel
from retriever import Retriever
from vector_store import VectorStore

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


class DocumentProcessingError(Exception):
    """Raised when PDF ingestion or indexing fails."""


def build_retriever_from_pages(
    pages: list[dict],
    embedding_model: Optional[EmbeddingModel] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Retriever:
    """Turn extracted pages into a searchable Retriever."""
    chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

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
) -> dict:
    return {
        "filename": filename,
        "page_count": len(pages),
        "chunk_count": retriever.vector_store.index.ntotal,
        "embedding_dimension": embedding_model.dimension,
    }


def index_document_from_bytes(
    pdf_bytes: bytes,
    filename: str = "uploaded.pdf",
    embedding_model: Optional[EmbeddingModel] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[Retriever, dict]:
    """
    Process an uploaded PDF once: extract → chunk → embed → FAISS.

    Returns (retriever, metadata) for session storage in the UI.
    """
    if not pdf_bytes:
        raise DocumentProcessingError("The uploaded file is empty.")

    try:
        pages = load_pdf_from_bytes(pdf_bytes)
    except ValueError as exc:
        raise DocumentProcessingError(str(exc)) from exc
    except fitz.FileDataError as exc:
        raise DocumentProcessingError(
            "Invalid PDF file. Please upload a valid PDF document."
        ) from exc
    except Exception as exc:
        raise DocumentProcessingError(
            "Failed to read the PDF. See terminal logs for details."
        ) from exc

    if embedding_model is None:
        embedding_model = EmbeddingModel()

    try:
        retriever = build_retriever_from_pages(
            pages,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except DocumentProcessingError:
        raise
    except Exception as exc:
        raise DocumentProcessingError(
            "Failed to index the document. See terminal logs for details."
        ) from exc

    metadata = _document_metadata(pages, retriever, embedding_model, filename)
    return retriever, metadata


def build_retriever_from_pdf(
    pdf_path: Union[str, Path],
    embedding_model: Optional[EmbeddingModel] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Retriever:
    """Load a PDF from disk and build a Retriever."""
    pages = load_pdf(pdf_path)
    return build_retriever_from_pages(
        pages,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def build_retriever_from_bytes(
    pdf_bytes: bytes,
    embedding_model: Optional[EmbeddingModel] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Retriever:
    """Load a PDF from uploaded bytes and build a Retriever (Phase 1 compatible)."""
    retriever, _ = index_document_from_bytes(
        pdf_bytes,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return retriever
