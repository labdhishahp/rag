"""
Build a retriever from a PDF — shared by the CLI (Phase 1) and Streamlit app (Phase 2).

Input:  PDF file path or raw bytes
Output: Retriever ready to search
"""

from pathlib import Path
from typing import Optional, Union

from chunker import chunk_pages
from document_loader import load_pdf, load_pdf_from_bytes
from embeddings import EmbeddingModel
from retriever import Retriever
from vector_store import VectorStore

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


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
    embeddings = embedding_model.embed_texts(texts)

    store = VectorStore(dimension=embedding_model.dimension)
    store.add(embeddings, chunks)

    return Retriever(embedding_model, store)


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
    """Load a PDF from uploaded bytes and build a Retriever."""
    pages = load_pdf_from_bytes(pdf_bytes)
    return build_retriever_from_pages(
        pages,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
