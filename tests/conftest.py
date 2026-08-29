"""
Shared fixtures.

Run everything:            ./.venv/bin/python -m pytest
Skip the slow embedding tests:  ./.venv/bin/python -m pytest -m "not embedding"
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

DATA = PROJECT_ROOT / "data"
REAL = DATA / "real"

FORMULA_PDF = DATA / "formula_sample.pdf"

REAL_DOCS = {
    "rag_survey": REAL / "rag_survey.pdf",
    "bert": REAL / "bert.pdf",
    "nist": REAL / "nist_sp800-207.pdf",
}


def pytest_configure(config):
    config.addinivalue_line("markers", "embedding: loads the sentence-transformer (slow)")
    config.addinivalue_line("markers", "llm: calls the Gemini API (needs key, costs quota)")


@pytest.fixture(scope="session")
def embedding_model():
    from embeddings import EmbeddingModel

    return EmbeddingModel()


@pytest.fixture(scope="session")
def formula_pages():
    from document_loader import load_pdf

    if not FORMULA_PDF.exists():
        pytest.skip("data/formula_sample.pdf missing — run scripts/create_formula_pdf.py")
    return load_pdf(FORMULA_PDF)


@pytest.fixture(scope="session")
def formula_chunks(formula_pages):
    from chunker import chunk_pages

    return chunk_pages(
        formula_pages, 500, 50, document_name="formula_sample.pdf", document_id="formula"
    )


@pytest.fixture(scope="session")
def formula_store(formula_chunks, embedding_model):
    from vector_store import VectorStore

    store = VectorStore(dimension=embedding_model.dimension)
    store.add(embedding_model.embed_texts([c["text"] for c in formula_chunks]), formula_chunks)
    return store


@pytest.fixture(scope="session")
def formula_retriever(formula_store, embedding_model):
    from retriever import Retriever

    return Retriever(embedding_model, formula_store)


def real_doc(name: str) -> Path:
    path = REAL_DOCS[name]
    if not path.exists():
        pytest.skip(f"{path.name} not present — run scripts/fetch_real_docs.py")
    return path
