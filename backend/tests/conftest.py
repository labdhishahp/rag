"""
Backend API test fixtures.

The RAG core is real (real embedding model, real FAISS, real evidence gate) —
only the LLM is faked, the same pattern tests/test_generation_logic.py
already uses, so these tests exercise the actual retrieval/expansion/gating
behavior through the HTTP boundary without spending Gemini quota.

Run from backend/:  python -m pytest
"""

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# These tests exercise the HTTP boundary and the real retrieval/gating logic
# against the primary embedding provider, so they need HF_TOKEN and network.
# Set before importing the app.
os.environ.setdefault("EMBEDDING_PROVIDER", "huggingface")

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from api.main import create_app  # noqa: E402
from api.rag_bridge import LLMClient, LLMError  # noqa: E402

PROJECT_ROOT = BACKEND_DIR.parent
FORMULA_PDF = PROJECT_ROOT / "data" / "formula_sample.pdf"


class FakeLLM(LLMClient):
    """Deterministic stand-in for GeminiClient — no network call, no quota."""

    def __init__(self, response: str = "The formula is A = P(1 + r/n)^(nt) [S1]."):
        self.response = response
        self.calls = 0
        self.active_model = "fake-llm"

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return self.response


class FailingLLM(LLMClient):
    active_model = "fake-llm-failing"

    def generate(self, prompt: str) -> str:
        raise LLMError("The Gemini API is temporarily unavailable. Please try again later.")


def pytest_configure(config):
    config.addinivalue_line("markers", "embedding: loads the sentence-transformer (slow)")


@pytest.fixture(scope="session")
def app():
    """
    The API wired to in-memory storage with the rate limiter switched off.

    Both are deliberate. A configured DATABASE_URL would otherwise point these
    tests at the real database — slow, order-dependent, and writing rows into
    somewhere that is not a test fixture. And the limiter counts every caller
    as one client (they all share the TestClient's address), so leaving it on
    would make a long suite start returning 429 partway through, for reasons
    that have nothing to do with what is being tested.

    test_postgres_storage.py talks to the real database on purpose, and
    test_security.py switches the limits back on for the tests that need them.
    """
    from api.config import settings

    settings.database_url = None
    settings.api_key = None
    settings.rate_limit_questions = 0
    settings.rate_limit_uploads = 0

    application = create_app()
    with TestClient(application):
        yield application


@pytest.fixture
def client(app):
    # Deterministic starting point; individual tests override app.state as needed.
    app.state.rag_state.llm = FakeLLM()
    app.state.rag_state.llm_init_error = None
    return TestClient(app)


@pytest.fixture
def formula_pdf_bytes():
    if not FORMULA_PDF.exists():
        pytest.skip("data/formula_sample.pdf missing — run scripts/create_formula_pdf.py")
    return FORMULA_PDF.read_bytes()


@pytest.fixture
def indexed_session(client, formula_pdf_bytes):
    response = client.post(
        "/api/documents",
        files={"file": ("formula_sample.pdf", formula_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200, response.text
    return response.json()["session_id"]
