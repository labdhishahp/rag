"""
Turn text into embedding vectors.

What is an embedding?
  A list of numbers (a vector) representing the *meaning* of a piece of text.
  Texts with similar meaning get vectors pointing in similar directions.

Why numbers?
  Computers cannot compare meaning directly. Vectors let us use geometry:
  similar meaning -> vectors close together -> high similarity score.

------------------------------------------------------------------------------
TWO PROVIDERS, AND WHY THE CHOICE IS PER DOCUMENT
------------------------------------------------------------------------------
    huggingface   BAAI/bge-small-en-v1.5 over the HF inference API.  384-d.
                  PRIMARY. Batches properly (32 texts -> 32 vectors), returns
                  unit-normalized vectors, and is measurably faster than the
                  alternative.

    gemini        gemini-embedding-001.  768-d.
                  FALLBACK, used only when Hugging Face cannot be reached.

An embedding is only meaningful inside the vector space that produced it. A
384-d BGE vector and a 768-d Gemini vector are not merely different lengths —
they are different coordinate systems, and a similarity between them is
nonsense even when the arithmetic happens to run.

So the rule this module enforces:

    FALLBACK HAPPENS WHEN INDEXING A DOCUMENT, NEVER WHEN QUERYING ONE.

At index time we may choose either provider and we RECORD which one was used
(pipeline.py writes it into the document's metadata). At query time we look
that record up and use exactly that provider. If it is unavailable the query
fails loudly, because silently answering with a vector from the wrong space
would return confident nonsense — the one failure mode this system exists to
avoid.

Mixing is therefore impossible by construction: a document's vectors and its
queries always come from the same provider, and documents indexed by different
providers live in separate stores keyed by document_id.

------------------------------------------------------------------------------
THRESHOLDS TRAVEL WITH THE PROVIDER
------------------------------------------------------------------------------
Cosine scores are not comparable across models either: each spreads "unrelated"
and "relevant" over its own range. So each provider carries the similarity
floors measured for it (see config.py), and the evidence gate uses the floors
belonging to the document being queried.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from config import FLOORS_BY_PROVIDER

# The API keys are read from the environment. Locally that comes from .env; a
# deployed function is given its environment by the platform, so the import is
# optional rather than required.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # pragma: no cover
    pass

PRIMARY_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface").strip().lower()
FALLBACK_PROVIDER = os.getenv("EMBEDDING_FALLBACK_PROVIDER", "gemini").strip().lower()

# --- Hugging Face (primary) -------------------------------------------------
HF_MODEL = "BAAI/bge-small-en-v1.5"
HF_DIMENSION = 384
HF_URL = "https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"
# Measured against the live endpoint: 32 texts return 32 vectors in ~0.5s.
HF_BATCH_SIZE = 32
HF_TIMEOUT = 60

# bge-small is trained with an instruction prepended to QUERIES only; documents
# are embedded bare. Measured on the gold set when this model ran locally:
# the documented prefix gave identical recall and a slightly NARROWER gap
# between absent and answerable scores, so it is deliberately left empty.
HF_QUERY_PREFIX = ""

# --- Gemini (fallback) ------------------------------------------------------
GEMINI_MODEL = "gemini-embedding-001"
GEMINI_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))
GEMINI_BATCH_SIZE = 32
GEMINI_MAX_RETRIES = 6
# The free tier's real constraint, read off an actual 429 body: 100 CONTENTS
# per minute, per project. A 32-text batch spends 32 of them.
GEMINI_REQUESTS_PER_MINUTE = int(os.getenv("EMBEDDING_RPM", "90"))
# Asymmetric task types: a question is not trying to resemble other questions,
# it is trying to resemble the passage that answers it.
TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"

_recent: list[float] = []


class EmbeddingError(Exception):
    """Raised when a provider cannot produce embeddings."""


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, so a dot product IS the cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # a zero vector would otherwise produce NaN
    return (matrix / norms).astype(np.float32)


# ---------------------------------------------------------------- Hugging Face
class HuggingFaceEmbeddings:
    """BAAI/bge-small-en-v1.5 through the Hugging Face inference API."""

    name = "huggingface"

    def __init__(self, model: str = HF_MODEL, dimension: int = HF_DIMENSION):
        token = os.getenv("HF_TOKEN")
        if not token:
            raise EmbeddingError(
                "HF_TOKEN not found. Add it to .env (see .env.example) or set it "
                "in the deployment environment."
            )
        self.model_name = model
        self.dimension = dimension
        self._url = HF_URL.format(model=model)
        self._token = token
        self.hard_floor, self.soft_floor = FLOORS_BY_PROVIDER[self.name]

    def _post(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"inputs": texts}).encode()
        request = urllib.request.Request(
            self._url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        # A model that is not resident returns 503 with an estimated load time.
        # That is a cold start, not a failure, so it is worth one patient retry.
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=HF_TIMEOUT) as response:
                    vectors = json.loads(response.read())
                break
            except urllib.error.HTTPError as exc:
                body = exc.read()[:200].decode(errors="replace")
                if exc.code == 503 and attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise EmbeddingError(f"Hugging Face returned {exc.code}: {body}") from exc
            except Exception as exc:  # noqa: BLE001 - network/timeout
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise EmbeddingError(f"Hugging Face request failed: {exc}") from exc
        else:  # pragma: no cover - loop always breaks or raises
            raise EmbeddingError("Hugging Face request failed after retries")

        # One vector per input is the whole reason this provider is primary.
        # If that ever stops being true, fail here rather than let a truncated
        # batch reach the vector store.
        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"Expected {len(texts)} embeddings, got {len(vectors)}. "
                "The endpoint is not returning one vector per input."
            )
        return vectors

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        out: list[list[float]] = []
        for start in range(0, len(texts), HF_BATCH_SIZE):
            out.extend(self._post(texts[start : start + HF_BATCH_SIZE]))
        return _normalize(np.asarray(out, dtype=np.float32))

    def embed_query(self, question: str) -> np.ndarray:
        return self.embed_texts([HF_QUERY_PREFIX + question])[0]


# ---------------------------------------------------------------------- Gemini
def _wait_for_quota(cost: int) -> None:
    """Sleep until `cost` more texts fit inside the last-60-seconds budget."""
    if GEMINI_REQUESTS_PER_MINUTE <= 0:
        return
    while True:
        now = time.monotonic()
        _recent[:] = [t for t in _recent if now - t < 60.0]
        if len(_recent) + cost <= GEMINI_REQUESTS_PER_MINUTE:
            _recent.extend([now] * cost)
            return
        time.sleep(max(0.05, 60.0 - (now - _recent[0]) + 0.05))


def _retry_delay(message: str) -> float | None:
    """Google states the wait in the 429 body ("'retryDelay': '48s'")."""
    match = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", message)
    return float(match.group(1)) if match else None


class GeminiEmbeddings:
    """gemini-embedding-001. Used only when Hugging Face is unavailable."""

    name = "gemini"

    def __init__(self, model: str = GEMINI_MODEL, dimension: int = GEMINI_DIMENSION):
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EmbeddingError(
                "GEMINI_API_KEY not found. Copy .env.example to .env and set your API key."
            )
        self.model_name = model
        self.dimension = dimension
        self._client = genai.Client(api_key=api_key)
        self._types = types
        self.hard_floor, self.soft_floor = FLOORS_BY_PROVIDER[self.name]

    def _batch(self, batch: list[str], task_type: str) -> list[list[float]]:
        config = self._types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=self.dimension
        )
        delay = 2.0
        for attempt in range(GEMINI_MAX_RETRIES):
            _wait_for_quota(len(batch))  # each text spends one unit of quota
            try:
                response = self._client.models.embed_content(
                    model=self.model_name, contents=batch, config=config
                )
                return [e.values for e in response.embeddings]
            except Exception as exc:  # noqa: BLE001 - retry depends on the message
                message = str(exc)
                low = message.lower()
                retryable = "429" in low or "resource_exhausted" in low or "503" in low
                if not retryable or attempt == GEMINI_MAX_RETRIES - 1:
                    raise EmbeddingError(f"Gemini embedding failed: {message[:200]}") from exc
                time.sleep(_retry_delay(message) or delay)
                delay *= 2
        raise EmbeddingError("unreachable")

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        out: list[list[float]] = []
        for start in range(0, len(texts), GEMINI_BATCH_SIZE):
            out.extend(self._batch(texts[start : start + GEMINI_BATCH_SIZE], TASK_DOCUMENT))
        # Only the full 3072-d vector arrives normalized; a truncated one does
        # not (768-d comes back with norm ~0.59), so normalizing is required.
        return _normalize(np.asarray(out, dtype=np.float32))

    def embed_query(self, question: str) -> np.ndarray:
        return _normalize(np.asarray(self._batch([question], TASK_QUERY), dtype=np.float32))[0]


# ------------------------------------------------------------------- selection
_BUILDERS = {"huggingface": HuggingFaceEmbeddings, "gemini": GeminiEmbeddings}


def create_provider(name: str):
    """Build one named provider. Raises EmbeddingError if it cannot be used."""
    try:
        builder = _BUILDERS[name]
    except KeyError:
        raise EmbeddingError(
            f"Unknown embedding provider {name!r}. Expected one of {sorted(_BUILDERS)}."
        ) from None
    return builder()


def provider_for_indexing(primary: str = PRIMARY_PROVIDER, fallback: str = FALLBACK_PROVIDER):
    """
    The provider to embed a NEW document with: primary, or the fallback if the
    primary cannot be built or reached.

    This is the ONLY place a fallback is allowed. The caller must record which
    provider came back (pipeline.py does) so that queries against this document
    use the same one.
    """
    try:
        return create_provider(primary)
    except EmbeddingError as exc:
        if not fallback or fallback == primary:
            raise
        print(f"\n=== EMBEDDINGS === primary provider {primary!r} unavailable ({exc}); "
              f"falling back to {fallback!r} for this document")
        return create_provider(fallback)


def provider_for_document(name: str | None):
    """
    The provider a stored document was indexed with. Never falls back: another
    provider's vectors could not be compared against this document's index.
    """
    if not name:
        # Documents indexed before providers were recorded predate the primary
        # switch, so they are Gemini-embedded.
        name = "gemini"
    return create_provider(name)


# Backwards-compatible name. Existing callers that just want "an embedder"
# (ingestion, scripts) get the primary-with-fallback choice.
def EmbeddingModel(provider: str | None = None):  # noqa: N802 - kept as a factory name
    return create_provider(provider) if provider else provider_for_indexing()
