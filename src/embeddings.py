"""
Turn text into embedding vectors, using Google's embedding API.

What is an embedding?
  A list of numbers (a vector) representing the *meaning* of a piece of text.
  Texts with similar meaning get vectors pointing in similar directions.

Why numbers?
  Computers cannot compare meaning directly. Vectors let us use geometry:
  similar meaning -> vectors close together -> high similarity score.

Same model for documents and questions:
  Both must live in the same vector space, or the scores are meaningless —
  like comparing coordinates from two different maps.

Why an API instead of running a model locally:
  A local sentence-transformers model needs torch, roughly 650MB installed.
  That does not fit in a serverless function, and loading a model on every
  cold start is slow. Calling the API keeps the deployment small and start-up
  fast, at the cost of a network round trip per batch.

IMPORTANT — the thresholds in config.py belong to THIS model:
  Cosine scores are not comparable across embedding models. If you change the
  model or its dimensionality, the similarity floors in config.py must be
  re-measured, not adjusted by feel.
"""

import os
import re
import time
from pathlib import Path

import numpy as np

# The API key is read from the environment. Locally that comes from .env; a
# deployed function is given its environment by the platform, so the import is
# optional rather than required.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # pragma: no cover
    pass

MODEL = "gemini-embedding-001"

# gemini-embedding-001 is a Matryoshka model: a truncated prefix of its full
# 3072-d vector is still a usable embedding. 768 keeps storage and comparison
# cheap and stays under pgvector's 2000-dimension ceiling for indexed columns.
#
# The catch, measured rather than assumed: only the full 3072-d vector comes
# back L2-normalized. A truncated one does not (768-d arrives with norm ~0.59),
# so it MUST be re-normalized here — otherwise every cosine score, and every
# threshold in config.py, is silently wrong.
DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))

# Measured against the live API: 100 texts per call is the documented hard
# ceiling, but a 100-text batch already trips the free tier's rate limit. 32
# goes through comfortably.
BATCH_SIZE = 32
MAX_RETRIES = 6

# The free tier's real constraint, read off an actual 429 body:
#   quotaId EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier
#   quotaValue 100
# It counts CONTENTS, not HTTP calls — a 32-text batch spends 32 of the 100. We
# pace below the ceiling instead of discovering it by failing. Raise
# EMBEDDING_RPM to match a paid plan.
REQUESTS_PER_MINUTE = int(os.getenv("EMBEDDING_RPM", "90"))

# Asymmetric task types: the model embeds a QUESTION differently from a
# PASSAGE on purpose, because a question is not trying to resemble other
# questions — it is trying to resemble the passage that answers it.
TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"

# Timestamps of recent embedded texts, used to stay under REQUESTS_PER_MINUTE.
_recent: list[float] = []


def _wait_for_quota(cost: int) -> None:
    """Sleep until `cost` more texts fit inside the last-60-seconds budget."""
    if REQUESTS_PER_MINUTE <= 0:
        return
    while True:
        now = time.monotonic()
        _recent[:] = [t for t in _recent if now - t < 60.0]
        if len(_recent) + cost <= REQUESTS_PER_MINUTE:
            _recent.extend([now] * cost)
            return
        time.sleep(max(0.05, 60.0 - (now - _recent[0]) + 0.05))


def _retry_delay(message: str) -> float | None:
    """Google states the wait in the 429 body ("'retryDelay': '48s'").

    Honouring the number the server gives us beats exponential backoff, which
    here would retry several times too early before ever waiting long enough.
    """
    match = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", message)
    return float(match.group(1)) if match else None


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, so a dot product IS the cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # a zero vector would otherwise produce NaN
    return (matrix / norms).astype(np.float32)


class EmbeddingModel:
    """
    Embeds text through the Gemini API.

        .embed_texts(list[str]) -> (N, DIMENSION) float32, L2-normalized
        .embed_query(str)       -> (DIMENSION,)   float32, L2-normalized
    """

    def __init__(self, dimension: int = DIMENSION):
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Copy .env.example to .env and set your API key."
            )
        self.model_name = MODEL
        self.dimension = dimension
        self._client = genai.Client(api_key=api_key)
        self._types = types

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed document passages."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            vectors.extend(self._embed_batch(texts[start : start + BATCH_SIZE], TASK_DOCUMENT))
        return _normalize(np.asarray(vectors, dtype=np.float32))

    def embed_query(self, question: str) -> np.ndarray:
        """Embed one user question. Note this is NOT embed_texts on one string —
        a query is deliberately encoded differently from a passage."""
        vectors = self._embed_batch([question], TASK_QUERY)
        return _normalize(np.asarray(vectors, dtype=np.float32))[0]

    def _embed_batch(self, batch: list[str], task_type: str) -> list[list[float]]:
        config = self._types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=self.dimension
        )
        delay = 2.0
        for attempt in range(MAX_RETRIES):
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
                if not retryable or attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(_retry_delay(message) or delay)
                delay *= 2
        raise RuntimeError("unreachable")
