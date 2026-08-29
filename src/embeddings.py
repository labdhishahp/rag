"""
Turn text into embedding vectors.

What is an embedding?
  A list of numbers (a vector) that represents the *meaning* of a piece of text.
  The model was trained so that texts with similar meaning end up with vectors
  that point in similar directions in high-dimensional space.

Why convert text to numbers?
  Computers cannot directly compare meaning of sentences. Numbers let us use
  geometry: similar meanings → vectors close together → high similarity score.

Same model for documents and queries:
  Both must live in the same vector space. If you embed chunks with model A
  and questions with model B, similarity scores are meaningless — like comparing
  coordinates from two different maps.

Similarity / semantic similarity:
  We measure how close two vectors are (cosine similarity: 1.0 = identical
  direction, 0.0 = unrelated). Semantic similarity means "means the same thing"
  even when the exact words differ — e.g. "2024 revenue" vs "how much did the
  company earn last year?"

------------------------------------------------------------------------------
TWO BACKENDS, ONE INTERFACE
------------------------------------------------------------------------------
    local    sentence-transformers, runs the model in this process.
             Free per call, no network, instant. Costs ~530MB of torch, which
             is why it cannot ship to a serverless function.

    gemini   Google's embedding API. Nothing to install, nothing to load, so
             the deployed bundle drops by ~800MB and cold start no longer
             includes loading a 128MB model. Costs a network round trip per
             batch and is subject to API rate limits.

The choice is environment-driven (EMBEDDING_BACKEND), not hardcoded, because
the right answer differs by context: tests and offline experiments want `local`
(hermetic, free, fast); the deployed API wants `gemini` (fits in the bundle).
Everything above this file — retriever, context builder, evaluation — is
written against EmbeddingModel and does not know or care which is in use.

------------------------------------------------------------------------------
WHY COSINE SIMILARITY THRESHOLDS ARE MODEL-SPECIFIC
------------------------------------------------------------------------------
Scores from different embedding models are NOT comparable. bge-small compresses
its scores upward (unrelated text ~0.4-0.5, relevant ~0.7-0.9); another model
will spread them differently. config.py's floors are calibrated against a
measured absent/answerable distribution, so CHANGING THE MODEL REQUIRES
RE-RUNNING eval/run_retrieval.py AND RE-DERIVING THOSE FLOORS. See config.py.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

# The API backend needs GEMINI_API_KEY, and this module is imported by callers
# (the evaluation harness, scripts) that never touch llm.py, so it cannot rely
# on llm.py having loaded the .env first. Optional on purpose: a deployed
# function gets its environment from the platform and has no .env file, and
# python-dotenv should not be a hard requirement there.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # pragma: no cover - depends on the install profile
    pass

# --- local backend ----------------------------------------------------------
#
# Chosen by measurement, not reputation (eval/run_retrieval.py, 23 answerable
# questions across 5 documents, top_k=3):
#
#     model                    gold in top-3   gold in context   size
#     all-MiniLM-L6-v2              0.61            0.74         22M params, 384-d
#     BAAI/bge-small-en-v1.5        0.83            0.87         33M params, 384-d
#
# Same dimension, same latency class, +13 points of recall.
LOCAL_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
LEGACY_MODEL = "all-MiniLM-L6-v2"

# Query-side instruction prefixes, per model, as documented by their authors.
# bge-small's documented prefix was measured on our gold set: identical recall
# (0.83) and a slightly NARROWER gap between absent and answerable scores, so
# it is left off.
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}

# --- gemini backend ---------------------------------------------------------
GEMINI_DEFAULT_MODEL = "gemini-embedding-001"

# Output dimensionality. gemini-embedding-001 is a Matryoshka (MRL) model: it
# is trained so that a TRUNCATED prefix of the full 3072-d vector is still a
# usable embedding. 768 keeps storage and comparison cheap and stays under
# pgvector's 2000-dimension ceiling for indexed columns.
#
# The catch, measured rather than assumed: only the full 3072-d vector comes
# back L2-normalized. Truncated outputs do not (768-d arrives with norm ~0.59),
# so they MUST be re-normalized here or every cosine score — and therefore every
# threshold in config.py — is silently wrong.
GEMINI_DEFAULT_DIMENSION = 768

# Measured against the live API: 100 texts per call is the documented hard
# ceiling (400 INVALID_ARGUMENT above it), but a 100-text batch already trips
# the free tier's rate limit (429). 32 goes through comfortably.
GEMINI_BATCH_SIZE = 32
GEMINI_MAX_RETRIES = 6

# The free tier's real constraint, read off an actual 429 body:
#   quotaId EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier
#   quotaValue 100
# It counts CONTENTS, not HTTP calls — a 32-text batch spends 32 of the 100.
# So we pace ourselves below the ceiling rather than discovering it by failing:
# a 200-chunk document is ~2 minutes of embedding on the free tier, and a paid
# tier raises the limit substantially. Set EMBEDDING_RPM to match your plan.
GEMINI_REQUESTS_PER_MINUTE = int(os.getenv("EMBEDDING_RPM", "90"))

# Asymmetric task types. This is Gemini's equivalent of the BGE query prefix:
# the model embeds a QUESTION and a PASSAGE differently on purpose, because a
# question is not trying to be similar to other questions — it is trying to be
# similar to the passage that answers it.
TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"


class _RateLimiter:
    """
    Sliding-window limiter over the last 60 seconds.

    Proactive rather than reactive: waiting for a 429 wastes the round trip and,
    worse, the retry storms of several callers tend to synchronise. Counting
    what we have spent and sleeping before the ceiling keeps a long indexing run
    predictable.
    """

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._spent: list[float] = []

    def acquire(self, cost: int) -> None:
        if self.per_minute <= 0:
            return
        while True:
            now = time.monotonic()
            self._spent = [t for t in self._spent if now - t < 60.0]
            if len(self._spent) + cost <= self.per_minute:
                self._spent.extend([now] * cost)
                return
            # Sleep until the oldest recorded request falls out of the window.
            time.sleep(max(0.05, 60.0 - (now - self._spent[0]) + 0.05))


def _retry_delay_from(message: str) -> float | None:
    """
    Google states the wait in the 429 body ("'retryDelay': '48s'"). Honouring the
    number the server gives us beats guessing with exponential backoff, which
    here would retry three times too early before ever waiting long enough.
    """
    import re

    match = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", message)
    return float(match.group(1)) if match else None


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to unit length so a dot product IS the cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Guard against a zero vector (an empty or degenerate input) producing NaN.
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class LocalBackend:
    """sentence-transformers, in-process. No network, no per-call cost."""

    def __init__(self, model_name: str, query_prefix: str | None = None):
        # Imported lazily so that a deployment WITHOUT torch installed can still
        # import this module and use the Gemini backend.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.query_prefix = (
            query_prefix if query_prefix is not None else QUERY_PREFIXES.get(model_name, "")
        )

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10,
        )
        return vectors.astype(np.float32)

    def embed_query_text(self, question: str) -> np.ndarray:
        return self.embed_documents([self.query_prefix + question])[0]


class GeminiBackend:
    """
    Google's embedding API. Nothing to install, nothing to load.

    Batches, re-normalizes truncated MRL outputs, and retries on the rate
    limits the free tier enforces.
    """

    def __init__(self, model_name: str, dimension: int, cache_dir: Path | None = None):
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Copy .env.example to .env and set your API key."
            )
        self.model_name = model_name
        self.dimension = dimension
        self.query_prefix = ""  # task_type does this job for Gemini.
        self._client = genai.Client(api_key=api_key)
        self._types = types
        self._cache = _EmbeddingCache(cache_dir, model_name, dimension) if cache_dir else None
        self._limiter = _RateLimiter(GEMINI_REQUESTS_PER_MINUTE)

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        out: list[list[float] | None] = [None] * len(texts)
        pending: list[int] = []
        for i, text in enumerate(texts):
            cached = self._cache.get(text, task_type) if self._cache else None
            if cached is not None:
                out[i] = cached
            else:
                pending.append(i)

        for start in range(0, len(pending), GEMINI_BATCH_SIZE):
            batch_idx = pending[start : start + GEMINI_BATCH_SIZE]
            batch = [texts[i] for i in batch_idx]
            vectors = self._embed_batch(batch, task_type)
            for i, vector in zip(batch_idx, vectors):
                out[i] = vector
                if self._cache:
                    self._cache.put(texts[i], task_type, vector)
        if self._cache:
            self._cache.flush()

        return _l2_normalize(np.asarray(out, dtype=np.float32))

    def _embed_batch(self, batch: list[str], task_type: str) -> list[list[float]]:
        config = self._types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=self.dimension
        )
        delay = 2.0
        for attempt in range(GEMINI_MAX_RETRIES):
            # Each content in the batch spends one unit of the per-minute quota.
            self._limiter.acquire(len(batch))
            try:
                response = self._client.models.embed_content(
                    model=self.model_name, contents=batch, config=config
                )
                return [e.values for e in response.embeddings]
            except Exception as exc:  # noqa: BLE001 - retry policy depends on the message
                message = str(exc)
                low = message.lower()
                retryable = "429" in low or "resource_exhausted" in low or "503" in low
                if not retryable or attempt == GEMINI_MAX_RETRIES - 1:
                    raise
                # Prefer the server's own retryDelay; fall back to doubling.
                time.sleep(_retry_delay_from(message) or delay)
                delay *= 2
        raise RuntimeError("unreachable")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, TASK_DOCUMENT)

    def embed_query_text(self, question: str) -> np.ndarray:
        return self._embed([question], TASK_QUERY)[0]


class _EmbeddingCache:
    """
    Content-addressed disk cache for API embeddings.

    Not a performance nicety: the retrieval evaluation re-embeds the same five
    documents on every run, which is hundreds of API calls against a rate-limited
    free tier. Caching makes an eval run repeatable and nearly free, which is what
    keeps "measure it" a cheap habit rather than an expensive one.
    """

    def __init__(self, cache_dir: Path, model_name: str, dimension: int):
        self.path = Path(cache_dir) / f"{model_name.replace('/', '_')}-{dimension}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._dirty = False
        try:
            self._data: dict[str, list[float]] = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self._data = {}

    @staticmethod
    def _key(text: str, task_type: str) -> str:
        return hashlib.sha256(f"{task_type}\x00{text}".encode()).hexdigest()

    def get(self, text: str, task_type: str) -> list[float] | None:
        return self._data.get(self._key(text, task_type))

    def put(self, text: str, task_type: str, vector: list[float]) -> None:
        self._data[self._key(text, task_type)] = list(vector)
        self._dirty = True

    def flush(self) -> None:
        if self._dirty:
            self.path.write_text(json.dumps(self._data))
            self._dirty = False


def _resolve_backend(model_name: str | None, backend: str | None) -> str:
    if backend:
        return backend
    if model_name:
        # An explicit model name identifies its own backend, so experiments like
        # EmbeddingModel("all-MiniLM-L6-v2") keep working regardless of the env.
        return "gemini" if model_name.startswith("gemini-") else "local"
    return os.getenv("EMBEDDING_BACKEND", "gemini").strip().lower()


class EmbeddingModel:
    """
    The embedding interface the rest of the system is written against.

    Public surface (unchanged across the backend swap):
        .model_name  .dimension  .query_prefix
        .embed_texts(list[str]) -> (N, dimension) float32, L2-normalized
        .embed_query(str)       -> (dimension,)  float32, L2-normalized
    """

    def __init__(
        self,
        model_name: str | None = None,
        query_prefix: str | None = None,
        backend: str | None = None,
        dimension: int | None = None,
        cache_dir: Path | None = None,
    ):
        kind = _resolve_backend(model_name, backend)
        if kind == "local":
            self._backend = LocalBackend(model_name or LOCAL_DEFAULT_MODEL, query_prefix)
        elif kind == "gemini":
            dim = dimension or int(os.getenv("EMBEDDING_DIMENSION", GEMINI_DEFAULT_DIMENSION))
            cache = cache_dir
            if cache is None and os.getenv("EMBEDDING_CACHE_DIR"):
                cache = Path(os.environ["EMBEDDING_CACHE_DIR"])
            self._backend = GeminiBackend(model_name or GEMINI_DEFAULT_MODEL, dim, cache)
        else:
            raise ValueError(
                f"Unknown EMBEDDING_BACKEND {kind!r}. Expected 'local' or 'gemini'."
            )

        self.backend_name = kind
        self.model_name = self._backend.model_name
        self.dimension = self._backend.dimension
        self.query_prefix = self._backend.query_prefix

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed document texts.

        Returns a 2D array of shape (num_texts, dimension), dtype float32.
        Vectors are L2-normalized so dot product equals cosine similarity.
        """
        return self._backend.embed_documents(texts)

    def embed_query(self, question: str) -> np.ndarray:
        """
        Embed a single user question.

        Returns a 1D array of shape (dimension,). Note this is NOT the same
        operation as embed_texts on one string: both backends deliberately
        encode a query differently from a passage (a prefix for BGE, a
        task_type for Gemini).
        """
        return self._backend.embed_query_text(question)
