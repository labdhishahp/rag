# Backend — Knowledge Assistant API

FastAPI HTTP boundary in front of the RAG core in `../src/`. No RAG logic lives
here: this is routing, request validation, persistence, and turning `src/`'s
dataclass results into JSON.

## Run locally

From the **repository root** (not this directory — the API imports the RAG core
from `../src`, so the root is the import root):

```bash
pip install -r requirements-dev.txt     # runtime deps + local extras
cp .env.example .env                    # set GEMINI_API_KEY
uvicorn api.main:app --reload --port 8000 --app-dir backend
```

With no `DATABASE_URL` set, the API uses in-memory storage — no database needed
for local work.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Readiness: embedding backend, LLM configured, storage reachable |
| POST | `/api/documents` | Upload + index a PDF/DOCX (multipart `file`); returns `session_id` |
| GET | `/api/documents/{session_id}` | Fetch a session's document metadata |
| POST | `/api/chat` | `{session_id, question, top_k?}` → answer + citations + retrieval detail |
| POST | `/api/sessions/{session_id}/reset` | Clear conversation memory, keep the document |
| DELETE | `/api/sessions/{session_id}` | Drop a session |

## How state works

A session is one indexed document plus its conversation. Neither is held in the
process between requests — see `api/storage.py` for why. Each request rebuilds
what it needs:

```
request -> touch session -> load document metadata
        -> hydrate a VectorStore from stored chunks+vectors (LRU-cached)
        -> load conversation turns
        -> answer -> append the new turns
```

That is what makes the API safe to run on a platform where the next request may
land on a different instance.

| | `MemoryStorage` | `PostgresStorage` |
|---|---|---|
| Selected when | `DATABASE_URL` unset | `DATABASE_URL` set |
| Used for | local dev, tests | deployment |
| Survives restart | no | yes |

Vectors are stored as raw `float32` bytes (exact round-trip, no text formatting
in between) and chunk metadata as JSONB (the chunker's schema has grown across
phases; one JSONB column means the next new field is not also a migration).

## Environment

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `GEMINI_API_KEY` | yes | — | Embeddings and generation |
| `DATABASE_URL` | deployment | — | Postgres; unset = in-memory |
| `ALLOWED_ORIGINS` | deployment | `localhost:3000` | CORS allowlist |
| `EMBEDDING_BACKEND` | no | `gemini` | `gemini` (API) or `local` (sentence-transformers) |
| `EMBEDDING_DIMENSION` | no | `768` | Gemini output dimensionality |
| `EMBEDDING_RPM` | no | `90` | Client-side embed rate cap (free tier allows 100/min) |
| `MAX_UPLOAD_BYTES` | no | `4194304` | Kept under Vercel's 4.5MB request-body limit |
| `LLM_PROVIDER` | no | `gemini` | See `src/llm.py` |

## Test

```bash
cd backend && python -m pytest
```

Real embedding model, real retrieval, real evidence gate; only the LLM is faked
(`tests/conftest.py`). Tests pin `EMBEDDING_BACKEND=local` so they stay
hermetic, free and offline — the deployed embedding backend's *quality* is
measured in `eval/`, which is where that belongs.
