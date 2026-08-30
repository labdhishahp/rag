# Backend — Knowledge Assistant API

FastAPI HTTP boundary in front of the RAG core in `src/`. No RAG logic lives
here: this is routing, request validation, persistence, and turning `src/`'s
dataclass results into JSON.

## Run locally

From the **repository root**:

```bash
pip install -r backend/requirements.txt
cp .env.example .env            # set HF_TOKEN and GEMINI_API_KEY
uvicorn api.main:app --reload --port 8000 --app-dir backend
```

`backend/` is deliberately self-contained — it holds `api/`, `src/`,
`requirements.txt` and `pyproject.toml` — because a Vercel service cannot read
files above its own root.

With no `DATABASE_URL` set the API uses in-memory storage, so no database is
needed to work locally.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Readiness: each embedding provider, LLM configured, storage reachable |
| POST | `/api/documents` | Upload + index a PDF/DOCX (multipart `file`); returns `session_id` |
| GET | `/api/documents/{session_id}` | A session's document metadata |
| POST | `/api/chat` | `{session_id, question, top_k?}` → answer + citations + retrieval detail |
| POST | `/api/sessions/{session_id}/reset` | Clear the conversation, keep the document |
| DELETE | `/api/sessions/{session_id}` | Drop a session |

## How state works

A session is one indexed document plus its conversation. Neither is kept in the
process between requests — a serverless instance cannot be assumed to still
exist on the next one. Each request rebuilds what it needs:

```
request -> touch session -> load document metadata
        -> rebuild a VectorStore from stored chunks + vectors (LRU-cached)
        -> load conversation turns
        -> answer -> append the new turns
```

| | `MemoryStorage` | `PostgresStorage` |
|---|---|---|
| Selected when | `DATABASE_URL` unset | `DATABASE_URL` set |
| Used for | local development | deployment |
| Survives restart | no | yes |

Vectors are stored as raw `float32` bytes, so they round-trip exactly with no
text formatting in between. Chunk metadata is stored as JSONB, because the
chunker's fields have grown over time and a new field should not also be a
database migration.

> `PostgresStorage` has not yet been exercised against a live database.

## Environment

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `HF_TOKEN` | yes | — | Primary embedding provider (bge-small, 384-d) |
| `GEMINI_API_KEY` | yes | — | Generation, and the 768-d embedding fallback |
| `DATABASE_URL` | deployment | — | Postgres; unset = in-memory |
| `ALLOWED_ORIGINS` | deployment | `localhost:3000` | CORS allowlist |
| `EMBEDDING_PROVIDER` | no | `huggingface` | Provider for new documents |
| `EMBEDDING_DIMENSION` | no | `768` | Gemini fallback output size |
| `EMBEDDING_RPM` | no | `90` | Client-side rate cap (free tier allows 100/min) |
| `MAX_UPLOAD_BYTES` | no | `4194304` | Under Vercel's 4.5MB request-body limit |
| `LLM_PROVIDER` | no | `gemini` | See `src/llm.py` |
