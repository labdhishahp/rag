# Backend — Knowledge Assistant API

FastAPI HTTP boundary in front of the RAG core in `../src/`. No RAG logic
lives here — this is routing, request validation, session bookkeeping, and
turning `src/`'s dataclass results into JSON.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env          # defaults work for local dev
uvicorn api.main:app --reload --port 8000
```

`GEMINI_API_KEY` is read from the project-root `.env` (`../.env`), same as it
always was for `app.py`. This directory's `.env` only holds API-layer settings
(CORS origins, upload limits, session TTL) — see `.env.example`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + readiness (embedding model loaded? LLM configured?) |
| POST | `/api/documents` | Upload + index a PDF/DOCX (multipart `file`); returns `session_id` |
| GET | `/api/documents/{session_id}` | Fetch a session's document metadata |
| POST | `/api/chat` | `{session_id, question, top_k?}` → full answer + citations + retrieval detail |
| POST | `/api/sessions/{session_id}/reset` | Clear conversation memory, keep the indexed document |
| DELETE | `/api/sessions/{session_id}` | Drop a session entirely |

A session is one indexed document plus its `Conversation` — held in an
in-memory dict (see `api/rag_bridge.py`), bounded by `MAX_SESSIONS` and
`SESSION_TTL_SECONDS`. No database; this mirrors what Streamlit's
`st.session_state` was doing, just addressed by an explicit id instead of a
browser session.

## Test

```bash
python -m pytest
```

Real embedding model and real FAISS/evidence-gate behavior; the LLM is faked
(`tests/conftest.py: FakeLLM` / `FailingLLM`) so tests are fast, free, and
deterministic — the same pattern `../tests/test_generation_logic.py` uses.
