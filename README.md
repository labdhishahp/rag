# Knowledge Assistant

Ask questions about your own documents and get answers grounded in them, with
citations back to the exact page and section. If the document does not contain
the answer, the system says so instead of inventing one.

## The flow

```
   PDF / DOCX upload
        ↓  document_loader.py + pdf_layout.py   read the PDF's real layout
   pages of text
        ↓  chunker.py                            split on sentence boundaries
   chunks (+ page, section, neighbours)
        ↓  embeddings.py                         text -> vectors (Gemini API)
   vectors
        ↓  vector_store.py                       store + cosine search
   top matches for a question
        ↓  context_builder.py                    add neighbours, merge, budget
   evidence passages [S1] [S2] ...
        ↓  rag.py                                evidence gate: enough to answer?
        ↓  prompt_builder.py + llm.py            grounded prompt -> Gemini
   answer + verified citations
```

Two things happen before retrieval, and one after:

- **`query_understanding.py`** reads the question first. "What is the formula?"
  and "Explain the formula in detail" need different amounts of evidence, so
  depth is decided before a single vector is compared.
- **`conversation.py`** resolves follow-ups. "Explain it in more detail" is
  expanded with the previous question *for retrieval only*; the model still
  answers the question the user actually asked.
- **`rag.py`** verifies citations afterwards, deleting any `[S7]` that points at
  a passage which does not exist.

## Project layout

```
src/                     the RAG pipeline, one file per stage
  pdf_layout.py            PDF geometry -> clean blocks (headings, no headers)
  document_loader.py       PDF/DOCX -> pages
  chunker.py               pages -> chunks with metadata
  embeddings.py            text -> vectors (Gemini embedding API)
  vector_store.py          vectors + metadata, exact cosine search
  retriever.py             question -> top-k chunks
  context_builder.py       hits -> expanded, merged, budgeted evidence
  query_understanding.py   how much detail does this question want?
  conversation.py          follow-up resolution
  prompt_builder.py        evidence + question -> prompt
  llm.py                   Gemini client (swappable)
  rag.py                   orchestration, evidence gate, citation checking
  pipeline.py              ingestion: upload -> indexed Retriever
  config.py                the measured similarity thresholds

backend/api/             FastAPI HTTP layer (no RAG logic)
  main.py                  app + CORS + startup
  config.py                environment settings
  rag_bridge.py            the only file that knows both API and RAG core
  storage.py               documents, vectors and conversations
  routers/                 health, documents, chat

frontend/                Next.js chat UI (talks to the API over HTTP only)
  app/ components/ lib/
```

## Running it

Two terminals, from the repository root.

```bash
# 1. backend -> http://localhost:8000
pip install -r requirements.txt
cp .env.example .env            # add your GEMINI_API_KEY
uvicorn api.main:app --reload --port 8000 --app-dir backend

# 2. frontend -> http://localhost:3000
cd frontend && npm install
cp .env.example .env.local      # points at http://localhost:8000
npm run dev
```

With no `DATABASE_URL` set the backend keeps everything in memory, so nothing
else needs installing to try it locally.

## Configuration

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `GEMINI_API_KEY` | yes | — | Embeddings and answer generation |
| `DATABASE_URL` | deployment | — | Postgres; unset means in-memory |
| `ALLOWED_ORIGINS` | deployment | `localhost:3000` | CORS allowlist |
| `EMBEDDING_DIMENSION` | no | `768` | Embedding size |
| `EMBEDDING_RPM` | no | `90` | Client-side rate cap (free tier allows 100/min) |
| `MAX_UPLOAD_BYTES` | no | `4194304` | Kept under Vercel's 4.5MB request limit |
| `NEXT_PUBLIC_API_URL` | frontend | `localhost:8000` | Backend URL, read at **build** time |

## Deployment

Two Vercel projects from this one repository:

- **frontend** — root directory `frontend/`. Set `NEXT_PUBLIC_API_URL` before
  building; it is compiled into the bundle, so changing it needs a redeploy.
- **backend** — root directory the repository root (`vercel.json` and
  `pyproject.toml` point at `backend/api/main.py`). It must be the root because
  the API imports the RAG core from `src/`.

The backend needs a Postgres database (`DATABASE_URL`) to keep documents and
conversations across invocations, since serverless instances do not persist.

Two known limits: uploads are capped at 4.5MB by the platform, and the Gemini
free tier allows 100 embeddings/minute and 20 generations/day per model.
