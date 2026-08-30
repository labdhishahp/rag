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
        ↓  embeddings.py            text -> vectors (bge-small via HF API)
   vectors
        ↓  vector_store.py                       store + cosine search
   top matches for a question
        ↓  context_builder.py                    add neighbours, merge, budget
   evidence passages [S1] [S2] ...
        ↓  rag.py                                evidence gate: enough to answer?
        ↓  prompt_builder.py + llm.py            grounded prompt -> Gemini
   answer + verified citations
```

**Embeddings come from two providers, and the choice is per document.**
`BAAI/bge-small-en-v1.5` over the Hugging Face API is primary (384-d); Gemini
`embedding-001` (768-d) is the fallback when Hugging Face is unreachable. A
384-d vector and a 768-d one are different coordinate systems, not just
different lengths, so **fallback happens when INDEXING a document and never
when querying one**: each document records the provider that embedded it and is
always searched with that same provider. The similarity thresholds travel with
it too, since a 0.62 score means different things under the two models.

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
  embeddings.py            text -> vectors (bge-small via HF; Gemini fallback)
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
cp .env.example .env            # add HF_TOKEN and GEMINI_API_KEY
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
| `HF_TOKEN` | yes | — | Primary embedding provider (bge-small) |
| `GEMINI_API_KEY` | yes | — | Answer generation, and the embedding fallback |
| `DATABASE_URL` | deployment | — | Postgres; unset means in-memory |
| `ALLOWED_ORIGINS` | deployment | `localhost:3000` | CORS allowlist |
| `EMBEDDING_PROVIDER` | no | `huggingface` | Provider for new documents |
| `EMBEDDING_FALLBACK_PROVIDER` | no | `gemini` | Used when the primary is unreachable |
| `EMBEDDING_DIMENSION` | no | `768` | Gemini fallback output size |
| `EMBEDDING_RPM` | no | `90` | Gemini rate cap (free tier allows 100/min) |
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
free tier allows 20 generations/day per model. Embeddings now go to Hugging
Face, so the Gemini embedding cap (1000/day, per project) only applies when the
fallback is in use.

## Retrieval quality

Measured on a 28-question gold set (23 answerable, 5 absent) across five
documents. Reproduce with `python eval/run_retrieval.py --provider huggingface`.

| Provider | Recall@3 | Gold in context |
|---|---|---|
| `huggingface` — bge-small @384 (primary) | **0.826** | 0.826 |
| `gemini` — embedding-001 @768 (fallback) | 0.870 | 0.870 |
