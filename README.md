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
backend/src/             the RAG pipeline, one file per stage
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
pip install -r backend/requirements.txt
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
| `API_KEY` | **yes on Vercel** | — | Shared secret required in `X-API-Key` |
| `RATE_LIMIT_QUESTIONS` | no | `20` | Per client, per window |
| `RATE_LIMIT_UPLOADS` | no | `10` | Per client, per window |
| `RATE_LIMIT_WINDOW_SECONDS` | no | `3600` | Rolling window |
| `BACKEND_URL` | frontend | `localhost:8000` | Python API, read at **request** time |
| `BACKEND_API_KEY` | frontend | — | Must match `API_KEY`; server-side only |

## Deployment

**One** Vercel project, using Vercel Services:

- **Framework Preset:** `Services`
- **Root Directory:** `.` (repository root, where `vercel.json` lives)

`vercel.json` declares two services — `web` (Next.js, root `frontend/`) and
`api` (FastAPI, root `backend/`) — and exactly one public rewrite, sending all
traffic to `web`.

There is deliberately **no public rewrite for `api`**. A Vercel service is
private by default, so the FastAPI service has no internet-facing route at all.
`web` reaches it through a service *binding*, which injects a private URL as
`API_INTERNAL_URL`; internal calls skip the public request pipeline. A service
without a binding cannot even derive that URL.

That is what lets the browser never hold a credential: it calls `/api/*` on the
Next.js service, which attaches the API key server-side and forwards over the
binding. The key is kept as a second layer — the binding grants access but does
not authenticate — so a public route added to `api` by mistake would still be
refused.

Because `api` has no public route, external uptime monitoring should use
`https://<app>/api/health`, which the proxy forwards.

The backend needs a Postgres database (`DATABASE_URL`) to keep documents and
conversations across invocations, since serverless instances do not persist.
Use Supabase's transaction pooler (port 6543).

### How the API is protected

The browser never talks to Python directly. It calls this app's own
`/api/[...path]` route handler, which runs server-side, attaches the API key
and forwards the request. That keeps the key out of the JS bundle — anything
the browser holds is readable — and makes the browser's requests same-origin,
so CORS never enters the picture. `ALLOWED_ORIGINS` therefore only governs
direct (non-browser) access, and the app refuses to boot on Vercel with a
wildcard origin or a missing `API_KEY`.

Rate limits are counted in Postgres rather than in memory: a serverless
deployment runs many instances, and a per-instance counter would allow the
limit once per instance. Uploads and questions have separate budgets because
they exhaust different quotas.

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
