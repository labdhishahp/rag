# Knowledge Assistant

Ask questions about your own documents and get answers grounded in them, with
citations back to the exact page and section. If the document does not contain
the answer, the system says so instead of inventing one.

---

## 1. Overview

Upload a PDF or Word document and ask questions about it in natural language.
Every answer cites the passages it came from, and the UI can show exactly what
was retrieved, what was added around it, and what text the model actually read.

The design goal was not "build a chatbot" but **never answer from nothing**.
Three mechanisms enforce that:

- a **deterministic evidence gate** declines before an LLM call when retrieval
  found nothing relevant,
- **citation verification** strips any `[S#]` label pointing at a passage that
  does not exist,
- **measured similarity thresholds**, keyed to the embedding model that
  produced them, rather than numbers chosen by feel.

## 2. Main features

| Feature | What it does |
|---|---|
| Layout-aware ingestion | Reads the PDF's real geometry — blocks, font sizes, positions — instead of guessing structure from character counts |
| Structure-preserving chunking | Chunk boundaries land between sentences, never inside a word |
| Two embedding providers | `bge-small` primary, Gemini fallback; the choice is recorded per document and never mixed |
| Neighbour expansion | Similarity finds the entry point, adjacency completes the thought, bounded by section |
| Evidence gate | Declines deterministically, with no API call, when the best match is below a measured floor |
| Adaptive depth | "What is the formula?" and "explain it in detail" retrieve *different amounts of evidence*, not just different wording |
| Follow-up resolution | "Explain it in more detail" is expanded with the previous question for retrieval only |
| Verified citations | Every `[S#]` is checked against the passages that actually exist |
| Retrieval inspector | Per answer: matched vs. expanded chunks, evidence level, and the exact text sent to the model |
| Chunk & embedding viewer | See what the document became — every chunk and its stored vector |

## 3. Architecture

```
Browser
   │  same-origin /api/*
   ▼
Next.js route handler  (frontend/app/api/[...path]/route.ts)
   │  attaches the API key server-side, forwards over a private service binding
   ▼
FastAPI  (backend/api/)          ← no public route; unreachable from the internet
   │
   ├── routers/     health · documents · chat
   ├── security.py  shared-key auth + database-backed rate limits
   ├── storage.py   Postgres (documents, chunks+vectors, sessions, turns)
   └── rag_bridge.py   the only file that knows both the API and the RAG core
          │
          ▼
    backend/src/   the RAG pipeline, one file per stage
```

Three layers, one seam. `backend/src/` knows nothing about HTTP;
`backend/api/` contains no RAG logic; the frontend imports no Python.

## 4. Frontend

Next.js 16 (App Router, TypeScript), no UI framework — one stylesheet.

```
frontend/
├── app/
│   ├── api/[...path]/route.ts   server-side proxy to the Python API
│   ├── page.tsx · layout.tsx · globals.css
├── components/
│   ├── KnowledgeAssistant.tsx   app shell, chat state, health probe
│   ├── DocumentUpload.tsx       upload + indexing status
│   ├── MessageBubble.tsx        one message, markdown-rendered
│   ├── SourcesPanel.tsx         per-answer retrieval detail
│   └── InspectorPanel.tsx       stored chunks and their vectors
└── lib/
    ├── api.ts                   typed HTTP client
    └── types.ts                 the contract with the Python API
```

The browser never holds a credential. It calls this app's own `/api/*` route,
which runs server-side, attaches the API key and forwards the request — so the
key stays out of the JS bundle and requests are same-origin, which keeps CORS
out of the picture entirely.

## 5. Backend

FastAPI, deployed as a private serverless service.

```
backend/
├── api/                    HTTP boundary — no RAG logic
│   ├── main.py             app, CORS, startup, deployment config guard
│   ├── config.py           environment settings
│   ├── security.py         API key + per-client rate limits
│   ├── storage.py          Postgres / in-memory persistence
│   ├── rag_bridge.py       API ↔ RAG core boundary
│   └── routers/            health · documents · chat
├── src/                    the RAG pipeline, one file per stage
└── requirements.txt        exactly what the deployed function installs
```

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Readiness: each embedding provider, LLM, storage |
| `POST` | `/api/documents` | Upload and index a PDF/DOCX; returns a `session_id` |
| `GET` | `/api/documents/{session_id}` | Document metadata |
| `GET` | `/api/documents/{session_id}/chunks` | Stored chunks + embedding previews |
| `POST` | `/api/chat` | Ask a question; returns answer, citations, retrieval detail |
| `POST` | `/api/sessions/{session_id}/reset` | Clear the conversation, keep the document |
| `DELETE` | `/api/sessions/{session_id}` | Drop a session |

Nothing is held in process between requests — a serverless instance cannot be
assumed to survive. Each request rebuilds what it needs from Postgres, with a
small LRU cache for warm instances.

## 6. RAG pipeline

```
   PDF / DOCX upload
        ↓  document_loader.py + pdf_layout.py   read the PDF's real layout
   pages of text
        ↓  chunker.py                            split on sentence boundaries
   chunks (+ page, section, neighbours)
        ↓  embeddings.py            text → vectors (bge-small via HF API)
   vectors
        ↓  vector_store.py                       store + exact cosine search
   top matches for a question
        ↓  context_builder.py                    add neighbours, merge, budget
   evidence passages [S1] [S2] ...
        ↓  rag.py                                evidence gate: enough to answer?
        ↓  prompt_builder.py + llm.py            grounded prompt → Gemini
   answer + verified citations
```

| Stage | File | Responsibility |
|---|---|---|
| Layout | `pdf_layout.py` | PDF geometry → clean blocks; removes running headers, joins hyphenated wraps, detects headings |
| Loading | `document_loader.py` | PDF/DOCX → pages. A `.docx` has no fixed pages, so its units are labelled "part", never "page" |
| Chunking | `chunker.py` | 500-char chunks, 50-char overlap, packed by whole sentences |
| Embedding | `embeddings.py` | `bge-small-en-v1.5` (384-d) primary, `gemini-embedding-001` (768-d) fallback |
| Storage | `vector_store.py` | numpy matrix; `vectors @ query` on unit rows *is* cosine similarity |
| Retrieval | `retriever.py` | Pure similarity search — one job, deliberately |
| Expansion | `context_builder.py` | Neighbours within the same section, merged, overlap removed, budgeted |
| Understanding | `query_understanding.py` | Reads depth *before* retrieving, so depth changes how much evidence is gathered |
| Memory | `conversation.py` | Resolves follow-ups deterministically, for retrieval only |
| Prompt | `prompt_builder.py` | Evidence + question → grounded prompt |
| Generation | `llm.py` | Gemini client behind an abstract `LLMClient` |
| Orchestration | `rag.py` | Evidence gate, citation verification |
| Config | `config.py` | The measured similarity floors, and where they came from |

**Embeddings come from two providers, and the choice is per document.** A 384-d
vector and a 768-d one are different coordinate systems, not just different
lengths — so **fallback happens when indexing a document and never when
querying one**. Each document records the provider that embedded it and is
always searched with that same provider. The similarity thresholds travel with
it too, since 0.62 means different things under the two models.

## 7. Technologies

| Layer | Choice | Why |
|---|---|---|
| Frontend | Next.js 16, React 19, TypeScript | Server-side route handler keeps the API key out of the browser |
| Backend | FastAPI, Python 3.10+ | Async, typed request validation, small serverless footprint |
| PDF | PyMuPDF | Exposes block/span geometry; the layout pipeline depends on it |
| Embeddings | `BAAI/bge-small-en-v1.5` via HF API; Gemini fallback | Small, fast, measurably adequate — and no 650 MB torch install |
| Vector search | numpy | Exact brute force; at this corpus size an ANN index buys nothing |
| LLM | Gemini via `google-genai` | Behind an abstract client, so the provider is swappable |
| Database | Postgres (Supabase) | Vectors as `BYTEA` round-trip exactly; chunk metadata as `JSONB` |
| Hosting | Vercel Services | One project, two services, a private backend |

## 8. Environment variables

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `HF_TOKEN` | yes | — | Primary embedding provider (bge-small) |
| `GEMINI_API_KEY` | yes | — | Answer generation, and the embedding fallback |
| `DATABASE_URL` | deployment | — | Postgres; unset means in-memory |
| `API_KEY` | **yes on Vercel** | — | Shared secret required in `X-API-Key` |
| `ALLOWED_ORIGINS` | deployment | `localhost:3000` | CORS allowlist |
| `EMBEDDING_PROVIDER` | no | `huggingface` | Provider for new documents |
| `EMBEDDING_FALLBACK_PROVIDER` | no | `gemini` | Used when the primary is unreachable |
| `EMBEDDING_DIMENSION` | no | `768` | Gemini fallback output size |
| `EMBEDDING_RPM` | no | `90` | Gemini rate cap |
| `LLM_PROVIDER` | no | `gemini` | See `backend/src/llm.py` |
| `MAX_UPLOAD_BYTES` | no | `4194304` | Kept under Vercel's 4.5 MB request limit |
| `RATE_LIMIT_QUESTIONS` | no | `20` | Per client, per window |
| `RATE_LIMIT_UPLOADS` | no | `10` | Per client, per window |
| `RATE_LIMIT_WINDOW_SECONDS` | no | `3600` | Rolling window |
| `BACKEND_URL` | frontend | `localhost:8000` | Python API, read at **request** time |
| `BACKEND_API_KEY` | frontend | — | Must match `API_KEY`; server-side only |

Copy `.env.example` → `.env` and `frontend/.env.example` → `frontend/.env.local`.
Never commit either; both are gitignored.

## 9. Running locally

Two terminals, from the repository root.

```bash
# 1. backend -> http://localhost:8000
pip install -r backend/requirements.txt
cp .env.example .env               # add HF_TOKEN and GEMINI_API_KEY
uvicorn api.main:app --reload --port 8000 --app-dir backend

# 2. frontend -> http://localhost:3000
cd frontend && npm install
cp .env.example .env.local         # points at http://localhost:8000
npm run dev
```

With no `DATABASE_URL` set the backend keeps everything in memory, so nothing
else needs installing to try it locally.

## 10. Deployment

**One** Vercel project, using Vercel Services. Framework preset `Services`,
root directory `.` (where `vercel.json` lives).

`vercel.json` declares two services — `web` (Next.js, root `frontend/`) and
`api` (FastAPI, root `backend/`) — and exactly one public rewrite, sending all
traffic to `web`.

There is deliberately **no public rewrite for `api`**. A Vercel service is
private by default, so the FastAPI service has no internet-facing route at all.
`web` reaches it through a service *binding*, which injects a private URL as
`API_INTERNAL_URL`. A service without a binding cannot even derive that URL.

That is what lets the browser never hold a credential: it calls `/api/*` on the
Next.js service, which attaches the API key server-side and forwards over the
binding. The key is kept as a second layer — the binding grants access but does
not authenticate — so a public route added to `api` by mistake would still be
refused.

Two operational notes:

- The backend needs Postgres (`DATABASE_URL`) to keep documents and
  conversations across invocations; serverless instances do not persist. Use
  Supabase's transaction pooler (port 6543). The app refuses to start on Vercel
  without it, rather than silently losing every upload.
- Because `api` has no public route, external uptime monitoring should use
  `https://<app>/api/health`, which the proxy forwards.

## 11. Project structure

```
.
├── backend/
│   ├── api/                  FastAPI HTTP layer (no RAG logic)
│   │   ├── main.py  config.py  security.py  storage.py  rag_bridge.py
│   │   └── routers/          health.py  documents.py  chat.py
│   ├── src/                  the RAG pipeline, one file per stage
│   │   ├── pdf_layout.py  document_loader.py  chunker.py
│   │   ├── embeddings.py  vector_store.py  retriever.py
│   │   ├── context_builder.py  query_understanding.py  conversation.py
│   │   ├── prompt_builder.py  llm.py  rag.py  pipeline.py
│   │   └── config.py         the measured similarity thresholds
│   ├── pyproject.toml        declares the Vercel entrypoint
│   └── requirements.txt      exactly what the deployed function installs
│
├── frontend/                 Next.js chat UI (talks to the API over HTTP only)
│   ├── app/  components/  lib/
│   ├── package.json  tsconfig.json  next.config.ts  eslint.config.mjs
│   └── .env.example
│
├── eval/                     retrieval quality measurement
│   ├── gold.jsonl            28 questions (23 answerable, 5 deliberately absent)
│   └── run_retrieval.py      measures recall and context quality, zero LLM calls
│
├── data/                     the two small documents the gold set refers to
├── vercel.json               two services, one public rewrite
├── .env.example
└── README.md
```

### A note on `eval/`

The similarity thresholds in `backend/src/config.py` were **measured, not
chosen**. `eval/run_retrieval.py` reproduces that measurement, and makes zero
LLM calls so it can be run on every change:

```bash
python eval/run_retrieval.py --provider huggingface
```

Measured on the 28-question gold set: **recall@3 = 0.826** with bge-small at
384 dimensions. Three of the five documents in the gold set are large public
papers that are not redistributed here; the harness reports them as skipped and
still measures the rest.
