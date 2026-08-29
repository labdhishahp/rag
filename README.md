# RAG Project — Phase 1 + Phase 2

Educational RAG pipeline: ingest a PDF, retrieve relevant chunks, and answer questions with an LLM grounded in retrieved context.

## Pipeline

```
PDF → extract → chunk → embed → FAISS → question → retrieve → prompt → LLM → answer + sources
```

## Architecture

The RAG core (`src/`) is a plain Python library with no UI opinions. Two presentation
layers sit in front of it:

```
Browser → frontend/ (Next.js)  → backend/ (FastAPI) → src/ (RAG core) → Gemini
Browser → app.py (Streamlit, legacy fallback)         → src/ (RAG core) → Gemini
```

- **`backend/`** — FastAPI HTTP API in front of `src/`. Upload/index a document,
  ask questions, reset/delete a session, check `/health`. See `backend/README.md`.
- **`frontend/`** — Next.js chat UI that talks to the backend only over HTTP
  (never imports Python). See `frontend/README.md`.
- **`app.py`** — the original Streamlit app. Kept as a fallback; see "Run Phase 2
  (Streamlit UI)" below. It imports `src/` directly and is unaffected by the
  backend/frontend split.

Quick start for the new stack, from the repo root (needs two terminals):

```bash
# terminal 1 — backend, http://localhost:8000
cd backend && pip install -r requirements.txt
cp .env.example .env   # defaults are fine for local dev
uvicorn api.main:app --reload --port 8000

# terminal 2 — frontend, http://localhost:3000
cd frontend && npm install
cp .env.example .env.local   # points at http://localhost:8000
npm run dev
```

The root `.env` (`GEMINI_API_KEY`) is read by `src/llm.py` exactly as before —
both the backend and the Streamlit app share it.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Optional: create a sample PDF for manual testing (CLI / TESTING.md only)
python3 scripts/create_sample_pdf.py

# API key for Phase 2 LLM
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your-key-here
```

### LLM provider

Phase 2 uses **Google Gemini `gemini-2.0-flash`** via the Gemini API:

- Free-tier friendly and suitable for learning projects
- Strong instruction-following for grounded Q&A
- Replaceable — implement `LLMClient` in `src/llm.py` to swap providers

## Run Phase 2 (Streamlit UI — legacy fallback)

The Streamlit app still works unchanged and remains available while the new
frontend/backend split above is verified in real use.

```bash
streamlit run app.py
```

1. Upload your own PDF or DOCX (indexed once automatically — no pre-loaded document)
2. Ask questions — only retrieval + LLM run per question
3. See answer, expandable sources, retrieved context, and RAG flow explanation

## Run Phase 1 (retrieval only, no LLM)

```bash
cd src
python3 main.py
```

Prints retrieved chunks for manual inspection — useful for debugging retrieval before trusting the LLM.

## Configuration

| Setting | Where | Default | Meaning |
|---------|-------|---------|---------|
| `CHUNK_SIZE` | `src/pipeline.py` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | `src/pipeline.py` | 50 | Overlap between chunks |
| `Top K` | Streamlit sidebar | 3 | Chunks sent to LLM (1, 3, 5, or 10) |
| Similarity threshold | Streamlit sidebar | 0.35 | Low-confidence warning cutoff |
| LLM model | `src/llm.py` | gemini-2.0-flash | Model used for answers |
| `GEMINI_API_KEY` | `.env` | — | Google Gemini API key |

## Project layout

```
├── app.py                    # Streamlit UI (Phase 2, legacy fallback)
├── backend/                  # FastAPI API layer in front of src/ (see backend/README.md)
│   ├── api/
│   └── tests/
├── frontend/                  # Next.js chat UI (see frontend/README.md)
│   ├── app/ components/ lib/
│   └── e2e/                  # live Playwright end-to-end test
├── data/sample.pdf
├── scripts/create_sample_pdf.py
├── src/
│   ├── document_loader.py    # PDF → page text
│   ├── chunker.py            # pages → chunks
│   ├── embeddings.py         # text → vectors
│   ├── vector_store.py       # FAISS index
│   ├── retriever.py          # question → top-k chunks
│   ├── pipeline.py           # PDF → Retriever + metadata
│   ├── prompt_builder.py     # chunks + question → prompt
│   ├── llm.py                # LLM API (replaceable)
│   ├── conversation.py       # conversation memory (Phase 4)
│   ├── rag.py                # full RAG orchestration
│   └── main.py               # Phase 1 CLI
├── TESTING.md                # manual test checklist
├── requirements.txt
└── README.md
```

See **TESTING.md** for the Phase 2 manual test checklist.

## Testing

```bash
python -m pytest                                  # RAG core (src/) — 77 tests
cd backend && python -m pytest                    # API layer, real RAG core + fake LLM — 15 tests
cd frontend && npx tsc --noEmit && npm run lint   # frontend static checks
cd frontend && npm run test:e2e                   # live browser -> API -> RAG -> Gemini -> browser
```

`test:e2e` starts both services itself and calls the real Gemini API — it
needs a working root `.env`, so run it deliberately rather than on every save.

## Not included (future phases)

Agents, hybrid search, reranking, multiple documents, OCR, production deployment.
