# RAG Project — Phase 1 + Phase 2

Educational RAG pipeline: ingest a PDF, retrieve relevant chunks, and answer questions with an LLM grounded in retrieved context.

## Pipeline

```
PDF → extract → chunk → embed → FAISS → question → retrieve → prompt → LLM → answer + sources
```

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

## Run Phase 2 (Streamlit UI)

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
├── app.py                    # Streamlit UI (Phase 2)
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
│   ├── rag.py                # full RAG orchestration
│   └── main.py               # Phase 1 CLI
├── TESTING.md                # manual test checklist
├── requirements.txt
└── README.md
```

See **TESTING.md** for the Phase 2 manual test checklist.

## Not included (future phases)

Agents, hybrid search, reranking, conversation memory, multiple documents, OCR, production deployment.
