# RAG Project — Phase 1 + Phase 2

Educational RAG pipeline: ingest a PDF, retrieve relevant chunks, and answer questions with an LLM grounded in retrieved context.

## Pipeline

```
PDF → extract → chunk → embed → FAISS → question → retrieve → prompt → LLM → answer + sources
```

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Sample PDF for testing
python scripts/create_sample_pdf.py

# API key for Phase 2 LLM (OpenAI)
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

## Run Phase 2 (Streamlit UI)

```bash
streamlit run app.py
```

1. Upload a PDF
2. Click **Process document**
3. Ask a question
4. See answer, source pages, and retrieved chunks

## Run Phase 1 (retrieval only, no LLM)

```bash
cd src
python main.py
```

Prints retrieved chunks for manual inspection — useful for debugging retrieval before trusting the LLM.

## Configuration

| Setting | Where | Default | Meaning |
|---------|-------|---------|---------|
| `CHUNK_SIZE` | `src/pipeline.py` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | `src/pipeline.py` | 50 | Overlap between chunks |
| `top_k` | Streamlit sidebar | 3 | Chunks sent to LLM |
| Similarity threshold | Streamlit sidebar | 0.35 | Low-confidence warning cutoff |
| OpenAI model | `src/llm.py` | gpt-4o-mini | LLM used for answers |

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
│   ├── pipeline.py           # PDF → Retriever (shared)
│   ├── prompt_builder.py     # chunks + question → prompt
│   ├── llm.py                # LLM API (replaceable)
│   ├── rag.py                # full RAG orchestration
│   └── main.py                 # Phase 1 CLI
├── TESTING.md                # manual test checklist
├── requirements.txt
└── README.md
```

See **TESTING.md** for the Phase 2 manual test checklist.

## Not included (future phases)

Agents, hybrid search, reranking, conversation memory, multiple documents, OCR, production deployment.
