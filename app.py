"""
Streamlit UI for Phase 2 RAG: upload PDF → ask questions → see answer + sources.

Run from project root:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Allow imports from src/
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from embeddings import EmbeddingModel  # noqa: E402
from llm import create_llm  # noqa: E402
from pipeline import build_retriever_from_bytes  # noqa: E402
from rag import RAGSystem, DEFAULT_SIMILARITY_THRESHOLD  # noqa: E402

# --- Page config ---
st.set_page_config(page_title="RAG Document Q&A", layout="wide")
st.title("RAG Document Q&A")
st.caption("Phase 2 — Retrieval + LLM answering (educational demo)")

# --- Sidebar settings ---
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top-k chunks", min_value=1, max_value=10, value=3)
similarity_threshold = st.sidebar.slider(
    "Low-confidence threshold",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_SIMILARITY_THRESHOLD,
    step=0.05,
    help=(
        "If the best retrieval score is below this, we warn that chunks may be "
        "irrelevant. This is a heuristic — not a perfect missing-info detector."
    ),
)

# --- Cached embedding model (slow to load; reuse across reruns) ---
@st.cache_resource
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()


@st.cache_resource
def get_llm():
    return create_llm("openai")


# --- Session state defaults ---
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = None

# --- Step 1: Upload PDF ---
st.header("1. Upload document")
uploaded_file = st.file_uploader("Upload one PDF", type=["pdf"])

if uploaded_file is not None:
    if st.button("Process document", type="primary"):
        with st.spinner("Extracting text, chunking, embedding... (first run may download the model)"):
            try:
                pdf_bytes = uploaded_file.read()
                embedding_model = get_embedding_model()
                retriever = build_retriever_from_bytes(
                    pdf_bytes,
                    embedding_model=embedding_model,
                )
                st.session_state.retriever = retriever
                st.session_state.document_name = uploaded_file.name
                st.session_state.chunk_count = retriever.vector_store.index.ntotal
            except Exception as exc:
                st.error(f"Failed to process PDF: {exc}")

# --- Step 2: Show processing status ---
st.header("2. Document status")
if st.session_state.retriever is not None:
    st.success(
        f"Processed **{st.session_state.document_name}** — "
        f"{st.session_state.chunk_count} chunks indexed and ready."
    )
else:
    st.info("Upload a PDF and click **Process document** to begin.")

# --- Step 3: Ask questions ---
st.header("3. Ask a question")

if st.session_state.retriever is not None:
    question = st.text_input(
        "Your question",
        placeholder="e.g. What was the company's revenue in 2024?",
    )

    if st.button("Get answer", type="primary") and question.strip():
        try:
            llm = get_llm()
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error(f"Could not initialize LLM: {exc}")
            st.stop()

        rag = RAGSystem(
            retriever=st.session_state.retriever,
            llm=llm,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        with st.spinner("Retrieving context and generating answer..."):
            try:
                result = rag.answer(question.strip())
            except Exception as exc:
                st.error(f"Error: {exc}")
                st.stop()

        # --- Answer ---
        st.subheader("Answer")
        st.write(result["answer"])

        # --- Low-confidence warning ---
        if result["low_confidence"]:
            st.warning(
                f"Retrieval confidence is low (best similarity: "
                f"{result['best_similarity']:.3f}, threshold: {similarity_threshold:.2f}). "
                "The retrieved passages may not contain the answer. "
                "Verify the answer against the chunks below."
            )

        # --- Sources ---
        st.subheader("Sources")
        if result["sources"]:
            source_lines = "\n".join(f"- Page {page}" for page in result["sources"])
            st.markdown(source_lines)
        else:
            st.write("No sources retrieved.")

        # --- Retrieved chunks (for learning / debugging) ---
        with st.expander("Retrieved chunks (for inspection)", expanded=False):
            for chunk in result["chunks"]:
                st.markdown(
                    f"**Chunk {chunk['rank']}** — Page {chunk['page_number']} — "
                    f"Similarity: {chunk['similarity']:.4f}"
                )
                st.text(chunk["text"])
                st.divider()

        with st.expander("Full prompt sent to LLM (for learning)", expanded=False):
            st.code(result["prompt"])

else:
    st.text_input("Your question", disabled=True, placeholder="Process a document first.")

# --- Helpful test hints ---
with st.sidebar.expander("Suggested test questions"):
    st.markdown(
        """
        **In document:**
        - What was the company's revenue in 2024?
        - Who is the CEO?

        **Rephrased:**
        - How much did the firm earn in 2024?

        **Not in document:**
        - What is the population of Japan?

        **Vague:**
        - Tell me about the company.
        """
    )
