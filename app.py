"""
RAG Document Chat — Streamlit frontend for Phase 2.

Run from project root:
    streamlit run app.py

The UI delegates all retrieval and LLM logic to src/rag.py and src/pipeline.py.
"""

import hashlib
import logging
import sys
from pathlib import Path

import streamlit as st

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from embeddings import EmbeddingModel  # noqa: E402
from llm import LLMError, create_llm  # noqa: E402
from pipeline import DocumentProcessingError, index_document_from_bytes  # noqa: E402
from rag import DEFAULT_SIMILARITY_THRESHOLD, RAGSystem  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOP_K_OPTIONS = [1, 3, 5, 10]

# ---------------------------------------------------------------------------
# Page config & header
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Document Chat",
    page_icon="📄",
    layout="wide",
)

st.title("RAG Document Chat")
st.caption("Upload a document and ask questions about its contents.")

# ---------------------------------------------------------------------------
# Sidebar — retrieval settings
# ---------------------------------------------------------------------------
st.sidebar.header("Retrieval settings")
top_k = st.sidebar.selectbox(
    "Top K",
    options=TOP_K_OPTIONS,
    index=TOP_K_OPTIONS.index(3),
    help="Number of document chunks retrieved for each question.",
)
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
        """
    )

# ---------------------------------------------------------------------------
# Cached resources (embedding model + LLM — not rebuilt per question)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()


@st.cache_resource
def get_llm():
    return create_llm("openai")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    defaults = {
        "retriever": None,
        "doc_metadata": None,
        "processed_file_hash": None,
        "chat_history": [],
        "last_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()


def _file_hash(name: str, data: bytes) -> str:
    return hashlib.sha256(name.encode() + data).hexdigest()


def _process_upload(uploaded_file) -> None:
    """Index document once; store retriever in session state."""
    pdf_bytes = uploaded_file.read()
    file_id = _file_hash(uploaded_file.name, pdf_bytes)

    if st.session_state.processed_file_hash == file_id:
        return

    with st.spinner("Processing document — extracting, chunking, embedding..."):
        try:
            embedding_model = get_embedding_model()
            retriever, metadata = index_document_from_bytes(
                pdf_bytes,
                filename=uploaded_file.name,
                embedding_model=embedding_model,
            )
            st.session_state.retriever = retriever
            st.session_state.doc_metadata = metadata
            st.session_state.processed_file_hash = file_id
            st.session_state.chat_history = []
            st.session_state.last_result = None
        except DocumentProcessingError as exc:
            logger.exception("Document processing failed")
            st.error(str(exc))
        except Exception as exc:
            logger.exception("Unexpected document processing error")
            st.error(
                "Something went wrong while processing the document. "
                "See terminal logs for details."
            )


def _render_document_status() -> None:
    st.subheader("Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="One PDF at a time. Uploading a new file replaces the current index.",
    )

    if uploaded_file is not None:
        _process_upload(uploaded_file)

    meta = st.session_state.doc_metadata
    if meta and st.session_state.retriever is not None:
        st.success("✅ Document processed")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Filename", meta["filename"])
        col2.metric("Pages", meta["page_count"])
        col3.metric("Chunks", meta["chunk_count"])
        col4.metric("Embedding dim", meta["embedding_dimension"])
    else:
        st.info("Upload a PDF to begin. The document is indexed once and reused for all questions.")


def _render_chat_history() -> None:
    if not st.session_state.chat_history:
        return

    st.subheader("Previous questions this session")
    for i, entry in enumerate(reversed(st.session_state.chat_history), start=1):
        with st.expander(f"Q{len(st.session_state.chat_history) - i + 1}: {entry['question'][:80]}", expanded=False):
            st.markdown(f"**Answer:** {entry['answer']}")
            if entry["sources"]:
                pages = ", ".join(f"Page {p}" for p in entry["sources"])
                st.caption(f"Sources: {pages}")


def _render_answer(result: dict) -> None:
    st.subheader("Answer")
    st.markdown(result["answer"])

    if result["low_confidence"]:
        st.warning(
            f"Retrieval confidence is low (best similarity: "
            f"{result['best_similarity']:.3f}, threshold: {similarity_threshold:.2f}). "
            "The retrieved passages may not contain the answer. "
            "This threshold is a hint, not proof that information is absent."
        )

    st.subheader("Sources")
    if result["chunks"]:
        for i, chunk in enumerate(result["chunks"], start=1):
            with st.expander(
                f"Source {i} — Page {chunk['page_number']} — "
                f"Similarity: {chunk['similarity']:.3f}",
                expanded=False,
            ):
                st.caption(f"Chunk ID: {chunk['chunk_id']} | Rank: {chunk['rank']}")
                st.markdown(f"_{chunk['text']}_")
    else:
        st.write("No sources retrieved.")

    with st.expander("🔍 Retrieved Context", expanded=False):
        if result["chunks"]:
            for chunk in result["chunks"]:
                st.markdown(
                    f"**Rank {chunk['rank']}** · Page {chunk['page_number']} · "
                    f"Chunk ID {chunk['chunk_id']} · Similarity {chunk['similarity']:.4f}"
                )
                st.text(chunk["text"])
                st.divider()
        else:
            st.write("No chunks were retrieved.")

    with st.expander("How did RAG answer this?", expanded=False):
        st.markdown(
            """
            ```
            Your question
                  ↓
            Question embedding (same model as document chunks)
                  ↓
            FAISS similarity search
                  ↓
            Top-k relevant chunks
                  ↓
            Chunks added to prompt as DOCUMENT CONTEXT
                  ↓
            LLM generates grounded answer
                  ↓
            Answer + source pages shown here
            ```
            """
        )
        meta = st.session_state.doc_metadata or {}
        st.markdown(
            f"- **Top K used:** {result['top_k']}\n"
            f"- **Chunks retrieved:** {result['num_retrieved_chunks']}\n"
            f"- **Embedding dimension:** {result['embedding_dimension']}\n"
            f"- **Source pages:** {', '.join(str(p) for p in result['sources']) or 'none'}\n"
            f"- **Best similarity:** {result['best_similarity']:.4f}\n"
            f"- **Document chunks indexed:** {meta.get('chunk_count', '—')}"
        )


def _render_question_area() -> None:
    st.subheader("Ask a question")

    if st.session_state.retriever is None:
        st.warning("Please upload a document first.")
        st.text_input(
            "Ask a question about your document...",
            disabled=True,
            key="question_disabled",
        )
        return

    question = st.text_input(
        "Ask a question about your document...",
        placeholder="e.g. What was the company's revenue in 2024?",
        key="question_input",
    )

    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked:
        if not question.strip():
            st.error("Please enter a question before clicking Ask.")
            return

        try:
            llm = get_llm()
        except ValueError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            logger.exception("LLM initialization failed")
            st.error("Could not initialize the LLM. See terminal logs for details.")
            return

        rag = RAGSystem(
            retriever=st.session_state.retriever,
            llm=llm,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            embedding_dimension=st.session_state.doc_metadata.get("embedding_dimension"),
        )

        with st.spinner("Retrieving context and generating answer..."):
            try:
                result = rag.answer(question.strip())
            except ValueError as exc:
                st.error(str(exc))
                return
            except LLMError as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                logger.exception("RAG answer failed")
                st.error(
                    "Something went wrong while generating the answer. "
                    "See terminal logs for details."
                )
                return

        st.session_state.last_result = result
        st.session_state.chat_history.append(
            {
                "question": result["question"],
                "answer": result["answer"],
                "sources": result["sources"],
            }
        )

    if st.session_state.last_result:
        _render_answer(st.session_state.last_result)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
_render_document_status()
st.divider()
_render_question_area()
st.divider()
_render_chat_history()
