"""
RAG Document Chat — Streamlit frontend for Phase 2.

Run from project root:
    streamlit run app.py

The app uses ONLY user-uploaded documents — no hardcoded sample.pdf.
Document processing is delegated to src/pipeline.py; Q&A to src/rag.py.
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
from pipeline import DocumentProcessingError, index_document_from_upload  # noqa: E402
from rag import DEFAULT_SIMILARITY_THRESHOLD, RAGSystem  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOP_K_OPTIONS = [1, 3, 5, 10]

# UI state constants
STATE_NO_DOCUMENT = "no_document"
STATE_PROCESSING = "processing"
STATE_READY = "ready"
STATE_ANSWERING = "answering"
STATE_ANSWER = "answer"

# ---------------------------------------------------------------------------
# Page config & header
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Document Chat",
    page_icon="📄",
    layout="wide",
)

_header_col, _chunks_col = st.columns([11, 1])
with _header_col:
    st.title("RAG Document Chat")
    st.caption("Upload a document and ask questions about its contents.")
with _chunks_col:
    _chunks_btn_slot = st.empty()

# ---------------------------------------------------------------------------
# Sidebar
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


@st.cache_resource
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()


@st.cache_resource
def get_llm():
    return create_llm("gemini")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    defaults = {
        "retriever": None,
        "doc_metadata": None,
        "processed_file_hash": None,
        "processing_error": None,
        "chat_history": [],
        "last_result": None,
        "ui_state": STATE_NO_DOCUMENT,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()


def _file_hash(name: str, data: bytes) -> str:
    return hashlib.sha256(name.encode() + data).hexdigest()


def _clear_document_state() -> None:
    """Discard previous document index and Q&A history."""
    st.session_state.retriever = None
    st.session_state.doc_metadata = None
    st.session_state.processed_file_hash = None
    st.session_state.processing_error = None
    st.session_state.chat_history = []
    st.session_state.last_result = None


def _process_upload(uploaded_file) -> None:
    """
    Process uploaded file immediately when it changes.

    Runs once per unique file (hash). Does NOT re-run on every question.
    """
    file_bytes = uploaded_file.getvalue()
    file_id = _file_hash(uploaded_file.name, file_bytes)

    if st.session_state.processed_file_hash == file_id:
        if st.session_state.retriever is not None:
            st.session_state.ui_state = STATE_READY
        return

    # New document — replace everything from the previous upload.
    _clear_document_state()
    st.session_state.ui_state = STATE_PROCESSING

    progress = st.empty()
    steps: list[str] = []

    def on_step(message: str) -> None:
        steps.append(message)
        progress.info("⏳ Processing your document...\n\n" + "\n".join(f"- {s}" for s in steps))

    try:
        embedding_model = get_embedding_model()
        retriever, metadata = index_document_from_upload(
            file_bytes,
            filename=uploaded_file.name,
            embedding_model=embedding_model,
            on_step=on_step,
        )
        st.session_state.retriever = retriever
        st.session_state.doc_metadata = metadata
        st.session_state.processed_file_hash = file_id
        st.session_state.ui_state = STATE_READY
        progress.empty()
    except DocumentProcessingError as exc:
        logger.exception("Document processing failed")
        st.session_state.processing_error = str(exc)
        st.session_state.ui_state = STATE_NO_DOCUMENT
        progress.empty()
        st.error(str(exc))
    except Exception:
        logger.exception("Unexpected document processing error")
        st.session_state.processing_error = (
            "Something went wrong while processing the document."
        )
        st.session_state.ui_state = STATE_NO_DOCUMENT
        progress.empty()
        st.error(
            "Something went wrong while processing the document. "
            "See terminal logs for details."
        )


def _render_upload_section() -> None:
    st.subheader("Upload your document")
    uploaded_file = st.file_uploader(
        "Choose a PDF or Word document",
        type=["pdf", "docx"],
        help="Upload one document at a time. A new upload replaces the previous index.",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        _process_upload(uploaded_file)

    meta = st.session_state.doc_metadata
    ui_state = st.session_state.ui_state

    if ui_state == STATE_PROCESSING:
        st.info("⏳ Processing your document...")
    elif ui_state == STATE_READY and meta and st.session_state.retriever is not None:
        st.success("✅ Document processed successfully")
        st.markdown(
            f"**File:** {meta['filename']}  \n"
            f"**Pages:** {meta['page_count']}  \n"
            f"**Chunks:** {meta['chunk_count']}  \n"
            f"**Embedding dimension:** {meta['embedding_dimension']}  \n"
            f"**Status:** Ready for questions"
        )
    elif st.session_state.processing_error:
        st.warning("Upload a PDF or DOCX to begin.")
    else:
        st.info("Upload a PDF to begin.")


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
        for chunk in result["chunks"]:
            st.markdown(
                f"**Rank {chunk['rank']}** · Page {chunk['page_number']} · "
                f"Chunk ID {chunk['chunk_id']} · Similarity {chunk['similarity']:.4f}"
            )
            st.text(chunk["text"])
            st.divider()

    with st.expander("How did RAG answer this?", expanded=False):
        st.markdown(
            """
            ```
            Your question
                  ↓
            Question embedding
                  ↓
            FAISS similarity search
                  ↓
            Top-k relevant chunks
                  ↓
            Prompt construction (DOCUMENT CONTEXT + USER QUESTION)
                  ↓
            LLM API
                  ↓
            Answer + source pages
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


def _render_question_section() -> None:
    st.subheader("Ask a question about your document")

    document_ready = (
        st.session_state.ui_state == STATE_READY
        and st.session_state.retriever is not None
    )

    if not document_ready:
        st.text_input(
            "Ask a question about your document...",
            disabled=True,
            placeholder="Upload and process a document first.",
            key="question_disabled",
        )
        return

    question = st.text_input(
        "Ask a question about your document...",
        placeholder="e.g. What was the company's revenue in 2024?",
        key="question_input",
        label_visibility="collapsed",
    )

    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked:
        if not question.strip():
            st.error("Please enter a question before clicking Ask.")
            return

        st.session_state.ui_state = STATE_ANSWERING

        try:
            llm = get_llm()
        except ValueError as exc:
            st.session_state.ui_state = STATE_READY
            st.error(str(exc))
            return
        except Exception:
            logger.exception("LLM initialization failed")
            st.session_state.ui_state = STATE_READY
            st.error("Could not initialize the LLM. See terminal logs for details.")
            return

        rag = RAGSystem(
            retriever=st.session_state.retriever,
            llm=llm,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            embedding_dimension=st.session_state.doc_metadata.get("embedding_dimension"),
        )

        with st.spinner("Searching document and generating answer..."):
            try:
                result = rag.answer(question.strip())
            except ValueError as exc:
                st.session_state.ui_state = STATE_READY
                st.error(str(exc))
                return
            except LLMError as exc:
                st.session_state.ui_state = STATE_READY
                st.error(str(exc))
                return
            except Exception:
                logger.exception("RAG answer failed")
                st.session_state.ui_state = STATE_READY
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
        st.session_state.ui_state = STATE_ANSWER

    if st.session_state.ui_state == STATE_ANSWERING:
        st.info("Searching document and generating answer...")

    if st.session_state.last_result and st.session_state.ui_state in (
        STATE_ANSWER,
        STATE_READY,
    ):
        _render_answer(st.session_state.last_result)


@st.dialog("Document Chunks")
def _show_chunks_dialog() -> None:
    meta = st.session_state.doc_metadata
    chunks = meta.get("chunks") if meta else None
    if not chunks:
        st.write("No document uploaded.")
        return

    for i, chunk in enumerate(chunks):
        st.markdown(f"**Chunk {chunk['chunk_id']}**")
        st.markdown(f"Page: {chunk['page_number']}")
        st.markdown(f"Length: {len(chunk['text'])} characters")
        st.markdown("")
        st.text(chunk["text"])
        if i < len(chunks) - 1:
            st.divider()


def _render_chunks_button() -> None:
    meta = st.session_state.doc_metadata
    chunks = meta.get("chunks") if meta else None
    document_ready = st.session_state.retriever is not None and chunks

    # TEMPORARY DIAGNOSTIC — remove after debugging
    st.warning(
        f"**Chunks button diagnostic**\n\n"
        f"Retriever exists: {'YES' if st.session_state.retriever is not None else 'NO'}\n\n"
        f"doc_metadata exists: {'YES' if meta is not None else 'NO'}\n\n"
        f"chunks exists: {'YES' if chunks is not None else 'NO'}\n\n"
        f"chunks type: {type(chunks).__name__}\n\n"
        f"chunks length: {len(chunks) if chunks is not None else 'N/A'}\n\n"
        f"state/status: ui_state={st.session_state.ui_state}"
        + (f", metadata status={meta.get('status')}" if meta else "")
    )

    with _chunks_btn_slot.container():
        if st.button("Chunks", disabled=not document_ready, key="chunks_viewer_btn"):
            _show_chunks_dialog()


def _render_chat_history() -> None:
    if not st.session_state.chat_history:
        return

    st.subheader("Previous questions this session")
    for i, entry in enumerate(reversed(st.session_state.chat_history), start=1):
        label = entry["question"][:80] + ("..." if len(entry["question"]) > 80 else "")
        with st.expander(
            f"Q{len(st.session_state.chat_history) - i + 1}: {label}",
            expanded=False,
        ):
            st.markdown(f"**Answer:** {entry['answer']}")
            if entry["sources"]:
                pages = ", ".join(f"Page {p}" for p in entry["sources"])
                st.caption(f"Sources: {pages}")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
_render_upload_section()
st.divider()
_render_question_section()
st.divider()
_render_chat_history()
_render_chunks_button()
