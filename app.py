"""
Knowledge Assistant — Streamlit chat frontend.

Run from project root:
    streamlit run app.py

What this file does and does not do:
  - Uploads documents and indexes them once (src/pipeline.py).
  - Keeps a real Conversation (src/conversation.py) and passes it to the RAG
    system on every turn, so follow-ups like "explain it in more detail" are
    resolved for retrieval AND shown to the model.
  - Renders answers with their sources and, for the curious, exactly what
    retrieval did: the rewritten query, matched vs. surrounding chunks,
    evidence level.
  - Contains no retrieval or generation logic of its own.
"""

import hashlib
import logging
import sys
from pathlib import Path

import streamlit as st

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from conversation import Conversation  # noqa: E402
from embeddings import EmbeddingModel  # noqa: E402
from llm import LLMError, create_llm  # noqa: E402
from pipeline import DocumentProcessingError, index_document_from_upload  # noqa: E402
from rag import RAGSystem  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Knowledge Assistant", page_icon="📚", layout="wide")


# ---------------------------------------------------------------------------
# Cached heavy resources
# ---------------------------------------------------------------------------
@st.cache_resource
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()


@st.cache_resource
def get_llm():
    return create_llm("gemini")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "retriever": None,
        "doc_metadata": None,
        "processed_file_hash": None,
        "conversation": Conversation(),
        "messages": [],          # [{"role", "content", "result"?}] for rendering
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


def _file_hash(name: str, data: bytes) -> str:
    return hashlib.sha256(name.encode() + data).hexdigest()


def _reset_for_new_document() -> None:
    st.session_state.retriever = None
    st.session_state.doc_metadata = None
    st.session_state.processed_file_hash = None
    st.session_state.conversation = Conversation()
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Sidebar: document upload + settings + pipeline visibility
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Document")
    uploaded = st.file_uploader(
        "Upload a PDF or Word document",
        type=["pdf", "docx"],
        help="One document at a time for now. A new upload starts a new conversation.",
    )

    if uploaded is not None:
        data = uploaded.getvalue()
        file_id = _file_hash(uploaded.name, data)
        if st.session_state.processed_file_hash != file_id:
            _reset_for_new_document()
            steps: list[str] = []
            progress = st.empty()

            def on_step(message: str) -> None:
                steps.append(message)
                progress.info("\n".join(f"- {s}" for s in steps))

            try:
                retriever, metadata = index_document_from_upload(
                    data, filename=uploaded.name,
                    embedding_model=get_embedding_model(), on_step=on_step,
                )
                st.session_state.retriever = retriever
                st.session_state.doc_metadata = metadata
                st.session_state.processed_file_hash = file_id
                progress.empty()
            except DocumentProcessingError as exc:
                progress.empty()
                st.error(str(exc))
            except Exception:  # noqa: BLE001
                logger.exception("Unexpected document processing error")
                progress.empty()
                st.error("Something went wrong while processing the document. See terminal logs.")

    meta = st.session_state.doc_metadata
    if meta:
        st.success("Indexed")
        st.markdown(
            f"**{meta['filename']}**  \n"
            f"{meta['page_count']} {meta['page_label']}s · {meta['chunk_count']} chunks · "
            f"{meta['embedding_dimension']}-d embeddings"
        )

    st.divider()
    st.header("Settings")
    top_k_choice = st.selectbox(
        "Matches to retrieve (top-k)",
        options=["auto", 1, 3, 5, 10],
        index=0,
        help="'auto' lets the request decide: short factual questions fetch more matches, "
             "detailed ones fetch fewer matches plus their neighbours.",
    )
    show_debug = st.toggle("Show retrieval details under each answer", value=True)

    if st.button("New conversation", disabled=not st.session_state.messages):
        st.session_state.conversation = Conversation()
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main: chat
# ---------------------------------------------------------------------------
st.title("Knowledge Assistant")
st.caption("Ask about your document. Follow-ups like “explain that in more detail” are understood.")


def _render_result_details(result: dict) -> None:
    """The inspectable pipeline: what retrieval did for this answer."""
    with st.expander("Sources and retrieval details", expanded=False):
        cited = set(result.get("cited_labels", []))
        st.markdown("**Sources**")
        for i, citation in enumerate(result["source_citations"], start=1):
            mark = "✅" if i in cited else "▫️"
            st.markdown(f"{mark} `[S{i}]` {citation}")
        if not result["source_citations"]:
            st.write("No passages retrieved.")

        st.markdown("**Retrieval**")
        lines = [
            f"- Request read as: `{result['depth']}`"
            + (" · asked for an example" if result["understanding"].wants_example else ""),
        ]
        if result.get("was_follow_up"):
            lines.append(f"- Follow-up detected. Retrieval searched for: *{result['retrieval_query']}*")
        lines += [
            f"- Matched chunks: `{result['entry_chunk_ids']}` · added neighbours: `{result['expanded_chunk_ids']}`"
            + (f" · dropped for budget: `{result['dropped_chunk_ids']}`" if result["dropped_chunk_ids"] else ""),
            f"- Evidence level: `{result['evidence_level']}` (best similarity {result['best_similarity']:.3f})",
            f"- Context sent: {result['context_chars']} chars in {len(result['context'].passages)} passage(s); "
            f"overlap removed: {result['duplicate_chars_removed']} chars",
            f"- LLM called: {'yes' if result['llm_called'] else 'no (declined on evidence)'}"
            + (f" · model `{result['llm_model']}`" if result.get("llm_model") else ""),
        ]
        st.markdown("\n".join(lines))

        with st.expander("Evidence passages exactly as sent to the model"):
            st.text(result["context"].formatted)


# Replay the conversation.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("result") is not None:
            if msg["result"]["low_confidence"] and not msg["result"]["refused"]:
                st.caption("⚠️ Low retrieval confidence — verify against the sources.")
            if show_debug:
                _render_result_details(msg["result"])

document_ready = st.session_state.retriever is not None
prompt = st.chat_input(
    "Ask a question about your document…" if document_ready else "Upload a document to begin",
    disabled=not document_ready,
)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            llm = get_llm()
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        rag = RAGSystem(
            retriever=st.session_state.retriever,
            llm=llm,
            top_k=None if top_k_choice == "auto" else int(top_k_choice),
            embedding_dimension=(st.session_state.doc_metadata or {}).get("embedding_dimension"),
            debug=True,   # pipeline prints go to the terminal running streamlit
        )
        with st.spinner("Retrieving evidence and writing the answer…"):
            try:
                result = rag.answer(prompt, st.session_state.conversation)
            except ValueError as exc:
                st.error(str(exc))
                st.stop()
            except LLMError as exc:
                st.error(str(exc))
                st.stop()
            except Exception:  # noqa: BLE001
                logger.exception("RAG answer failed")
                st.error("Something went wrong while generating the answer. See terminal logs.")
                st.stop()

        st.markdown(result["answer"])
        if result["low_confidence"] and not result["refused"]:
            st.caption("⚠️ Low retrieval confidence — verify against the sources.")
        if show_debug:
            _render_result_details(result)

    RAGSystem.record_turn(st.session_state.conversation, result)
    st.session_state.messages.append({"role": "assistant", "content": result["answer"], "result": result})
