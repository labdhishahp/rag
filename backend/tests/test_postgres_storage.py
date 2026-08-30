"""
PostgresStorage against a real database (Supabase in practice).

Skipped unless DATABASE_URL is set, so the rest of the suite still runs with no
database. These are the tests that were missing while the Postgres path was
written but never executed — every one of them covers something that only
fails against a real server: DDL that the driver has to accept, a float32 round
trip through BYTEA, JSONB coming back as dicts, and a connection that survives
being reused.

Each test works inside its own document/session ids and cleans up after itself,
so the suite is safe to run against a database that holds other data.
"""

import os
import uuid

import numpy as np
import pytest

pytest.importorskip("psycopg")

from api.rag_bridge import Conversation  # noqa: E402
from api.storage import PostgresStorage  # noqa: E402

DATABASE_URL = os.getenv("DATABASE_URL")


pytestmark = pytest.mark.skipif(
    not DATABASE_URL, reason="DATABASE_URL not set — Postgres integration tests skipped"
)


@pytest.fixture(scope="module")
def store():
    storage = PostgresStorage(DATABASE_URL)
    storage.init_schema()
    yield storage
    storage.close()


@pytest.fixture
def document(store):
    """A throwaway document, removed afterwards along with everything cascading."""
    document_id = f"test_{uuid.uuid4().hex[:10]}"
    chunks = [
        {
            "chunk_id": i,
            "document_id": document_id,
            "document_name": "t.pdf",
            "text": f"chunk {i} body text",
            "page_number": 1 + i // 2,
            "page_label": "page",
            "section": "Compound Interest" if i < 2 else None,
            "char_start": i * 100,
            "char_end": (i + 1) * 100,
            "prev_chunk_id": i - 1 if i else None,
            "next_chunk_id": i + 1 if i < 3 else None,
        }
        for i in range(4)
    ]
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((4, 384)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    metadata = {
        "document_id": document_id,
        "filename": "t.pdf",
        "page_label": "page",
        "page_count": 2,
        "chunk_count": 4,
        "embedding_provider": "huggingface",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dimension": 384,
        "status": "ready",
    }
    store.save_document(metadata, chunks, vectors)
    yield {"id": document_id, "metadata": metadata, "chunks": chunks, "vectors": vectors}
    with store._cursor() as cur:  # noqa: SLF001 - test cleanup
        cur.execute("DELETE FROM documents WHERE document_id = %s", (document_id,))


# ---- schema ----------------------------------------------------------------

def test_schema_initialises_and_is_idempotent(store):
    """init_schema runs on a cold instance and must survive running twice."""
    store._schema_ready = False  # noqa: SLF001 - force the DDL to run again
    store.init_schema()
    with store._cursor() as cur:  # noqa: SLF001
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = ANY(%s)",
            (["documents", "chunks", "sessions", "turns"],),
        )
        assert {r[0] for r in cur.fetchall()} == {"documents", "chunks", "sessions", "turns"}


def test_health_check_reports_reachable(store):
    ok, error = store.healthy()
    assert ok is True and error is None


# ---- documents and chunks --------------------------------------------------

def test_document_metadata_round_trips_as_a_dict(store, document):
    """JSONB must come back as a dict, including the provider binding."""
    got = store.get_document(document["id"])
    assert isinstance(got, dict)
    assert got["embedding_provider"] == "huggingface"
    assert got["embedding_dimension"] == 384
    assert got["filename"] == "t.pdf"


def test_vectors_round_trip_bit_for_bit(store, document):
    """
    The reason embeddings are stored as raw float32 BYTEA: what comes back must
    be *identical*, or every similarity — and so every calibrated threshold —
    shifts after a restart.
    """
    chunks, vectors = store.load_chunks(document["id"])
    assert vectors.dtype == np.float32
    assert vectors.shape == (4, 384)
    assert np.array_equal(vectors, document["vectors"])


def test_chunk_metadata_survives_intact(store, document):
    """Expansion depends on prev/next links and section labels being exact."""
    chunks, _ = store.load_chunks(document["id"])
    assert [c["chunk_id"] for c in chunks] == [0, 1, 2, 3]
    assert chunks[0]["section"] == "Compound Interest"
    assert chunks[2]["section"] is None
    assert chunks[1]["prev_chunk_id"] == 0 and chunks[1]["next_chunk_id"] == 2
    assert chunks[0]["char_end"] == 100


def test_reindexing_replaces_chunks_rather_than_appending(store, document):
    store.save_document(document["metadata"], document["chunks"], document["vectors"])
    chunks, vectors = store.load_chunks(document["id"])
    assert len(chunks) == 4 and vectors.shape[0] == 4


def test_unknown_document_returns_nothing(store):
    assert store.get_document("does-not-exist") is None
    assert store.load_chunks("does-not-exist") == ([], None)


# ---- sessions and conversation ---------------------------------------------

def test_session_lifecycle(store, document):
    session_id = store.create_session(document["id"])
    assert store.touch_session(session_id) == document["id"]
    assert store.delete_session(session_id) is True
    assert store.touch_session(session_id) is None
    assert store.delete_session(session_id) is False


def test_conversation_persists_and_reloads_in_order(store, document):
    session_id = store.create_session(document["id"])
    conversation = Conversation()
    conversation.add_user("What is the compound interest formula?")
    conversation.add_assistant(
        "It is A = P(1 + r/n)^(nt) [S1].",
        retrieved_chunk_ids=[0, 1],
        sources=["t.pdf — page 1"],
        retrieval_query="What is the compound interest formula?",
    )
    store.append_turns(session_id, conversation, from_index=0)

    reloaded = store.load_conversation(session_id)
    assert [t.role for t in reloaded.turns] == ["user", "assistant"]
    assert reloaded.last_user_question() == "What is the compound interest formula?"
    assistant = reloaded.turns[1]
    assert assistant.retrieved_chunk_ids == [0, 1]
    assert assistant.sources == ["t.pdf — page 1"]
    assert assistant.retrieval_query == "What is the compound interest formula?"
    store.delete_session(session_id)


def test_append_turns_only_writes_the_new_ones(store, document):
    """A second turn must not rewrite or duplicate the first."""
    session_id = store.create_session(document["id"])
    conversation = Conversation()
    conversation.add_user("first")
    conversation.add_assistant("answer one")
    store.append_turns(session_id, conversation, from_index=0)

    already = len(conversation.turns)
    conversation.add_user("second")
    conversation.add_assistant("answer two")
    store.append_turns(session_id, conversation, from_index=already)

    reloaded = store.load_conversation(session_id)
    assert [t.content for t in reloaded.turns] == ["first", "answer one", "second", "answer two"]
    store.delete_session(session_id)


def test_clear_conversation_keeps_the_document(store, document):
    session_id = store.create_session(document["id"])
    conversation = Conversation()
    conversation.add_user("hello")
    store.append_turns(session_id, conversation, from_index=0)

    assert store.clear_conversation(session_id) is True
    assert store.load_conversation(session_id).is_empty
    assert store.touch_session(session_id) == document["id"]
    assert store.clear_conversation("nope") is False
    store.delete_session(session_id)


def test_deleting_a_document_cascades_to_sessions_and_turns(store):
    document_id = f"test_{uuid.uuid4().hex[:10]}"
    vectors = np.zeros((1, 384), dtype=np.float32)
    store.save_document(
        {"document_id": document_id, "filename": "x.pdf", "embedding_dimension": 384},
        [{"chunk_id": 0, "document_id": document_id, "text": "x"}],
        vectors,
    )
    session_id = store.create_session(document_id)
    conversation = Conversation()
    conversation.add_user("hi")
    store.append_turns(session_id, conversation, from_index=0)

    with store._cursor() as cur:  # noqa: SLF001
        cur.execute("DELETE FROM documents WHERE document_id = %s", (document_id,))
    assert store.touch_session(session_id) is None
    assert store.load_conversation(session_id).is_empty


# ---- retrieval state -------------------------------------------------------

def test_retriever_rehydrates_from_storage_and_searches(store, document):
    """
    The whole point of persisting vectors: a cold instance must be able to
    rebuild a working index from the database alone.
    """
    from api.storage import RetrieverCache

    class StubProvider:
        name = "huggingface"
        model_name = "BAAI/bge-small-en-v1.5"
        dimension = 384
        hard_floor, soft_floor = 0.55, 0.65

        def embed_query(self, text):
            return document["vectors"][2]  # pretend the query matches chunk 2

    cache = RetrieverCache(store)
    retriever = cache.get(document["id"], StubProvider())
    assert retriever is not None
    assert retriever.vector_store.ntotal == 4

    hits = retriever.retrieve("anything", top_k=2)
    assert hits[0]["chunk_id"] == 2
    assert hits[0]["similarity"] == pytest.approx(1.0, abs=1e-5)
    # Neighbour lookup must work off the rehydrated store, since context
    # expansion depends on it.
    assert retriever.vector_store.get_chunk(3, document["id"])["chunk_id"] == 3


# ---- connection usage ------------------------------------------------------

def test_one_connection_is_reused_across_many_operations(store, document):
    """
    Answering a question touches storage ~5 times. Those must share a
    connection: five connects per answer means five TLS handshakes and five
    pooler slots for one request.
    """
    first = store._connection()  # noqa: SLF001
    store.get_document(document["id"])
    store.load_chunks(document["id"])
    session_id = store.create_session(document["id"])
    store.touch_session(session_id)
    store.load_conversation(session_id)
    assert store._connection() is first  # noqa: SLF001 - same object, not reconnected
    assert not first.closed
    store.delete_session(session_id)


def test_a_dropped_connection_is_reopened_transparently(store, document):
    """Poolers recycle idle connections; a frozen instance wakes to a dead socket."""
    store._connection().close()  # noqa: SLF001 - simulate the pooler hanging up
    assert store.get_document(document["id"])["filename"] == "t.pdf"
    assert not store._connection().closed  # noqa: SLF001


def test_prepared_statements_are_disabled_for_transaction_pooling(store):
    """
    Supabase's transaction pooler cannot carry server-side prepared statements
    between statements. psycopg would create them automatically after a few
    executions, so preparation must stay off.
    """
    assert store._connection().prepare_threshold is None  # noqa: SLF001
    # Prove it by running the same parameterised query well past the default
    # threshold of 5 — this is exactly what used to fail through a pooler.
    for _ in range(12):
        store.get_document("does-not-exist")
