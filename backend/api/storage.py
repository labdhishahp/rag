"""
Persistence for documents, their vectors, and conversations.

------------------------------------------------------------------------------
WHY THIS EXISTS
------------------------------------------------------------------------------
The API used to hold a Retriever per session in a process-local dict. That is
correct for one long-lived server and impossible on a serverless platform, where
the process handling the next request may not be the process that handled the
last one. Nothing in memory can be assumed to survive.

So state moves out of the process, and the request path becomes:

    request -> load the document's chunks+vectors -> search -> answer

------------------------------------------------------------------------------
WHY NOT SEARCH IN THE DATABASE
------------------------------------------------------------------------------
pgvector could run the similarity search in SQL. It is deliberately not used
for that here:

  * The retrieval behaviour is already measured and tested against VectorStore.
    Hydrating that same class from the database keeps expansion, neighbour
    lookup, budgeting and every existing test bit-for-bit unchanged; moving the
    ranking into SQL would mean re-proving all of it.
  * Context expansion needs get_chunk(neighbour_id) anyway, so the chunks have
    to be reachable regardless — and a document's whole vector set is small
    (200 chunks x 768 dims x 4 bytes = ~600KB).

The trade is real but bounded: a cold instance transfers a document's vectors
once, then caches them. When a corpus grows past the point where that is cheap,
add a pgvector column and push the ranking down — the storage interface below
is the seam for it.

------------------------------------------------------------------------------
WHY BYTEA AND JSONB RATHER THAN COLUMNS
------------------------------------------------------------------------------
embedding BYTEA   float32 round-trips exactly through np.frombuffer, with no
                  text formatting in the middle, and needs no database
                  extension — so this runs on Neon, Supabase, RDS or a local
                  Postgres without special provisioning.
metadata JSONB    the chunker's schema has grown across phases (section,
                  char offsets and neighbour links were all added later). One
                  JSONB column means the next field the chunker learns to
                  produce is not also a database migration.

------------------------------------------------------------------------------
TWO BACKENDS
------------------------------------------------------------------------------
    memory      the previous behaviour. Zero setup, used by tests and local
                development. Loses everything on restart, which is fine there.
    postgres    the deployed default (DATABASE_URL).

Chosen by DATABASE_URL being set, so local development and the test suite need
no database and no configuration.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from typing import Optional

import numpy as np

from .rag_bridge import Conversation, Retriever, VectorStore

# Hydrated retrievers, keyed by document_id. A warm serverless instance serving
# consecutive turns of one conversation should not re-read the same vectors from
# the database every time; a cold one pays once. Small because instances are
# memory-capped and a document is ~1MB.
_RETRIEVER_CACHE_SIZE = 8


class Storage(ABC):
    """What the API needs to persist. Deliberately small."""

    # ---- documents -------------------------------------------------------
    @abstractmethod
    def save_document(self, metadata: dict, chunks: list[dict], embeddings: np.ndarray) -> None: ...

    @abstractmethod
    def get_document(self, document_id: str) -> Optional[dict]: ...

    @abstractmethod
    def load_chunks(self, document_id: str) -> tuple[list[dict], Optional[np.ndarray]]: ...

    # ---- sessions --------------------------------------------------------
    @abstractmethod
    def create_session(self, document_id: str) -> str: ...

    @abstractmethod
    def touch_session(self, session_id: str) -> Optional[str]:
        """Mark the session used and return its document_id, or None if unknown."""

    @abstractmethod
    def delete_session(self, session_id: str) -> bool: ...

    @abstractmethod
    def session_count(self) -> int: ...

    # ---- conversation ----------------------------------------------------
    @abstractmethod
    def load_conversation(self, session_id: str) -> Conversation: ...

    @abstractmethod
    def append_turns(self, session_id: str, conversation: Conversation, from_index: int) -> None: ...

    @abstractmethod
    def clear_conversation(self, session_id: str) -> bool: ...

    # ---- rate limiting ---------------------------------------------------
    @abstractmethod
    def record_and_count(self, client_id: str, kind: str, window_seconds: int) -> int:
        """
        Record one use and return how many the client has made in the window.

        Counted here rather than in the process because a serverless deployment
        runs many instances, and a per-instance counter would allow the limit
        once per instance.
        """

    # ---- shared ----------------------------------------------------------
    def healthy(self) -> tuple[bool, Optional[str]]:
        return True, None


class MemoryStorage(Storage):
    """Process-local. Same lifetime guarantees as before: none across restarts."""

    def __init__(self) -> None:
        self._documents: dict[str, dict] = {}
        self._chunks: dict[str, tuple[list[dict], np.ndarray]] = {}
        self._sessions: dict[str, dict] = {}
        self._conversations: dict[str, Conversation] = {}
        self._usage: dict[tuple[str, str], list[float]] = {}

    def save_document(self, metadata, chunks, embeddings):
        document_id = metadata["document_id"]
        self._documents[document_id] = dict(metadata)
        self._chunks[document_id] = ([dict(c) for c in chunks], np.asarray(embeddings, dtype=np.float32))

    def get_document(self, document_id):
        found = self._documents.get(document_id)
        return dict(found) if found else None

    def load_chunks(self, document_id):
        found = self._chunks.get(document_id)
        return ([dict(c) for c in found[0]], found[1]) if found else ([], None)

    def create_session(self, document_id):
        session_id = uuid.uuid4().hex
        now = time.time()
        self._sessions[session_id] = {"document_id": document_id, "created_at": now, "last_used_at": now}
        self._conversations[session_id] = Conversation()
        return session_id

    def touch_session(self, session_id):
        session = self._sessions.get(session_id)
        if session is None:
            return None
        session["last_used_at"] = time.time()
        return session["document_id"]

    def delete_session(self, session_id):
        self._conversations.pop(session_id, None)
        return self._sessions.pop(session_id, None) is not None

    def session_count(self):
        return len(self._sessions)

    def load_conversation(self, session_id):
        return self._conversations.get(session_id) or Conversation()

    def append_turns(self, session_id, conversation, from_index):
        self._conversations[session_id] = conversation

    def clear_conversation(self, session_id):
        if session_id not in self._sessions:
            return False
        self._conversations[session_id] = Conversation()
        return True

    def record_and_count(self, client_id, kind, window_seconds):
        now = time.time()
        events = self._usage.setdefault((client_id, kind), [])
        events[:] = [t for t in events if now - t < window_seconds]
        events.append(now)
        return len(events)


SCHEMA_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS documents (
           document_id   TEXT PRIMARY KEY,
           metadata      JSONB       NOT NULL,
           created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
       )""",
    """CREATE TABLE IF NOT EXISTS chunks (
           document_id   TEXT  NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
           chunk_id      INT   NOT NULL,
           metadata      JSONB NOT NULL,
           embedding     BYTEA NOT NULL,
           PRIMARY KEY (document_id, chunk_id)
       )""",
    """CREATE TABLE IF NOT EXISTS sessions (
           session_id    TEXT PRIMARY KEY,
           document_id   TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
           created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
           last_used_at  TIMESTAMPTZ NOT NULL DEFAULT now()
       )""",
    """CREATE TABLE IF NOT EXISTS turns (
           session_id    TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
           position      INT  NOT NULL,
           payload       JSONB NOT NULL,
           PRIMARY KEY (session_id, position)
       )""",
    "CREATE INDEX IF NOT EXISTS sessions_last_used_idx ON sessions (last_used_at)",
    """CREATE TABLE IF NOT EXISTS usage_events (
           client_id   TEXT        NOT NULL,
           kind        TEXT        NOT NULL,
           created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
       )""",
    "CREATE INDEX IF NOT EXISTS usage_events_lookup_idx ON usage_events (client_id, kind, created_at)",
)


# Query parameters that appear in connection strings copied from hosting
# dashboards but mean nothing to libpq, which rejects any parameter it does not
# recognise ("invalid URI query parameter"). Supabase's transaction-pooler URI
# carries pgbouncer=true, and its Prisma variants add pool sizing; both are
# hints for other clients, so dropping them changes no behaviour here.
_NON_LIBPQ_PARAMS = {"pgbouncer", "connection_limit", "pool_timeout", "schema", "connect_timeout_ms"}


def _clean_dsn(dsn: str) -> tuple[str, list[str]]:
    """Strip parameters libpq would refuse. Returns the DSN and what was removed."""
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    parts = urlsplit(dsn)
    if not parts.query:
        return dsn, []
    kept, dropped = [], []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        (dropped if key.lower() in _NON_LIBPQ_PARAMS else kept).append((key, value))
    if not dropped:
        return dsn, []
    cleaned = urlunsplit(parts._replace(query=urlencode(kept)))
    return cleaned, [k for k, _ in dropped]


class PostgresStorage(Storage):
    """
    Postgres via psycopg 3. Tested against Supabase.

    ------------------------------------------------------------------------
    TWO THINGS A POOLED, SERVERLESS POSTGRES NEEDS
    ------------------------------------------------------------------------
    1. prepare_threshold=None.
       psycopg silently promotes a query to a server-side PREPARED STATEMENT
       after it has run a few times. Supabase's transaction-mode pooler (port
       6543) hands each statement to whichever backend is free, so the prepared
       statement is rarely there when it is needed and the connection errors
       with "prepared statement _pg3_0 already exists" or "does not exist".
       Disabling preparation costs a little planning time and buys a connection
       that works through the pooler.

    2. One connection per REQUEST, not per query.
       Answering one question touches storage five times (touch the session,
       read the document, load its chunks, load the conversation, append the
       new turns). Connecting five times means five TLS handshakes and five
       pooler slots for one answer. The connection is therefore created once
       and reused, and re-opened transparently if the pooler or a serverless
       freeze has dropped it.

    autocommit is on, so a reused connection never holds a transaction open
    between calls — which is what would make holding one dangerous.
    """

    def __init__(self, dsn: str):
        self.dsn, dropped = _clean_dsn(dsn)
        if dropped:
            print(f"=== STORAGE === ignoring non-libpq connection parameters: {', '.join(dropped)}")
        self._schema_ready = False
        self._conn = None

    # -- plumbing ----------------------------------------------------------
    def _connection(self):
        """The live connection, opened on first use and reused afterwards."""
        import psycopg

        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(
                self.dsn,
                autocommit=True,
                # See the class docstring: required for transaction pooling.
                prepare_threshold=None,
            )
        return self._conn

    @contextmanager
    def _cursor(self):
        """
        A cursor on the shared connection.

        Retries once on a dropped connection, which is normal rather than
        exceptional here: poolers recycle idle connections and a serverless
        instance can be frozen for minutes between requests.
        """
        import psycopg

        try:
            with self._connection().cursor() as cur:
                yield cur
        except (psycopg.OperationalError, psycopg.InterfaceError):
            try:
                if self._conn is not None:
                    self._conn.close()
            except Exception:  # noqa: BLE001 - already broken; nothing to salvage
                pass
            self._conn = None
            with self._connection().cursor() as cur:
                yield cur

    def init_schema(self) -> None:
        if self._schema_ready:
            return
        # One statement per execute(): psycopg's extended query protocol does
        # not accept several statements in a single call.
        with self._cursor() as cur:
            for statement in SCHEMA_STATEMENTS:
                cur.execute(statement)
        self._schema_ready = True

    def close(self) -> None:
        """Release the connection (FastAPI shutdown)."""
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None

    def healthy(self):
        try:
            with self._cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return True, None
        except Exception as exc:  # noqa: BLE001 - surfaced on /health, never raised
            return False, str(exc)

    # -- documents ---------------------------------------------------------
    def save_document(self, metadata, chunks, embeddings):
        self.init_schema()
        vectors = np.asarray(embeddings, dtype=np.float32)
        document_id = metadata["document_id"]
        rows = [
            (document_id, chunk["chunk_id"], json.dumps(chunk), vectors[i].tobytes())
            for i, chunk in enumerate(chunks)
        ]
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO documents (document_id, metadata) VALUES (%s, %s)
                   ON CONFLICT (document_id) DO UPDATE SET metadata = EXCLUDED.metadata""",
                (document_id, json.dumps(metadata)),
            )
            # Re-indexing the same document replaces its chunks rather than
            # appending a second copy of them.
            cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
            cur.executemany(
                """INSERT INTO chunks (document_id, chunk_id, metadata, embedding)
                   VALUES (%s, %s, %s, %s)""",
                rows,
            )

    def get_document(self, document_id):
        self.init_schema()
        with self._cursor() as cur:
            cur.execute("SELECT metadata FROM documents WHERE document_id = %s", (document_id,))
            row = cur.fetchone()
        return row[0] if row else None

    def load_chunks(self, document_id):
        self.init_schema()
        with self._cursor() as cur:
            cur.execute(
                "SELECT metadata, embedding FROM chunks WHERE document_id = %s ORDER BY chunk_id",
                (document_id,),
            )
            rows = cur.fetchall()
        if not rows:
            return [], None
        chunks = [row[0] for row in rows]
        vectors = np.stack([np.frombuffer(bytes(row[1]), dtype=np.float32) for row in rows])
        return chunks, vectors

    # -- sessions ----------------------------------------------------------
    def create_session(self, document_id):
        self.init_schema()
        session_id = uuid.uuid4().hex
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (session_id, document_id) VALUES (%s, %s)",
                (session_id, document_id),
            )
        return session_id

    def touch_session(self, session_id):
        self.init_schema()
        with self._cursor() as cur:
            cur.execute(
                """UPDATE sessions SET last_used_at = now() WHERE session_id = %s
                   RETURNING document_id""",
                (session_id,),
            )
            row = cur.fetchone()
        return row[0] if row else None

    def delete_session(self, session_id):
        self.init_schema()
        with self._cursor() as cur:
            cur.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))
            return cur.rowcount > 0

    def session_count(self):
        self.init_schema()
        with self._cursor() as cur:
            cur.execute("SELECT count(*) FROM sessions")
            return int(cur.fetchone()[0])

    # -- conversation ------------------------------------------------------
    def load_conversation(self, session_id):
        self.init_schema()
        with self._cursor() as cur:
            cur.execute(
                "SELECT payload FROM turns WHERE session_id = %s ORDER BY position",
                (session_id,),
            )
            rows = cur.fetchall()
        conversation = Conversation()
        for (payload,) in rows:
            if payload["role"] == "user":
                conversation.add_user(payload["content"])
            else:
                conversation.add_assistant(
                    payload["content"],
                    retrieved_chunk_ids=payload.get("retrieved_chunk_ids"),
                    sources=payload.get("sources"),
                    retrieval_query=payload.get("retrieval_query"),
                )
        return conversation

    def append_turns(self, session_id, conversation, from_index):
        """Persist only the turns added since from_index — a turn is never rewritten."""
        self.init_schema()
        new = conversation.turns[from_index:]
        if not new:
            return
        rows = [
            (
                session_id,
                from_index + offset,
                json.dumps(
                    {
                        "role": turn.role,
                        "content": turn.content,
                        "retrieved_chunk_ids": turn.retrieved_chunk_ids,
                        "sources": turn.sources,
                        "retrieval_query": turn.retrieval_query,
                    }
                ),
            )
            for offset, turn in enumerate(new)
        ]
        with self._cursor() as cur:
            cur.executemany(
                """INSERT INTO turns (session_id, position, payload) VALUES (%s, %s, %s)
                   ON CONFLICT (session_id, position) DO NOTHING""",
                rows,
            )

    def clear_conversation(self, session_id):
        self.init_schema()
        with self._cursor() as cur:
            cur.execute("SELECT 1 FROM sessions WHERE session_id = %s", (session_id,))
            if cur.fetchone() is None:
                return False
            cur.execute("DELETE FROM turns WHERE session_id = %s", (session_id,))
        return True


    def record_and_count(self, client_id, kind, window_seconds):
        self.init_schema()
        with self._cursor() as cur:
            # Housekeeping first so the table cannot grow without bound; it is
            # cheap because the index covers exactly this predicate.
            cur.execute(
                "DELETE FROM usage_events WHERE created_at < now() - make_interval(secs => %s)",
                (window_seconds * 10,),
            )
            cur.execute(
                "INSERT INTO usage_events (client_id, kind) VALUES (%s, %s)", (client_id, kind)
            )
            cur.execute(
                """SELECT count(*) FROM usage_events
                   WHERE client_id = %s AND kind = %s
                     AND created_at > now() - make_interval(secs => %s)""",
                (client_id, kind, window_seconds),
            )
            return int(cur.fetchone()[0])


class RetrieverCache:
    """
    document_id -> hydrated Retriever, most-recently-used last.

    Rebuilding a VectorStore is cheap but not free (a database round trip plus
    a numpy stack), and consecutive turns of one conversation hit the same
    document every time.
    """

    def __init__(self, storage: Storage, capacity: int = _RETRIEVER_CACHE_SIZE):
        self.storage = storage
        self.capacity = capacity
        self._cache: OrderedDict[str, Retriever] = OrderedDict()

    def get(self, document_id: str, embedding_model) -> Optional[Retriever]:
        cached = self._cache.get(document_id)
        if cached is not None:
            self._cache.move_to_end(document_id)
            return cached

        chunks, vectors = self.storage.load_chunks(document_id)
        if not chunks or vectors is None:
            return None

        store = VectorStore(dimension=vectors.shape[1])
        store.add(vectors, chunks)
        retriever = Retriever(embedding_model, store)

        self._cache[document_id] = retriever
        self._cache.move_to_end(document_id)
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return retriever

    def invalidate(self, document_id: str) -> None:
        self._cache.pop(document_id, None)


def create_storage(database_url: Optional[str]) -> Storage:
    """
    Postgres when DATABASE_URL is configured, in-memory otherwise.

    In-memory is a legitimate choice locally and the reason the test suite needs
    no database. On a serverless platform it is never legitimate: instances do
    not persist, so an upload would appear to succeed and then 404 on the next
    request when a different instance served it. That failure is invisible in
    logs and looks like a bug in the application, so refuse to start instead.
    """
    if database_url:
        return PostgresStorage(database_url)

    # Vercel sets VERCEL=1 in every deployment and build environment.
    if os.getenv("VERCEL"):
        raise RuntimeError(
            "DATABASE_URL is not set. A serverless deployment cannot use in-memory "
            "storage: each request may land on a different instance, so documents "
            "and conversations would vanish between requests. Set DATABASE_URL to a "
            "Supabase connection string (use the transaction pooler on port 6543)."
        )
    return MemoryStorage()
