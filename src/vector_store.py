"""
Store chunk embeddings and map search results back to chunks.

What does vector similarity search do?
  Given a query vector, find the stored vectors with the highest similarity
  (here: cosine similarity via inner product on normalized vectors).

Why not FAISS any more?
  It used to be faiss.IndexFlatIP. "Flat" means FAISS stored the vectors in a
  plain matrix and compared the query against EVERY one of them — exact search,
  no approximation, no index structure. That is a single matrix multiply, which
  numpy already does, so FAISS was a 30 MB compiled dependency computing
  `matrix @ query` on our behalf.

  It earned its place while the app was a long-lived local process. It stopped
  earning it once the target became a serverless function, where the process
  does not survive between requests: an in-memory index has to be rebuilt or
  reloaded every time regardless, so the only thing FAISS contributed was
  install weight and a C extension.

  The maths is unchanged, and deliberately so. `scores = vectors @ query` on
  L2-normalized rows IS cosine similarity, and IS what IndexFlatIP computed.
  The retrieval evaluation is expected to produce byte-identical numbers before
  and after this swap; if it does not, something else broke.

  FAISS becomes the right answer again at a scale this project does not have
  (roughly 10^5-10^6 vectors, where an approximate index like IVF or HNSW beats
  brute force). At hundreds to low thousands of chunks, brute force wins on
  simplicity and loses nothing on speed.

Why a vector store at all, rather than comparing text directly?
  Meaning is not string equality. Embedding turns "2024 revenue" and "how much
  did the company earn last year?" into nearby vectors, and geometry does the
  rest. See embeddings.py.

Why map position -> chunk?
  The matrix only holds numbers. It does not know page numbers or original
  text. We keep a parallel list `chunks` where chunks[i] describes vector i.
"""

import numpy as np


class VectorStore:
    """Embedding matrix + parallel chunk metadata."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        # (N, dimension) float32, one L2-normalized row per chunk. None until
        # the first add() — numpy has no natural empty-with-shape starting
        # point that vstack treats cleanly.
        self._vectors: np.ndarray | None = None
        self.chunks: list[dict] = []
        # (document_id, chunk_id) -> row in self._vectors / self.chunks.
        #
        # Why not just use chunk_id as the list index? Today they happen to
        # match, because the chunker numbers chunks 0..N-1 and we add them in
        # order. But add() can be called more than once, and a second document
        # restarts its chunk_ids at 0 — so position and chunk_id would diverge
        # and silently return the wrong chunk. An explicit map cannot drift.
        self._id_to_position: dict[tuple[str | None, int], int] = {}

    @property
    def ntotal(self) -> int:
        """How many vectors are stored. (Was index.ntotal under FAISS.)"""
        return len(self.chunks)

    def add(self, embeddings: np.ndarray, chunks: list[dict]) -> None:
        """
        Add embeddings and their corresponding chunk metadata.

        embeddings: shape (num_chunks, dimension)
        chunks:     same length as embeddings; order must match row order.
        """
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected dimension {self.dimension}, got {embeddings.shape[1]}"
            )

        base = len(self.chunks)
        block = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._vectors = block if self._vectors is None else np.vstack([self._vectors, block])
        self.chunks.extend(chunks)

        for offset, chunk in enumerate(chunks):
            key = (chunk.get("document_id"), chunk["chunk_id"])
            self._id_to_position[key] = base + offset

    def get_chunk(self, chunk_id: int, document_id: str | None = None) -> dict | None:
        """
        Look up one stored chunk by its ID, scoped to its document.

        This is what makes neighbour expansion possible: given a chunk that
        similarity search found, we can fetch chunk_id-1 and chunk_id+1 without
        another vector search — because adjacency is a property of the document,
        not of the embedding space.

        Returns a COPY, for the same reason search() does: callers annotate
        results, and the store's own metadata must not be mutated.
        """
        position = self._id_to_position.get((document_id, chunk_id))
        if position is None:
            return None
        return dict(self.chunks[position])

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[dict]:
        """
        Find the top_k most similar chunks to the query embedding.

        Returns a COPY of each matching chunk's full metadata, plus two
        search-specific fields:
            "similarity": float  — cosine similarity to the query
            "rank": int          — 1 = best match

        Why a pass-through instead of naming the fields explicitly:
          This method used to build a new dict with five hardcoded keys. Any
          other metadata the chunker produced (document name, page label, char
          offsets, neighbour links) was silently dropped here — so enriching the
          chunker had no visible effect. Copying the chunk means new metadata
          reaches the retriever automatically, with no change needed here.

        Why dict(chunk) and not chunk:
          Callers add fields like "similarity" to the result. Returning the
          stored dict itself would let a caller mutate the store's own metadata,
          and a later search would return a chunk polluted with a stale
          similarity score from a previous query.
        """
        if self.ntotal == 0:
            raise ValueError("Vector store is empty. Add chunks before searching.")

        top_k = min(top_k, self.ntotal)

        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        # Rows are L2-normalized (embeddings.py) and so is the query, so the
        # inner product is the cosine similarity. This one line is the entirety
        # of what IndexFlatIP did.
        scores = self._vectors @ query

        # A stable descending sort: ties resolve to the lower row index, which
        # is the order a flat index reports them in. Only the top_k slice is
        # ordered, but at this corpus size a full argsort is not worth avoiding.
        order = np.argsort(-scores, kind="stable")[:top_k]

        results: list[dict] = []
        for rank, position in enumerate(order, start=1):
            result = dict(self.chunks[int(position)])
            result["similarity"] = float(scores[position])
            result["rank"] = rank
            results.append(result)

        return results
