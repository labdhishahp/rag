"""
Store chunk embeddings in FAISS and map search results back to chunks.

What is FAISS?
  Facebook AI Similarity Search — a library for fast nearest-neighbor search
  over many vectors. Instead of comparing a question to every chunk one by one
  in Python loops, FAISS finds the closest vectors efficiently.

Why a vector index?
  With thousands or millions of chunks, brute-force comparison is slow.
  Even for small documents, an index keeps the pattern clear and scales later.

What does vector similarity search do?
  Given a query vector, find the stored vectors with the highest similarity
  (here: cosine similarity via inner product on normalized vectors).

What happens when we query?
  FAISS returns:
    - indices: positions in our index (0, 1, 2, ...) pointing to which vectors matched
    - scores: similarity values (higher = more similar)

Why map index → chunk?
  FAISS only stores numbers. It does not know page numbers or original text.
  We keep a parallel list `chunks` where chunks[i] is the metadata for vector i.
"""

import faiss
import numpy as np


class VectorStore:
    """FAISS index + parallel chunk metadata."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        # IndexFlatIP = exact search using inner product.
        # With normalized vectors, inner product == cosine similarity.
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[dict] = []
        # chunk_id -> position in self.chunks.
        #
        # Why not just use chunk_id as the list index? Today they happen to
        # match, because the chunker numbers chunks 0..N-1 and we add them in
        # order. But add() can be called more than once, and a second document
        # restarts its chunk_ids at 0 — so position and chunk_id would diverge
        # and silently return the wrong chunk. An explicit map cannot drift.
        self._id_to_position: dict[tuple[str | None, int], int] = {}

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
        self.index.add(embeddings)
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
        results, and the index's own metadata must not be mutated.
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
          stored dict itself would let a caller mutate the index's own metadata,
          and a later search would return a chunk polluted with a stale
          similarity score from a previous query.
        """
        if self.index.ntotal == 0:
            raise ValueError("Vector store is empty. Add chunks before searching.")

        top_k = min(top_k, self.index.ntotal)

        # FAISS expects shape (1, dimension) for a single query.
        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query, top_k)

        results: list[dict] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            result = dict(self.chunks[idx])
            result["similarity"] = float(score)
            result["rank"] = rank
            results.append(result)

        return results
