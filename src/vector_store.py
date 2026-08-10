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

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[dict]:
        """
        Find the top_k most similar chunks to the query embedding.

        Returns a list of dicts:
            {
                "chunk_id": ...,
                "page_number": ...,
                "text": ...,
                "similarity": float,
                "rank": int,  # 1 = best match
            }
        """
        if self.index.ntotal == 0:
            raise ValueError("Vector store is empty. Add chunks before searching.")

        top_k = min(top_k, self.index.ntotal)

        # FAISS expects shape (1, dimension) for a single query.
        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query, top_k)

        results: list[dict] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            chunk = self.chunks[idx]
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "page_number": chunk["page_number"],
                    "text": chunk["text"],
                    "similarity": float(score),
                    "rank": rank,
                }
            )

        return results
