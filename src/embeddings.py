"""
Turn text into embedding vectors using sentence-transformers.

What is an embedding?
  A list of numbers (a vector) that represents the *meaning* of a piece of text.
  The model was trained so that texts with similar meaning end up with vectors
  that point in similar directions in high-dimensional space.

Why convert text to numbers?
  Computers cannot directly compare meaning of sentences. Numbers let us use
  geometry: similar meanings → vectors close together → high similarity score.

What do the dimensions mean?
  Each dimension is not a human-readable feature like "mentions revenue."
  Together, hundreds of dimensions encode patterns the model learned from
  massive text datasets. Think of it as a compressed semantic fingerprint.

Same model for documents and queries:
  Both must live in the same vector space. If you embed chunks with model A
  and questions with model B, similarity scores are meaningless — like comparing
  coordinates from two different maps.

Similarity / semantic similarity:
  We measure how close two vectors are (cosine similarity: 1.0 = identical
  direction, 0.0 = unrelated). Semantic similarity means "means the same thing"
  even when the exact words differ — e.g. "2024 revenue" vs "how much did the
  company earn last year?"
"""

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingModel:
    """Thin wrapper around sentence-transformers for consistent encoding."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()  #It asks how many numbers will your embedding contain?"

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of strings.

        Returns a 2D array of shape (num_texts, dimension), dtype float32.
        Vectors are L2-normalized so dot product equals cosine similarity.
        """
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10,  #progress bar means it will show a progress bar if the number of texts is greater than 10.
        )
        return vectors.astype(np.float32)

    def embed_query(self, question: str) -> np.ndarray:
        """
        Embed a single user question.

        Returns a 1D array of shape (dimension,).
        """
        vector = self.embed_texts([question])[0]
        return vector
