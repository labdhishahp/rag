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


# Chosen by measurement, not reputation (eval/run_retrieval.py, 23 answerable
# questions across 5 documents, top_k=3):
#
#     model                    gold in top-3   gold in context   size
#     all-MiniLM-L6-v2              0.61            0.74         22M params, 384-d
#     BAAI/bge-small-en-v1.5        0.83            0.87         33M params, 384-d
#
# Same dimension, same latency class, +13 points of recall. The misses that
# remained with MiniLM were on the dense real documents (survey 0.33, BERT 0.33
# in top-3); bge-small doubled both. Switch back with EmbeddingModel("all-MiniLM-L6-v2")
# and re-run the eval if you want to see it yourself.
#
# Note: bge scores are compressed upward (unrelated text ~0.4-0.5, relevant
# ~0.7-0.9), so the similarity floors in config.py are calibrated per model.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
LEGACY_MODEL = "all-MiniLM-L6-v2"

# Query-side instruction prefixes, per model, as documented by their authors.
# bge-small's documented prefix was measured on our gold set: identical recall
# (0.83) and a slightly NARROWER gap between absent and answerable scores
# (0.511 vs 0.574, against 0.506 vs 0.622 without it), so it is left off.
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


class EmbeddingModel:
    """Thin wrapper around sentence-transformers for consistent encoding."""

    def __init__(self, model_name: str = DEFAULT_MODEL, query_prefix: str | None = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()  #It asks how many numbers will your embedding contain?"
        # Some retrieval models (the BGE family) are trained with an
        # instruction prepended to QUERIES only — documents are embedded bare.
        # It tells the model "this is a search query", which sharpens the
        # separation between relevant and irrelevant passages.
        self.query_prefix = (
            query_prefix if query_prefix is not None else QUERY_PREFIXES.get(model_name, "")
        )

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
        vector = self.embed_texts([self.query_prefix + question])[0]
        return vector
