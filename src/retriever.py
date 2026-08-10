"""
Retrieve the most relevant document chunks for a user question.

Pipeline for one question:
  question → embedding → FAISS search → top-k chunks with scores
"""

from embeddings import EmbeddingModel
from vector_store import VectorStore


class Retriever:
    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(self, question: str, top_k: int = 3) -> list[dict]:
        """
        Embed the question and return the top_k matching chunks.

        Input:  natural language question
        Output: list of result dicts (see VectorStore.search)
        """
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        query_vector = self.embedding_model.embed_query(question)
        return self.vector_store.search(query_vector, top_k=top_k)
