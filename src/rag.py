"""
Full RAG orchestration: retrieve → prompt → LLM → answer + sources.

Why retrieval quality matters:
  RAG can only answer from what retrieval returns. Wrong chunks → wrong or
  missing answers, even with a perfect LLM.

Similarity threshold (limitations):
  We use a threshold as a *hint* that retrieval may be weak — NOT as proof that
  an answer does or does not exist in the document. The LLM prompt is the
  primary guard against inventing answers.
"""

from llm import LLMClient
from retriever import Retriever

DEFAULT_SIMILARITY_THRESHOLD = 0.35


class RAGSystem:
    """
    Connects retrieval, prompt building, and LLM generation.

    Input:  user question
    Output: answer, source pages, retrieved chunks, confidence metadata
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        top_k: int = 3,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        embedding_dimension: int | None = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.embedding_dimension = (
            embedding_dimension or retriever.embedding_model.dimension
        )

    def answer(self, question: str) -> dict:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        # Step 1: Retrieve relevant chunks (Phase 1 retriever — unchanged)
        chunks = self.retriever.retrieve(question, top_k=self.top_k)
        best_similarity = chunks[0]["similarity"] if chunks else 0.0
        low_confidence = best_similarity < self.similarity_threshold

        # Step 2 + 3: Build prompt internally and generate answer via LLM module
        answer = self.llm.answer_with_context(
            question, chunks, low_confidence=low_confidence
        )

        # Step 4: Source pages from chunk metadata (Phase 1) — NOT from the LLM
        source_pages = extract_source_pages(chunks)

        return {
            "question": question,
            "answer": answer,
            "sources": source_pages,
            "chunks": chunks,
            "best_similarity": best_similarity,
            "low_confidence": low_confidence,
            "top_k": self.top_k,
            "embedding_dimension": self.embedding_dimension,
            "num_retrieved_chunks": len(chunks),
        }


def extract_source_pages(chunks: list[dict]) -> list[int]:
    """Unique page numbers from retrieved chunks, sorted."""
    return sorted({chunk["page_number"] for chunk in chunks})
