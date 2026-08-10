"""
Full RAG orchestration: retrieve → prompt → LLM → answer + sources.

Why retrieval quality matters:
  RAG can only answer from what retrieval returns. Wrong chunks → wrong or
  missing answers, even with a perfect LLM. "Garbage in, garbage out."

Similarity threshold (limitations):
  We use a threshold as a *hint* that retrieval may be weak — NOT as proof that
  an answer does or does not exist in the document. A relevant chunk can score
  lower than expected; an irrelevant chunk can score surprisingly high. The LLM
  prompt is the primary guard against inventing answers.
"""

from llm import LLMClient
from prompt_builder import build_rag_prompt
from retriever import Retriever

# Below this best-match score, we flag retrieval as "low confidence".
# This is a heuristic — not a perfect detector of missing information.
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
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def answer(self, question: str) -> dict:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        # Step 1: Retrieve relevant chunks
        chunks = self.retriever.retrieve(question, top_k=self.top_k)
        best_similarity = chunks[0]["similarity"] if chunks else 0.0
        low_confidence = best_similarity < self.similarity_threshold

        # Step 2: Build grounded prompt
        prompt = build_rag_prompt(question, chunks, low_confidence=low_confidence)

        # Step 3: Generate answer from LLM
        answer = self.llm.generate(prompt)

        # Step 4: Collect source pages from chunk metadata (Phase 1)
        source_pages = sorted({chunk["page_number"] for chunk in chunks})

        return {
            "question": question,
            "answer": answer,
            "sources": source_pages,
            "chunks": chunks,
            "best_similarity": best_similarity,
            "low_confidence": low_confidence,
            "prompt": prompt,  # useful for debugging / learning
        }


def extract_source_pages(chunks: list[dict]) -> list[int]:
    """Unique page numbers from retrieved chunks, sorted."""
    return sorted({chunk["page_number"] for chunk in chunks})
