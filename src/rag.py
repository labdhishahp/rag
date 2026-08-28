"""
Full RAG orchestration: retrieve → expand → build context → LLM → answer + sources.

Why retrieval quality matters:
  RAG can only answer from what retrieval returns. Wrong chunks → wrong or
  missing answers, even with a perfect LLM.

Why retrieval alone is not enough (Step 3):
  Similarity finds the passage that RESEMBLES the question. It does not find the
  passages needed to ANSWER it. A formula resembles "what is the formula?"; the
  list defining its symbols does not resemble anything a user would type. So we
  use similarity to find an entry point, then adjacency to complete the thought.
  See context_builder.py for the full reasoning and the tradeoff.

Similarity threshold (limitations):
  We use a threshold as a *hint* that retrieval may be weak — NOT as proof that
  an answer does or does not exist in the document. The LLM prompt is the
  primary guard against inventing answers.
"""

from config import DEFAULT_TOP_K, SIMILARITY_SOFT_FLOOR
from context_builder import (
    DEFAULT_CONTEXT_BUDGET_CHARS,
    DEFAULT_NEIGHBOUR_WINDOW,
    build_context,
    citation_for,
)
from llm import LLMClient
from retriever import Retriever

# Kept as a name for the Streamlit sidebar; the value lives in config.py.
DEFAULT_SIMILARITY_THRESHOLD = SIMILARITY_SOFT_FLOOR


class RAGSystem:
    """
    Connects retrieval, context building, and LLM generation.

    Input:  user question
    Output: answer, sources, retrieved chunks, expansion detail, confidence
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        embedding_dimension: int | None = None,
        neighbour_window: int = DEFAULT_NEIGHBOUR_WINDOW,
        context_budget_chars: int = DEFAULT_CONTEXT_BUDGET_CHARS,
        debug: bool = True,
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.embedding_dimension = (
            embedding_dimension or retriever.embedding_model.dimension
        )
        self.neighbour_window = neighbour_window
        self.context_budget_chars = context_budget_chars
        self.debug = debug

    def answer(self, question: str) -> dict:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        if self.debug:
            print("\n=== QUESTION ===")
            print(f"  {question}")

        # Step 1: Retrieve entry-point chunks by pure similarity.
        chunks = self.retriever.retrieve(question, top_k=self.top_k)
        best_similarity = chunks[0]["similarity"] if chunks else 0.0
        low_confidence = best_similarity < self.similarity_threshold

        # Step 2: Expand to neighbours, merge overlaps, enforce the budget.
        context = build_context(
            chunks,
            self.retriever.vector_store,
            neighbour_window=self.neighbour_window,
            budget_chars=self.context_budget_chars,
            debug=self.debug,
        )

        # Step 3: Generate the answer from the assembled evidence.
        answer = self.llm.answer_with_context(
            question, context.formatted, low_confidence=low_confidence
        )

        # Step 4: Sources come from chunk metadata, never from the LLM.
        #
        # Note these are drawn from the FULL context, not just the similarity
        # hits: an expanded neighbour that supplied the variable definitions
        # genuinely contributed to the answer, so citing it is honest and
        # omitting it would not be.
        source_pages = extract_source_pages(context)
        source_citations = [citation_for(p) for p in context.passages]

        if self.debug:
            print("\n=== SOURCES ===")
            for citation in source_citations:
                print(f"  {citation}")
            print()

        return {
            "question": question,
            "answer": answer,
            "sources": source_pages,
            "source_citations": source_citations,
            # Entry-point similarity hits. Kept under the original key so the
            # Streamlit UI keeps working unchanged.
            "chunks": chunks,
            "best_similarity": best_similarity,
            "low_confidence": low_confidence,
            "top_k": self.top_k,
            "embedding_dimension": self.embedding_dimension,
            "num_retrieved_chunks": len(chunks),
            # Step 3 additions — what expansion actually did.
            "context": context,
            "entry_chunk_ids": context.entry_chunk_ids,
            "expanded_chunk_ids": context.expanded_chunk_ids,
            "dropped_chunk_ids": context.dropped_chunk_ids,
            "context_chars": context.total_chars,
            "duplicate_chars_removed": context.duplicate_chars_removed,
            "evidence_level": context.evidence_level,
        }


def extract_source_pages(context) -> list[int]:
    """Unique page numbers across every passage in the context, sorted."""
    return sorted({passage.page_number for passage in context.passages})
