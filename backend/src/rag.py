"""
Full RAG orchestration: understand → retrieve → expand → gate → prompt → LLM → answer + sources.

Why retrieval quality matters:
  RAG can only answer from what retrieval returns. Wrong chunks → wrong or
  missing answers, even with a perfect LLM.

Why retrieval alone is not enough (Step 3):
  Similarity finds the passage that RESEMBLES the question. It does not find the
  passages needed to ANSWER it. So we use similarity to find an entry point, then
  adjacency (within the same section) to complete the thought. See context_builder.py.

Why the request is read first (Phase 3):
  "What is the formula?" and "Explain the formula in detail" need DIFFERENT
  evidence, not just different wording. Depth sets the retrieval config before a
  single vector is compared. See query_understanding.py.

Why the evidence gate is deterministic (Phase 3):
  When the best hit is below the hard floor, retrieval found nothing relevant.
  Sending near-noise to the LLM with a stern note relies on the model obeying;
  declining in code costs no API call and behaves the same every time. The
  in-prompt refusal rule remains as the second line of defence for the cases
  no similarity threshold can catch (see config.py on the a05 hard negative).
"""

import re

from config import DEFAULT_TOP_K, SIMILARITY_HARD_FLOOR, SIMILARITY_SOFT_FLOOR
from conversation import Conversation
from context_builder import (
    DEFAULT_CONTEXT_BUDGET_CHARS,
    DEFAULT_NEIGHBOUR_WINDOW,
    build_context,
    citation_for,
    labels_for,
)
from llm import LLMClient
from prompt_builder import REFUSAL_TEXT, build_rag_prompt
from query_understanding import retrieval_config_for, understand
from retriever import Retriever

# Exposed as a name for callers; the value itself lives in config.py.
DEFAULT_SIMILARITY_THRESHOLD = SIMILARITY_SOFT_FLOOR

# One or more labels inside one pair of brackets: [S1], [A2], [S1, S3], [B1,B2].
# The prefix is a letter group so a comparison's [A#]/[B#] labels match too, and
# grouping is supported because models write "[S1, S3]" unprompted.
#
# \d+ rather than \d{1,2}: a bounded digit count means an oversized label like
# [S999] does not MATCH, so it is never examined and survives verification —
# exactly the invented source citations exist to prevent. A summary can easily
# carry more than 99 passages, which is how this was found.
_CITATION = re.compile(r"\[((?:[A-Z]{1,2}\d+)(?:[,\s]+[A-Z]{1,2}\d+)*)\]")


class RAGSystem:
    """
    Connects query understanding, retrieval, context building, and generation.

    Input:  user question
    Output: answer, sources, retrieved chunks, expansion detail, confidence
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        embedding_dimension: int | None = None,
        neighbour_window: int | None = None,
        context_budget_chars: int | None = None,
        debug: bool = True,
    ):
        self.retriever = retriever
        self.llm = llm
        # None means "let the request decide" (query_understanding); a value
        # pins it (an explicit caller override).
        self.top_k = top_k

        # The similarity floors belong to whichever embedding provider built
        # this retriever's index — a 0.62 hit means different things under
        # bge-small and gemini-embedding-001. Taking them from the provider
        # rather than from a module constant is what keeps the evidence gate
        # correct when documents in the system were embedded by different
        # providers. See embeddings.py and config.py.
        provider = retriever.embedding_model
        self.hard_floor = getattr(provider, "hard_floor", SIMILARITY_HARD_FLOOR)
        self.soft_floor = getattr(provider, "soft_floor", SIMILARITY_SOFT_FLOOR)
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None else self.soft_floor
        )
        self.embedding_dimension = (
            embedding_dimension or retriever.embedding_model.dimension
        )
        # None means "let the request decide" (query_understanding). A value
        # pins it — useful for experiments and for the CLI.
        self.neighbour_window = neighbour_window
        self.context_budget_chars = context_budget_chars
        self.debug = debug

    def answer(self, question: str, conversation: Conversation | None = None) -> dict:
        """
        Answer one user message.

        conversation — optional. When given, follow-up messages ("explain it
        more") are resolved against it for RETRIEVAL, and the recent turns are
        shown to the model for reference resolution. The conversation is NOT
        updated here; the caller records the turn (see record_turn), so a
        failed call never leaves a half-written history.
        """
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        # Step 0: Read the request.
        understanding = understand(question)
        config = retrieval_config_for(understanding)
        top_k = self.top_k if self.top_k is not None else config["top_k"]
        window = self.neighbour_window if self.neighbour_window is not None else config["neighbour_window"]
        budget = self.context_budget_chars if self.context_budget_chars is not None else config["budget_chars"]

        # Step 0b: Decide what retrieval should actually search for.
        # A standalone question searches for itself. A follow-up searches for
        # itself PLUS the question it follows — deterministically, no LLM.
        follow_up = bool(conversation) and not conversation.is_empty and understanding.needs_context
        retrieval_query = (
            conversation.retrieval_query_for(question, needs_context=True)
            if follow_up
            else question
        )
        conversation_text = conversation.format_for_prompt() if (conversation and not conversation.is_empty) else None

        if self.debug:
            print("\n=== QUESTION ===")
            print(f"  {question}")
            print(f"  depth={understanding.describe()}  ->  top_k={top_k} neighbour_window={window} budget={budget}")
            if follow_up:
                print("\n=== REWRITTEN QUERY (retrieval only) ===")
                print(f"  {retrieval_query}")

        # Step 1: Retrieve entry-point chunks by pure similarity.
        chunks = self.retriever.retrieve(retrieval_query, top_k=top_k)
        if follow_up:
            # Also retrieve with the raw message and keep its best hits. A poor
            # augmentation must never hide a match the bare message would have
            # found ("And in 2023?" matches the 2023 chunk on its own; glued to
            # the 2024 question it drifts to 2024).
            raw_hits = self.retriever.retrieve(question, top_k=top_k)
            chunks = _merge_hits(chunks, raw_hits, keep_secondary=2)
        best_similarity = chunks[0]["similarity"] if chunks else 0.0
        low_confidence = best_similarity < self.similarity_threshold

        # Step 2: Expand to same-section neighbours, merge, enforce the budget.
        context = build_context(
            chunks,
            self.retriever.vector_store,
            neighbour_window=window,
            budget_chars=budget,
            hard_floor=self.hard_floor,
            soft_floor=self.soft_floor,
            debug=self.debug,
        )

        # Step 3: Evidence gate — nothing relevant found, so do not ask the LLM.
        if context.evidence_level == "none":
            if self.debug:
                print("\n=== EVIDENCE GATE === best similarity below hard floor; declining without an LLM call")
            return self._result(
                question, understanding, REFUSAL_TEXT, chunks, context,
                best_similarity, low_confidence, llm_called=False, cited=[],
                retrieval_query=retrieval_query,
            )

        # Step 4: Generate from the assembled evidence (+ conversation for reference resolution).
        answer = self.llm.answer_with_context(
            question,
            context.formatted,
            low_confidence=low_confidence,
            depth=understanding.depth,
            wants_example=understanding.wants_example,
            conversation=conversation_text,
        )

        # Step 5: Verify citations against the labels that actually exist.
        answer, cited = check_citations(answer, labels_for(context.passages))

        result = self._result(
            question, understanding, answer, chunks, context,
            best_similarity, low_confidence, llm_called=True, cited=cited,
            retrieval_query=retrieval_query,
        )
        if self.debug:
            print("\n=== SOURCES ===")
            for i, citation in enumerate(result["source_citations"], start=1):
                used = "cited" if i in cited else "     "
                print(f"  [S{i}] {used}  {citation}")
            print()
        return result

    @staticmethod
    def record_turn(conversation: Conversation, result: dict) -> None:
        """Append the user message and the answer (with its evidence) to the conversation."""
        conversation.add_user(result["question"])
        conversation.add_assistant(
            result["answer"],
            retrieved_chunk_ids=[cid for p in result["context"].passages for cid in p.chunk_ids],
            sources=result["source_citations"],
            retrieval_query=result["retrieval_query"],
        )

    def _result(self, question, understanding, answer, chunks, context,
                best_similarity, low_confidence, llm_called, cited,
                retrieval_query=None) -> dict:
        return build_result(
            question=question, understanding=understanding, answer=answer,
            context=context, llm=self.llm, chunks=chunks,
            best_similarity=best_similarity, low_confidence=low_confidence,
            llm_called=llm_called, cited=cited, retrieval_query=retrieval_query,
            embedding_dimension=self.embedding_dimension, task="answer",
        )


def build_result(
    *,
    question,
    understanding,
    answer,
    context,
    llm,
    task: str = "answer",
    chunks=None,
    best_similarity: float = 0.0,
    low_confidence: bool = False,
    llm_called: bool = True,
    cited=(),
    retrieval_query=None,
    embedding_dimension=None,
) -> dict:
    """
    The one result shape every task kind returns.

    Shared rather than duplicated on purpose: the API serializer, the
    conversation recorder and the UI all read these keys, so a second builder
    would be a second contract to keep in sync — and drifting from it is how
    "one pipeline" quietly becomes two.

    `cited` holds 1-based positions into the label list that was verified, and
    `context.passages` is in the SAME order, which is what makes cited_sources
    correct for a comparison whose labels are [A1..][B1..].
    """
    chunks = list(chunks or [])
    source_citations = [citation_for(p) for p in context.passages]
    return {
        "question": question,
        "task": task,
        "retrieval_query": retrieval_query or question,
        "was_follow_up": (retrieval_query or question) != question,
        "answer": answer,
        "depth": understanding.depth,
        "understanding": understanding,
        "sources": extract_source_pages(context),
        "source_citations": source_citations,
        # Passages the model actually cited, 1-based, in label order.
        "cited_sources": [source_citations[i - 1] for i in cited if 0 < i <= len(source_citations)],
        "cited_labels": list(cited),
        # Entry-point similarity hits. Empty for tasks that do not search.
        "chunks": chunks,
        "best_similarity": best_similarity,
        "low_confidence": low_confidence,
        "top_k": len(chunks),
        "llm_provider": getattr(llm, "name", None),
        "llm_model": getattr(llm, "active_model", None),
        "embedding_dimension": embedding_dimension,
        "num_retrieved_chunks": len(chunks),
        "context": context,
        "entry_chunk_ids": context.entry_chunk_ids,
        "expanded_chunk_ids": context.expanded_chunk_ids,
        "dropped_chunk_ids": context.dropped_chunk_ids,
        "context_chars": context.total_chars,
        "duplicate_chars_removed": context.duplicate_chars_removed,
        "evidence_level": context.evidence_level,
        "llm_called": llm_called,
        "refused": answer.strip().startswith(REFUSAL_TEXT),
    }


def _merge_hits(primary: list[dict], secondary: list[dict], keep_secondary: int = 2) -> list[dict]:
    """
    All primary hits plus the best `keep_secondary` secondary hits not already
    present, ordered by similarity.

    Measured reason for GUARANTEEING secondary hits rather than pooling and
    truncating: the raw message's hits usually score lower than the augmented
    query's (vaguer text), so a pooled top-k silently dropped exactly the hits
    the union existed to protect. Two dependent follow-ups regressed until the
    raw hits were kept unconditionally.
    """
    seen = {(h.get("document_id"), h["chunk_id"]) for h in primary}
    extra = []
    for hit in secondary:
        key = (hit.get("document_id"), hit["chunk_id"])
        if key not in seen:
            seen.add(key)
            extra.append(hit)
        if len(extra) >= keep_secondary:
            break
    merged = sorted(primary + extra, key=lambda h: h["similarity"], reverse=True)
    for rank, hit in enumerate(merged, start=1):
        hit["rank"] = rank
    return merged


def check_citations(answer: str, valid_labels: list[str]) -> tuple[str, list[int]]:
    """
    Keep citations that point at real passages; strip the ones that do not.

    A model can emit "[S7]" when only four passages exist. Leaving that in would
    show the user a source that does not exist — the one thing a citation must
    never do. Grouped citations ("[S1, S3]") are handled per label, so a valid
    one survives sharing brackets with an invalid one.

    Takes the exact labels rather than a count, because a comparison produces
    two independent sets ([A1..] and [B1..]) and a label from one side must not
    validate against the other. That is what stops a claim about subject A
    citing subject B's evidence and passing verification.

    Returns the cleaned answer and the sorted 1-based positions into
    valid_labels of the labels actually used.
    """
    valid = {label: i + 1 for i, label in enumerate(valid_labels)}
    used: set[int] = set()

    def rewrite(match: re.Match) -> str:
        kept = []
        for token in re.split(r"[,\s]+", match.group(1)):
            if token in valid:
                used.add(valid[token])
                kept.append(token)
        return f"[{', '.join(kept)}]" if kept else ""

    cleaned = _CITATION.sub(rewrite, answer)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    # Removing a citation can leave a space before its sentence's punctuation.
    cleaned = re.sub(r" ([.,;:])", r"\1", cleaned)
    return cleaned.strip(), sorted(used)


def extract_source_pages(context) -> list[int]:
    """Unique page numbers across every passage in the context, sorted."""
    return sorted({passage.page_number for passage in context.passages})
