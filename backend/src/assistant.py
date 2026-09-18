"""
Knowledge Assistant — deterministic orchestration above the RAG core.

    message + conversation
          │
          ▼
    task_router.route            what kind of task, which documents
          │
          ├── answer     ──▶ RAGSystem.answer(message, conversation, document_ids)
          ├── compare    ──▶ two scoped retrievals → [A#]/[B#] evidence → one LLM call
          ├── summarize  ──▶ walk the document in reading order → one LLM call
          └── clarify    ──▶ ask the user; no retrieval, no LLM call

Everything here is a FIXED plan chosen by pattern matching. There is no loop
and no model deciding what to do next — that is deliberately left to Phase 6,
which will sit above this class and reuse its pieces as tools. Keeping the
plans deterministic here means Phase 6 has a baseline to beat.

Every result has the same shape as RAGSystem.answer's, so the UI, the
conversation recorder and the evaluators treat all task kinds alike.
"""

from __future__ import annotations

from context_builder import (
    ContextResult,
    build_context,
    citation_for,
    format_passages,
    labels_for,
)
from conversation import Conversation
from knowledge_base import DocumentInfo, KnowledgeBase
from prompt_builder import REFUSAL_TEXT, build_compare_prompt, build_summary_prompt
from query_understanding import understand
from rag import RAGSystem, check_citations
from task_router import Task, route

COMPARE_SIDE_BUDGET_CHARS = 3000
SUMMARY_BUDGET_CHARS = 6500
SUMMARY_SNIPPET_CHARS = 420


class KnowledgeAssistant:
    def __init__(self, kb: KnowledgeBase, llm, top_k: int | None = None, debug: bool = True):
        self.kb = kb
        self.llm = llm
        self.debug = debug
        self.rag = RAGSystem(retriever=kb.retriever, llm=llm, top_k=top_k, debug=debug)

    # ------------------------------------------------------------------
    def answer(self, message: str, conversation: Conversation | None = None) -> dict:
        message = message.strip()
        if not message:
            raise ValueError("Question cannot be empty.")
        if self.kb.is_empty:
            return self._plain_result(message, "No document is loaded yet. Upload one first.",
                                      kind="clarify", clarification=True)

        task = route(message, self.kb, conversation)
        if self.debug:
            print(f"\n=== TASK === {task.describe()}   ({'; '.join(task.reasons)})")

        if task.kind == "clarify":
            return self._plain_result(message, task.clarification, kind="clarify", clarification=True)
        if task.kind == "compare":
            return self._compare(task, conversation)
        if task.kind == "summarize":
            return self._summarize(task)
        result = self.rag.answer(message, conversation, document_ids=task.document_ids)
        result["task"] = "answer"
        result["task_reasons"] = task.reasons
        return result

    # ------------------------------------------------------------------
    def _compare(self, task: Task, conversation: Conversation | None) -> dict:
        """
        Two subjects, two retrievals, one answer.

        Each side gets its own similarity search (scoped to its own document
        when one is named), its own section-scoped expansion and its own
        budget, then the two evidence sets are labelled [A#] and [B#] so the
        model — and the citation checker — can tell them apart.
        """
        understanding = understand(task.message)
        sides = []
        for label, part, doc_ids in zip(("A", "B"), task.parts, task.part_document_ids):
            hits = self.kb.retriever.retrieve(part, top_k=3, document_ids=doc_ids or task.document_ids)
            ctx = build_context(hits, self.kb.vector_store, neighbour_window=1,
                                budget_chars=COMPARE_SIDE_BUDGET_CHARS, debug=False)
            sides.append({"label": label, "subject": part, "context": ctx, "hits": hits})
            if self.debug:
                print(f"\n=== COMPARE SIDE {label}: {part!r} === entries {ctx.entry_chunk_ids} "
                      f"expanded {ctx.expanded_chunk_ids} evidence={ctx.evidence_level}")

        if all(s["context"].evidence_level == "none" for s in sides):
            combined = _combine(sides)
            return self._compose(task.message, REFUSAL_TEXT, combined, [], llm_called=False,
                                 kind="compare", understanding=understanding, refused=True)

        blocks = []
        missing = []
        for s in sides:
            if s["context"].evidence_level == "none":
                missing.append(s["subject"])
                blocks.append(f"=== {s['label']}: {s['subject']} ===\n(no relevant evidence found)")
            else:
                blocks.append(f"=== {s['label']}: {s['subject']} ===\n"
                              + format_passages(s["context"].passages, label_prefix=s["label"]))
        evidence = "\n\n".join(blocks)
        prompt = build_compare_prompt(task.message, sides[0]["subject"], sides[1]["subject"],
                                      evidence, missing=missing, depth=understanding.depth,
                                      conversation=conversation.format_for_prompt() if conversation and not conversation.is_empty else None)
        answer = self.llm.generate(prompt)

        valid = []
        for s in sides:
            valid += labels_for(s["context"].passages, label_prefix=s["label"])
        answer, cited = check_citations(answer, valid)
        combined = _combine(sides)
        return self._compose(task.message, answer, combined, cited, llm_called=True,
                             kind="compare", understanding=understanding, valid_labels=valid)

    # ------------------------------------------------------------------
    def _summarize(self, task: Task) -> dict:
        """
        Summaries are not retrieval. Walk the document in reading order,
        section by section, taking the opening of each section until the
        budget is spent. The model then summarises a faithful skeleton of the
        whole document rather than the three chunks most similar to the word
        "summary".
        """
        doc: DocumentInfo = task.target_document
        chunks = self.kb.chunks_for(doc.document_id)
        understanding = understand(task.message)

        # First chunk of every section (in order), then fill remaining budget
        # with further chunks of the longest sections.
        picked: list[dict] = []
        seen_sections: set = set()
        for c in chunks:
            key = c["section"] or f"__page{c['page_number']}"
            if key not in seen_sections:
                seen_sections.add(key)
                picked.append(c)
        used = sum(min(len(c["text"]), SUMMARY_SNIPPET_CHARS) for c in picked)
        for c in chunks:
            if used >= SUMMARY_BUDGET_CHARS:
                break
            if c not in picked:
                picked.append(c)
                used += min(len(c["text"]), SUMMARY_SNIPPET_CHARS)
        picked.sort(key=lambda c: c["chunk_id"])

        # Build passages through the normal machinery so citations and the UI work.
        entries = [dict(c, similarity=1.0, rank=i + 1) for i, c in enumerate(picked)]
        ctx = build_context(entries, self.kb.vector_store, neighbour_window=0,
                            budget_chars=SUMMARY_BUDGET_CHARS + 2000, debug=False)
        # Trim each passage to its opening so the skeleton stays within budget.
        for p in ctx.passages:
            if len(p.text) > SUMMARY_SNIPPET_CHARS:
                p.text = p.text[:SUMMARY_SNIPPET_CHARS].rsplit(" ", 1)[0] + " …"
        ctx.formatted = format_passages(ctx.passages)
        ctx.total_chars = sum(len(p.text) for p in ctx.passages)
        ctx.evidence_level = "ok"
        if self.debug:
            print(f"\n=== SUMMARIZE {doc.name} === {len(chunks)} chunks -> {len(ctx.passages)} passages, "
                  f"{ctx.total_chars} chars, sections: {doc.sections[:8]}{'…' if len(doc.sections) > 8 else ''}")

        prompt = build_summary_prompt(task.message, doc.name, doc.sections, ctx.formatted,
                                      depth=understanding.depth)
        answer = self.llm.generate(prompt)
        answer, cited = check_citations(answer, labels_for(ctx.passages))
        return self._compose(task.message, answer, ctx, cited, llm_called=True,
                             kind="summarize", understanding=understanding)

    # ------------------------------------------------------------------
    def _compose(self, message, answer, context: ContextResult, cited, llm_called, kind,
                 understanding, refused=False, valid_labels=None) -> dict:
        source_citations = [citation_for(p) for p in context.passages]
        return {
            "question": message,
            "retrieval_query": message,
            "was_follow_up": False,
            "answer": answer,
            "depth": understanding.depth,
            "understanding": understanding,
            "task": kind,
            "sources": sorted({p.page_number for p in context.passages}),
            "source_citations": source_citations,
            "cited_sources": [source_citations[i - 1] for i in cited if 0 < i <= len(source_citations)],
            "cited_labels": cited,
            "citation_labels": valid_labels or labels_for(context.passages),
            "chunks": [],
            "best_similarity": context.best_similarity,
            "low_confidence": context.evidence_level == "weak",
            "top_k": 0,
            "llm_model": getattr(self.llm, "active_model", None),
            "embedding_dimension": self.kb.embedding_model.dimension,
            "num_retrieved_chunks": len(context.entry_chunk_ids),
            "context": context,
            "entry_chunk_ids": context.entry_chunk_ids,
            "expanded_chunk_ids": context.expanded_chunk_ids,
            "dropped_chunk_ids": context.dropped_chunk_ids,
            "context_chars": context.total_chars,
            "duplicate_chars_removed": context.duplicate_chars_removed,
            "evidence_level": context.evidence_level,
            "llm_called": llm_called,
            "refused": refused or answer.strip().startswith(REFUSAL_TEXT),
            "clarification": False,
        }

    def _plain_result(self, message, text, kind, clarification=False) -> dict:
        ctx = ContextResult()
        understanding = understand(message)
        r = self._compose(message, text, ctx, [], llm_called=False, kind=kind, understanding=understanding)
        r["clarification"] = clarification
        r["refused"] = False
        return r


def _combine(sides) -> ContextResult:
    """One ContextResult spanning both comparison sides, for the UI and recorder."""
    passages = []
    for s in sides:
        passages += s["context"].passages
    combined = ContextResult(
        passages=passages,
        formatted="\n\n".join(
            f"=== {s['label']}: {s['subject']} ===\n" + format_passages(s["context"].passages, s["label"])
            for s in sides
        ),
        entry_chunk_ids=[cid for s in sides for cid in s["context"].entry_chunk_ids],
        expanded_chunk_ids=[cid for s in sides for cid in s["context"].expanded_chunk_ids],
        dropped_chunk_ids=[cid for s in sides for cid in s["context"].dropped_chunk_ids],
        total_chars=sum(s["context"].total_chars for s in sides),
        duplicate_chars_removed=sum(s["context"].duplicate_chars_removed for s in sides),
        best_similarity=max((s["context"].best_similarity for s in sides), default=0.0),
    )
    levels = [s["context"].evidence_level for s in sides]
    combined.evidence_level = "none" if all(l == "none" for l in levels) else ("weak" if "none" in levels or "weak" in levels else "ok")
    return combined
