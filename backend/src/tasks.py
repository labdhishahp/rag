"""
The two task plans that need a different SHAPE of evidence than a question does.

Both reuse the existing machinery rather than reimplementing it: the same
Passage/ContextResult types, the same format_passages and citation_for, the
same check_citations verification, and the same build_result shape. What differs
is only HOW the evidence is chosen — one search, a walk, or two searches.

    rag.answer      one search  → expand → gate → answer
    summarize       no search   → walk the document in reading order
    compare         two searches → two labelled evidence sets → one answer
"""

from __future__ import annotations

from config import SIMILARITY_HARD_FLOOR, SIMILARITY_SOFT_FLOOR
from context_builder import (
    ContextResult,
    Passage,
    build_context,
    format_passages,
    labels_for,
)
from prompt_builder import REFUSAL_TEXT, build_compare_prompt, build_summary_prompt
from query_understanding import understand
from rag import build_result, check_citations

# --------------------------------------------------------------------------
# Summarize
# --------------------------------------------------------------------------

# Ceiling on the skeleton sent to the model. Matches the "detailed" answer
# budget: a summary is the one request that legitimately wants the most context.
SUMMARY_BUDGET_CHARS = 6500

# Below this, a section's opening is too short to carry meaning — roughly one
# clause. When the budget divided by the section count falls under it, we stop
# shrinking and start SAMPLING sections instead, and say so in the answer.
#
# Measured section counts that motivate the number:
#     formula_sample.pdf    5 sections -> 1300 chars each
#     storage.py            6          -> 1083
#     sample.pdf           11          ->  590
#     rag_survey.pdf       36          ->  180
#     nist_sp800-207.pdf   79          ->   82   <- under the floor, samples
SUMMARY_MIN_SNIPPET_CHARS = 150

# Upper bound on one passage, so a short document does not send one enormous
# block that buries the rest.
SUMMARY_MAX_SNIPPET_CHARS = 1400


def _section_groups(chunks: list[dict]) -> list[list[dict]]:
    """
    Chunks grouped into the units a summary should cover, in reading order.

    With headings, a group is a run of chunks under one section. Without any
    (a plain .txt), every chunk is its own group — grouping by page instead
    would put a ten-chunk single-page file into ONE group and summarise a
    tenth of it.
    """
    if not any(c.get("section") for c in chunks):
        return [[c] for c in chunks]

    groups: list[list[dict]] = []
    current_key = object()
    for chunk in chunks:
        key = chunk.get("section") or f"__{chunk.get('page_label', 'page')}{chunk['page_number']}"
        if key != current_key:
            groups.append([])
            current_key = key
        groups[-1].append(chunk)
    return groups


def _trim(text: str, limit: int) -> str:
    """The opening of a chunk, cut on a word boundary."""
    if len(text) <= limit:
        return text
    head = text[:limit].rsplit(" ", 1)[0]
    return (head or text[:limit]) + " …"


def summarize(retriever, llm, question: str, *, embedding_dimension=None,
              debug: bool = False) -> dict:
    """
    Summarise the session's document by WALKING it, not searching it.

    Similarity cannot retrieve "all of it": there is no query vector whose
    nearest neighbours are a representative sample of a whole document. So the
    evidence is built by walking reading order and taking the opening of every
    section — the place authors say what a section is about.

    Two properties this guarantees, both of which a naive implementation loses:

      EVERY SECTION IS REPRESENTED. The snippet length is derived from the
      section count rather than fixed, so the budget is spent evenly instead of
      being consumed by the first N sections. A long document does not silently
      lose its later half.

      TRIM BEFORE BUDGET. Passages are trimmed first and the budget measured on
      the trimmed text. Budgeting on full chunk lengths and trimming afterwards
      drops entries that would in fact have fitted.
    """
    store = retriever.vector_store
    chunks = sorted(store.chunks, key=lambda c: c["chunk_id"])
    if not chunks:
        raise ValueError("This document has no indexed chunks to summarise.")

    understanding = understand(question)
    groups = _section_groups(chunks)

    # Outline of the WHOLE document, even when the evidence samples it. The
    # model is told the real shape so it cannot invent a section.
    outline: list[str] = []
    for chunk in chunks:
        section = chunk.get("section")
        if section and (not outline or outline[-1] != section):
            outline.append(section)

    # Size the snippet to fit every group; sample only when that would make the
    # snippets meaningless.
    total_groups = len(groups)
    snippet = SUMMARY_BUDGET_CHARS // max(1, total_groups)
    if snippet < SUMMARY_MIN_SNIPPET_CHARS:
        keep = max(1, SUMMARY_BUDGET_CHARS // SUMMARY_MIN_SNIPPET_CHARS)
        if keep >= total_groups:
            indexes = list(range(total_groups))
        elif keep == 1:
            indexes = [0]
        else:
            # Spread across the FULL span, endpoints included. `int(i * n/keep)`
            # looks equivalent and is not: its largest index falls short of the
            # last group, so the final section of a long document is silently
            # dropped — the exact failure this sampling exists to avoid.
            # Dividing by (keep - 1) pins the first and last groups.
            step = (total_groups - 1) / (keep - 1)
            indexes = sorted({round(i * step) for i in range(keep)})
        groups = [groups[i] for i in indexes]
        snippet = SUMMARY_BUDGET_CHARS // len(groups)
    snippet = min(snippet, SUMMARY_MAX_SNIPPET_CHARS)
    coverage = (len(groups), total_groups)

    # One opening per group, then fill any leftover budget round-robin so extra
    # depth is spread across the document rather than poured into section one.
    picked = [group[0] for group in groups]
    used = sum(len(_trim(c["text"], snippet)) for c in picked)
    depth_index = 1
    while used < SUMMARY_BUDGET_CHARS:
        added = False
        for group in groups:
            if depth_index >= len(group):
                continue
            chunk = group[depth_index]
            cost = len(_trim(chunk["text"], snippet))
            if used + cost > SUMMARY_BUDGET_CHARS:
                continue
            picked.append(chunk)
            used += cost
            added = True
        if not added:
            break
        depth_index += 1

    picked.sort(key=lambda c: c["chunk_id"])

    # One passage per chunk: for a summary that is exactly the citation
    # granularity wanted — [S3] means "this point came from that section".
    passages = [
        Passage(
            text=_trim(chunk["text"], snippet),
            document_name=chunk.get("document_name"),
            document_id=chunk.get("document_id"),
            page_number=chunk["page_number"],
            page_label=chunk.get("page_label", "page"),
            section=chunk.get("section"),
            chunk_ids=[chunk["chunk_id"]],
            entry_chunk_ids=[chunk["chunk_id"]],
            best_similarity=0.0,
            char_start=chunk["char_start"],
            char_end=chunk["char_end"],
        )
        for chunk in picked
    ]

    context = ContextResult(
        passages=passages,
        formatted=format_passages(passages),
        entry_chunk_ids=[c["chunk_id"] for c in picked],
        total_chars=sum(len(p.text) for p in passages),
        # The evidence gate is bypassed DELIBERATELY. It compares a similarity
        # score against a measured floor, and there is no query here to be
        # similar to — the document IS the evidence. Declining a summary
        # because "similarity was too low" would be nonsense.
        evidence_level="ok",
    )

    document_name = picked[0].get("document_name") or "this document"
    if debug:
        print(f"\n=== SUMMARIZE {document_name} === {len(chunks)} chunks, "
              f"{total_groups} sections -> {len(passages)} passages, "
              f"{context.total_chars} chars, snippet={snippet}, coverage={coverage}")

    answer = llm.generate(
        build_summary_prompt(question, document_name, outline, context.formatted,
                             depth=understanding.depth, coverage=coverage)
    )
    answer, cited = check_citations(answer, labels_for(passages))

    return build_result(
        question=question, understanding=understanding, answer=answer,
        context=context, llm=llm, task="summarize", cited=cited,
        embedding_dimension=embedding_dimension,
    )


# --------------------------------------------------------------------------
# Compare
# --------------------------------------------------------------------------

COMPARE_TOP_K = 3
# Half a normal answer's budget per side, so a two-sided answer costs about the
# same as a one-sided one and neither subject can crowd the other out.
COMPARE_SIDE_BUDGET_CHARS = 3000


def compare(retriever, llm, question: str, subject_a: str, subject_b: str, *,
            embedding_dimension=None, debug: bool = False) -> dict:
    """
    Compare two subjects found in one document.

    One search for "simple vs compound interest" returns whichever subject the
    document discusses more, and the answer comes out one-sided. So each subject
    gets its OWN embedding, its own ranking, its own section-scoped expansion
    and its own evidence budget.

    The two evidence sets are labelled [A#] and [B#] and verified against their
    own label lists, which is what makes "facts about A must cite A" a checkable
    property rather than a request in the prompt.
    """
    provider = retriever.embedding_model
    hard_floor = getattr(provider, "hard_floor", SIMILARITY_HARD_FLOOR)
    soft_floor = getattr(provider, "soft_floor", SIMILARITY_SOFT_FLOOR)
    understanding = understand(question)

    sides = []
    for label, subject in (("A", subject_a), ("B", subject_b)):
        hits = retriever.retrieve(subject, top_k=COMPARE_TOP_K)
        context = build_context(
            hits, retriever.vector_store,
            neighbour_window=1,
            budget_chars=COMPARE_SIDE_BUDGET_CHARS,
            hard_floor=hard_floor, soft_floor=soft_floor,
            debug=False,
        )
        sides.append({"label": label, "subject": subject, "context": context, "hits": hits})
        if debug:
            print(f"\n=== COMPARE {label}: {subject!r} === entries {context.entry_chunk_ids} "
                  f"expanded {context.expanded_chunk_ids} evidence={context.evidence_level}")

    combined = _combine(sides)
    valid_labels = [
        label
        for side in sides
        for label in labels_for(side["context"].passages, side["label"])
    ]

    # The evidence gate, applied PER SIDE. Both sides empty is the same
    # situation the answer path declines on, so it declines here too — without
    # an LLM call.
    if all(side["context"].evidence_level == "none" for side in sides):
        refused = build_result(
            question=question, understanding=understanding, answer=REFUSAL_TEXT,
            context=combined, llm=llm, task="compare", cited=[], llm_called=False,
            embedding_dimension=embedding_dimension,
        )
        refused["compare_sides"] = _side_report(sides)
        return refused

    missing = [s["subject"] for s in sides if s["context"].evidence_level == "none"]
    blocks = []
    for side in sides:
        header = f"=== {side['label']}: {side['subject']} ==="
        if side["context"].evidence_level == "none":
            blocks.append(f"{header}\n(no relevant evidence found)")
        else:
            blocks.append(
                f"{header}\n"
                + format_passages(side["context"].passages, label_prefix=side["label"])
            )

    answer = llm.generate(
        build_compare_prompt(
            question, subject_a, subject_b, "\n\n".join(blocks),
            missing=missing, depth=understanding.depth,
        )
    )
    answer, cited = check_citations(answer, valid_labels)

    result = build_result(
        question=question, understanding=understanding, answer=answer,
        context=combined, llm=llm, task="compare", cited=cited,
        best_similarity=combined.best_similarity,
        low_confidence=combined.evidence_level == "weak",
        embedding_dimension=embedding_dimension,
    )
    result["compare_sides"] = _side_report(sides)
    return result


def _side_report(sides) -> list[dict]:
    """
    Per-side evidence status.

    _combine folds both sides into one ContextResult for the UI, which loses
    the fact that ONE side found nothing — and "one subject is absent from this
    document" is exactly what a comparison most needs to report honestly. A
    below-floor side still has passages, so the combined block cannot be
    inspected for it after the fact.
    """
    return [
        {
            "label": side["label"],
            "subject": side["subject"],
            "evidence_level": side["context"].evidence_level,
            "passages": len(side["context"].passages),
        }
        for side in sides
    ]


def _combine(sides) -> ContextResult:
    """
    One ContextResult spanning both sides, so the UI and the conversation
    recorder treat a comparison like any other answer.

    Passage order is A's then B's, matching the order the labels were generated
    in — that is what makes build_result's cited_sources lookup correct.
    """
    passages = [p for side in sides for p in side["context"].passages]
    combined = ContextResult(
        passages=passages,
        formatted="\n\n".join(
            f"=== {s['label']}: {s['subject']} ===\n"
            + format_passages(s["context"].passages, s["label"])
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
    combined.evidence_level = (
        "none" if all(level == "none" for level in levels)
        else "weak" if ("none" in levels or "weak" in levels)
        else "ok"
    )
    return combined
