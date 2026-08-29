"""
Vector store, retrieval and context expansion on the synthetic document.

Pins the Step 3 behaviour that fixed the Stage 0 false negative: the chunk
holding the variable definitions must reach the context, the worked example
must reach the context for a detailed question, and an unanswerable question
must NOT cause expansion around noise.
"""

import pytest

from context_builder import DEFAULT_CONTEXT_BUDGET_CHARS, build_context

pytestmark = pytest.mark.embedding


def _covered(ctx):
    return {cid for p in ctx.passages for cid in p.chunk_ids}


def _holder(chunks, needle):
    n = " ".join(needle.split())
    return [c["chunk_id"] for c in chunks if n in " ".join(c["text"].split())]


def test_metadata_survives_search(formula_retriever, formula_chunks):
    top = formula_retriever.retrieve("What is the compound interest formula?", top_k=3)[0]
    original = next(c for c in formula_chunks if c["chunk_id"] == top["chunk_id"])
    for field in original:
        assert top[field] == original[field]
    assert "similarity" in top and "rank" in top


def test_search_returns_copies(formula_retriever, formula_store):
    result = formula_retriever.retrieve("compound interest", top_k=1)[0]
    result["similarity"] = 999.0
    stored = formula_store.get_chunk(result["chunk_id"], result["document_id"])
    assert "similarity" not in stored


def test_get_chunk_by_id(formula_store, formula_chunks):
    c = formula_chunks[2]
    assert formula_store.get_chunk(c["chunk_id"], c["document_id"])["text"] == c["text"]
    assert formula_store.get_chunk(999, c["document_id"]) is None


def test_variable_definitions_reach_context(formula_retriever, formula_store, formula_chunks):
    gold = _holder(formula_chunks, "r = the annual nominal interest rate")
    entries = formula_retriever.retrieve(
        "What do each of the variables in the compound interest formula mean?", top_k=3
    )
    ctx = build_context(entries, formula_store, debug=False)
    assert any(g in _covered(ctx) for g in gold)


def test_worked_example_reaches_context_for_detailed_question(
    formula_retriever, formula_store, formula_chunks
):
    gold = _holder(formula_chunks, "Suppose P = 1000, r = 0.05, n = 12")
    entries = formula_retriever.retrieve(
        "What is the compound interest formula and explain it in detail?", top_k=3
    )
    ctx = build_context(entries, formula_store, debug=False)
    assert any(g in _covered(ctx) for g in gold)
    assert ctx.total_chars <= DEFAULT_CONTEXT_BUDGET_CHARS


def test_merged_passages_are_contiguous_and_overlap_removed(formula_retriever, formula_store):
    entries = formula_retriever.retrieve("What is the compound interest formula?", top_k=3)
    ctx = build_context(entries, formula_store, debug=False)
    merged = [p for p in ctx.passages if len(p.chunk_ids) > 1]
    assert merged, "expected at least one merged passage"
    for p in merged:
        assert p.chunk_ids == list(range(p.chunk_ids[0], p.chunk_ids[-1] + 1))
    assert ctx.duplicate_chars_removed > 0


def test_absent_information_does_not_expand(formula_retriever, formula_store, formula_chunks):
    entries = formula_retriever.retrieve("What is the parental leave policy?", top_k=3)
    from config import SIMILARITY_HARD_FLOOR
    assert entries[0]["similarity"] < SIMILARITY_HARD_FLOOR
    ctx = build_context(entries, formula_store, debug=False)
    assert ctx.expanded_chunk_ids == []
    assert len(_covered(ctx)) < len(formula_chunks) / 2
