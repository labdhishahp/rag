"""
STEP 3 TEST — neighbour/context expansion.

Retrieval only (no LLM, no API key, no cost). The question this answers:

    When similarity search finds ONE relevant chunk, does adjacency supply the
    neighbouring chunks that actually complete the answer?

For each test query we show:
  - which chunks similarity found            (the entry points)
  - which chunks adjacency added             (the expansion)
  - which chunks were dropped by the budget
  - whether the three pieces of the compound interest section are now present:
        the formula, the variable definitions, the worked example
  - how much duplicated overlap text was removed by merging
  - the final context size against the budget

Run:  ./.venv/bin/python scripts/test_expansion.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chunker import chunk_pages  # noqa: E402
from context_builder import (  # noqa: E402
    DEFAULT_CONTEXT_BUDGET_CHARS,
    DEFAULT_NEIGHBOUR_WINDOW,
    build_context,
)
from document_loader import load_pdf  # noqa: E402
from embeddings import EmbeddingModel  # noqa: E402
from pipeline import make_document_id  # noqa: E402
from retriever import Retriever  # noqa: E402
from vector_store import VectorStore  # noqa: E402

PDF_PATH = PROJECT_ROOT / "data" / "formula_sample.pdf"
TOP_K = 3

# The three pieces a complete answer about the formula needs.
PIECES = {
    "FORMULA": "A = P(1 + r/n)^(nt)",
    "VARIABLE DEFINITIONS": "r = the annual nominal interest rate",
    "WORKED EXAMPLE": "Suppose P = 1000, r = 0.05, n = 12",
    "HOW IT WORKS": "that interest itself begins earning interest",
}

QUERIES = [
    "What is the compound interest formula?",
    "What is the compound interest formula and explain it in detail?",
    "What do each of the variables in the compound interest formula mean?",
    "What is the parental leave policy?",
]


def build_retriever():
    file_bytes = PDF_PATH.read_bytes()
    pages = load_pdf(PDF_PATH)
    chunks = chunk_pages(
        pages,
        chunk_size=500,
        chunk_overlap=50,
        document_name=PDF_PATH.name,
        document_id=make_document_id(file_bytes),
    )
    model = EmbeddingModel()
    embeddings = model.embed_texts([c["text"] for c in chunks])
    store = VectorStore(dimension=model.dimension)
    store.add(embeddings, chunks)
    return Retriever(model, store), chunks


def locate_pieces(chunks) -> dict:
    """Which chunk holds each piece of the answer."""
    located = {}
    for label, needle in PIECES.items():
        norm = " ".join(needle.split())
        located[label] = [
            c["chunk_id"] for c in chunks if norm in " ".join(c["text"].split())
        ]
    return located


def main() -> None:
    print("\n" + "#" * 72)
    print("# STEP 3 TEST — NEIGHBOUR / CONTEXT EXPANSION")
    print("#" * 72)

    retriever, chunks = build_retriever()
    located = locate_pieces(chunks)

    print(f"\nIndexed {len(chunks)} chunks.")
    print(f"neighbour_window = {DEFAULT_NEIGHBOUR_WINDOW}"
          f"   budget = {DEFAULT_CONTEXT_BUDGET_CHARS} chars\n")

    print("Where the pieces of a complete answer live:")
    for label, ids in located.items():
        print(f"  {label:<22} -> chunk {ids}")

    all_ok = True

    for question in QUERIES:
        print("\n" + "=" * 72)
        print(f"QUERY: {question}")
        print("=" * 72)

        entry = retriever.retrieve(question, top_k=TOP_K)

        # Compare with and without expansion so the delta is unambiguous.
        without = build_context(
            entry, retriever.vector_store, neighbour_window=0, debug=False
        )
        with_expansion = build_context(
            entry, retriever.vector_store, debug=True
        )

        print("\n  --- WITHOUT expansion (Stage 0 behaviour) ---")
        print(f"    chunks: {sorted(without.entry_chunk_ids)}")
        print(f"    context: {without.total_chars} chars")

        print("\n  --- WITH expansion (Step 3) ---")
        covered = {
            cid
            for p in with_expansion.passages
            for cid in p.chunk_ids
        }
        print(f"    chunks: {sorted(covered)}")
        print(f"    context: {with_expansion.total_chars} chars")

        print("\n  --- DID WE GET THE PIECES? ---")
        for label, ids in located.items():
            if not ids:
                continue
            before = any(i in without.entry_chunk_ids for i in ids)
            after = any(i in covered for i in ids)
            change = ""
            if after and not before:
                change = "  <-- ADDED BY EXPANSION"
            elif before and not after:
                change = "  <-- LOST (regression!)"
                all_ok = False
            print(
                f"    {'YES' if after else 'NO ':<3} {label:<22}"
                f" (chunk {ids}){change}"
            )

        # Budget sanity.
        if with_expansion.total_chars > DEFAULT_CONTEXT_BUDGET_CHARS:
            print(f"\n    BUDGET EXCEEDED: {with_expansion.total_chars} chars")
            all_ok = False

    print("\n" + "#" * 72)
    print("# STEP 3 CHECKS")
    print("#" * 72)

    # Targeted assertion: the Stage 0 failure must be fixed.
    variable_chunks = located["VARIABLE DEFINITIONS"]
    entry = retriever.retrieve(
        "What do each of the variables in the compound interest formula mean?",
        top_k=TOP_K,
    )
    expanded = build_context(entry, retriever.vector_store, debug=False)
    covered = {cid for p in expanded.passages for cid in p.chunk_ids}
    got_variables = any(i in covered for i in variable_chunks)

    detail_entry = retriever.retrieve(
        "What is the compound interest formula and explain it in detail?",
        top_k=TOP_K,
    )
    detail_ctx = build_context(detail_entry, retriever.vector_store, debug=False)
    detail_covered = {cid for p in detail_ctx.passages for cid in p.chunk_ids}
    got_example = any(i in detail_covered for i in located["WORKED EXAMPLE"])

    # An unanswerable question must NOT cause the whole document to be pulled
    # in. Expanding around near-noise entry points just gives the model more
    # opportunity to invent something.
    absent_entry = retriever.retrieve("What is the parental leave policy?", top_k=TOP_K)
    absent_ctx = build_context(absent_entry, retriever.vector_store, debug=False)
    absent_covered = {cid for p in absent_ctx.passages for cid in p.chunk_ids}

    checks = [
        ("variable-definition query reaches the definitions chunk", got_variables),
        ("detailed formula query reaches the worked example", got_example),
        ("no piece was lost relative to no-expansion", all_ok),
        (
            "context stays within budget",
            detail_ctx.total_chars <= DEFAULT_CONTEXT_BUDGET_CHARS,
        ),
        (
            "absent-info query does NOT expand around noise",
            absent_ctx.expanded_chunk_ids == [],
        ),
        (
            "absent-info query stays small (< half the index)",
            len(absent_covered) < len(chunks) / 2,
        ),
    ]
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    passed = sum(1 for _, ok in checks if ok)
    print(f"\n  STEP 3 RESULT: {passed}/{len(checks)} checks passed\n")
    if passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
