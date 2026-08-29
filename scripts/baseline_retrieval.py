"""
STAGE 0 BASELINE — measure how the CURRENT retrieval system behaves.

This script changes nothing. It exists to record, in numbers, the problems we
are about to fix in Stage 1, so that after each change we can re-run it and see
whether the change actually helped.

It runs retrieval ONLY (no LLM, no API key needed), because the core problem is
a retrieval problem: the chunks needed to explain an answer are never fetched,
so no amount of prompt tuning can recover them.

What it reports:
  1. Every chunk the current chunker produces (ID, page, length, preview)
  2. Where the formula / variable definitions / explanation actually live
  3. For each test question: which chunks similarity search returns
  4. A verdict — were the explanation chunks retrieved, yes or no?

Run:  ./venv/bin/python scripts/baseline_retrieval.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chunker import chunk_pages  # noqa: E402
from document_loader import load_pdf  # noqa: E402
from embeddings import EmbeddingModel  # noqa: E402
from retriever import Retriever  # noqa: E402
from vector_store import VectorStore  # noqa: E402

# Current production settings — do not change these here. They are the baseline.
PDF_PATH = PROJECT_ROOT / "data" / "formula_sample.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

# Marker text we search for, to locate which chunk holds which piece of the
# compound interest section. These are substrings of the source document.
MARKERS = {
    "THE FORMULA": "A = P(1 + r/n)^(nt)",
    "THE VARIABLES": "P = the principal",
    "THE EXPLANATION": "How the formula works",
    "THE WORKED EXAMPLE": "Worked example",
}

TEST_QUERIES = [
    # 1. Direct formula question — the chunk with the formula should win.
    "What is the compound interest formula?",
    # 2. Explicit request for detail — does retrieval fetch MORE evidence?
    "Explain the compound interest formula in detail, including every variable.",
    # 3. Asking directly about the variables — can similarity find them at all?
    "What do the variables in the compound interest formula mean?",
    # 4. A bare follow-up, as a user would actually type it after Q1.
    #    This is the Stage 2 problem, measured here as a baseline.
    "Explain this in more detail.",
    # 5. Something genuinely absent from the document.
    "What is the company's parental leave policy?",
]


def find_marker_chunks(chunks: list[dict]) -> dict:
    """Locate which chunk IDs contain each marker string."""
    located = {}
    for label, needle in MARKERS.items():
        # clean_text collapses whitespace, so normalise the needle the same way.
        needle_norm = " ".join(needle.split())
        located[label] = [
            chunk["chunk_id"]
            for chunk in chunks
            if needle_norm in " ".join(chunk["text"].split())
        ]
    return located


def print_all_chunks(chunks: list[dict]) -> None:
    print("\n" + "=" * 72)
    print(f"ALL CHUNKS PRODUCED BY THE CURRENT CHUNKER  (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print("=" * 72)
    for chunk in chunks:
        preview = chunk["text"][:70].replace("\n", " ")
        print(
            f"  chunk {chunk['chunk_id']:>3} | page {chunk['page_number']:>2} "
            f"| {len(chunk['text']):>4} chars | {preview}..."
        )


def print_marker_map(located: dict) -> None:
    print("\n" + "=" * 72)
    print("WHERE THE ANSWER PIECES ACTUALLY LIVE")
    print("=" * 72)
    for label, ids in located.items():
        where = ", ".join(f"chunk {i}" for i in ids) if ids else "NOT FOUND"
        print(f"  {label:<20} → {where}")


def print_chunk_boundary_check(chunks: list[dict]) -> None:
    """Show whether chunks end mid-sentence — a chunking-quality signal."""
    print("\n" + "=" * 72)
    print("CHUNK BOUNDARY QUALITY (does the chunk end at a sentence end?)")
    print("=" * 72)
    clean_endings = 0
    for chunk in chunks:
        text = chunk["text"].rstrip()
        ends_clean = text.endswith((".", "!", "?", ":"))
        if ends_clean:
            clean_endings += 1
        flag = "clean " if ends_clean else "MID-SENTENCE"
        tail = text[-45:].replace("\n", " ")
        print(f"  chunk {chunk['chunk_id']:>3} | {flag:<12} | ...{tail}")
    total = len(chunks)
    print(
        f"\n  {clean_endings}/{total} chunks end at a sentence boundary "
        f"({clean_endings / total * 100:.0f}%)"
    )


def run_query(retriever: Retriever, question: str, located: dict) -> None:
    print("\n" + "=" * 72)
    print(f"QUERY: {question}")
    print("=" * 72)

    results = retriever.retrieve(question, top_k=TOP_K)
    retrieved_ids = [r["chunk_id"] for r in results]

    print(f"\n  RETRIEVED (top_k={TOP_K}): chunks {retrieved_ids}\n")
    for r in results:
        preview = r["text"][:100].replace("\n", " ")
        print(
            f"    rank {r['rank']} | chunk {r['chunk_id']:>3} | page {r['page_number']} "
            f"| similarity {r['similarity']:.4f}"
        )
        print(f"      {preview}...")

    # The verdict: for the compound-interest queries, did we get the pieces
    # needed to actually EXPLAIN the formula?
    print("\n  DID WE RETRIEVE THE PIECES NEEDED TO EXPLAIN THE ANSWER?")
    for label, ids in located.items():
        if not ids:
            continue
        hit = any(i in retrieved_ids for i in ids)
        mark = "YES" if hit else "NO "
        print(f"    {mark}  {label:<20} (lives in chunk {ids})")


def main() -> None:
    if not PDF_PATH.exists():
        raise SystemExit(
            f"Test document missing: {PDF_PATH}\n"
            "Create it first:  ./venv/bin/python scripts/create_formula_pdf.py"
        )

    print("\n" + "#" * 72)
    print("# STAGE 0 BASELINE — CURRENT SYSTEM, NO CHANGES")
    print("#" * 72)
    print(f"\nDocument: {PDF_PATH.name}")

    pages = load_pdf(PDF_PATH)
    print(f"Pages extracted: {len(pages)}")

    chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"Chunks created:  {len(chunks)}")

    print("\nMetadata fields currently present on each chunk:")
    print(f"  {sorted(chunks[0].keys())}")

    print_all_chunks(chunks)
    print_chunk_boundary_check(chunks)

    located = find_marker_chunks(chunks)
    print_marker_map(located)

    print("\nLoading embedding model (first run downloads weights)...")
    embedding_model = EmbeddingModel()
    print(f"  Embedding dimension: {embedding_model.dimension}")

    embeddings = embedding_model.embed_texts([c["text"] for c in chunks])
    store = VectorStore(dimension=embedding_model.dimension)
    store.add(embeddings, chunks)
    retriever = Retriever(embedding_model, store)
    print(f"  Vectors indexed: {store.index.ntotal}")

    print("\nMetadata fields returned by vector_store.search():")
    sample = store.search(embedding_model.embed_query("test"), top_k=1)[0]
    print(f"  {sorted(sample.keys())}")

    for question in TEST_QUERIES:
        run_query(retriever, question, located)

    print("\n" + "#" * 72)
    print("# END BASELINE")
    print("#" * 72 + "\n")


if __name__ == "__main__":
    main()
