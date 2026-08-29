"""
STAGE 0 BASELINE (part 2) — measure the CURRENT system's end-to-end answers.

scripts/baseline_retrieval.py measures retrieval only. This script runs the full
current pipeline (retrieve -> prompt -> Gemini -> answer) so we can record the
answers we get TODAY, before any Stage 1 change.

It changes no project code. It calls the same RAGSystem the Streamlit app calls.

The key measurement is a PAIR of questions:
    A. "What is the compound interest formula?"          -> short answer expected
    B. "Give me the equation AND explain it in detail."  -> should be much longer

If B is not meaningfully longer or richer than A, we have reproduced the exact
bug described in the brief: asking for detail does not produce detail.

Requires a real GEMINI_API_KEY in .env. Costs a few Gemini calls.

Run:  ./.venv/bin/python scripts/baseline_answers.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chunker import chunk_pages  # noqa: E402
from document_loader import load_pdf  # noqa: E402
from embeddings import EmbeddingModel  # noqa: E402
from llm import LLMError, create_llm  # noqa: E402
from rag import RAGSystem  # noqa: E402
from retriever import Retriever  # noqa: E402
from vector_store import VectorStore  # noqa: E402

# Current production settings — this is the baseline, do not tune them here.
PDF_PATH = PROJECT_ROOT / "data" / "formula_sample.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

QUESTIONS = [
    ("A", "What is the compound interest formula?"),
    ("B", "Give me the compound interest equation and explain it in detail."),
    ("C", "What do each of the variables in the compound interest formula mean?"),
    ("D", "What is the company's parental leave policy?"),  # absent from document
]


def build_system() -> RAGSystem:
    pages = load_pdf(PDF_PATH)
    chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    embedding_model = EmbeddingModel()
    embeddings = embedding_model.embed_texts([c["text"] for c in chunks])

    store = VectorStore(dimension=embedding_model.dimension)
    store.add(embeddings, chunks)

    print(f"Indexed {store.index.ntotal} chunks from {len(pages)} pages.")

    return RAGSystem(
        retriever=Retriever(embedding_model, store),
        llm=create_llm("gemini"),
        top_k=TOP_K,
    )


def main() -> None:
    if not PDF_PATH.exists():
        raise SystemExit(
            f"Test document missing: {PDF_PATH}\n"
            "Create it first:  ./.venv/bin/python scripts/create_formula_pdf.py"
        )

    print("\n" + "#" * 72)
    print("# STAGE 0 BASELINE (part 2) — END-TO-END ANSWERS, CURRENT SYSTEM")
    print("#" * 72 + "\n")

    rag = build_system()
    lengths: dict[str, int] = {}

    for tag, question in QUESTIONS:
        print("\n" + "=" * 72)
        print(f"[{tag}] {question}")
        print("=" * 72)

        try:
            result = rag.answer(question)
        except LLMError as exc:
            print(f"  LLM ERROR: {exc}")
            continue

        retrieved = [c["chunk_id"] for c in result["chunks"]]
        context_chars = sum(len(c["text"]) for c in result["chunks"])

        print(f"\n  retrieved chunks   : {retrieved}")
        print(f"  source pages       : {result['sources']}")
        print(f"  best similarity    : {result['best_similarity']:.4f}")
        print(f"  low_confidence     : {result['low_confidence']}")
        print(f"  context sent to LLM: {context_chars} chars")

        answer = result["answer"]
        lengths[tag] = len(answer)
        print(f"  answer length      : {len(answer)} chars")
        print("\n  --- ANSWER ---")
        for line in answer.splitlines():
            print(f"  {line}")

    # The headline measurement.
    print("\n" + "#" * 72)
    print("# THE KEY COMPARISON")
    print("#" * 72)
    if "A" in lengths and "B" in lengths:
        a, b = lengths["A"], lengths["B"]
        ratio = b / a if a else 0
        print(f"\n  [A] plain question      : {a:>5} chars")
        print(f"  [B] 'explain in detail' : {b:>5} chars")
        print(f"  ratio B/A               : {ratio:.2f}x")
        print(
            "\n  If the ratio is near 1.0, asking for detail produced no extra detail —\n"
            "  which is the Stage 1 problem we are trying to fix."
        )
    print()


if __name__ == "__main__":
    main()
