"""
STAGE 0 vs STEPS 1-3 — end-to-end, side by side, on the same four questions.

Both pipelines run live against the same document, the same embedding model,
the same top_k, and the SAME PROMPT TEMPLATE. The only differences are the ones
Steps 1-3 introduced:

    OLD                                 NEW
    ---------------------------------   -----------------------------------------
    character-window chunking           sentence/block packing        (Step 2)
    whitespace-flattened text           structure-preserving text     (Step 2)
    3 metadata fields                   13 metadata fields            (Step 1)
    top-k chunks, score order           + neighbour expansion         (Step 3)
    no overlap removal                  merged, overlap deleted       (Step 3)
    no budget                           5000-char budget              (Step 3)
    page numbers only                   document / page / section     (Step 1+2)

Holding the prompt fixed is deliberate. If the prompt changed too, we could not
tell whether a better answer came from better evidence or better instructions.
This isolates the retrieval and context work.

Costs 8 Gemini calls. Requires a real GEMINI_API_KEY in .env.

Run:  ./.venv/bin/python scripts/compare_stage0.py
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunker import chunk_pages  # noqa: E402
from context_builder import build_context  # noqa: E402
from document_loader import load_pdf  # noqa: E402
from embeddings import EmbeddingModel  # noqa: E402
from legacy_stage0 import old_chunk_pages, old_format_context  # noqa: E402
from llm import LLMError, create_llm  # noqa: E402
from pipeline import make_document_id  # noqa: E402
from prompt_builder import build_rag_prompt  # noqa: E402
from rag import DEFAULT_SIMILARITY_THRESHOLD  # noqa: E402
from retriever import Retriever  # noqa: E402
from vector_store import VectorStore  # noqa: E402

PDF_PATH = PROJECT_ROOT / "data" / "formula_sample.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

QUESTIONS = [
    ("1", "What is the compound interest formula?"),
    ("2", "What is the compound interest formula and explain it in detail?"),
    ("3", "What do each of the variables in the compound interest formula mean?"),
    ("4", "What is the parental leave policy?"),
]

# Facts that a complete answer to questions 1-3 should be able to state.
# Presence is checked in the ANSWER TEXT, not the context — we care whether the
# information reached the user, not merely whether it reached the model.
ANSWER_FACTS = {
    "formula": ["P(1 + r/n)", "P(1+r/n)"],
    "all 5 variables": None,  # handled specially below
    "compounding mechanism": ["earning interest", "earns interest", "exponential"],
    "worked example": ["1647", "647.01", "0.05/12"],
}
VARIABLE_TOKENS = ["principal", "interest rate", "compounded", "time", "final amount"]


# The Gemini free tier allows 5 generate_content requests per minute per model.
# This script makes 8 (two pipelines x four questions), so it must pace itself
# or it will 429 partway through and produce an incomplete comparison.
_MIN_SECONDS_BETWEEN_CALLS = 13.0
_last_call_at = [0.0]


def paced_generate(llm, prompt: str, attempts: int = 4) -> str:
    """Call the LLM, respecting the free-tier rate limit, retrying on 429."""
    for attempt in range(attempts):
        wait = _MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - _last_call_at[0])
        if wait > 0:
            time.sleep(wait)
        _last_call_at[0] = time.monotonic()
        try:
            return llm.generate(prompt)
        except LLMError as exc:
            message = str(exc).lower()
            # Both are transient: 429 (quota) and 503 (service unavailable).
            transient = "rate limit" in message or "unavailable" in message
            if not transient or attempt == attempts - 1:
                raise
            backoff = _MIN_SECONDS_BETWEEN_CALLS * (attempt + 2)
            print(f"    (transient API error; waiting {backoff:.0f}s and retrying)")
            time.sleep(backoff)
    raise LLMError("Exhausted retries against the Gemini rate limit.")


def build_old(pages, model):
    # The Stage 0 pipeline also used the old layout-blind text extraction.
    from document_loader import load_pdf_raw_text

    raw_pages = load_pdf_raw_text(PDF_PATH.read_bytes())
    chunks = old_chunk_pages(raw_pages, CHUNK_SIZE, CHUNK_OVERLAP)
    embeddings = model.embed_texts([c["text"] for c in chunks])
    store = VectorStore(dimension=model.dimension)
    store.add(embeddings, chunks)
    return Retriever(model, store), chunks


def build_new(pages, model, file_bytes):
    chunks = chunk_pages(
        pages,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        document_name=PDF_PATH.name,
        document_id=make_document_id(file_bytes),
    )
    embeddings = model.embed_texts([c["text"] for c in chunks])
    store = VectorStore(dimension=model.dimension)
    store.add(embeddings, chunks)
    return Retriever(model, store), chunks


def facts_present(answer: str) -> dict:
    low = answer.lower()
    found = {}
    for label, needles in ANSWER_FACTS.items():
        if label == "all 5 variables":
            found[label] = all(t in low for t in VARIABLE_TOKENS)
        else:
            found[label] = any(n.lower() in low for n in needles)
    return found


def run_old(retriever, llm, question):
    chunks = retriever.retrieve(question, top_k=TOP_K)
    best = chunks[0]["similarity"] if chunks else 0.0
    low_conf = best < DEFAULT_SIMILARITY_THRESHOLD
    context = old_format_context(chunks)
    prompt = build_rag_prompt(question, context, low_confidence=low_conf)
    answer = paced_generate(llm, prompt)
    return {
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "expanded": [],
        "context_chars": len(context),
        "best_similarity": best,
        "low_confidence": low_conf,
        "answer": answer,
        "sources": sorted({c["page_number"] for c in chunks}),
    }


def run_new(retriever, llm, question):
    chunks = retriever.retrieve(question, top_k=TOP_K)
    best = chunks[0]["similarity"] if chunks else 0.0
    low_conf = best < DEFAULT_SIMILARITY_THRESHOLD
    ctx = build_context(chunks, retriever.vector_store, debug=False)
    prompt = build_rag_prompt(question, ctx.formatted, low_confidence=low_conf)
    answer = paced_generate(llm, prompt)
    covered = sorted({cid for p in ctx.passages for cid in p.chunk_ids})
    return {
        "chunk_ids": ctx.entry_chunk_ids,
        "expanded": ctx.expanded_chunk_ids,
        "covered": covered,
        "context_chars": ctx.total_chars,
        "duplicates_removed": ctx.duplicate_chars_removed,
        "best_similarity": best,
        "low_confidence": low_conf,
        "answer": answer,
        "sources": sorted({p.page_number for p in ctx.passages}),
        "citations": [
            f"{p.document_name} — {p.page_label} {p.page_number}"
            + (f" — {p.section}" if p.section else "")
            for p in ctx.passages
        ],
    }


def main() -> None:
    if not PDF_PATH.exists():
        raise SystemExit(f"Missing {PDF_PATH}; run scripts/create_formula_pdf.py")

    print("\n" + "#" * 74)
    print("# STAGE 0  vs  STEPS 1-3      (same document, same prompt, same top_k)")
    print("#" * 74)

    file_bytes = PDF_PATH.read_bytes()
    pages = load_pdf(PDF_PATH)
    model = EmbeddingModel()

    old_retriever, old_chunks = build_old(pages, model)
    new_retriever, new_chunks = build_new(pages, model, file_bytes)
    llm = create_llm("gemini")

    print(f"\nOLD chunker: {len(old_chunks)} chunks"
          f"   NEW chunker: {len(new_chunks)} chunks")

    summary = []

    # Optional filter so a single question can be re-run without spending
    # free-tier quota on the others:  compare_stage0.py --only 4
    questions = QUESTIONS
    if "--only" in sys.argv:
        wanted = sys.argv[sys.argv.index("--only") + 1].split(",")
        questions = [q for q in QUESTIONS if q[0] in wanted]
        print(f"\n(running only question(s) {wanted})")

    for tag, question in questions:
        print("\n" + "=" * 74)
        print(f"Q{tag}: {question}")
        print("=" * 74)

        try:
            old = run_old(old_retriever, llm, question)
            new = run_new(new_retriever, llm, question)
        except LLMError as exc:
            print(f"  LLM ERROR: {exc}")
            raise SystemExit(1)

        print("\n  RETRIEVAL")
        print(f"    OLD  chunks {old['chunk_ids']}"
              f"  | context {old['context_chars']} chars"
              f"  | best sim {old['best_similarity']:.3f}")
        print(f"    NEW  entry {new['chunk_ids']}"
              f"  + expanded {new['expanded'] or '[]'}"
              f"  = {new['covered']}")
        print(f"         context {new['context_chars']} chars"
              f"  | best sim {new['best_similarity']:.3f}"
              f"  | overlap removed {new['duplicates_removed']} chars")

        print("\n  SOURCES")
        print(f"    OLD  pages {old['sources']}")
        print(f"    NEW  {'; '.join(new['citations'])}")

        print(f"\n  --- OLD ANSWER ({len(old['answer'])} chars) ---")
        for line in old["answer"].splitlines():
            print(f"    {line}")

        print(f"\n  --- NEW ANSWER ({len(new['answer'])} chars) ---")
        for line in new["answer"].splitlines():
            print(f"    {line}")

        # Fact coverage only makes sense for the answerable questions.
        if tag in ("1", "2", "3"):
            old_facts = facts_present(old["answer"])
            new_facts = facts_present(new["answer"])
            print("\n  FACTS PRESENT IN THE ANSWER")
            for label in ANSWER_FACTS:
                o = "YES" if old_facts[label] else "no "
                n = "YES" if new_facts[label] else "no "
                delta = ""
                if new_facts[label] and not old_facts[label]:
                    delta = "  <-- GAINED"
                elif old_facts[label] and not new_facts[label]:
                    delta = "  <-- LOST"
                print(f"    {label:<24} OLD {o}   NEW {n}{delta}")
            summary.append((tag, old_facts, new_facts, old, new))
        else:
            # The absent-information question: the ONLY correct behaviour is a
            # refusal. Length and citations are irrelevant if it invents facts.
            refusal_markers = ["not available", "not contain", "no information",
                               "does not", "couldn't find", "could not find",
                               "not mentioned", "not provided", "not found"]
            old_refused = any(m in old["answer"].lower() for m in refusal_markers)
            new_refused = any(m in new["answer"].lower() for m in refusal_markers)
            print("\n  CORRECTLY DECLINED TO ANSWER?")
            print(f"    OLD {'YES' if old_refused else 'NO — HALLUCINATED'}")
            print(f"    NEW {'YES' if new_refused else 'NO — HALLUCINATED'}")
            summary.append((tag, {"refused": old_refused},
                            {"refused": new_refused}, old, new))

    # ------------------------------------------------------------------
    print("\n" + "#" * 74)
    print("# SUMMARY")
    print("#" * 74)
    print(f"\n  {'Q':<3} {'metric':<26} {'OLD':>10} {'NEW':>10}")
    for tag, old_facts, new_facts, old, new in summary:
        print(f"  {tag:<3} {'context chars':<26} {old['context_chars']:>10}"
              f" {new['context_chars']:>10}")
        print(f"  {'':<3} {'answer chars':<26} {len(old['answer']):>10}"
              f" {len(new['answer']):>10}")
        if "refused" in old_facts:
            print(f"  {'':<3} {'correctly declined':<26}"
                  f" {str(old_facts['refused']):>10} {str(new_facts['refused']):>10}")
        else:
            o = sum(1 for v in old_facts.values() if v)
            n = sum(1 for v in new_facts.values() if v)
            print(f"  {'':<3} {'facts present (of 4)':<26} {o:>10} {n:>10}")
        print()


if __name__ == "__main__":
    main()
