"""
STEP 1 TEST — does chunk metadata survive the round trip through retrieval?

The question this answers:
    We enriched the chunker. But metadata is only useful if it still exists by
    the time a chunk comes BACK out of FAISS. Before Step 1, vector_store.search
    rebuilt each result from five hardcoded keys, so anything else was silently
    dropped. This test proves that no longer happens.

The data path being tested:
    chunk_pages()  ->  embed  ->  VectorStore.add  ->  FAISS
                                                         |
    retriever.retrieve()  <-  VectorStore.search  <-------+

FAISS itself stores ONLY numbers. It has no idea what a page number is. The
metadata never enters the index — it rides alongside in VectorStore.chunks, and
search() maps a matched vector position back to its chunk. This test checks that
mapping preserves everything.

Run:  ./.venv/bin/python scripts/test_metadata.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chunker import chunk_pages  # noqa: E402
from document_loader import load_pdf  # noqa: E402
from embeddings import EmbeddingModel  # noqa: E402
from pipeline import make_document_id  # noqa: E402
from retriever import Retriever  # noqa: E402
from vector_store import VectorStore  # noqa: E402

PDF_PATH = PROJECT_ROOT / "data" / "formula_sample.pdf"

# Every field the Step 1 chunker promises to attach to each chunk.
EXPECTED_CHUNK_FIELDS = [
    "chunk_id",
    "text",
    "document_name",
    "document_id",
    "page_number",
    "page_label",
    "section",
    "position_in_page",
    "char_start",
    "char_end",
    "prev_chunk_id",
    "next_chunk_id",
    "total_chunks",
]

# search() adds these two on top of the chunk's own metadata.
EXPECTED_SEARCH_EXTRAS = ["similarity", "rank"]


def check(label: str, passed: bool, detail: str = "") -> bool:
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    return passed


def main() -> None:
    print("\n" + "#" * 72)
    print("# STEP 1 TEST — METADATA FOUNDATION")
    print("#" * 72)

    file_bytes = PDF_PATH.read_bytes()
    document_id = make_document_id(file_bytes)

    pages = load_pdf(PDF_PATH)
    chunks = chunk_pages(
        pages,
        chunk_size=500,
        chunk_overlap=50,
        document_name=PDF_PATH.name,
        document_id=document_id,
    )

    print(f"\nDocument : {PDF_PATH.name}")
    print(f"Doc ID   : {document_id}  (sha256 of file bytes, first 12 hex)")
    print(f"Pages    : {len(pages)}   Chunks: {len(chunks)}")

    results: list[bool] = []

    # ---------------------------------------------------------------
    print("\n--- 1. Does the chunker attach every promised field? ---")
    # ---------------------------------------------------------------
    missing = [f for f in EXPECTED_CHUNK_FIELDS if f not in chunks[0]]
    results.append(
        check(
            f"all {len(EXPECTED_CHUNK_FIELDS)} fields present on chunk 0",
            not missing,
            f"missing: {missing}" if missing else f"{len(chunks[0])} fields total",
        )
    )
    consistent = all(set(c.keys()) == set(chunks[0].keys()) for c in chunks)
    results.append(check("schema identical across all chunks", consistent))

    # ---------------------------------------------------------------
    print("\n--- 2. Is the metadata actually CORRECT? ---")
    # ---------------------------------------------------------------
    # Wrong metadata is worse than no metadata: it produces confident,
    # verifiable-looking citations that point at the wrong place.
    ids_sequential = [c["chunk_id"] for c in chunks] == list(range(len(chunks)))
    results.append(check("chunk_ids are sequential from 0", ids_sequential))

    offsets_ok = True
    offset_detail = ""
    for page in pages:
        # Step 2 moved chunking onto structure-preserving cleaned text, so
        # offsets are into THAT coordinate space, not the flattened one.
        from document_loader import clean_text_structured

        page_text = clean_text_structured(page["text"])
        for c in chunks:
            if c["page_number"] != page["page_number"]:
                continue
            sliced = page_text[c["char_start"]:c["char_end"]]
            if sliced != c["text"]:
                offsets_ok = False
                offset_detail = f"chunk {c['chunk_id']} offsets do not match its text"
                break
    results.append(
        check(
            "char_start/char_end slice the page back to the exact chunk text",
            offsets_ok,
            offset_detail,
        )
    )

    links_ok = True
    for i, c in enumerate(chunks):
        expected_prev = chunks[i - 1]["chunk_id"] if i > 0 else None
        expected_next = chunks[i + 1]["chunk_id"] if i < len(chunks) - 1 else None
        if c["prev_chunk_id"] != expected_prev or c["next_chunk_id"] != expected_next:
            links_ok = False
            break
    results.append(check("prev/next chunk links form an unbroken chain", links_ok))
    results.append(
        check(
            "first chunk has prev=None, last has next=None",
            chunks[0]["prev_chunk_id"] is None
            and chunks[-1]["next_chunk_id"] is None,
        )
    )

    pos_ok = True
    seen: dict[int, int] = {}
    for c in chunks:
        expected = seen.get(c["page_number"], 0)
        if c["position_in_page"] != expected:
            pos_ok = False
            break
        seen[c["page_number"]] = expected + 1
    results.append(check("position_in_page restarts at 0 on each page", pos_ok))

    results.append(
        check(
            "page_label is honest for a PDF",
            all(c["page_label"] == "page" for c in chunks),
            "'page' (a real PDF page); DOCX would say 'section'",
        )
    )

    # ---------------------------------------------------------------
    print("\n--- 3. THE KEY TEST: does metadata survive FAISS? ---")
    # ---------------------------------------------------------------
    embedding_model = EmbeddingModel()
    embeddings = embedding_model.embed_texts([c["text"] for c in chunks])
    store = VectorStore(dimension=embedding_model.dimension)
    store.add(embeddings, chunks)
    retriever = Retriever(embedding_model, store)

    retrieved = retriever.retrieve("What is the compound interest formula?", top_k=3)
    top = retrieved[0]

    lost = [f for f in EXPECTED_CHUNK_FIELDS if f not in top]
    results.append(
        check(
            f"all {len(EXPECTED_CHUNK_FIELDS)} chunk fields survive retrieval",
            not lost,
            f"lost in transit: {lost}" if lost else "none dropped",
        )
    )
    extras_present = [f for f in EXPECTED_SEARCH_EXTRAS if f in top]
    results.append(
        check(
            "search() adds similarity + rank",
            len(extras_present) == 2,
            f"present: {extras_present}",
        )
    )

    original = next(c for c in chunks if c["chunk_id"] == top["chunk_id"])
    values_match = all(top[f] == original[f] for f in EXPECTED_CHUNK_FIELDS)
    results.append(
        check("retrieved values are IDENTICAL to the indexed chunk", values_match)
    )

    # ---------------------------------------------------------------
    print("\n--- 4. Does search() protect the index from mutation? ---")
    # ---------------------------------------------------------------
    # If search() returned the stored dict itself, a caller writing to a result
    # would corrupt the index, and the next query would return a chunk carrying
    # a stale similarity score from a previous question.
    retrieved[0]["similarity"] = 999.0
    stored = next(c for c in store.chunks if c["chunk_id"] == top["chunk_id"])
    results.append(
        check(
            "mutating a result does not pollute the stored chunk",
            "similarity" not in stored,
            "stored chunk has no 'similarity' key — good",
        )
    )

    # ---------------------------------------------------------------
    print("\n--- 5. What a citation can now say ---")
    # ---------------------------------------------------------------
    print("\n  Before Step 1, a retrieved chunk could only support: 'Page 2'")
    print("  Now, from real metadata (nothing invented):\n")
    # Re-retrieve: the mutation check above deliberately corrupted the earlier
    # result dicts, and printing those would show a fake similarity of 999.
    retrieved_fresh = retriever.retrieve(
        "What is the compound interest formula?", top_k=3
    )
    for r in retrieved_fresh:
        section = r["section"] or "(none detected)"
        print(
            f"    {r['document_name']} — {r['page_label']} {r['page_number']}"
            f" — section '{section}'"
        )
        print(
            f"      chunk {r['chunk_id']}, position {r['position_in_page']} on that"
            f" {r['page_label']}, chars {r['char_start']}-{r['char_end']},"
            f" similarity {r['similarity']:.3f}"
        )
        print(
            f"      neighbours available for expansion: "
            f"prev={r['prev_chunk_id']} next={r['next_chunk_id']}"
        )

    # ---------------------------------------------------------------
    print("\n--- 6. Does metadata reach the CONTEXT BUILDER intact? ---")
    # ---------------------------------------------------------------
    # The full path is: chunk -> vector store -> retrieval -> context builder.
    # A passage is what the LLM finally reads, so citation metadata has to
    # survive all the way here or the citation cannot be produced.
    from context_builder import build_context, citation_for

    context = build_context(retrieved_fresh, store, debug=False)

    results.append(
        check(
            "context builder produced passages",
            context.has_evidence,
            f"{len(context.passages)} passages",
        )
    )

    passage_fields_ok = all(
        p.document_name is not None
        and p.document_id is not None
        and p.page_number is not None
        and p.page_label is not None
        and p.chunk_ids
        for p in context.passages
    )
    results.append(
        check("every passage carries document/page/label/chunk_ids", passage_fields_ok)
    )

    # A merged passage must report every chunk it absorbed, or a citation
    # would under-report its own sources.
    merged = [p for p in context.passages if len(p.chunk_ids) > 1]
    results.append(
        check(
            "merged passages list all constituent chunk ids",
            all(
                p.chunk_ids == list(range(p.chunk_ids[0], p.chunk_ids[-1] + 1))
                for p in merged
            ),
            f"{len(merged)} merged passage(s)",
        )
    )

    sections_ok = all(
        p.section is None or isinstance(p.section, str) for p in context.passages
    )
    results.append(check("section is a real string or None, never invented", sections_ok))

    print("\n  Citations built purely from surviving metadata:")
    for number, passage in enumerate(context.passages, start=1):
        print(f"    [S{number}] {citation_for(passage)}"
              f"   (chunks {passage.chunk_ids})")

    # ---------------------------------------------------------------
    print("\n" + "#" * 72)
    passed, total = sum(results), len(results)
    print(f"# STEP 1 RESULT: {passed}/{total} checks passed")
    print("#" * 72 + "\n")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
