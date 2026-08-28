"""
STEP 2 TEST — structure-aware chunking, measured against the old chunker.

Runs BOTH algorithms over the same document so every difference is visible:

  OLD: text[start : start+500] on whitespace-collapsed text
  NEW: pack whole sentences/blocks, on structure-preserving text

Metrics that matter:
  1. chunks starting or ending mid-WORD        -> must reach 0
  2. chunks ending mid-SENTENCE                -> must drop sharply
  3. runt chunks (tiny, mostly duplicated)     -> must reach 0
  4. is the variable list intact in one chunk? -> the Step 0 failure
  5. char offsets still exact                  -> Step 1 invariant preserved

Run:  ./.venv/bin/python scripts/test_chunking.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chunker import chunk_pages  # noqa: E402
from document_loader import (  # noqa: E402
    clean_text,
    clean_text_structured,
    load_pdf,
)

PDF_PATH = PROJECT_ROOT / "data" / "formula_sample.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# The five variable definitions from the compound interest section.
VARIABLE_MARKERS = [
    "P = the principal",
    "r = the annual nominal interest rate",
    "n = the number of times interest is compounded",
    "t = the total time the money is invested",
    "A = the final amount accumulated",
]
FORMULA = "A = P(1 + r/n)^(nt)"


# --------------------------------------------------------------------------
# The OLD algorithm, reproduced here so we can compare fairly.
# This is the character-window slicer that chunker.py used before Step 2.
# --------------------------------------------------------------------------
def old_chunk_pages(pages, chunk_size, chunk_overlap):
    chunks = []
    chunk_id = 0
    for page in pages:
        text = clean_text(page["text"])
        if not text:
            continue
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page_number": page["page_number"],
                        "text": chunk_text,
                    }
                )
                chunk_id += 1
            if end >= len(text):
                break
            start = end - chunk_overlap
    return chunks


# --------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------
def starts_mid_word(chunk_text: str, source: str) -> bool:
    """True if the chunk begins part-way through a word."""
    index = source.find(chunk_text[:40])
    if index <= 0:
        return False
    return source[index - 1].isalnum() and chunk_text[0].isalnum()


def ends_mid_word(chunk_text: str, source: str) -> bool:
    """True if the chunk stops part-way through a word."""
    tail = chunk_text[-40:]
    index = source.find(tail)
    if index < 0:
        return False
    after = index + len(tail)
    if after >= len(source):
        return False
    return source[after].isalnum() and chunk_text[-1].isalnum()


def ends_mid_sentence(chunk_text: str) -> bool:
    return not chunk_text.rstrip().endswith((".", "!", "?", ":", ";"))


def measure(chunks, sources_by_page, label):
    mid_word_start = mid_word_end = mid_sentence = 0
    for c in chunks:
        source = sources_by_page[c["page_number"]]
        if starts_mid_word(c["text"], source):
            mid_word_start += 1
        if ends_mid_word(c["text"], source):
            mid_word_end += 1
        if ends_mid_sentence(c["text"]):
            mid_sentence += 1

    lengths = [len(c["text"]) for c in chunks]
    runts = [c for c in chunks if len(c["text"]) < 120]

    print(f"\n  {label}")
    print(f"    chunks                    : {len(chunks)}")
    print(f"    length min / mean / max   : {min(lengths)} / "
          f"{sum(lengths)//len(lengths)} / {max(lengths)}")
    print(f"    START mid-word            : {mid_word_start}")
    print(f"    END mid-word              : {mid_word_end}")
    print(f"    END mid-sentence          : {mid_sentence}"
          f"  ({mid_sentence/len(chunks)*100:.0f}%)")
    print(f"    runt chunks (<120 chars)  : {len(runts)}"
          + (f"  -> {[c['chunk_id'] for c in runts]}" if runts else ""))
    return {
        "mid_word": mid_word_start + mid_word_end,
        "mid_sentence": mid_sentence,
        "runts": len(runts),
        "count": len(chunks),
    }


def find_marker_chunk(chunks, needle):
    """Which chunk IDs contain this text (whitespace-insensitive)."""
    needle_norm = " ".join(needle.split())
    return [
        c["chunk_id"]
        for c in chunks
        if needle_norm in " ".join(c["text"].split())
    ]


def main() -> None:
    print("\n" + "#" * 72)
    print("# STEP 2 TEST — STRUCTURE-AWARE CHUNKING")
    print("#" * 72)

    pages = load_pdf(PDF_PATH)

    old_sources = {p["page_number"]: clean_text(p["text"]) for p in pages}
    new_sources = {p["page_number"]: clean_text_structured(p["text"]) for p in pages}

    old = old_chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    new = chunk_pages(
        pages,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        document_name=PDF_PATH.name,
        document_id="test",
    )

    print("\n=== 1. BOUNDARY QUALITY ===")
    old_stats = measure(old, old_sources, "OLD — character-window slicing")
    new_stats = measure(new, new_sources, "NEW — sentence/block packing")

    print("\n=== 2. THE MID-WORD BREAK, SHOWN DIRECTLY ===")
    print("\n  OLD chunk 1 ENDS:")
    print(f"    ...{old[1]['text'][-70:]!r}")
    print("  OLD chunk 2 STARTS:")
    print(f"    {old[2]['text'][:70]!r}")
    print("    ^ the word 'variables' is severed across the boundary")

    print("\n  NEW chunk boundaries at the same place:")
    for c in new:
        if c["page_number"] == 2:
            print(f"    chunk {c['chunk_id']} ENDS  : ...{c['text'][-60:]!r}")

    print("\n=== 3. IS THE VARIABLE LIST INTACT IN ONE CHUNK? ===")
    print("\n  This is the Step 0 failure: query C could not be answered because")
    print("  no single retrievable chunk held the variable definitions.\n")

    for label, chunks in (("OLD", old), ("NEW", new)):
        located = {m: find_marker_chunk(chunks, m) for m in VARIABLE_MARKERS}
        all_ids = [ids for ids in located.values() if ids]
        # Which chunk (if any) contains ALL five definitions?
        complete = None
        for c in chunks:
            norm = " ".join(c["text"].split())
            if all(" ".join(m.split()) in norm for m in VARIABLE_MARKERS):
                complete = c["chunk_id"]
                break
        print(f"  {label}:")
        for marker, ids in located.items():
            print(f"    {marker[:42]:<44} -> chunk {ids}")
        spread = sorted({i for ids in all_ids for i in ids})
        print(f"    spread across chunks: {spread}")
        print(f"    ALL FIVE in one chunk: "
              f"{'YES -> chunk ' + str(complete) if complete is not None else 'NO'}")
        formula_in = find_marker_chunk(chunks, FORMULA)
        print(f"    formula '{FORMULA}' -> chunk {formula_in}")
        print()

    print("=== 4. SECTION METADATA (new in Step 2) ===")
    print("\n  Headings are now detectable because newlines survive cleaning.\n")
    for c in new:
        section = c["section"] if c["section"] else "(none detected)"
        print(f"    chunk {c['chunk_id']:>2} | {c['page_label']} {c['page_number']}"
              f" | section: {section}")

    print("\n=== 5. STEP 1 INVARIANT: are char offsets still exact? ===")
    bad = [
        c["chunk_id"]
        for c in new
        if new_sources[c["page_number"]][c["char_start"]:c["char_end"]] != c["text"]
    ]
    print(f"\n  chunks whose offsets do not reproduce their text: {bad if bad else 0}")
    offsets_ok = not bad
    print(f"  invariant holds: {offsets_ok}")

    print("\n" + "#" * 72)
    print("# STEP 2 SUMMARY")
    print("#" * 72)
    rows = [
        ("mid-word breaks", old_stats["mid_word"], new_stats["mid_word"], 0),
        ("mid-sentence endings", old_stats["mid_sentence"], new_stats["mid_sentence"], None),
        ("runt chunks", old_stats["runts"], new_stats["runts"], 0),
        ("total chunks", old_stats["count"], new_stats["count"], None),
    ]
    print(f"\n  {'metric':<26} {'OLD':>6} {'NEW':>6}   {'target':>7}")
    for name, o, n, target in rows:
        goal = str(target) if target is not None else "-"
        print(f"  {name:<26} {o:>6} {n:>6}   {goal:>7}")

    ok = (
        new_stats["mid_word"] == 0
        and new_stats["runts"] == 0
        and new_stats["mid_sentence"] < old_stats["mid_sentence"]
        and offsets_ok
    )
    print(f"\n  STEP 2 {'PASSED' if ok else 'FAILED'}")
    print()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
