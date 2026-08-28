"""
Real-document validation: how does ingestion behave on PDFs we did not write?

The synthetic test document (data/formula_sample.pdf) is single-column, one
font, no headers, no tables. Passing on it proves very little. This script runs
the ingestion path on real PDFs fetched by scripts/fetch_real_docs.py and
reports the properties that went wrong the first time we tried:

    wrap joining        did wrapped lines get rejoined into paragraphs?
    hyphenation         'impres- sive' artifacts
    boilerplate         running headers/footers leaking into chunks
    headings            how many blocks were accepted as section headings
    section labels      what actually got attached to chunks
    boundaries          mid-sentence endings, mid-word starts, runts
    offsets             does text[char_start:char_end] == chunk text?

Run:  ./.venv/bin/python scripts/validate_real_docs.py [data/real/some.pdf ...]
"""

import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chunker import chunk_pages, page_text_for_chunking  # noqa: E402
from document_loader import load_pdf  # noqa: E402

REAL_DIR = PROJECT_ROOT / "data" / "real"
HYPHEN_ARTIFACT = re.compile(r"[a-z]- [a-z]")


def analyse(pdf_path: Path, show_headings: int = 40) -> dict:
    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages, 500, 50, document_name=pdf_path.name, document_id="val")
    sources = {p["page_number"]: page_text_for_chunking(p) for p in pages}

    lengths = [len(c["text"]) for c in chunks]
    mid_sentence = sum(
        not c["text"].rstrip().endswith((".", "!", "?", ":", ";")) for c in chunks
    )
    runts = sum(len(c["text"]) < 120 for c in chunks)
    mid_word_start = sum(
        1
        for c in chunks
        if c["char_start"] > 0
        and sources[c["page_number"]][c["char_start"] - 1].isalnum()
        and c["text"][:1].isalnum()
    )
    offset_violations = sum(
        sources[c["page_number"]][c["char_start"]:c["char_end"]] != c["text"] for c in chunks
    )
    hyphen_hits = sum(len(HYPHEN_ARTIFACT.findall(c["text"])) for c in chunks)
    headings = [(p["page_number"], h) for p in pages for h in p.get("headings", [])]
    sections = Counter(c["section"] for c in chunks)

    # Lines recurring on many pages of the RAW extraction are header/footer
    # candidates; count how many chunks still contain any of them.
    from document_loader import load_pdf_raw_text

    raw = load_pdf_raw_text(pdf_path.read_bytes())
    recur: Counter = Counter()
    for p in raw:
        for line in {l.strip() for l in p["text"].split("\n") if len(l.strip()) > 8}:
            recur[line] += 1
    boilerplate = [l for l, n in recur.items() if n >= max(3, len(raw) * 0.3)]
    leaked = sum(any(b in c["text"] for b in boilerplate) for c in chunks)

    print(f"\n{'#' * 74}\n# {pdf_path.name}: {len(pages)} pages -> {len(chunks)} chunks\n{'#' * 74}")
    print(f"  chunk length min/mean/max     : {min(lengths)}/{sum(lengths) // len(lengths)}/{max(lengths)}")
    print(f"  END mid-sentence              : {mid_sentence}/{len(chunks)} ({mid_sentence / len(chunks) * 100:.0f}%)")
    print(f"  START mid-word                : {mid_word_start}")
    print(f"  runt chunks (<120)            : {runts}")
    print(f"  offset invariant violations   : {offset_violations}")
    print(f"  hyphenation artifacts         : {hyphen_hits}")
    print(f"  recurring header/footer lines : {len(boilerplate)}  -> leaked into {leaked} chunks")
    print(f"  headings accepted             : {len(headings)}  ({len(headings) / len(pages):.1f} per page)")
    print(f"  chunks with section=None      : {sections.get(None, 0)}")

    print(f"\n  first {show_headings} headings:")
    for pn, h in headings[:show_headings]:
        print(f"    p{pn:<3} {h[:90]!r}")
    if len(headings) > show_headings:
        print(f"    ... {len(headings) - show_headings} more")

    print("\n  most common section labels on chunks:")
    for sec, n in sections.most_common(12):
        print(f"    {n:>3}  {str(sec)[:90]!r}")

    return {
        "chunks": len(chunks),
        "mid_sentence_pct": mid_sentence / len(chunks) * 100,
        "mid_word_start": mid_word_start,
        "runts": runts,
        "offset_violations": offset_violations,
        "hyphen_hits": hyphen_hits,
        "leaked_boilerplate_chunks": leaked,
        "headings": len(headings),
        "section_none": sections.get(None, 0),
    }


def main() -> None:
    targets = [Path(a) for a in sys.argv[1:]] or sorted(REAL_DIR.glob("*.pdf"))
    if not targets:
        raise SystemExit(
            f"No PDFs found in {REAL_DIR}. Run: ./.venv/bin/python scripts/fetch_real_docs.py"
        )
    for t in targets:
        analyse(t)
    print()


if __name__ == "__main__":
    main()
