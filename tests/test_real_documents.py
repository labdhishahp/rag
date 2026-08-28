"""
Ingestion regression tests on REAL PDFs (data/real/, fetched by
scripts/fetch_real_docs.py, never committed).

Why these exist:
  The synthetic test document passed 100% while ingestion was failing on real
  PDFs — 779 fake headings on a 21-page paper, a running header attached as
  the section of 27% of a report's chunks, 'impres- sive' hyphenation
  artifacts in every paragraph. These tests pin the properties that broke so
  they cannot silently return.

Each test is parametrised over the three documents and skips if a file is
absent, so the suite still runs on a fresh clone.
"""

import re

import pytest

from chunker import chunk_pages, page_text_for_chunking
from conftest import REAL_DOCS, real_doc
from document_loader import load_pdf

# 'impres- sive' style artifacts. Legitimate English uses a hanging hyphen
# before a coordinating word ("Public- or Customer-Facing"), so those are
# excluded from the pattern.
HYPHEN_ARTIFACT = re.compile(r"[a-z]- (?!(or|and|versus|vs|to|nor)\b)[a-z]")

# Headings that MUST be found — chosen by reading the documents.
KNOWN_HEADINGS = {
    "rag_survey": {"I. INTRODUCTION", "C. Modular RAG", "III. RETRIEVAL", "REFERENCES"},
    "bert": {"1 Introduction", "3.1 Pre-training BERT", "References"},
    "nist": {"1 Introduction", "2.1 Tenets of Zero Trust", "3.3 Trust Algorithm"},
}

# Text that must NEVER appear inside a chunk (running headers/footers).
FORBIDDEN_IN_CHUNKS = {
    "nist": [
        "This publication is available free of charge from",
        "NIST SP 800-207\nZERO TRUST ARCHITECTURE",
    ],
    "rag_survey": ["arXiv:2312.10997"],
    "bert": [],
}


@pytest.fixture(scope="module", params=list(REAL_DOCS))
def doc(request):
    path = real_doc(request.param)
    pages = load_pdf(path)
    chunks = chunk_pages(pages, 500, 50, document_name=path.name, document_id=request.param)
    return request.param, pages, chunks


def test_offsets_reproduce_text(doc):
    _, pages, chunks = doc
    sources = {p["page_number"]: page_text_for_chunking(p) for p in pages}
    bad = [c["chunk_id"] for c in chunks if sources[c["page_number"]][c["char_start"]:c["char_end"]] != c["text"]]
    assert bad == []


def test_no_mid_word_starts_and_no_runts(doc):
    _, pages, chunks = doc
    sources = {p["page_number"]: page_text_for_chunking(p) for p in pages}
    mid_word = [
        c["chunk_id"]
        for c in chunks
        if c["char_start"] > 0
        and sources[c["page_number"]][c["char_start"] - 1].isalnum()
        and c["text"][:1].isalnum()
    ]
    assert mid_word == []
    assert [c["chunk_id"] for c in chunks if len(c["text"]) < 120] == []


def test_mid_sentence_rate_bounded(doc):
    _, _, chunks = doc
    mid = sum(not c["text"].rstrip().endswith((".", "!", "?", ":", ";")) for c in chunks)
    assert mid / len(chunks) <= 0.30, f"{mid}/{len(chunks)} chunks end mid-sentence"


def test_no_hyphenation_artifacts(doc):
    _, _, chunks = doc
    hits = [m.group(0) for c in chunks for m in HYPHEN_ARTIFACT.finditer(c["text"])]
    assert hits == []


def test_boilerplate_not_in_chunks(doc):
    name, _, chunks = doc
    # The same sentence can legitimately appear as body text on a cover or
    # authority page; what must never happen is the RUNNING footer leaking into
    # body-page chunks. So only body pages are checked.
    body = [c for c in chunks if c["page_number"] >= 10]
    for needle in FORBIDDEN_IN_CHUNKS[name]:
        leaked = [c["page_number"] for c in body if needle in c["text"]]
        assert leaked == [], f"{needle!r} leaked into body-page chunks on pages {leaked}"


def test_heading_count_is_plausible(doc):
    _, pages, _ = doc
    headings = [h for p in pages for h in p.get("headings", [])]
    per_page = len(headings) / len(pages)
    assert 0.5 <= per_page <= 4.0, f"{len(headings)} headings on {len(pages)} pages"


def test_known_headings_detected(doc):
    name, pages, _ = doc
    found = {h for p in pages for h in p.get("headings", [])}
    missing = KNOWN_HEADINGS[name] - found
    assert not missing, f"expected headings not detected: {missing}"


def test_section_labels_are_sane(doc):
    _, _, chunks = doc
    labels = {c["section"] for c in chunks} - {None}
    for label in labels:
        assert "://" not in label, label
        assert len(label) > 2, label
        assert not re.fullmatch(r"[ivxlcIVXLC]+", label), label
        assert not label.rstrip().endswith((",", "-")), label
    # A section, once seen, carries forward: nothing after the first heading is None.
    first_labelled = next(i for i, c in enumerate(chunks) if c["section"] is not None)
    assert all(c["section"] is not None for c in chunks[first_labelled:])


def test_running_header_is_not_a_section(doc):
    name, _, chunks = doc
    if name == "nist":
        assert "ZERO TRUST ARCHITECTURE" not in {c["section"] for c in chunks}
