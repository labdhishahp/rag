"""
Chunking + metadata on the synthetic formula document (data/formula_sample.pdf).

These pin the Stage 1 guarantees that fixed the original failures:
  - no chunk starts or ends mid-word
  - no runt chunks
  - the five variable definitions live in ONE chunk
  - section labels are the five real headings, nothing invented
  - char offsets slice the page text back to the exact chunk text
"""

from chunker import page_text_for_chunking

VARIABLE_MARKERS = [
    "P = the principal",
    "r = the annual nominal interest rate",
    "n = the number of times interest is compounded",
    "t = the total time the money is invested",
    "A = the final amount accumulated",
]
EXPECTED_SECTIONS = {
    "Introduction to Financial Mathematics",
    "Compound Interest",
    "Simple Interest",
    "Present Value",
    "Risk and Diversification",
}
EXPECTED_FIELDS = {
    "chunk_id", "text", "document_name", "document_id", "page_number", "page_label",
    "section", "position_in_page", "char_start", "char_end", "prev_chunk_id",
    "next_chunk_id", "total_chunks",
}


def _norm(s: str) -> str:
    return " ".join(s.split())


def test_schema_complete_and_uniform(formula_chunks):
    assert set(formula_chunks[0]) == EXPECTED_FIELDS
    assert all(set(c) == EXPECTED_FIELDS for c in formula_chunks)


def test_ids_sequential_and_linked(formula_chunks):
    ids = [c["chunk_id"] for c in formula_chunks]
    assert ids == list(range(len(formula_chunks)))
    assert formula_chunks[0]["prev_chunk_id"] is None
    assert formula_chunks[-1]["next_chunk_id"] is None
    for a, b in zip(formula_chunks, formula_chunks[1:]):
        assert a["next_chunk_id"] == b["chunk_id"]
        assert b["prev_chunk_id"] == a["chunk_id"]


def test_offsets_reproduce_text(formula_pages, formula_chunks):
    sources = {p["page_number"]: page_text_for_chunking(p) for p in formula_pages}
    for c in formula_chunks:
        assert sources[c["page_number"]][c["char_start"]:c["char_end"]] == c["text"]


def test_no_mid_word_boundaries(formula_pages, formula_chunks):
    sources = {p["page_number"]: page_text_for_chunking(p) for p in formula_pages}
    for c in formula_chunks:
        src = sources[c["page_number"]]
        if c["char_start"] > 0:
            assert not (src[c["char_start"] - 1].isalnum() and c["text"][0].isalnum())
        if c["char_end"] < len(src):
            assert not (src[c["char_end"]].isalnum() and c["text"][-1].isalnum())


def test_no_runts_and_no_mid_sentence_endings(formula_chunks):
    assert all(len(c["text"]) >= 120 for c in formula_chunks)
    assert all(c["text"].rstrip().endswith((".", "!", "?", ":", ";")) for c in formula_chunks)


def test_variable_definitions_in_one_chunk(formula_chunks):
    holders = [
        c["chunk_id"]
        for c in formula_chunks
        if all(_norm(m) in _norm(c["text"]) for m in VARIABLE_MARKERS)
    ]
    assert holders, "no single chunk contains all five variable definitions"


def test_section_labels_are_exactly_the_real_headings(formula_chunks):
    assert {c["section"] for c in formula_chunks} == EXPECTED_SECTIONS
    # A formula line must never be mistaken for a heading.
    assert not any("=" in (c["section"] or "") for c in formula_chunks)


def test_page_label_is_honest(formula_chunks):
    assert all(c["page_label"] == "page" for c in formula_chunks)
