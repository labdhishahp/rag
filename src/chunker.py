"""
Split document text into chunks that respect sentence and block structure.

Why not embed the whole PDF at once?
  Embedding models have input length limits. More importantly, a single vector
  for a long document averages away detail — a question about page 7 would be
  compared against one blob representing the entire file, and the signal gets
  diluted.

Why chunk?
  Smaller pieces let us retrieve only the passages that match a question.
  Each chunk becomes one searchable unit in the vector index.

Chunk size:
  How many characters each chunk contains (we use characters as a simple
  proxy for tokens). Larger chunks = more context per result, but less precise.

Overlap:
  Trailing sentences from one chunk repeat at the start of the next, so a fact
  spanning a boundary is complete in at least one chunk.

Too small: fragments lose meaning ("revenue was" without the number).
Too large: chunks mix unrelated topics; similarity scores become vague.

------------------------------------------------------------------------------
WHY THIS FILE WAS REWRITTEN (Step 2)
------------------------------------------------------------------------------
The previous chunker sliced on raw character counts: text[start : start + 500].
Measured on our test document, that produced:

  - chunk 1 ending "...Where the variables are defined as follows: P = the principal"
  - chunk 2 starting "ariables are defined as follows: P = the principal, meaning..."
                      ^^^^^^^^ the word "variables" cut in half

  - 4 of 9 chunks ending mid-sentence
  - chunk 4: a 78-character runt that was almost entirely duplicated overlap

A chunk that starts mid-word embeds badly. The embedding model has never seen
"ariables" as a word, so the vector is noisy, and the chunk loses similarity
contests it should win. That is a direct cause of the Step 0 finding where the
chunk containing every variable definition was never retrieved.

The fix: choose boundaries the document already offers.

  1. Recover block structure  (document_loader.clean_text_structured)
  2. Split blocks into sentences
  3. PACK whole sentences into chunks, never splitting one
  4. Only fall back to word boundaries if a single sentence exceeds chunk_size
  5. Merge runt tail chunks into their predecessor

A chunk boundary can now only land between sentences or between blocks. It can
never land inside a word.
"""

import re

from document_loader import clean_text_structured, is_heading


def page_text_for_chunking(page: dict) -> str:
    """
    The exact text the chunker slices, and therefore the coordinate space of
    every chunk's char_start/char_end.

    Pages from pdf_layout / the DOCX loader are already structured (one block
    per line, wraps joined) and carry "structured": True — use them verbatim.
    Anything else (plain text, the legacy raw-PDF path) goes through the older
    character-width heuristic as a fallback.

    Tests must reconstruct chunk text through THIS function, never by calling
    a cleaner directly, or the offset invariant is being checked against the
    wrong string.
    """
    if page.get("structured"):
        lines = [" ".join(line.split()) for line in page["text"].split("\n")]
        return "\n".join(line for line in lines if line)
    return clean_text_structured(page["text"])


def _page_heading_test(page: dict):
    """Return a predicate deciding whether a block line is a heading on this page."""
    if page.get("structured"):
        known = {" ".join(h.split()) for h in page.get("headings", [])}
        return lambda line: line in known
    return is_heading

# A sentence ends at . ! or ? followed by whitespace and then something that
# starts a new sentence (capital letter, digit, or an opening bracket).
#
# The lookahead is what protects decimals: "0.05/12" has no whitespace after
# the period, and "1647.01. The" does — so only the second one splits.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\d])")

# A tail chunk shorter than this is merged backwards instead of being indexed
# on its own. Below roughly this length a chunk is usually a fragment that
# duplicates its predecessor's overlap and adds no retrievable meaning.
DEFAULT_MIN_CHUNK_CHARS = 120


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    document_name: str | None = None,
    document_id: str | None = None,
    min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
) -> list[dict]:
    """
    Split page text into chunks with metadata.

    Input:  pages from load_pdf() — [{"page_number": 1, "text": "..."}, ...]
    Output: chunks with the metadata schema documented in _make_chunk().

    We chunk within each page first, then assign global chunk IDs.

    document_name / document_id identify which file a chunk came from. They are
    optional so existing callers (src/main.py, the Phase 1 CLI) keep working;
    when omitted the fields are present but None rather than absent, so
    downstream code can rely on the schema being stable.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[dict] = []
    chunk_id = 0
    # A section continues until the next heading, even across a page break.
    # Without this, every page that happens to contain no heading of its own
    # would report section=None for all its chunks.
    current_section: str | None = None

    for page in pages:
        # One block per line: headings, paragraphs, list items, tables.
        text = page_text_for_chunking(page)
        if not text:
            continue

        page_number = page["page_number"]
        page_label = page.get("page_label", "page")

        units = _unit_spans(text, chunk_size)
        if not units:
            continue

        spans = _pack_units(units, text, chunk_size, chunk_overlap)
        spans = _merge_runt_tail(spans, text, min_chunk_chars, chunk_size)
        headings = _heading_positions(text, _page_heading_test(page))

        for position_in_page, (start, end) in enumerate(spans):
            chunk_text = text[start:end]
            section_here = _section_for(headings, start)
            if section_here is not None:
                current_section = section_here
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "document_name": document_name,
                    "document_id": document_id,
                    "page_number": page_number,
                    "page_label": page_label,
                    # The nearest heading at or above this chunk's start,
                    # carried forward from earlier pages when this page has
                    # none. None only before the first heading in the
                    # document — we never invent one.
                    "section": current_section,
                    "position_in_page": position_in_page,
                    "char_start": start,
                    "char_end": end,
                }
            )
            chunk_id += 1

    if not chunks:
        raise ValueError("No chunks created — all pages were empty after cleaning.")

    return _link_neighbours(chunks)


# --------------------------------------------------------------------------
# Splitting text into packing units
# --------------------------------------------------------------------------


def _block_spans(text: str) -> list[tuple[int, int]]:
    """Spans of each structural block (the lines produced by the cleaner)."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for line in text.split("\n"):
        if line:
            spans.append((cursor, cursor + len(line)))
        cursor += len(line) + 1  # +1 for the newline we split on
    return spans


def _sentence_spans(block: str, offset: int) -> list[tuple[int, int]]:
    """Spans of each sentence inside one block, in absolute coordinates."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for piece in _SENTENCE_BOUNDARY.split(block):
        if not piece:
            continue
        start = block.index(piece, cursor)
        end = start + len(piece)
        spans.append((offset + start, offset + end))
        cursor = end
    return spans


def _split_long_span(text: str, start: int, end: int, limit: int) -> list[tuple[int, int]]:
    """
    Break an over-long unit on WORD boundaries.

    Reached only when one sentence is longer than chunk_size — a table row, a
    run-on line, or text with no punctuation. Splitting on spaces means the
    worst case is a broken sentence, never a broken word.
    """
    spans: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        if end - cursor <= limit:
            spans.append((cursor, end))
            break
        # Last space at or before the limit.
        window_end = cursor + limit
        split_at = text.rfind(" ", cursor, window_end)
        if split_at <= cursor:
            # A single word longer than the limit; hard-split it as a last
            # resort rather than emitting an oversized chunk.
            split_at = window_end
        spans.append((cursor, split_at))
        cursor = split_at + 1 if text[split_at:split_at + 1] == " " else split_at
    return spans


def _unit_spans(text: str, chunk_size: int) -> list[tuple[int, int]]:
    """
    The indivisible pieces we pack into chunks: one sentence, or one short block.

    A chunk boundary can only ever fall between two units, which is what
    guarantees no chunk starts or ends mid-word.
    """
    units: list[tuple[int, int]] = []
    for block_start, block_end in _block_spans(text):
        block = text[block_start:block_end]
        for start, end in _sentence_spans(block, block_start):
            if end - start > chunk_size:
                units.extend(_split_long_span(text, start, end, chunk_size))
            else:
                units.append((start, end))
    return units


# --------------------------------------------------------------------------
# Packing units into chunks
# --------------------------------------------------------------------------


def _pack_units(
    units: list[tuple[int, int]],
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[int, int]]:
    """
    Greedily fill chunks with whole units, then step back for overlap.

    Chunk length is measured as (last_unit_end - first_unit_start), i.e. the
    real slice length including the separators between units. That keeps
    char_start/char_end exact: text[char_start:char_end] IS the chunk text.
    """
    spans: list[tuple[int, int]] = []
    first = 0

    while first < len(units):
        last = first
        # Extend while the resulting slice still fits.
        while last + 1 < len(units):
            candidate_end = units[last + 1][1]
            if candidate_end - units[first][0] > chunk_size:
                break
            last += 1

        spans.append((units[first][0], units[last][1]))

        if last + 1 >= len(units):
            break

        # Overlap: walk back from the end, taking whole units, while they fit
        # in the overlap budget. Two guarantees:
        #   - next_first > first, so we always make progress
        #   - the next chunk can still hold unit last+1 (something NEW). Without
        #     this check, a long unit following short ones produced chunks made
        #     entirely of overlap text: 'Stage / Retrieval / process' then
        #     'Retrieval / process' — pure duplicates that then embedded and
        #     competed in search.
        next_first = last + 1
        while next_first - 1 > first:
            candidate = next_first - 1
            if units[last][1] - units[candidate][0] > chunk_overlap:
                break
            if units[last + 1][1] - units[candidate][0] > chunk_size:
                break
            next_first = candidate

        first = next_first

    return spans


def _merge_runt_tail(
    spans: list[tuple[int, int]],
    text: str,
    min_chunk_chars: int,
    chunk_size: int,
) -> list[tuple[int, int]]:
    """
    Fold too-short chunks into a neighbour.

    Step 0 measured a 78-character chunk whose content was almost entirely the
    overlap window repeating the previous chunk. It carried no new meaning, yet
    it competed in every similarity search — and actually ranked FIRST for an
    unrelated question. Merging it away removes that noise.

    A runt can appear anywhere a very long unit (a collapsed table, a run-on
    line) sits next to a few tiny ones, not only at the end of a page, so every
    span is checked. Prefer merging backwards; fall back to forwards. Allow the
    merged chunk to exceed chunk_size somewhat — one slightly long chunk is
    better than one useless one.
    """
    limit = chunk_size * 1.5
    merged: list[tuple[int, int]] = []
    pending_runt: tuple[int, int] | None = None

    for start, end in spans:
        if pending_runt is not None:
            # Try to attach the previous runt to the FRONT of this span.
            if end - pending_runt[0] <= limit:
                start = pending_runt[0]
            else:
                merged.append(pending_runt)  # could not place it; keep as-is
            pending_runt = None

        if end - start >= min_chunk_chars:
            merged.append((start, end))
            continue

        # Runt: attach to the previous span if that stays within the limit.
        if merged and end - merged[-1][0] <= limit:
            merged[-1] = (merged[-1][0], end)
        else:
            pending_runt = (start, end)

    if pending_runt is not None:
        merged.append(pending_runt)

    return merged


# --------------------------------------------------------------------------
# Section headings
# --------------------------------------------------------------------------


def _heading_positions(text: str, heading_test=is_heading) -> list[tuple[int, str]]:
    """(start offset, heading text) for every block the predicate accepts."""
    return [
        (start, text[start:end])
        for start, end in _block_spans(text)
        if heading_test(text[start:end])
    ]


def _section_for(headings: list[tuple[int, str]], chunk_start: int) -> str | None:
    """
    The nearest heading at or before this chunk's start.

    Returns None when no heading precedes the chunk. We would rather report no
    section than guess one — a wrong section label produces a citation that
    looks verifiable and is not.
    """
    section = None
    for start, heading in headings:
        if start <= chunk_start:
            section = heading
        else:
            break
    return section


def _link_neighbours(chunks: list[dict]) -> list[dict]:
    """
    Add prev/next chunk IDs and the total count.

    Why a second pass: a chunk cannot know its successor until the whole
    document has been chunked. These links are what make Step 3's neighbour
    expansion possible — they turn a flat list of chunks into a chain that can
    be walked forwards and backwards in reading order.

    Neighbours never cross a document boundary, because chunk_ids restart per
    ingested document and we only ever link within this call's chunk list.
    """
    total = len(chunks)
    for index, chunk in enumerate(chunks):
        chunk["prev_chunk_id"] = chunks[index - 1]["chunk_id"] if index > 0 else None
        chunk["next_chunk_id"] = (
            chunks[index + 1]["chunk_id"] if index < total - 1 else None
        )
        chunk["total_chunks"] = total
    return chunks
