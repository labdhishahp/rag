"""
EXPERIMENTAL recursive chunker. Not wired into the application.

Lives outside backend/src/ on purpose: nothing in the running system imports
it, so measuring it cannot change production behaviour.

------------------------------------------------------------------------------
WHAT IS DIFFERENT FROM backend/src/chunker.py
------------------------------------------------------------------------------
The production chunker treats every SENTENCE as the indivisible packing unit.
It splits each block into sentences up front, then greedily packs sentences
until the next one would not fit. A chunk boundary therefore lands between two
sentences, wherever that happens to fall relative to the document's own
structure.

This chunker uses the classic recursive strategy instead: try a hierarchy of
separators, largest first, and only descend when a piece is still too big.

    "\n\n"  paragraph break
    "\n"    block break (our cleaner emits one block per line)
    ". "    sentence
    "? "    sentence
    "! "    sentence
    " "     word
    ""      hard split (last resort)

The consequence, and the thing worth measuring: a block that already fits in
chunk_size is kept WHOLE and never divided into sentences. Descent to sentence
level happens only for blocks that overflow. Boundaries therefore prefer the
document's paragraph structure, and fall back to sentences only under pressure.

------------------------------------------------------------------------------
WHAT IS DELIBERATELY IDENTICAL
------------------------------------------------------------------------------
Everything else, so that an A/B result is attributable to the split strategy
and nothing else. This module imports the production helpers rather than
reimplementing them:

    page_text_for_chunking   same input text, so char offsets are comparable
    _page_heading_test       same heading detection
    _heading_positions       same section assignment
    _section_for
    _merge_runt_tail         same runt handling
    _link_neighbours         same prev/next chain

Output schema is byte-identical to chunk_pages(), including exact offsets:
text[char_start:char_end] IS the chunk text.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "backend" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chunker import (  # noqa: E402
    DEFAULT_MIN_CHUNK_CHARS,
    _heading_positions,
    _link_neighbours,
    _merge_runt_tail,
    _page_heading_test,
    _section_for,
    page_text_for_chunking,
)

# Largest structural separator first. Descent stops as soon as a piece fits.
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]


def _recursive_spans(
    text: str,
    start: int,
    end: int,
    chunk_size: int,
    separators: list[str],
) -> list[tuple[int, int]]:
    """
    Split text[start:end] into pieces no longer than chunk_size.

    Returns spans in absolute coordinates and in document order. Spans are
    contiguous-or-gapped exactly as the separators fall: a separator's own
    characters stay attached to the piece that precedes them, so joining
    consecutive spans reconstructs the source without loss.
    """
    if end - start <= chunk_size:
        return [(start, end)]

    segment = text[start:end]

    # Pick the first separator that actually occurs, so we always descend the
    # hierarchy rather than skipping levels that could have done the job.
    for depth, sep in enumerate(separators):
        if sep == "":
            # Nothing left to split on: hard-cut on width. Only reachable for a
            # single token longer than chunk_size.
            return [
                (cursor, min(cursor + chunk_size, end))
                for cursor in range(start, end, chunk_size)
            ]
        if sep not in segment:
            continue

        rest = separators[depth + 1:]
        spans: list[tuple[int, int]] = []
        cursor = start
        # str.split would lose offsets, so walk the separator positions.
        while cursor < end:
            hit = text.find(sep, cursor, end)
            if hit == -1:
                piece_end = end
            else:
                # Keep the separator with the preceding piece: that is what
                # makes the spans reconstruct the original text.
                piece_end = hit + len(sep)
            if piece_end > cursor:
                if piece_end - cursor > chunk_size:
                    spans.extend(_recursive_spans(text, cursor, piece_end, chunk_size, rest))
                else:
                    spans.append((cursor, piece_end))
            cursor = piece_end
            if hit == -1:
                break
        return spans

    return [(start, end)]


def _merge_pieces(
    pieces: list[tuple[int, int]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[int, int]]:
    """
    Merge adjacent pieces into chunks up to chunk_size, then carry overlap.

    Overlap is taken by stepping back over WHOLE pieces while they fit in the
    budget — the same discipline the production packer uses, so neither variant
    can start a chunk mid-word. That keeps the comparison about where
    boundaries fall, not about one side emitting broken text.
    """
    if not pieces:
        return []

    spans: list[tuple[int, int]] = []
    first = 0

    while first < len(pieces):
        last = first
        while last + 1 < len(pieces):
            if pieces[last + 1][1] - pieces[first][0] > chunk_size:
                break
            last += 1

        spans.append((pieces[first][0], pieces[last][1]))

        if last + 1 >= len(pieces):
            break

        # Step back for overlap, but never so far that the next chunk cannot
        # hold at least one NEW piece — otherwise a long piece after short ones
        # produces chunks made purely of repeated text.
        next_first = last + 1
        while next_first - 1 > first:
            candidate = next_first - 1
            if pieces[last][1] - pieces[candidate][0] > chunk_overlap:
                break
            if pieces[last + 1][1] - pieces[candidate][0] > chunk_size:
                break
            next_first = candidate

        first = next_first

    return spans


def chunk_pages_recursive(
    pages: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    document_name: str | None = None,
    document_id: str | None = None,
    min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
) -> list[dict]:
    """Drop-in replacement for chunk_pages() using recursive separator descent."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[dict] = []
    chunk_id = 0
    current_section: str | None = None

    for page in pages:
        text = page_text_for_chunking(page)
        if not text:
            continue

        pieces = _recursive_spans(text, 0, len(text), chunk_size, SEPARATORS)
        # Trailing separator characters would otherwise show up as leading
        # whitespace on the next chunk and shift every offset by a space.
        pieces = [(s, e) for s, e in pieces if text[s:e].strip()]
        if not pieces:
            continue

        spans = _merge_pieces(pieces, chunk_size, chunk_overlap)
        spans = _merge_runt_tail(spans, text, min_chunk_chars, chunk_size)
        headings = _heading_positions(text, _page_heading_test(page))

        for position_in_page, (start, end) in enumerate(spans):
            # Trim trailing separator whitespace so char_end lands on content,
            # matching the production chunker's behaviour.
            while end > start and text[end - 1].isspace():
                end -= 1
            section_here = _section_for(headings, start)
            if section_here is not None:
                current_section = section_here
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": text[start:end],
                    "document_name": document_name,
                    "document_id": document_id,
                    "page_number": page["page_number"],
                    "page_label": page.get("page_label", "page"),
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
