"""
Layout-aware PDF extraction using the geometry PyMuPDF already provides.

------------------------------------------------------------------------------
WHY THIS MODULE EXISTS
------------------------------------------------------------------------------
Step 2 originally inferred document structure from CHARACTER COUNTS: a line
was "wrapped" if it was at least 75% as long as the widest line on the page.
On real documents that failed hard (see scripts/validate_real_docs.py):

    two-column survey paper   779 "headings" on 21 pages   (true count ~30)
    section labels attached   'https://towardsdatascience.com/', 'preprint', 'L'
    NIST report               running header became the section of 27% of chunks
    hyphenated wraps          'impres- sive', 'intrin- sic'  (105 per paper)

The cause was structural: any page containing ONE wide line (an author list, a
figure caption, a table) pushed the threshold above every body line, so no
wrap was ever joined, and the heading detector then saw hundreds of short,
unpunctuated fragments.

The fix is to stop guessing. A PDF already contains its layout, and
page.get_text("dict") exposes it:

    block   = a paragraph-like unit the renderer grouped   -> paragraph boundary
    line    = one visual line inside a block               -> a WRAP, by construction
    span    = a run of text with one font                  -> size + bold flag
    bbox    = position on the page                         -> header/footer/footnote zones
    dir     = writing direction                            -> rotated margin text

Analogy: we were measuring shadows to guess the shape of a building whose
blueprint was in our hands the whole time.

------------------------------------------------------------------------------
WHAT THIS MODULE DOES, IN ORDER
------------------------------------------------------------------------------
  1. Read every text block on every page, with font size, boldness, bbox.
  2. Drop rotated lines (vertical watermarks, sideways footers).
  3. Join the lines of each block into one paragraph, de-hyphenating wraps.
  4. Find the document's BODY font size (char-weighted mode across pages).
  5. Remove running headers/footers: short text recurring on many pages.
  6. Remove bare page numbers near the top/bottom edge.
  7. Merge drop caps ("L" + "ARGE language models" -> "LARGE language models").
  8. Collapse table-like runs of tiny blocks into a single pipe-joined block.
  9. Move footnotes (small font, low on the page) to the end of the page.
 10. Decide which blocks are headings: short + (bold | larger | numbered).
 11. Emit per page: text (one block per line) and the list of heading strings.

What it deliberately does NOT do:
  - Re-order columns. PyMuPDF's native block order already reads the left
    column then the right for LaTeX/Word two-column PDFs (verified on two
    papers). A geometric re-sort would risk misplacing full-width captions.
  - Parse tables into cells. PyMuPDF's find_tables() flagged every text column
    on every page as a table with strategy="text" and missed the borderless
    real table with strategy="lines". Collapsing table regions into one block
    is the honest, low-risk alternative for now.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

import pymupdf

# --- tunables --------------------------------------------------------------

BOLD_FLAG = 16  # PyMuPDF span flag bit for bold

# A block whose dominant font is this much larger than body text is a heading
# candidate regardless of boldness.
HEADING_SIZE_RATIO = 1.15

# Footnotes are noticeably smaller than body text and sit low on the page.
FOOTNOTE_SIZE_RATIO = 0.85
FOOTNOTE_MIN_Y_FRACTION = 0.70

# Page numbers live within this fraction of the page's top or bottom edge.
EDGE_FRACTION = 0.12

# Text recurring on at least this share of pages (and at least this many pages)
# is a running header/footer, not content.
RECURRENCE_MIN_SHARE = 0.30
RECURRENCE_MIN_PAGES = 3
RECURRENCE_MAX_CHARS = 120

# A heading is short.
HEADING_MAX_LINES = 2
HEADING_MAX_WORDS = 14
HEADING_MAX_CHARS = 120

# A table-like region is a run of many consecutive tiny blocks.
TABLE_RUN_MIN_BLOCKS = 12
TABLE_CELL_MAX_WORDS = 4

_PAGE_NUMBER = re.compile(r"^(\d{1,4}|[ivxlcdm]{1,6}|[IVXLCDM]{1,6})$")
_NUMBERED_HEADING = re.compile(
    r"^(?:\d{1,2}(?:\.\d{1,2}){0,3}\.?|[IVX]{1,5}\.|[A-Z]\.|[A-Z]\d?\))\s+\S"
)
_CAPTION = re.compile(r"^(fig\.?|figure|table)\s*[\divxlc]+", re.IGNORECASE)
_ROMAN_ONLY = re.compile(r"^[ivxlcdmIVXLCDM]+$")


@dataclass
class Block:
    text: str
    size: float
    bold: bool
    bbox: tuple[float, float, float, float]
    line_count: int
    is_table: bool = False
    is_footnote: bool = False
    is_heading: bool = False

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def y1(self) -> float:
        return self.bbox[3]


@dataclass
class LayoutPage:
    page_number: int
    height: float
    blocks: list[Block] = field(default_factory=list)


# --------------------------------------------------------------------------
# Step 1-3: read blocks, drop rotated lines, join wrapped lines
# --------------------------------------------------------------------------


def _join_wrapped_lines(lines: list[str]) -> str:
    """
    Join the visual lines of one block into a single paragraph.

    De-hyphenation: a line ending in '-' whose successor starts lowercase is a
    word broken at the margin ("impres-" + "sive" -> "impressive"). If the
    successor starts uppercase the hyphen is real ("Pre-" + "Training" ->
    "Pre-Training").
    """
    out = ""
    for line in lines:
        # NFKC folds typographic ligatures into plain letters: "ﬁne-tuning" ->
        # "fine-tuning", "classiﬁcation" -> "classification". Left alone, the
        # ligature is a different Unicode character, so the embedding model
        # sees an unknown token and exact-text matching silently fails.
        line = " ".join(unicodedata.normalize("NFKC", line).split())
        if not line:
            continue
        if not out:
            out = line
            continue
        if out.endswith("-") and line[0].islower():
            out = out[:-1] + line
        elif out.endswith("-") and line[0].isupper():
            out = out + line
        else:
            out = out + " " + line
    return out


def _read_blocks(page: pymupdf.Page) -> list[Block]:
    blocks: list[Block] = []
    for raw in page.get_text("dict")["blocks"]:
        if raw.get("type") != 0:
            continue  # image block
        lines_text: list[str] = []
        size_weight: Counter = Counter()
        bold_chars = 0
        total_chars = 0
        for line in raw["lines"]:
            dx, dy = line.get("dir", (1.0, 0.0))
            if abs(dy) > 0.1:
                continue  # rotated text: watermark, sideways footer
            text = "".join(span["text"] for span in line["spans"])
            if not text.strip():
                continue
            lines_text.append(text)
            for span in line["spans"]:
                n = len(span["text"])
                size_weight[round(span["size"], 1)] += n
                total_chars += n
                if span["flags"] & BOLD_FLAG:
                    bold_chars += n
        if not lines_text:
            continue
        text = _join_wrapped_lines(lines_text)
        if not text:
            continue
        size = size_weight.most_common(1)[0][0]
        bold = total_chars > 0 and bold_chars / total_chars >= 0.6
        blocks.append(
            Block(
                text=text,
                size=size,
                bold=bold,
                bbox=tuple(raw["bbox"]),
                line_count=len(lines_text),
            )
        )
    return blocks


# --------------------------------------------------------------------------
# Step 4-6: body size, running headers/footers, page numbers
# --------------------------------------------------------------------------


def _body_font_size(pages: list[LayoutPage]) -> float:
    weight: Counter = Counter()
    for page in pages:
        for block in page.blocks:
            weight[block.size] += len(block.text)
    return weight.most_common(1)[0][0] if weight else 10.0


def _normalise(text: str) -> str:
    # Page-specific digits vary ("Page 3" / "Page 4"); collapse them so the
    # recurring shape is detected.
    return re.sub(r"\d+", "#", text.strip().lower())


def _recurring_texts(pages: list[LayoutPage]) -> set[str]:
    if len(pages) < RECURRENCE_MIN_PAGES:
        return set()
    seen: Counter = Counter()
    for page in pages:
        page_texts = {
            _normalise(b.text) for b in page.blocks if len(b.text) <= RECURRENCE_MAX_CHARS
        }
        for t in page_texts:
            seen[t] += 1
    threshold = max(RECURRENCE_MIN_PAGES, int(len(pages) * RECURRENCE_MIN_SHARE))
    return {t for t, n in seen.items() if n >= threshold}


def _is_page_number(block: Block, page_height: float) -> bool:
    if not _PAGE_NUMBER.match(block.text.strip()):
        return False
    near_top = block.y0 <= page_height * EDGE_FRACTION
    near_bottom = block.y1 >= page_height * (1 - EDGE_FRACTION)
    return near_top or near_bottom


def _strip_boilerplate(pages: list[LayoutPage]) -> int:
    recurring = _recurring_texts(pages)
    removed = 0
    for page in pages:
        kept = []
        for block in page.blocks:
            if _normalise(block.text) in recurring or _is_page_number(block, page.height):
                removed += 1
                continue
            kept.append(block)
        page.blocks = kept
    return removed


# --------------------------------------------------------------------------
# Step 7: drop caps
# --------------------------------------------------------------------------


def _merge_drop_caps(page: LayoutPage) -> None:
    """
    A drop cap arrives as a lone capital letter, either as its own block or as
    the last line of the preceding block, followed by a block whose first word
    is the rest of the word in capitals ("ARGE language models").
    """
    blocks = page.blocks
    i = 0
    while i < len(blocks) - 1:
        cur, nxt = blocks[i], blocks[i + 1]
        words = cur.text.split()
        if words and len(words[-1]) == 1 and words[-1].isupper() and nxt.text[:1].isupper():
            letter = words[-1]
            remainder = " ".join(words[:-1])
            nxt.text = letter + nxt.text
            if remainder:
                cur.text = remainder
                i += 1
            else:
                del blocks[i]
            continue
        i += 1


# --------------------------------------------------------------------------
# Step 8: table-like regions
# --------------------------------------------------------------------------


def _collapse_table_runs(page: LayoutPage) -> None:
    """
    A borderless table extracts as dozens of consecutive tiny blocks (one per
    cell). Left alone they become chunks of "Text / Chunk / Once / Wikipedia"
    and each cell is a heading candidate. Collapse the run into one block,
    joined with ' | ', prefixed by the caption when one immediately precedes it.
    """
    blocks = page.blocks
    out: list[Block] = []
    i = 0
    while i < len(blocks):
        j = i
        while j < len(blocks) and len(blocks[j].text.split()) <= TABLE_CELL_MAX_WORDS:
            j += 1
        run = blocks[i:j]
        if len(run) >= TABLE_RUN_MIN_BLOCKS:
            caption = ""
            if out and _CAPTION.match(out[-1].text):
                caption = out.pop().text + ": "
            merged = Block(
                text=caption + " | ".join(b.text for b in run),
                size=run[0].size,
                bold=False,
                bbox=(
                    min(b.bbox[0] for b in run),
                    min(b.bbox[1] for b in run),
                    max(b.bbox[2] for b in run),
                    max(b.bbox[3] for b in run),
                ),
                line_count=len(run),
                is_table=True,
            )
            out.append(merged)
            i = j
        else:
            out.extend(run if run else [blocks[i]])
            i = j if run else i + 1
    page.blocks = out


# --------------------------------------------------------------------------
# Step 9: footnotes
# --------------------------------------------------------------------------


def _relocate_footnotes(page: LayoutPage, body_size: float) -> None:
    body: list[Block] = []
    notes: list[Block] = []
    for block in page.blocks:
        small = block.size < body_size * FOOTNOTE_SIZE_RATIO
        low = block.y0 >= page.height * FOOTNOTE_MIN_Y_FRACTION
        if small and low and not block.is_table:
            block.is_footnote = True
            notes.append(block)
        else:
            body.append(block)
    page.blocks = body + notes


# --------------------------------------------------------------------------
# Step 10: headings
# --------------------------------------------------------------------------


def _looks_like_heading_text(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 2 or len(stripped) > HEADING_MAX_CHARS:
        return False
    if len(stripped.split()) > HEADING_MAX_WORDS:
        return False
    if stripped.endswith((".", ",", ";", ":")) and not _NUMBERED_HEADING.match(stripped):
        # "1. Introduction" style is fine; "...as shown below:" is prose.
        if not (stripped.endswith(".") and _NUMBERED_HEADING.match(stripped)):
            return False
    if "://" in stripped or "@" in stripped or "=" in stripped:
        return False
    if _ROMAN_ONLY.match(stripped) or stripped.isdigit():
        return False
    if _CAPTION.match(stripped):
        return False
    alpha_like = sum(1 for ch in stripped if ch.isalpha() or ch.isspace())
    if alpha_like / len(stripped) < 0.6:
        return False
    return stripped[0].isalnum()


_TOC_LEADER = re.compile(r"\.{4,}|(?:\. ){4,}|(?:\.{2,}|\s\.\s)\s*\d{1,3}\s*$")
_LETTERED_HEADING = re.compile(r"^[A-Z]\.\s+\S")
_ALLCAPS_WORD = re.compile(r"^[A-Z]{6,}$")


def _mark_headings(page: LayoutPage, body_size: float, doc_page_counts: Counter) -> None:
    # A heading is unique. The same short text three times on a page is a
    # table column label; the same text on several pages is a figure legend or
    # a cover-page repeat. Real section headings appear exactly once.
    occurrences = Counter(b.text for b in page.blocks)

    for block in page.blocks:
        if block.is_table or block.is_footnote or block.line_count > HEADING_MAX_LINES:
            continue
        if not _looks_like_heading_text(block.text):
            continue
        if occurrences[block.text] > 1 or doc_page_counts[block.text] > 1:
            continue
        if _TOC_LEADER.search(block.text):
            continue  # "1.1 History of ..........  7" is a table-of-contents line
        larger = block.size >= body_size * HEADING_SIZE_RATIO
        numbered = bool(_NUMBERED_HEADING.match(block.text))
        # "A. Naive RAG" is a heading; "Z. Chen, H. Zhang, and L. Zhao. 2018." is
        # a reference entry. Initials-style lists contain commas or a year.
        if numbered and _LETTERED_HEADING.match(block.text):
            if "," in block.text or re.search(r"\b(19|20)\d{2}\b", block.text):
                numbered = False
        # Bold body-size text that is a full sentence is emphasis, not a heading;
        # the word/char caps above already exclude long ones.
        #
        # Third signal, for documents set in a single font (no size or bold
        # information at all): a one-line, 2-6 word, mostly-capitalised block
        # with no terminal punctuation. Safe now that wraps are joined — a
        # genuine paragraph is never that short — where the same rule applied
        # to raw lines produced hundreds of false headings.
        words = block.text.split()
        # Judge capitalisation on alphabetic words only: "Financial Highlights
        # — 2024" is a title even though the dash and the year are not capitals.
        alpha_words = [w for w in words if any(ch.isalpha() for ch in w)]
        capitalised = sum(1 for w in alpha_words if w[:1].isupper())
        plain_title = (
            block.line_count == 1
            and 2 <= len(words) <= 6
            and alpha_words
            and capitalised / len(alpha_words) >= 0.6
            and "," not in block.text
            and not block.text.rstrip().endswith((".", ",", ";", ":", "?", "!", "-"))
        )
        # A lone all-caps word of 6+ letters ("REFERENCES", "ABSTRACT",
        # "APPENDIX") is a section heading in small-caps journal styles.
        allcaps_word = len(words) == 1 and bool(_ALLCAPS_WORD.match(words[0]))
        # A lone Title-case word of 6+ letters on its own line, unique in the
        # document ("Sustainability", "Introduction"). Table cells are excluded
        # by the uniqueness checks above and the table-run collapse.
        single_title = (
            len(words) == 1
            and block.line_count == 1
            and words[0].isalpha()
            and len(words[0]) >= 6
            and words[0][0].isupper()
            and words[0][1:].islower()
        )
        if larger or block.bold or numbered or plain_title or allcaps_word or single_title:
            block.is_heading = True


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def extract_pages(pdf_bytes: bytes, debug: bool = False) -> list[dict]:
    """
    PDF bytes -> [{"page_number", "page_label", "text", "headings", "structured"}]

    "text" has exactly one block per line; wraps are already joined.
    "headings" lists the block strings judged to be section headings, in order.
    "structured": True tells the chunker not to re-guess the layout.
    """
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.page_count == 0:
            raise ValueError("PDF has no pages.")
        pages = [
            LayoutPage(page_number=i + 1, height=doc[i].rect.height, blocks=_read_blocks(doc[i]))
            for i in range(doc.page_count)
        ]

    body_size = _body_font_size(pages)
    removed = _strip_boilerplate(pages)
    for page in pages:
        _merge_drop_caps(page)
        _collapse_table_runs(page)
        _relocate_footnotes(page, body_size)

    # How many distinct pages each block text appears on (after boilerplate
    # removal). Used to reject repeated figure labels as headings.
    doc_page_counts: Counter = Counter()
    for page in pages:
        for text in {b.text for b in page.blocks}:
            doc_page_counts[text] += 1

    for page in pages:
        _mark_headings(page, body_size, doc_page_counts)

    if debug:
        total_heads = sum(b.is_heading for p in pages for b in p.blocks)
        print(f"\n=== PDF LAYOUT === pages={len(pages)} body_font={body_size} "
              f"boilerplate_blocks_removed={removed} headings={total_heads}")

    out: list[dict] = []
    for page in pages:
        lines = [b.text for b in page.blocks if b.text.strip()]
        out.append(
            {
                "page_number": page.page_number,
                "page_label": "page",
                "text": "\n".join(lines),
                "headings": [b.text for b in page.blocks if b.is_heading],
                "structured": True,
            }
        )

    if not any(p["text"] for p in out):
        raise ValueError(
            "No extractable text found in the PDF. "
            "It may be scanned images only (needs OCR) or empty."
        )
    return out
