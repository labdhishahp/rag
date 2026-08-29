"""
Load documents and extract text into a common page-based format.

All loaders return the same structure:
    [{"page_number": 1, "text": "..."}, ...]

PDF  → one record per PDF page
DOCX → paragraphs grouped into pseudo-pages (~1500 chars each)

Why keep page numbers?
  When retrieval returns a chunk, knowing its page/section helps you verify
  the result and display source citations.
"""

import io
import re
from pathlib import Path
from typing import Union

import fitz  # PyMuPDF

from pdf_layout import extract_pages

# Target size for grouping DOCX paragraphs into pseudo-pages.
_DOCX_PAGE_TARGET_CHARS = 1500

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def load_pdf(pdf_path: Union[str, Path]) -> list[dict]:    #Create a function called load_pdf that accepts a file path and returns a list of dictionaries.
    """Extract text from every page of a PDF file on disk."""
    pdf_path = Path(pdf_path)   ## convert the file path to a Path object.- path ma convert karse
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with open(pdf_path, "rb") as f:    #open means open the file in binary mode.,  rb--read in binary mode.
        return load_pdf_from_bytes(f.read())
 

def load_pdf_from_bytes(pdf_bytes: bytes) -> list[dict]:   #this creates func that accepts raw Pdf bytes  than file name  (why not filename- as u dont have to save it fitst)
    """
    Extract text from a PDF provided as raw bytes.

    Delegates to pdf_layout.extract_pages, which reads the PDF's own layout
    (blocks, font sizes, positions) instead of guessing structure from
    character counts. Each returned page carries:
        page_number   real, verifiable PDF page number
        page_label    "page"
        text          one paragraph/heading/table per line, wraps already joined
        headings      block strings judged to be section headings
        structured    True — the chunker must not re-derive the layout
    """
    if not pdf_bytes:    #agar bytes empty hai to error throw karega.
        raise ValueError("PDF bytes are empty.")

    return extract_pages(pdf_bytes)


def load_pdf_raw_text(pdf_bytes: bytes) -> list[dict]:
    """
    The pre-Step-2b extraction: plain text per page, newline at every visual
    line break, no layout analysis. Kept so the old and new pipelines can be
    compared on the same document (scripts/legacy_stage0.py).
    """
    if not pdf_bytes:
        raise ValueError("PDF bytes are empty.")

    pages: list[dict] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.page_count == 0:
            raise ValueError("PDF has no pages.")
        for page_index in range(doc.page_count):
            text = doc[page_index].get_text("text").strip()
            pages.append({"page_number": page_index + 1, "page_label": "page", "text": text})

    if not any(page["text"] for page in pages):
        raise ValueError("No extractable text found in the PDF.")
    return pages


def load_docx_from_bytes(docx_bytes: bytes) -> list[dict]:
    """
    Extract text from a Word document (.docx).

    Word files do not have fixed pages like PDFs, so we group paragraphs
    into pseudo-pages of roughly _DOCX_PAGE_TARGET_CHARS characters each.
    Page numbers are section indices (1, 2, 3...) for citation purposes.
    """
    if not docx_bytes:
        raise ValueError("DOCX bytes are empty.")

    try:
        from docx import Document
    except ImportError as exc:
        raise ImportError(
            "python-docx is required for .docx files. Install with: pip install python-docx"
        ) from exc

    doc = Document(io.BytesIO(docx_bytes))  #docx_bytes are raw bytes. ,,,io.BytesIO(...) makes them behave like a file.

    # Word documents carry their structure explicitly: every paragraph has a
    # style, and headings use "Heading 1", "Heading 2", "Title", ... So unlike
    # PDFs we never need to guess what a heading is — we read the style name.
    paragraphs: list[str] = []
    headings: set[str] = set()
    for para in doc.paragraphs:
        text = " ".join(para.text.split())   #Get paragraph text and collapse whitespace.
        if not text:
            continue
        paragraphs.append(text)
        style = (para.style.name if para.style is not None else "") or ""
        if style.lower().startswith(("heading", "title")):
            headings.add(text)

    for table in doc.tables:     #Take all the non-empty cells in this row, clean their text, and join them together with | between them.
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                paragraphs.append(" ".join(row_text.split()))

    if not paragraphs:
        raise ValueError(
            "No extractable text found in the Word document. "
            "The file may be empty or contain only images."
        )

    pages: list[dict] = []   #completed pseudo-pages
    page_num = 1      #current pseudo-page number
    buffer: list[str] = []    #paragraphs currently being collected
    char_count = 0            #how many characters we've collected so far.

    def close_page() -> None:
        pages.append(
            {
                "page_number": page_num,
                # NOT a real page. A .docx has no fixed pagination, so this
                # number is our own invention. Labelling it "part" keeps
                # citations honest — we must never tell the user "page 3"
                # when no such page exists in their file. ("part", not
                # "section", so it cannot be confused with the heading-based
                # section metadata on each chunk.)
                "page_label": "part",
                "text": "\n".join(buffer),
                "headings": [p for p in buffer if p in headings],
                "structured": True,
            }
        )

    for para in paragraphs:    #Take paragraphs one at a time.
        buffer.append(para)     #Put the paragraph into the current pseudo-page.
        char_count += len(para)     #Count how many characters we've collected.
        if char_count >= _DOCX_PAGE_TARGET_CHARS:   #So when we reach approximately 1500 characters: Close this pseudo-page.
            close_page()
            page_num += 1  #Move to the next pseudo-page.
            buffer = []
            char_count = 0  #Reset the character count for the next pseudo-page.

    if buffer:    #If there are any remaining paragraphs that didn't fit in the LAST pseudo-page:
        close_page()

    return pages


def load_document_from_bytes(file_bytes: bytes, filename: str) -> list[dict]:
    """
    Route uploaded bytes to the correct loader based on file extension.

    Input:  raw file bytes + original filename (e.g. "report.pdf")
    Output: [{"page_number": 1, "text": "..."}, ...]
    """
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return load_pdf_from_bytes(file_bytes)
    if ext == ".docx":
        return load_docx_from_bytes(file_bytes)

    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise ValueError(
        f"Unsupported file type '{ext or '(none)'}'. Supported types: {supported}"
    )


def clean_text(text: str) -> str:
    """
    Collapse ALL whitespace, including newlines, into single spaces.

    Kept for the Phase 1 CLI and for comparing against the old chunker.
    New code should prefer clean_text_structured(), which keeps the line
    structure that headings, formulas and list items depend on.
    """
    return " ".join(text.split())


# --------------------------------------------------------------------------
# Structure-preserving cleaning
# --------------------------------------------------------------------------
#
# The problem clean_text() causes:
#   PyMuPDF's "text" extraction emits a newline at every VISUAL line break.
#   A wrapped paragraph therefore arrives as several lines, and a heading, a
#   standalone formula, and each item of a list also arrive as their own lines.
#   clean_text() flattens all of them into one long line, so by the time the
#   chunker runs there is no way to tell a heading from mid-sentence text.
#
# What we cannot rely on:
#   Blank lines. Many PDFs (including our own test document) contain none at
#   all — verified by inspecting the raw extraction. So paragraph detection
#   cannot be based on blank lines.
#
# The signal we use instead — three questions per line:
#   1. Is the line FULL?  A wrapped line runs close to the page's widest line.
#      A line much shorter than that ended early on purpose.
#   2. Does it end with sentence-terminal punctuation?  Then it is complete.
#   3. Is it (or the next line) a list item?  List items own their own line.
#
# If a line is full, unpunctuated, and not part of a list, the newline after it
# is a mere wrap and we join it to the next line with a space. Otherwise the
# newline is meaningful and we keep it.

# A list item: "P = the principal", "1. First", "- bullet", "iv) point".
_LIST_ITEM = re.compile(
    r"^\s*(?:[-*•·]|\(?\d{1,3}[.)\]]|\(?[A-Za-z][.)\]]|[A-Za-z]\s*=)\s"
)

_SENTENCE_END = (".", "!", "?", ":", ";")

# A line shorter than this fraction of the page's widest line ended on purpose.
_FULL_LINE_RATIO = 0.75

# Below this width we cannot infer a wrap column, so keep every newline.
_MIN_WRAP_WIDTH = 40

# A heading is short, unpunctuated, and few words.
_HEADING_MAX_CHARS = 60
_HEADING_MAX_WORDS = 8


def _is_list_item(line: str) -> bool:
    return bool(_LIST_ITEM.match(line))


# A heading is prose. This much of it must be letters or spaces.
# Rejects formulas and symbol-heavy lines that pass every other test.
_HEADING_MIN_ALPHA_RATIO = 0.80


def is_heading(line: str) -> bool:
    """
    Does this line look like a section heading?

    Conservative on purpose. A false heading attaches a wrong section label to
    every chunk beneath it, which is worse than having no label at all — a
    citation that looks verifiable and is not.

    Why the equals-sign and alpha-ratio rules exist:
      A standalone formula line looks exactly like a heading by every other
      measure: short, no terminal punctuation, few words, starts with a letter.
      Our own test caught "PV = FV / (1 + r)^n" being labelled a section.
      The _is_list_item() check happened to reject "A = P(1 + r/n)^(nt)" (single
      letter, then '='), but not the two-letter "PV =" — an inconsistency that
      only a content-based rule fixes properly.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > _HEADING_MAX_CHARS:
        return False
    if stripped.endswith(_SENTENCE_END):
        return False
    if _is_list_item(stripped):
        return False
    if len(stripped.split()) > _HEADING_MAX_WORDS:
        return False
    # Headings do not assign or compute anything.
    if "=" in stripped:
        return False
    # Headings are mostly words, not symbols, digits or operators.
    alpha_like = sum(1 for ch in stripped if ch.isalpha() or ch.isspace())
    if alpha_like / len(stripped) < _HEADING_MIN_ALPHA_RATIO:
        return False
    # Headings start with a letter or digit, not punctuation or a symbol.
    return stripped[0].isalnum()


def clean_text_structured(text: str) -> str:
    """
    Normalise whitespace while KEEPING structurally meaningful line breaks.

    Input:  raw page text from PyMuPDF (newline at every visual line break)
    Output: text where '\\n' appears only at real structural boundaries

    Guarantees:
      - no leading/trailing whitespace on any line
      - no runs of spaces or tabs inside a line
      - no blank lines
      - wrapped lines are rejoined with a single space
    """
    # Normalise line endings, squeeze intra-line whitespace, drop blank lines.
    lines = [" ".join(raw.split()) for raw in text.replace("\r\n", "\n").split("\n")]
    lines = [line for line in lines if line]

    if not lines:
        return ""

    widest = max(len(line) for line in lines)

    # Very narrow content (a title page, a short list): no wrap column to infer.
    if widest < _MIN_WRAP_WIDTH:
        return "\n".join(lines)

    full_line_threshold = widest * _FULL_LINE_RATIO

    out: list[str] = [lines[0]]
    for index in range(1, len(lines)):
        previous, current = lines[index - 1], lines[index]

        wrapped = (
            len(previous) >= full_line_threshold      # previous line ran to the margin
            and not previous.endswith(_SENTENCE_END)  # ...and was left unfinished
            and not _is_list_item(previous)           # list items end their own line
            and not _is_list_item(current)            # a new item starts a new line
            and not is_heading(current)               # a heading starts a new line
        )

        if wrapped:
            out[-1] = f"{out[-1]} {current}"
        else:
            out.append(current)

    return "\n".join(out)
