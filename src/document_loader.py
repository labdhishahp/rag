"""
Load a PDF and extract text page by page.

Why extract text?
  PDFs store layout, fonts, and images — not plain text. We need the actual
  words so we can chunk and embed them for search.

Why keep page numbers?
  When retrieval returns a chunk, knowing its page helps you verify the result
  against the original document and is useful later for citations.
"""

from pathlib import Path
from typing import Union

import fitz  # PyMuPDF


def load_pdf(pdf_path: Union[str, Path]) -> list[dict]:
    """
    Extract text from every page of a PDF.

    Returns a list of page records:
        [{"page_number": 1, "text": "..."}, ...]

    Page numbers are 1-based (page 1 = first page), which matches how humans
    read PDFs.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: list[dict] = []

    with fitz.open(pdf_path) as doc:
        if doc.page_count == 0:
            raise ValueError(f"PDF has no pages: {pdf_path}")

        for page_index in range(doc.page_count):
            page = doc[page_index]
            text = page.get_text("text").strip()
            pages.append(
                {
                    "page_number": page_index + 1,
                    "text": text,
                }
            )

    if not any(page["text"] for page in pages):
        raise ValueError(
            f"No extractable text found in {pdf_path}. "
            "The PDF may be scanned images only (needs OCR) or empty."
        )

    return pages


def load_pdf_from_bytes(pdf_bytes: bytes) -> list[dict]:
    """
    Extract text from a PDF provided as raw bytes (e.g. Streamlit file upload).

    Same output format as load_pdf().
    """
    if not pdf_bytes:
        raise ValueError("PDF bytes are empty.")

    pages: list[dict] = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.page_count == 0:
            raise ValueError("PDF has no pages.")

        for page_index in range(doc.page_count):
            page = doc[page_index]
            text = page.get_text("text").strip()
            pages.append(
                {
                    "page_number": page_index + 1,
                    "text": text,
                }
            )

    if not any(page["text"] for page in pages):
        raise ValueError(
            "No extractable text found in uploaded PDF. "
            "It may be scanned images only (needs OCR) or empty."
        )

    return pages


def clean_text(text: str) -> str:
    """
    Basic text cleanup before chunking.

    - Collapse runs of whitespace (PDF extractors often add extra newlines).
    - Strip leading/trailing space.

    We keep this minimal on purpose — aggressive cleaning can remove useful
    structure or punctuation that helps embeddings.
    """
    return " ".join(text.split())
