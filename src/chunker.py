"""
Split document text into overlapping chunks.

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
  The last N characters of one chunk repeat at the start of the next.
  This prevents sentences or facts from being cut in half at a boundary,
  so important text is not split across two weak partial matches.

Too small: fragments lose meaning ("revenue was" without the number).
Too large: chunks mix unrelated topics; similarity scores become vague.
"""

from document_loader import clean_text


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    """
    Split page text into chunks with metadata.

    Input:  pages from load_pdf() — [{"page_number": 1, "text": "..."}, ...]
    Output: chunks — [{"chunk_id": 0, "page_number": 1, "text": "..."}, ...]

    We chunk within each page first, then assign global chunk IDs.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[dict] = []
    chunk_id = 0

    for page in pages:
        text = clean_text(page["text"])
        if not text:
            continue

        page_number = page["page_number"]
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page_number": page_number,
                        "text": chunk_text,
                    }
                )
                chunk_id += 1

            if end >= len(text):
                break

            # Move forward, but keep overlap from the previous chunk.
            start = end - chunk_overlap

    if not chunks:
        raise ValueError("No chunks created — all pages were empty after cleaning.")

    return chunks
