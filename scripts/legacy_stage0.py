"""
The Stage 0 pipeline, preserved so we can measure against it.

This is NOT used by the application. It is a frozen copy of how chunking and
context assembly worked before Steps 1-3, kept so that every claim of
improvement can be re-verified on demand rather than taken on trust.

Two pieces are reproduced:
  old_chunk_pages()    — character-window slicing on whitespace-flattened text
  old_format_context() — the flat "[Page N | similarity X]" evidence block
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from document_loader import clean_text  # noqa: E402


def old_chunk_pages(
    pages: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    """
    The pre-Step-2 chunker: slice every chunk_size characters.

    Measured consequences on our test document:
      - chunks that begin mid-word ("ariables are defined as follows")
      - 4 of 9 chunks ending mid-sentence
      - a 78-character runt that was almost entirely duplicated overlap
    """
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

            start = end - chunk_overlap

    if not chunks:
        raise ValueError("No chunks created — all pages were empty after cleaning.")

    return chunks


def old_build_rag_prompt(question: str, context: str, low_confidence: bool = False) -> str:
    """
    The pre-Phase-3 prompt, verbatim. Its unconditional rule 4 ("directly and
    concisely") is why a worked example that WAS in the context never reached
    the answer; rule 5 is why citations were impossible.
    """
    confidence_note = ""
    if low_confidence:
        confidence_note = (
            "\nIMPORTANT: The retrieved passages may NOT be relevant to this question "
            "(similarity scores were low). If the context does not contain the answer, "
            "you MUST say the information is not available in the document.\n"
        )
    return f"""You are a document Q&A assistant. Answer the user's question using ONLY the document context below.

Rules:
1. Use ONLY facts from the DOCUMENT CONTEXT. Do not use outside knowledge.
2. If the answer is not in the DOCUMENT CONTEXT, say clearly: "This information is not available in the document."
3. Do not invent or guess numbers, names, dates, or facts.
4. Answer the question directly and concisely.
5. Do not mention "the context" or "the document" unless explaining missing information.
{confidence_note}
DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:"""


def old_format_context(chunks: list[dict]) -> str:
    """
    The pre-Step-3 evidence block.

    No expansion, no merging, no overlap removal, no budget: whatever the top-k
    similarity search returned, in score order.
    """
    if not chunks:
        return "(No document passages were retrieved.)"

    parts = []
    for chunk in chunks:
        header = f"[Page {chunk['page_number']} | similarity {chunk['similarity']:.3f}]"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(parts)
