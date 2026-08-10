"""
Build the RAG prompt sent to the LLM.

Why not send the entire document to the LLM?
  - Context window limits (models cannot read unlimited text).
  - Cost and latency grow with input size.
  - Irrelevant text increases the chance the model picks wrong information.

Why give retrieved chunks as "context"?
  Context = the specific passages we believe are relevant to the question.
  The LLM uses this as its source of truth instead of its general training memory.

Why separate DOCUMENT CONTEXT and USER QUESTION?
  Clear boundaries help the model know what is evidence vs what it must answer.
  Mixing them makes it easier for the model to confuse the question with facts
  or to treat its own assumptions as document content.
"""

from typing import List


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context block."""
    if not chunks:
        return "(No document passages were retrieved.)"

    parts: List[str] = []
    for chunk in chunks:
        header = f"[Page {chunk['page_number']} | similarity {chunk['similarity']:.3f}]"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(parts)


def build_rag_prompt(
    question: str,
    chunks: list[dict],
    low_confidence: bool = False,
) -> str:
    """
    Build the full prompt for the LLM.

    Input:  user question + retrieved chunks (+ optional low-confidence flag)
    Output: single string prompt
    """
    context = format_context(chunks)

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
