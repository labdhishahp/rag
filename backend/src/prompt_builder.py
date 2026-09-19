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

------------------------------------------------------------------------------
WHAT CHANGED IN PHASE 3, AND WHY
------------------------------------------------------------------------------
The previous prompt had one fixed rule: "Answer the question directly and
concisely." Measured effect: asked to explain the compound interest formula in
detail, with the worked example ALREADY IN THE CONTEXT, the model left the
example out. Retrieval had done its job; the instruction told the model to stop
early. It also forbade mentioning "the document", which made citations
impossible.

Now the prompt is assembled from the user's request:

  depth        brief / normal / detailed  ->  how much to say, and whether to
                                              actively pull in definitions,
                                              mechanisms and worked examples
                                              that are present in the evidence
  citations    every passage is labelled [S1], [S2]... and the model is asked
               to cite them inline. Citations are checked afterwards against
               the labels that actually exist.
  evidence     weak evidence -> an explicit caution; no evidence never reaches
               the LLM at all (rag.py declines deterministically).
"""

REFUSAL_TEXT = (
    "I couldn't find enough information in the provided documents to answer that reliably."
)

_DEPTH_INSTRUCTIONS = {
    "brief": (
        "The user wants a SHORT answer. Give the direct answer in one or two sentences "
        "(or the bare formula/value if that is what was asked). Do not add explanation, "
        "background, or examples unless the question cannot be answered without them."
    ),
    "normal": (
        "Answer the question directly, then add the explanation needed to understand the "
        "answer. Use the evidence's own definitions where they exist. Keep it focused."
    ),
    "detailed": (
        "The user wants a DETAILED explanation. Use ALL relevant evidence provided. If the "
        "evidence contains definitions of terms or variables, the mechanism of how something "
        "works, assumptions, or a worked example, INCLUDE them — do not summarise them away. "
        "Structure the answer with short headings such as: the answer/formula itself, what it "
        "means, definitions, how it works, example, caveats — using only the parts the "
        "evidence supports."
    ),
}

_EXAMPLE_INSTRUCTION = (
    "The user asked for an example. If the evidence contains a worked example, reproduce it "
    "with its numbers. If it does not, say so explicitly rather than inventing one."
)

_WEAK_EVIDENCE_NOTE = (
    "CAUTION: retrieval confidence is low; the passages below may not contain the answer. "
    "If they do not directly answer the question, say so instead of guessing."
)


def build_summary_prompt(question, document_name, sections, evidence,
                        depth="normal", coverage=None) -> str:
    """
    Summary: the evidence is a SKELETON of the whole document in reading order —
    the opening of every section — not the chunks most similar to the word
    "summary".

    The outline is passed separately from the evidence so the model knows the
    document's true shape even where the evidence is thin, and so it cannot
    invent a section that does not exist.

    coverage — (shown, total) sections when the document was too large to
    represent every section; the model is told to say so rather than imply the
    summary is complete.
    """
    outline = "\n".join(f"- {s}" for s in sections[:60]) if sections else "(no headings detected)"
    length = {
        "brief": "Three to five sentences.",
        "normal": "One short paragraph on the purpose, then the main points as a bulleted list (one per major section).",
        "detailed": "A paragraph on the purpose, then a section-by-section account of the key points.",
    }.get(depth, "One short paragraph on the purpose, then the main points as a bulleted list.")
    coverage_note = ""
    if coverage and coverage[0] < coverage[1]:
        coverage_note = (
            f"\nNOTE: this document has {coverage[1]} sections and the evidence below samples "
            f"{coverage[0]} of them, spread across the document. Say plainly that the summary is "
            "based on a sample; do not imply it covers everything.\n"
        )
    return f"""You are a knowledge assistant summarising a document using ONLY the evidence below.

The document is "{document_name}". Its sections, in order:
{outline}

Rules:
1. Summarise ONLY what the EVIDENCE contains. Do not add background knowledge about the topic.
2. Cite the passage each point comes from, e.g. [S3].
3. Do not invent sections, numbers, or conclusions that are not in the evidence.
4. The evidence is the OPENING of each section, not its full text. Summarise what is there; do not guess how a section continues.
{coverage_note}
Length: {length}

EVIDENCE (section openings, in reading order):
{evidence}

USER REQUEST:
{question}

SUMMARY:"""


def build_compare_prompt(question, subject_a, subject_b, evidence, missing=None,
                         depth="normal", conversation=None) -> str:
    """
    Comparison: two labelled evidence sets, one structured answer.

    Evidence arrives as "=== A: <subject> ===" with [A#] labels and
    "=== B: <subject> ===" with [B#] labels, so the model attributes each claim
    to the right side — and rag.check_citations verifies it against that side's
    labels only.
    """
    missing_note = ""
    if missing:
        missing_note = (
            f"\nNOTE: no relevant evidence was found for: {', '.join(missing)}. Say so plainly for "
            "that side and do not invent anything about it; still describe the other side from its evidence.\n"
        )
    length = {
        "brief": "Keep it short: the two or three most important differences.",
        "normal": "Cover the main differences and any stated similarities.",
        "detailed": "Be thorough: definitions, mechanisms, and every difference the evidence supports.",
    }.get(depth, "Cover the main differences and any stated similarities.")
    conv = f"\nCONVERSATION SO FAR (for reference resolution only, not evidence):\n{conversation}\n" if conversation else ""
    return f"""You are a knowledge assistant comparing two subjects using ONLY the evidence below.

Rules:
1. Use ONLY facts from the EVIDENCE. No outside knowledge, no assumptions.
2. Cite inline with the passage labels ([A1], [B2]...). Facts about "{subject_a}" must cite A-labels; facts about "{subject_b}" must cite B-labels.
3. If the evidence does not support a claimed difference, do not state it. If a side has no evidence, say so.
4. Never invent numbers, names, formulas or facts.
{missing_note}
Answer structure:
**{subject_a}** — what the evidence says (with citations)
**{subject_b}** — what the evidence says (with citations)
**Key differences** — point by point
**Similarities** — only if the evidence supports any
{length}
{conv}
EVIDENCE:
{evidence}

USER QUESTION:
{question}

ANSWER:"""


_CONVERSATION_RULE = (
    "7. The CONVERSATION SO FAR is there so you can resolve references like \"it\", \"that\" or "
    "\"the previous one\" in the user's latest message. It is NOT evidence: never cite it and "
    "never treat something the assistant said earlier as a fact from the documents."
)


def build_rag_prompt(
    question: str,
    context: str,
    low_confidence: bool = False,
    depth: str = "normal",
    wants_example: bool = False,
    conversation: str | None = None,
) -> str:
    """
    Build the full prompt for the LLM.

    Input:  user question, formatted evidence block ([S#]-labelled passages),
            low-confidence flag, requested depth, example flag, and — for
            follow-up turns — the recent conversation as plain text
    Output: single string prompt

    Conversation and evidence are separate blocks on purpose. The model needs
    the conversation to know what "it" means; it must not mistake the
    conversation for a source. See conversation.py.
    """
    depth_instruction = _DEPTH_INSTRUCTIONS.get(depth, _DEPTH_INSTRUCTIONS["normal"])
    example_note = f"\n{_EXAMPLE_INSTRUCTION}\n" if wants_example else ""
    confidence_note = f"\n{_WEAK_EVIDENCE_NOTE}\n" if low_confidence else ""
    conversation_rule = f"\n{_CONVERSATION_RULE}" if conversation else ""
    conversation_block = (
        f"\nCONVERSATION SO FAR (for reference resolution only, not evidence):\n{conversation}\n"
        if conversation
        else ""
    )

    return f"""You are a knowledge assistant answering questions about the user's documents. Answer using ONLY the evidence passages below.

Rules:
1. Use ONLY facts from the EVIDENCE. Do not use outside knowledge and do not fill gaps with assumptions.
2. If the evidence does not contain the answer, reply exactly: "{REFUSAL_TEXT}" You may add one sentence saying what the evidence does cover, if that is useful.
3. Never invent or guess numbers, names, dates, formulas, or facts. If a detail is not in the evidence, leave it out rather than approximate it.
4. Cite your sources inline using the passage labels, e.g. "...compounded monthly [S2]." Cite the passage a fact actually came from. Use only labels that appear below.
5. Passages marked (matched) are the ones retrieval matched to the question; passages marked (surrounding context) are their neighbours in the document, included because they often hold the definitions or examples that go with a match.
6. Do not describe the passages or narrate what you are doing ("the context says..."). Just answer, with citations.{conversation_rule}

Answer style: {depth_instruction}
{example_note}{confidence_note}{conversation_block}
EVIDENCE:
{context}

USER QUESTION:
{question}

ANSWER:"""
