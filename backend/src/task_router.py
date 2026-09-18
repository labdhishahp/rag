"""
Which KIND of knowledge task is this message, and which documents does it concern?

Most questions are answered by the standard path (retrieve → expand → answer).
Two kinds need a DIFFERENT RETRIEVAL SHAPE, which is why they are routed
rather than handled by the prompt alone:

    compare     "Compare simple and compound interest."
                Two subjects → two retrievals → two labelled evidence sets.
                One retrieval for "simple vs compound interest" tends to
                return only the dominant subject.

    summarize   "Summarize this document."
                Similarity search cannot retrieve "the whole document". The
                evidence must be walked in reading order, section by section.

Everything else — direct, detailed, examples, evidence questions — is depth,
not task type, and is already handled by query_understanding.

Clarification is a guard, not a route: it fires when the message needs a
document we cannot identify (several documents loaded, none named, task needs
one) or names a document ambiguously.

All of this is deterministic pattern matching. Analogy: the front desk that
decides whether you need the reference librarian, the archivist who pulls two
files side by side, or the person who writes the executive summary — from the
words you used, before anyone goes to the shelves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_SUMMARIZE = re.compile(
    r"^\s*(please\s+)?(summari[sz]e|give (me )?(a |an )?(brief |short |quick )?(summary|overview)( of)?|"
    r"what (is|are) (this|the) (document|paper|report|file)s? (about|cover)|"
    r"(overview|summary) of (this|the) (document|paper|report|file)|"
    r"what does (this|the) (document|paper|report|file) (say|cover|discuss))",
    re.IGNORECASE,
)

_COMPARE_PATTERNS = (
    re.compile(r"\bcompare\s+(?:the\s+)?(.+?)\s+(?:and|with|to|vs\.?|versus|against)\s+(?:the\s+)?(.+?)[\s?.!]*$", re.IGNORECASE),
    re.compile(r"\b(?:what (?:is|are) the )?differences?\s+between\s+(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+?)[\s?.!]*$", re.IGNORECASE),
    re.compile(r"\bhow (?:does|do|is|are)\s+(?:the\s+)?(.+?)\s+(?:differ|different)\s+from\s+(?:the\s+)?(.+?)[\s?.!]*$", re.IGNORECASE),
    re.compile(r"^\s*(?:the\s+)?(.+?)\s+(?:vs\.?|versus)\s+(?:the\s+)?(.+?)[\s?.!]*$", re.IGNORECASE),
)

_DOC_WORDS = re.compile(r"\b(document|paper|report|file|handbook|guideline|policy|spec)\b", re.IGNORECASE)


@dataclass
class Task:
    kind: str                                  # "answer" | "compare" | "summarize" | "clarify"
    message: str
    parts: list[str] = field(default_factory=list)          # compare: [A, B]
    document_ids: set[str] | None = None                    # scope for retrieval (None = all)
    part_document_ids: list[set[str] | None] = field(default_factory=list)  # compare: per side
    target_document: object | None = None                   # summarize: DocumentInfo
    clarification: str | None = None
    reasons: list[str] = field(default_factory=list)        # why we routed this way (debug)

    def describe(self) -> str:
        bits = [self.kind]
        if self.parts:
            bits.append(f"parts={self.parts}")
        if self.document_ids:
            bits.append(f"docs={len(self.document_ids)}")
        return " ".join(bits)


def _clean_part(text: str) -> str:
    text = re.sub(r"^(the|a|an)\s+", "", text.strip(), flags=re.IGNORECASE)
    return text.strip(" ,;:")


def route(message: str, kb, conversation=None) -> Task:
    """Decide the task kind and document scope for this message."""
    msg = " ".join(message.split())
    named = kb.resolve_documents(msg) if kb is not None else []
    n_docs = len(kb.documents) if kb is not None else 0
    reasons: list[str] = []

    # ---- summarize ------------------------------------------------------
    if _SUMMARIZE.search(msg):
        reasons.append("summarize cue")
        if len(named) == 1:
            return Task("summarize", msg, target_document=named[0],
                        document_ids={named[0].document_id}, reasons=reasons + ["document named"])
        if len(named) > 1:
            names = ", ".join(d.name for d in named)
            return Task("clarify", msg, reasons=reasons + ["several documents matched"],
                        clarification=f"Which document should I summarise? You mentioned several: {names}.")
        if n_docs == 1:
            only = next(iter(kb.documents.values()))
            return Task("summarize", msg, target_document=only,
                        document_ids={only.document_id}, reasons=reasons + ["only one document"])
        if n_docs == 0:
            return Task("clarify", msg, clarification="No document is loaded yet. Upload one first.",
                        reasons=reasons + ["empty knowledge base"])
        names = ", ".join(d.name for d in kb.documents.values())
        return Task("clarify", msg, reasons=reasons + ["multiple documents, none named"],
                    clarification=f"Which document would you like summarised? I have: {names}.")

    # ---- compare ---------------------------------------------------------
    for pattern in _COMPARE_PATTERNS:
        m = pattern.search(msg)
        if not m:
            continue
        a, b = _clean_part(m.group(1)), _clean_part(m.group(2))
        if not a or not b or len(a) > 80 or len(b) > 80:
            continue
        reasons.append(f"compare cue: {pattern.pattern[:25]}...")
        # Per-side document scope: if each side names a different document,
        # retrieve each side from its own document.
        side_docs: list[set[str] | None] = []
        for side in (a, b):
            docs = kb.resolve_documents(side) if kb is not None else []
            side_docs.append({docs[0].document_id} if len(docs) == 1 else None)
        return Task("compare", msg, parts=[a, b], part_document_ids=side_docs,
                    document_ids={d.document_id for d in named} or None, reasons=reasons)

    # ---- plain answer, possibly scoped to a named document --------------
    if len(named) == 1:
        reasons.append(f"scoped to {named[0].name}")
        return Task("answer", msg, document_ids={named[0].document_id}, reasons=reasons)
    if len(named) > 1:
        reasons.append("several documents named -> search all of them")
        return Task("answer", msg, document_ids={d.document_id for d in named}, reasons=reasons)
    if n_docs > 1 and _DOC_WORDS.search(msg) and re.search(r"\b(this|the)\s+(document|paper|report|file)\b", msg, re.IGNORECASE):
        names = ", ".join(d.name for d in kb.documents.values())
        return Task("clarify", msg, reasons=["'the document' with several loaded"],
                    clarification=f"Which document do you mean? I have: {names}.")
    return Task("answer", msg, reasons=reasons or ["default"])
