"""
Which KIND of task is this message? Decided before any retrieval happens.

Most questions take the standard path: retrieve → expand → gate → answer. Two
kinds need a DIFFERENT RETRIEVAL SHAPE, which is why they are routed rather
than handled by wording the prompt differently:

    summarize   "Summarize this document."
                Similarity search cannot retrieve "the whole document" — there
                is no query whose nearest neighbours are a representative
                sample of everything. The evidence has to be WALKED in reading
                order instead. See tasks.summarize.

    compare     "Compare simple and compound interest."
                One search for "simple vs compound interest" returns whichever
                subject the document discusses more, and the answer comes out
                one-sided. Two subjects need two retrievals with their own
                evidence budgets. See tasks.compare.

Everything else — direct questions, explanations, examples, follow-ups — is
DEPTH, not task kind, and query_understanding already handles it.

This router deliberately knows nothing about documents, providers or sessions.
It reads text and returns a plan:

  * no knowledge base, because a session holds exactly one document today. When
    multi-document lands, document resolution is added HERE and the plans below
    gain a filter argument; nothing else has to move.
  * no provider. Which model answers is chosen separately and independently —
    task selection and provider selection never touch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_SUMMARIZE = re.compile(
    r"^\s*(please\s+)?(summari[sz]e|give (me )?(a |an )?(brief |short |quick )?(summary|overview)( of)?|"
    r"what (is|are) (this|the) (document|paper|report|file|code)s? (about|cover)|"
    r"(overview|summary) of (this|the) (document|paper|report|file|code)|"
    r"what does (this|the) (document|paper|report|file|code) (say|cover|discuss|do))",
    re.IGNORECASE,
)

# Ordered most specific first. The bare "X vs Y" form is LAST because it is the
# loosest: any sentence containing " vs " would match it, so it only gets a turn
# once the explicit forms have declined.
_COMPARE_PATTERNS = (
    re.compile(r"\bcompare\s+(?:the\s+)?(.+?)\s+(?:and|with|to|vs\.?|versus|against)\s+(?:the\s+)?(.+?)[\s?.!]*$", re.IGNORECASE),
    re.compile(r"\b(?:what (?:is|are) the )?differences?\s+between\s+(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+?)[\s?.!]*$", re.IGNORECASE),
    re.compile(r"\bhow (?:does|do|is|are)\s+(?:the\s+)?(.+?)\s+(?:differ|different)\s+from\s+(?:the\s+)?(.+?)[\s?.!]*$", re.IGNORECASE),
    re.compile(r"^\s*(?:the\s+)?(.+?)\s+(?:vs\.?|versus)\s+(?:the\s+)?(.+?)[\s?.!]*$", re.IGNORECASE),
)

# A comparison subject is a noun phrase, not a clause. Anything longer is almost
# certainly a sentence that merely contains the word "and", so the match is
# rejected and the message falls through to the normal answer path — which is
# the safe direction to fail in.
_MAX_SUBJECT_CHARS = 80
_MIN_SUBJECT_CHARS = 2


@dataclass
class Task:
    kind: str                                       # "answer" | "summarize" | "compare"
    message: str
    parts: list[str] = field(default_factory=list)  # compare: [subject A, subject B]
    reasons: list[str] = field(default_factory=list)  # why we routed this way

    def describe(self) -> str:
        return self.kind + (f" {self.parts}" if self.parts else "")


def _resolve_shared_head(a: str, b: str) -> tuple[str, str]:
    """
    Restore the head noun the user elided from the first subject.

    "the difference between simple and compound interest" means
    simple INTEREST vs compound interest — English drops the repeated head
    from the first conjunct. Taken literally the first subject is the bare
    word "simple", which as a dense query matches almost anything.

    Fires only in the shape where that reading is the likely one: a
    single-word first subject against a multi-word second, where the first
    does not already end in the second's head.

        "simple" + "compound interest"          -> "simple interest"   fires
        "cats" + "dogs"                          -> unchanged (b is one word)
        "simple interest" + "compound interest"  -> unchanged (a is complete)

    The rule is deliberately asymmetric in what it risks. A wrong expansion
    ("apples" + "ripe oranges" -> "apples oranges") adds one token to a dense
    query, which retrieval absorbs. NOT expanding leaves a one-word query that
    matches weakly. Measured on formula_sample.pdf with bge-small:

        "simple"           top hits [4, 1, 2]   best similarity 0.614
        "simple interest"  top hits [4, 1, 3]   best similarity 0.744

    Different chunks, and 0.614 sits BELOW the 0.65 soft floor — so the bare
    word would have been answered under a low-confidence flag while the
    restored phrase is comfortably "ok".
    """
    a_words, b_words = a.split(), b.split()
    if len(a_words) == 1 and len(b_words) >= 2:
        head = b_words[-1]
        if a_words[-1].lower() != head.lower():
            return f"{a} {head}", b
    return a, b


def _clean_subject(text: str) -> str:
    text = re.sub(r"^(the|a|an)\s+", "", text.strip(), flags=re.IGNORECASE)
    return text.strip(" ,;:")


def route(message: str) -> Task:
    """Decide the task kind for this message. Pure text matching, no I/O."""
    msg = " ".join(message.split())

    if _SUMMARIZE.search(msg):
        return Task("summarize", msg, reasons=["summarize cue"])

    for pattern in _COMPARE_PATTERNS:
        match = pattern.search(msg)
        if not match:
            continue
        a, b = _clean_subject(match.group(1)), _clean_subject(match.group(2))
        if not (_MIN_SUBJECT_CHARS <= len(a) <= _MAX_SUBJECT_CHARS):
            continue
        if not (_MIN_SUBJECT_CHARS <= len(b) <= _MAX_SUBJECT_CHARS):
            continue
        raw_a = a
        a, b = _resolve_shared_head(a, b)
        reasons = [f"compare cue: {pattern.pattern[:28]}…"]
        if a != raw_a:
            reasons.append(f"restored elided head: {raw_a!r} -> {a!r}")
        return Task("compare", msg, parts=[a, b], reasons=reasons)

    return Task("answer", msg, reasons=["default"])
