"""
Read the user's request BEFORE retrieving or generating.

Two questions, both answered deterministically:

  1. How much detail does the user want?         -> depth
       "What is the formula?"                        brief
       "Explain the formula."                        normal
       "Explain the formula in detail."              detailed
       "Give me an example."                         normal + wants_example

  2. Is the message a follow-up that leans on the conversation?  -> needs_context
       "Explain it in more detail."   "What about the previous one?"
       (Used from Phase 4 onward; defined here because both are properties of
        the same sentence and are read in the same place.)

Why deterministic:
  These are cheap lexical judgements. A word list gets ~90% of them right, is
  free, runs in microseconds, and — most importantly — you can read exactly why
  it decided what it decided. An LLM classifier would cost a call per question
  against a 5-request-per-minute quota and hide its reasoning. If evaluation
  shows the heuristics missing real cases, upgrade THEN, with the miss list as
  the evidence.

Why depth matters for RETRIEVAL, not just wording:
  "Give me just the equation" should retrieve less (no neighbours), and
  "explain in detail" should retrieve more (neighbours + a larger budget).
  Depth is the switch that turns the same retriever into two different
  retrievers. Making the LLM merely "talk more" would not add a single piece of
  evidence it did not already have.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- depth ---------------------------------------------------------------

_BRIEF_CUES = (
    r"\bjust\b", r"\bonly\b", r"\bbriefly\b", r"\bin short\b", r"\bone[- ]line\b",
    r"\bone sentence\b", r"\bquick(ly)?\b", r"\bshort answer\b", r"\btl;?dr\b",
    r"\bwhat is the (formula|equation|definition|value|name|number)\b",
    # Short singular factoid ("Who is the CEO?"). Plural "what are the ..." asks
    # for a list, which needs the normal depth to come back complete.
    r"^(what|who|when|where|which) (is|was) [^,]{1,40}\?$",
)
_DETAILED_CUES = (
    r"\bin[- ]detail\b", r"\bdetailed\b", r"\bthoroughly\b", r"\bcomprehensive(ly)?\b",
    r"\bstep[- ]by[- ]step\b", r"\belaborate\b", r"\bexplain\b.*\b(how|why)\b",
    r"\bwalk me through\b", r"\bin depth\b", r"\beverything (about|on)\b",
    r"\bfull explanation\b", r"\bexplain (it|this|that)\b.*\b(more|further|fully)\b",
)
_EXPLAIN_CUES = (
    r"\bexplain\b", r"\bhow does\b", r"\bhow do\b", r"\bwhy\b", r"\bdescribe\b",
    r"\bwhat does .* mean\b", r"\bhow (it|this|that) works\b", r"\bmechanism\b",
)
_EXAMPLE_CUES = (
    r"\bexample\b", r"\bfor instance\b", r"\bworked\b", r"\billustrat", r"\bshow me\b",
    r"\bsample calculation\b", r"\bwalk(ed)? example\b",
)

# --- follow-up detection (Phase 4) --------------------------------------

_REFERENT_CUES = (
    r"^\s*(and|also|but|so|then|what about|how about)\b",
    r"\b(it|its|this|that|these|those|they|them|he|she|him|her|his|the (previous|former|latter|second|first|other|same) one)\b",
    r"\b(more|further|again|elaborate|expand)\b",
    r"\b(the (previous|above|earlier|last) (one|answer|point|method|formula))\b",
    r"^\s*(why|how|when|where)\s*[.?!]*\s*$",
    r"\b(compare|difference) (it|that|this|them)\b",
    r"\bwhich (one|is better|would you)\b",
    # A request for an example with no topic named: "Can you give me an example?"
    #
    # The terminal-punctuation class is deliberate, not cosmetic. These two cues
    # used to end in `\??\s*$`, which accepts a question mark or nothing but NOT
    # a full stop — so "Give me an example?" was read as a follow-up and "Give
    # me an example." was not. The miss was invisible for a long time because
    # retrieval on the bare sentence happened to land near the right chunk
    # anyway; it only became a visible refusal once the scores shifted. A cue
    # about sentence MEANING should never hinge on which terminator was typed.
    r"^\s*(can you |could you |please |would you )?(give|show|provide)( me)? (an|another|one more|a second|some) examples?\s*[.?!]*\s*$",
    # A short definite reference with no topic: "What do the variables mean?"
    r"^\s*(what|how|why|when)\b.{0,30}\bthe (variables?|formula|equation|method|approach|steps?|process|policy|terms?|components?|parts?)\b",
)


@dataclass
class QueryUnderstanding:
    question: str
    depth: str                       # "brief" | "normal" | "detailed"
    wants_example: bool
    wants_explanation: bool
    needs_context: bool              # leans on prior turns (Phase 4)
    cues: list[str] = field(default_factory=list)   # which rules fired — for debug prints

    def describe(self) -> str:
        flags = []
        if self.wants_example:
            flags.append("example")
        if self.wants_explanation:
            flags.append("explanation")
        if self.needs_context:
            flags.append("follow-up")
        return f"{self.depth}" + (f" (+{', '.join(flags)})" if flags else "")


def _any(patterns, text: str) -> list[str]:
    return [p for p in patterns if re.search(p, text, flags=re.IGNORECASE)]


def understand(question: str) -> QueryUnderstanding:
    q = " ".join(question.strip().split())
    low = q.lower()

    brief_hits = _any(_BRIEF_CUES, low)
    detailed_hits = _any(_DETAILED_CUES, low)
    explain_hits = _any(_EXPLAIN_CUES, low)
    example_hits = _any(_EXAMPLE_CUES, low)
    referent_hits = _any(_REFERENT_CUES, low)

    # Detailed wins over brief when both fire ("just explain it in detail").
    if detailed_hits:
        depth = "detailed"
    elif brief_hits and not explain_hits:
        depth = "brief"
    else:
        depth = "normal"

    wants_example = bool(example_hits)
    wants_explanation = bool(explain_hits) or depth == "detailed"

    # A short message full of pronouns and no content noun is leaning on the
    # conversation. Long, specific questions stand on their own even if they
    # contain "it".
    word_count = len(low.split())
    needs_context = bool(referent_hits) and (word_count <= 12 or len(referent_hits) >= 2)

    cues = brief_hits + detailed_hits + explain_hits + example_hits + referent_hits
    return QueryUnderstanding(
        question=q,
        depth=depth,
        wants_example=wants_example,
        wants_explanation=wants_explanation,
        needs_context=needs_context,
        cues=cues,
    )


def retrieval_config_for(u: QueryUnderstanding) -> dict:
    """
    Depth -> how much surrounding evidence to gather.

      brief     no neighbours; smaller budget. The user asked for one thing.
      normal    one neighbour each side (the Step 3 default).
      detailed  one neighbour each side and a larger budget, so the variable
                list AND the worked example both fit alongside the formula.

    An explicit request for an example is treated like "detailed" for
    retrieval purposes: the example is almost never in the matched chunk.

    Why brief uses a WIDER top_k:
      Measured: "What was Acme's total revenue in 2024?" had its gold chunk at
      rank 4. With neighbours on, expansion happened to rescue it; brief mode
      switched neighbours off and the answer became a false refusal. Depth may
      change how much SURROUNDING text we gather, but it must never lower the
      recall of the matches themselves — so brief takes five entries instead of
      three. Five bare chunks (~2200 chars) still cost less than three chunks
      plus their neighbours.
    """
    if u.depth == "brief" and not u.wants_example:
        return {"top_k": 5, "neighbour_window": 0, "budget_chars": 2500}
    if u.depth == "detailed" or u.wants_example:
        return {"top_k": 3, "neighbour_window": 1, "budget_chars": 6500}
    return {"top_k": 3, "neighbour_window": 1, "budget_chars": 5000}
