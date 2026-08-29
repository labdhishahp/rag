"""
Conversation memory — what were we just talking about?

------------------------------------------------------------------------------
THREE THINGS THAT ARE EASY TO CONFUSE
------------------------------------------------------------------------------
    conversation memory   "What were we discussing?"    lives here
    document knowledge    "What does the file say?"     lives in FAISS
    retrieved evidence    "What is relevant THIS turn?" rebuilt every turn

Before Phase 4 the Streamlit app kept a list of past questions and answers
and displayed them — but never passed them to retrieval or to the LLM. So
"explain it in more detail" was embedded literally (near-noise similarity) and
the model saw a question whose "it" referred to nothing.

------------------------------------------------------------------------------
HOW FOLLOW-UPS ARE RESOLVED — DETERMINISTICALLY FIRST
------------------------------------------------------------------------------
Analogy: a librarian who hears "explain that more" doesn't need a translator;
they remember what you asked thirty seconds ago and walk to the same shelf.

When the new message leans on the conversation (query_understanding.needs_context):

    retrieval query = previous user question + " " + current message
                      (+ the first sentence of the previous answer when the
                       message points at the answer: "the second one",
                       "that formula", "which is better")

No LLM call, no hallucinated rewrite, fully inspectable — the augmented query
is printed and shown in the UI. Retrieval runs on BOTH the augmented and the
raw message and unions the hits, so a wrong augmentation cannot hide a right
raw match. An LLM rewriter is the fallback if evaluation shows this failing,
not the default.

The ORIGINAL message is what the model answers; augmentation is for retrieval
only. Otherwise the model answers a question the user did not ask.

------------------------------------------------------------------------------
WINDOWING
------------------------------------------------------------------------------
The prompt receives the last N turns verbatim (default 6 messages = 3
exchanges). Older turns are dropped, not summarised — no measured need yet,
and a summary is one more thing that can drift.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

DEFAULT_WINDOW_MESSAGES = 6

# Phrases that point at the assistant's previous ANSWER rather than the user's
# previous question ("which one is better" refers to options the answer listed).
_ANSWER_REFERENT = re.compile(
    r"\b(the (first|second|third|last|previous|former|latter|other) one|which (one|is better)|"
    r"that (formula|method|approach|option|policy|answer)|compare (them|those|these)|"
    r"both of (them|those)|either of (them|those))\b",
    re.IGNORECASE,
)


@dataclass
class Turn:
    role: str            # "user" | "assistant"
    content: str
    # For assistant turns: what evidence produced the answer. Lets a later
    # "what about the previous one?" know which chunks/sources were involved.
    retrieved_chunk_ids: list[int] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    retrieval_query: str | None = None


@dataclass
class Conversation:
    turns: list[Turn] = field(default_factory=list)
    window_messages: int = DEFAULT_WINDOW_MESSAGES

    # ---- recording -------------------------------------------------------

    def add_user(self, content: str) -> None:
        self.turns.append(Turn(role="user", content=content.strip()))

    def add_assistant(self, content: str, retrieved_chunk_ids=None, sources=None,
                      retrieval_query=None) -> None:
        self.turns.append(
            Turn(
                role="assistant",
                content=content.strip(),
                retrieved_chunk_ids=list(retrieved_chunk_ids or []),
                sources=list(sources or []),
                retrieval_query=retrieval_query,
            )
        )

    # ---- reading ---------------------------------------------------------

    @property
    def is_empty(self) -> bool:
        return not self.turns

    def last_user_question(self) -> str | None:
        for turn in reversed(self.turns):
            if turn.role == "user":
                return turn.content
        return None

    def last_assistant_answer(self) -> str | None:
        for turn in reversed(self.turns):
            if turn.role == "assistant":
                return turn.content
        return None

    def window(self) -> list[Turn]:
        """The most recent messages that go into the prompt."""
        return self.turns[-self.window_messages:]

    def format_for_prompt(self) -> str:
        """
        The CONVERSATION block. Kept visually distinct from EVIDENCE so the
        model does not cite the conversation as if it were a document.
        """
        lines = []
        for turn in self.window():
            speaker = "User" if turn.role == "user" else "Assistant"
            text = _truncate(turn.content, 600)
            lines.append(f"{speaker}: {text}")
        return "\n".join(lines) if lines else "(none)"

    # ---- retrieval query construction -----------------------------------

    def retrieval_query_for(self, message: str, needs_context: bool) -> str:
        """
        The query retrieval should embed for this message.

        Standalone message -> the message itself.
        Follow-up          -> previous question + message
                              (+ lead of previous answer if the message points
                               at the answer's content)
        """
        message = " ".join(message.split())
        if not needs_context or self.is_empty:
            return message

        parts: list[str] = []
        previous_q = self.last_user_question()
        if previous_q:
            parts.append(previous_q)
        if _ANSWER_REFERENT.search(message):
            previous_a = self.last_assistant_answer()
            if previous_a:
                parts.append(_lead(previous_a, 300))
        parts.append(message)
        return " ".join(parts)


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _lead(text: str, limit: int) -> str:
    """First sentence(s) of an answer, up to `limit` chars, citations stripped."""
    text = re.sub(r"\[S\d+\]", "", text)
    text = re.sub(r"[*#_`]", "", text)          # markdown noise
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0]
