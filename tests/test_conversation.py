"""
Phase 4 — conversation memory and follow-up resolution, without an LLM.

The behaviours pinned here:
  - a standalone question is NOT augmented
  - a follow-up is augmented with the previous question (deterministically)
  - a message that points at the previous ANSWER also pulls in its lead
  - the prompt window is bounded
  - end to end with a fake LLM: "Explain it in more detail" after a compound
    interest question retrieves compound-interest chunks, not noise
  - the recorded turn carries the evidence used (for "the previous one")
"""

import pytest

from conversation import Conversation
from llm import LLMClient
from prompt_builder import build_rag_prompt
from query_understanding import understand
from rag import RAGSystem


def _conv_after_formula_question() -> Conversation:
    c = Conversation()
    c.add_user("What is the compound interest formula?")
    c.add_assistant(
        "The compound interest formula is A = P(1 + r/n)^(nt) [S1].",
        retrieved_chunk_ids=[1, 2],
        sources=["formula_sample.pdf — page 2 — section: Compound Interest"],
    )
    return c


def test_standalone_question_is_not_augmented():
    c = _conv_after_formula_question()
    q = "What is the present value formula?"
    assert not understand(q).needs_context
    assert c.retrieval_query_for(q, needs_context=False) == q


def test_follow_up_is_augmented_with_previous_question():
    c = _conv_after_formula_question()
    q = "Explain it in more detail."
    assert understand(q).needs_context
    rq = c.retrieval_query_for(q, needs_context=True)
    assert rq.startswith("What is the compound interest formula?")
    assert rq.endswith("Explain it in more detail.")


def test_answer_referent_pulls_in_previous_answer_lead():
    c = _conv_after_formula_question()
    rq = c.retrieval_query_for("Which one is better?", needs_context=True)
    assert "A = P(1 + r/n)^(nt)" in rq          # from the previous answer
    assert "[S1]" not in rq                      # citations stripped


def test_empty_conversation_never_augments():
    c = Conversation()
    assert c.retrieval_query_for("Explain it more.", needs_context=True) == "Explain it more."


def test_window_is_bounded():
    c = Conversation(window_messages=4)
    for i in range(10):
        c.add_user(f"question {i}")
        c.add_assistant(f"answer {i}")
    text = c.format_for_prompt()
    assert "question 9" in text and "answer 9" in text
    assert "question 0" not in text
    assert text.count("\n") == 3  # exactly 4 lines


def test_prompt_keeps_conversation_separate_from_evidence():
    p = build_rag_prompt("Explain it more.", "[S1] evidence text", conversation="User: What is X?\nAssistant: X is Y.")
    assert "CONVERSATION SO FAR" in p and "EVIDENCE:" in p
    assert p.index("CONVERSATION SO FAR") < p.index("EVIDENCE:")
    assert "never cite it" in p
    assert "CONVERSATION SO FAR" not in build_rag_prompt("q", "[S1] e")


class FakeLLM(LLMClient):
    def __init__(self):
        self.prompts = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "fake answer [S1]"


@pytest.mark.embedding
def test_follow_up_retrieves_the_previous_topic(formula_retriever, formula_chunks):
    llm = FakeLLM()
    rag = RAGSystem(retriever=formula_retriever, llm=llm, debug=False)
    conv = Conversation()

    first = rag.answer("What is the compound interest formula?", conv)
    RAGSystem.record_turn(conv, first)

    second = rag.answer("Explain it in more detail.", conv)
    assert second["was_follow_up"]
    assert "compound interest formula" in second["retrieval_query"].lower()

    covered = {cid for p in second["context"].passages for cid in p.chunk_ids}
    compound = {c["chunk_id"] for c in formula_chunks if c["section"] == "Compound Interest"}
    assert covered & compound, "follow-up did not reach the compound interest chunks"
    # The worked example (detailed request) should be there too.
    assert any("1647" in p.text for p in second["context"].passages)
    # The model saw the conversation, and the ORIGINAL question was answered.
    assert "CONVERSATION SO FAR" in llm.prompts[-1]
    assert "USER QUESTION:\nExplain it in more detail." in llm.prompts[-1]


@pytest.mark.embedding
def test_recorded_turn_carries_evidence(formula_retriever):
    rag = RAGSystem(retriever=formula_retriever, llm=FakeLLM(), debug=False)
    conv = Conversation()
    result = rag.answer("What is the present value formula?", conv)
    RAGSystem.record_turn(conv, result)
    last = conv.turns[-1]
    assert last.role == "assistant"
    assert last.retrieved_chunk_ids
    assert any("Present Value" in s for s in last.sources)


@pytest.mark.embedding
def test_standalone_second_turn_is_not_augmented(formula_retriever):
    rag = RAGSystem(retriever=formula_retriever, llm=FakeLLM(), debug=False)
    conv = Conversation()
    RAGSystem.record_turn(conv, rag.answer("What is the compound interest formula?", conv))
    second = rag.answer("What is the present value formula?", conv)
    assert not second["was_follow_up"]
    assert second["retrieval_query"] == "What is the present value formula?"
