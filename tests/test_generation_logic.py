"""
Phase 3 logic that must work WITHOUT calling an LLM:

  - depth / example / follow-up detection from the question text
  - depth -> retrieval configuration
  - prompt assembly (right instruction for the right depth, refusal text,
    citation rule present)
  - the evidence gate declines without an LLM call
  - citation checking strips labels that point at nothing

The LLM is replaced by a fake that records calls, so these run in milliseconds
and cost no quota.
"""

import pytest

from context_builder import build_context
from llm import LLMClient
from prompt_builder import REFUSAL_TEXT, build_rag_prompt
from query_understanding import retrieval_config_for, understand
from rag import RAGSystem, _check_citations


# ---- query understanding -------------------------------------------------

@pytest.mark.parametrize(
    "question, depth",
    [
        ("What is the compound interest formula?", "brief"),
        ("Just give me the equation.", "brief"),
        ("Who is the CEO?", "brief"),
        ("Explain the compound interest formula.", "normal"),
        ("How does compound interest work?", "normal"),
        ("Explain the compound interest formula in detail.", "detailed"),
        ("Give me the equation and explain it in detail.", "detailed"),
        ("Walk me through how the trust algorithm works.", "detailed"),
        ("What are the tenets of zero trust?", "normal"),
    ],
)
def test_depth_detection(question, depth):
    assert understand(question).depth == depth


def test_example_detection():
    assert understand("Give me a worked example of compound interest.").wants_example
    assert understand("Can you show me an example?").wants_example
    assert not understand("What is the present value formula?").wants_example


@pytest.mark.parametrize(
    "question, needs_context",
    [
        ("Explain it in more detail.", True),
        ("What about the previous one?", True),
        ("Why?", True),
        ("Can you give me an example of that?", True),
        ("Which one is better?", True),
        # Terminal punctuation must not decide whether a sentence leans on the
        # conversation: all three of these are the same request.
        ("Give me an example.", True),
        ("Give me an example", True),
        ("Show me an example.", True),
        ("Why.", True),
        ("What is the compound interest formula?", False),
        ("What are the tenets of zero trust?", False),
        ("How does the trust algorithm decide whether to grant access to a resource?", False),
    ],
)
def test_follow_up_detection(question, needs_context):
    assert understand(question).needs_context == needs_context


def test_retrieval_config_follows_depth():
    brief = retrieval_config_for(understand("What is the formula?"))
    normal = retrieval_config_for(understand("Explain the formula."))
    detailed = retrieval_config_for(understand("Explain the formula in detail."))
    example = retrieval_config_for(understand("Give me an example."))
    assert brief["neighbour_window"] == 0 and brief["budget_chars"] < normal["budget_chars"]
    # Brief narrows the SURROUNDING text, never the matches: it takes more entries.
    assert brief["top_k"] > normal["top_k"]
    assert normal["neighbour_window"] == 1
    assert detailed["budget_chars"] > normal["budget_chars"]
    assert example["neighbour_window"] == 1  # examples live next door, never in the match


def test_daily_quota_error_classification():
    from llm import is_daily_quota_error

    daily = "429 RESOURCE_EXHAUSTED ... 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'"
    minute = "429 RESOURCE_EXHAUSTED ... 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'"
    assert is_daily_quota_error(daily)
    assert not is_daily_quota_error(minute)


# ---- prompt assembly -----------------------------------------------------

def test_prompt_contains_depth_instruction_and_citation_rule():
    brief = build_rag_prompt("q", "[S1] x", depth="brief")
    detailed = build_rag_prompt("q", "[S1] x", depth="detailed", wants_example=True)
    assert "SHORT answer" in brief and "worked example" not in brief
    assert "DETAILED explanation" in detailed and "worked example" in detailed
    for prompt in (brief, detailed):
        assert REFUSAL_TEXT in prompt
        assert "[S2]" in prompt  # the citation example in the rules
        assert "(matched)" in prompt


def test_prompt_weak_evidence_note():
    assert "CAUTION" in build_rag_prompt("q", "c", low_confidence=True)
    assert "CAUTION" not in build_rag_prompt("q", "c", low_confidence=False)


# ---- citation checking ---------------------------------------------------

def test_check_citations_keeps_valid_and_strips_invalid():
    answer = "The rate is 5% [S1]. It compounds monthly [S2]. Unknown claim [S7]."
    cleaned, used = _check_citations(answer, n_passages=3)
    assert "[S7]" not in cleaned
    assert "[S1]" in cleaned and "[S2]" in cleaned
    assert used == [1, 2]


# ---- evidence gate with a fake LLM ---------------------------------------

class FakeLLM(LLMClient):
    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return "fake answer [S1] with a bad cite [S9]"


@pytest.mark.embedding
def test_gate_declines_absent_question_without_llm_call(formula_retriever):
    llm = FakeLLM()
    rag = RAGSystem(retriever=formula_retriever, llm=llm, debug=False)
    result = rag.answer("What is the parental leave policy?")
    assert llm.calls == 0
    assert result["refused"] is True
    assert result["llm_called"] is False
    assert result["answer"].startswith(REFUSAL_TEXT)
    assert result["evidence_level"] == "none"


@pytest.mark.embedding
def test_answerable_question_calls_llm_once_and_checks_citations(formula_retriever):
    llm = FakeLLM()
    rag = RAGSystem(retriever=formula_retriever, llm=llm, debug=False)
    result = rag.answer("What is the compound interest formula and explain it in detail?")
    assert llm.calls == 1
    assert result["depth"] == "detailed"
    assert "[S9]" not in result["answer"]
    assert result["cited_labels"] == [1]
    assert len(result["cited_sources"]) == 1


@pytest.mark.embedding
def test_brief_question_retrieves_without_expansion(formula_retriever):
    rag = RAGSystem(retriever=formula_retriever, llm=FakeLLM(), debug=False)
    brief = rag.answer("What is the compound interest formula?")
    detailed = rag.answer("Explain the compound interest formula in detail.")
    # Brief: no neighbours, but MORE matches (recall of the hits is never reduced).
    assert brief["expanded_chunk_ids"] == []
    assert brief["top_k"] == 5
    # Detailed: neighbours pulled in, worked example present.
    assert detailed["expanded_chunk_ids"] != []
    assert any("1647" in p.text for p in detailed["context"].passages)
