"""
Phase 5 — multi-document knowledge base, task routing, compare / summarise
plans and clarification. Fake LLM; no API calls.
"""

from pathlib import Path

import pytest

from assistant import KnowledgeAssistant
from conftest import DATA
from knowledge_base import DocumentInfo, KnowledgeBase
from llm import LLMClient
from rag import check_citations
from task_router import route

pytestmark = pytest.mark.embedding


class FakeLLM(LLMClient):
    def __init__(self, reply="fake [A1] and [B1] and [S1]"):
        self.reply = reply
        self.prompts: list[str] = []
        self.active_model = "fake"

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


@pytest.fixture(scope="module")
def kb(embedding_model):
    kb = KnowledgeBase(embedding_model)
    kb.add_document((DATA / "formula_sample.pdf").read_bytes(), "formula_sample.pdf")
    kb.add_document((DATA / "sample.pdf").read_bytes(), "sample.pdf")
    return kb


# ---- knowledge base --------------------------------------------------------

def test_two_documents_share_one_index(kb):
    assert len(kb.documents) == 2
    total = sum(d.chunk_count for d in kb.documents.values())
    assert kb.vector_store.ntotal == total


def test_re_adding_same_bytes_is_a_noop(kb):
    before = kb.vector_store.ntotal
    kb.add_document((DATA / "sample.pdf").read_bytes(), "renamed_copy.pdf")
    assert kb.vector_store.ntotal == before
    assert len(kb.documents) == 2


def test_resolve_documents_by_name_fragment(kb):
    assert [d.name for d in kb.resolve_documents("what does the formula sample say")] == ["formula_sample.pdf"]
    assert [d.name for d in kb.resolve_documents("according to sample.pdf")] == ["sample.pdf"]
    assert kb.resolve_documents("what is compound interest") == []


# ---- name resolution, in isolation ----------------------------------------
#
# These build a registry directly instead of indexing files: name matching is
# pure string work, so making it wait on an embedding API would only make the
# suite slow and flaky without testing anything more.

def _named(embedding_model, *names) -> KnowledgeBase:
    kb = KnowledgeBase(embedding_model)
    for i, name in enumerate(names):
        kb.documents[f"id{i}"] = DocumentInfo(
            document_id=f"id{i}", name=name, page_label="page",
            page_count=1, chunk_count=1,
        )
    return kb


def test_exact_filename_beats_a_shared_fragment(embedding_model):
    """'sample.pdf' must not also match formula_sample.pdf, and vice versa."""
    kb = _named(embedding_model, "formula_sample.pdf", "sample.pdf")
    assert [d.name for d in kb.resolve_documents("according to sample.pdf")] == ["sample.pdf"]
    # 'sample.pdf' occurs INSIDE 'formula_sample.pdf' as a substring; the
    # underscore is a word character, which is what stops it matching.
    assert [d.name for d in kb.resolve_documents("summarize formula_sample.pdf")] == ["formula_sample.pdf"]


def test_longest_stem_wins_when_names_overlap(embedding_model):
    kb = _named(embedding_model, "formula_sample.pdf", "sample.pdf")
    assert [d.name for d in kb.resolve_documents("what does the formula sample say")] == ["formula_sample.pdf"]
    # The separator the user typed should not matter.
    assert [d.name for d in kb.resolve_documents("in formula-sample")] == ["formula_sample.pdf"]


def test_partial_name_matching_still_works(embedding_model):
    kb = _named(embedding_model, "nist_sp800-207.pdf", "bert.pdf")
    assert [d.name for d in kb.resolve_documents("what does the NIST report say")] == ["nist_sp800-207.pdf"]
    assert [d.name for d in kb.resolve_documents("in the BERT paper")] == ["bert.pdf"]
    assert kb.resolve_documents("how does retrieval work") == []


def test_generic_fragments_do_not_scope_a_question(embedding_model):
    """'report', 'final' and 'version' must not identify a document alone."""
    kb = _named(embedding_model, "q3_report_final.pdf", "bert.pdf")
    assert kb.resolve_documents("is this the final version of the report?") == []
    assert [d.name for d in kb.resolve_documents("what is in q3_report_final.pdf")] == ["q3_report_final.pdf"]


def test_genuinely_ambiguous_names_return_every_candidate(embedding_model):
    """A real tie must stay a tie, so the router can ask which one."""
    kb = _named(embedding_model, "budget_2023.pdf", "budget_2024.pdf")
    names = sorted(d.name for d in kb.resolve_documents("what does the budget say?"))
    assert names == ["budget_2023.pdf", "budget_2024.pdf"]
    # Naming the year resolves it.
    assert [d.name for d in kb.resolve_documents("what does budget_2024.pdf say?")] == ["budget_2024.pdf"]


def test_filtered_retrieval_stays_in_document(kb):
    sample_id = next(d.document_id for d in kb.documents.values() if d.name == "sample.pdf")
    hits = kb.retriever.retrieve("What is the compound interest formula?", top_k=3, document_ids={sample_id})
    assert hits and all(h["document_id"] == sample_id for h in hits)


def test_unfiltered_retrieval_finds_the_right_document(kb):
    hits = kb.retriever.retrieve("What was Acme's total revenue in 2024?", top_k=5)
    assert hits[0]["document_name"] == "sample.pdf"
    hits = kb.retriever.retrieve("What is the compound interest formula?", top_k=5)
    assert hits[0]["document_name"] == "formula_sample.pdf"


# ---- routing --------------------------------------------------------------

def test_routing_kinds(kb):
    assert route("Compare simple interest and compound interest.", kb).kind == "compare"
    assert route("What is the difference between simple and compound interest?", kb).kind == "compare"
    assert route("Simple interest vs compound interest", kb).kind == "compare"
    assert route("Summarize the formula sample document.", kb).kind == "summarize"
    assert route("What is the compound interest formula?", kb).kind == "answer"
    assert route("Explain the formula in detail.", kb).kind == "answer"


def test_compare_parts_are_extracted(kb):
    t = route("What is the difference between simple interest and compound interest?", kb)
    assert t.parts == ["simple interest", "compound interest"]


def test_summarize_without_naming_a_document_asks_which(kb):
    t = route("Summarize this document.", kb)
    assert t.kind == "clarify"
    assert "formula_sample.pdf" in t.clarification and "sample.pdf" in t.clarification


def test_summarize_with_named_document_is_scoped(kb):
    t = route("Give me a summary of sample.pdf", kb)
    assert t.kind == "summarize"
    assert t.target_document.name == "sample.pdf"


def test_named_document_scopes_plain_answer(kb):
    t = route("According to sample.pdf, who is the CTO?", kb)
    assert t.kind == "answer"
    assert t.document_ids == {next(d.document_id for d in kb.documents.values() if d.name == "sample.pdf")}


def test_single_document_kb_never_asks_which(embedding_model):
    kb1 = KnowledgeBase(embedding_model)
    kb1.add_document((DATA / "formula_sample.pdf").read_bytes(), "formula_sample.pdf")
    assert route("Summarize this document.", kb1).kind == "summarize"


# ---- plans end to end with a fake LLM -------------------------------------

def test_compare_builds_two_labelled_evidence_sets(kb):
    llm = FakeLLM("**Simple** ... [A1]. **Compound** ... [B1]. Bad [S9] [C1].")
    assistant = KnowledgeAssistant(kb, llm, debug=False)
    result = assistant.answer("Compare simple interest and compound interest.")
    assert result["task"] == "compare"
    assert len(llm.prompts) == 1
    prompt = llm.prompts[0]
    assert "=== A: simple interest ===" in prompt and "=== B: compound interest ===" in prompt
    assert "[A1]" in prompt and "[B1]" in prompt
    texts = " ".join(p.text for p in result["context"].passages)
    assert "I = P * r * t" in texts and "A = P(1 + r/n)^(nt)" in texts
    # invalid labels stripped, valid kept
    assert "[S9]" not in result["answer"] and "[C1]" not in result["answer"]
    assert "[A1]" in result["answer"] and "[B1]" in result["answer"]


def test_summarize_walks_every_section_in_order(kb):
    llm = FakeLLM("Summary [S1] [S2].")
    assistant = KnowledgeAssistant(kb, llm, debug=False)
    result = assistant.answer("Summarize formula_sample.pdf")
    assert result["task"] == "summarize"
    assert len(llm.prompts) == 1
    sections = [p.section for p in result["context"].passages]
    for s in ["Introduction to Financial Mathematics", "Compound Interest", "Simple Interest",
              "Present Value", "Risk and Diversification"]:
        assert s in sections
    ids = [cid for p in result["context"].passages for cid in p.chunk_ids]
    assert ids == sorted(ids)
    assert "Its section headings, in order" in llm.prompts[0]


def test_clarification_makes_no_llm_call(kb):
    llm = FakeLLM()
    assistant = KnowledgeAssistant(kb, llm, debug=False)
    result = assistant.answer("Summarize this document.")
    assert result["clarification"] and not result["llm_called"] and llm.prompts == []
    assert "Which document" in result["answer"]


def test_scoped_answer_uses_only_named_document(kb):
    llm = FakeLLM("answer [S1]")
    assistant = KnowledgeAssistant(kb, llm, debug=False)
    result = assistant.answer("According to sample.pdf, what is the compound interest formula?")
    docs = {p.document_name for p in result["context"].passages}
    assert docs == {"sample.pdf"}


def test_empty_kb_asks_for_upload(embedding_model):
    assistant = KnowledgeAssistant(KnowledgeBase(embedding_model), FakeLLM(), debug=False)
    r = assistant.answer("What is compound interest?")
    assert r["clarification"] and not r["llm_called"]


# ---- citation checker handles prefixed and grouped labels ---------------

def test_check_citations_grouped_and_prefixed():
    cleaned, used = check_citations("x [A1, B2] y [S3] z [B9]", ["A1", "A2", "B1", "B2"])
    assert cleaned == "x [A1, B2] y z"
    assert used == [1, 4]
