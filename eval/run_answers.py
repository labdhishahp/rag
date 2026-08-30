"""
End-to-end ANSWER evaluation. Calls Gemini — paced for the free tier.

Measures what the retrieval eval cannot: did the information reach the USER?

Per gold question:
    fact_coverage      share of expected facts present in the answer text
    refused            answer is the refusal sentence
    correct_refusal    refused == (not answerable)
    citations_valid    every [S#] in the answer points at a real passage
    cited_any          at least one citation for an answerable question
    answer_chars       length (brief questions should be short)
    llm_called         the evidence gate may skip the call entirely

Aggregates: fact coverage on answerable, refusal confusion matrix, citation
validity, mean length by depth.

Usage:
    ./.venv/bin/python eval/run_answers.py --tag post-step5 [--only f01,f02] [--docs formula,sample]
    ./.venv/bin/python eval/run_answers.py --compare pre-step5 post-step5
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from embeddings import EmbeddingModel  # noqa: E402
from llm import LLMError, create_llm  # noqa: E402
from rag import RAGSystem  # noqa: E402
from run_retrieval import DocIndex, load_gold  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
MIN_SECONDS_BETWEEN_CALLS = 13.0  # free tier: 5 requests / minute


class PacedLLM:
    """Wraps the real client so the harness never trips the rate limit."""

    def __init__(self, inner):
        self.inner = inner
        self.last = 0.0
        self.calls = 0

    def answer_with_context(self, *args, **kwargs):
        for attempt in range(4):
            wait = MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()
            try:
                self.calls += 1
                return self.inner.answer_with_context(*args, **kwargs)
            except LLMError as exc:
                msg = str(exc).lower()
                if attempt == 3 or not ("rate limit" in msg or "unavailable" in msg):
                    raise
                time.sleep(MIN_SECONDS_BETWEEN_CALLS * (attempt + 2))
        raise LLMError("retries exhausted")

    def generate(self, prompt):
        return self.inner.generate(prompt)


def evaluate(tag: str, only: set[str] | None, docs: set[str] | None) -> dict:
    gold = load_gold()
    model = EmbeddingModel()
    llm = PacedLLM(create_llm("gemini"))
    indexes: dict[str, DocIndex] = {}
    rows = []

    for item in gold:
        if only and item["id"] not in only:
            continue
        if docs and not any(d in item["doc"] for d in docs):
            continue
        path = PROJECT_ROOT / item["doc"]
        if not path.exists():
            continue
        if item["doc"] not in indexes:
            indexes[item["doc"]] = DocIndex(path, model)
        idx = indexes[item["doc"]]

        rag = RAGSystem(retriever=idx.retriever, llm=llm, debug=False)
        t0 = time.perf_counter()
        try:
            r = rag.answer(item["question"])
        except LLMError as exc:
            print(f"  {item['id']}: LLM ERROR {exc}")
            continue
        elapsed = time.perf_counter() - t0

        answer_low = r["answer"].lower()
        facts = item.get("facts", [])
        # A fact may list alternative renderings separated by "|" (e.g. plain vs LaTeX).
        facts_found = [f for f in facts if any(alt.lower() in answer_low for alt in f.split("|"))]
        n_passages = len(r["context"].passages)

        rows.append(
            {
                "id": item["id"],
                "doc": item["doc"],
                "type": item["type"],
                "answerable": item["answerable"],
                "depth": r["depth"],
                "answer": r["answer"],
                "answer_chars": len(r["answer"]),
                "facts": facts,
                "facts_found": facts_found,
                "fact_coverage": (len(facts_found) / len(facts)) if facts else None,
                "refused": r["refused"],
                "correct_refusal": r["refused"] == (not item["answerable"]),
                "cited_labels": r["cited_labels"],
                "cited_any": bool(r["cited_labels"]),
                "n_passages": n_passages,
                "context_chars": r["context_chars"],
                "evidence_level": r["evidence_level"],
                "llm_called": r["llm_called"],
                "elapsed_s": elapsed,
            }
        )
        status = "REFUSED" if r["refused"] else f"facts {len(facts_found)}/{len(facts)}"
        print(f"  {item['id']:<4} {r['depth']:<8} {status:<12} cites={r['cited_labels']} "
              f"chars={len(r['answer']):<5} gate={r['evidence_level']}")

    result = {"tag": tag, "rows": rows, "summary": summarise(rows), "llm_calls": llm.calls}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"answers-{tag}.json").write_text(json.dumps(result, indent=2))
    return result


def summarise(rows):
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else 0.0

    ans = [r for r in rows if r["answerable"]]
    abs_ = [r for r in rows if not r["answerable"]]
    by_depth = {}
    for d in ("brief", "normal", "detailed"):
        sub = [r for r in ans if r["depth"] == d and not r["refused"]]
        by_depth[d] = {"n": len(sub), "mean_chars": mean([r["answer_chars"] for r in sub])}
    return {
        "n_answerable": len(ans),
        "n_absent": len(abs_),
        "fact_coverage": mean([r["fact_coverage"] for r in ans]),
        "full_fact_coverage_rate": mean([1.0 if r["fact_coverage"] == 1.0 else 0.0 for r in ans if r["fact_coverage"] is not None]),
        "false_refusals": sum(1 for r in ans if r["refused"]),
        "false_answers": sum(1 for r in abs_ if not r["refused"]),
        "cited_any_rate": mean([1.0 if r["cited_any"] else 0.0 for r in ans if not r["refused"]]),
        "gate_skipped_llm": sum(1 for r in rows if not r["llm_called"]),
        "answer_chars_by_depth": by_depth,
        "mean_elapsed_s": mean([r["elapsed_s"] for r in rows]),
    }


def print_report(result):
    s = result["summary"]
    print(f"\n=== ANSWER EVAL [{result['tag']}]   LLM calls: {result['llm_calls']} ===")
    print(f"  answerable {s['n_answerable']}  absent {s['n_absent']}")
    print(f"  fact coverage (mean)        : {s['fact_coverage']:.2f}")
    print(f"  all facts present (rate)    : {s['full_fact_coverage_rate']:.2f}")
    print(f"  false refusals (answerable) : {s['false_refusals']}")
    print(f"  false answers (absent)      : {s['false_answers']}")
    print(f"  cited at least one source   : {s['cited_any_rate']:.2f}")
    print(f"  gate skipped LLM            : {s['gate_skipped_llm']}")
    for d, v in s["answer_chars_by_depth"].items():
        print(f"  mean answer chars [{d:<8}] : {v['mean_chars']:.0f}  (n={v['n']})")


def compare(a_tag, b_tag):
    a = json.loads((RESULTS_DIR / f"answers-{a_tag}.json").read_text())
    b = json.loads((RESULTS_DIR / f"answers-{b_tag}.json").read_text())
    print(f"\n=== COMPARE ANSWERS  {a_tag} -> {b_tag} ===")
    for k in ["fact_coverage", "full_fact_coverage_rate", "false_refusals", "false_answers", "cited_any_rate"]:
        print(f"  {k:<26} {a['summary'][k]:>8.2f} -> {b['summary'][k]:>8.2f}")
    ra = {r["id"]: r for r in a["rows"]}
    rb = {r["id"]: r for r in b["rows"]}
    for qid in sorted(set(ra) & set(rb)):
        x, y = ra[qid], rb[qid]
        if x["facts_found"] != y["facts_found"] or x["refused"] != y["refused"]:
            print(f"    {qid}: facts {x['facts_found']} -> {y['facts_found']}  refused {x['refused']}->{y['refused']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="adhoc")
    ap.add_argument("--only", help="comma-separated question ids")
    ap.add_argument("--docs", help="comma-separated substrings of document names")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    only = set(args.only.split(",")) if args.only else None
    docs = set(args.docs.split(",")) if args.docs else None
    print_report(evaluate(args.tag, only, docs))


if __name__ == "__main__":
    main()
