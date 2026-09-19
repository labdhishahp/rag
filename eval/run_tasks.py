"""
Measure the task plans: routing, evidence selection, citations, groundedness.

eval/run_retrieval.py answers "did retrieval find the gold chunk?", which does
not apply to a summary (no search) or a comparison (two searches). This harness
answers the questions those plans actually raise:

    task detection      did the router pick the right plan?
    evidence selection  summarize: is every section represented, first to last?
                        compare:   did BOTH sides get their own evidence?
    citation validity   did every label the model emitted survive verification?
    groundedness        are the expected facts present in the answer?
    A/B isolation       can an A-label only ever cite A-side evidence?

Two modes, because the structural half needs no model:

    --provider none        (default) routing, evidence, coverage, isolation.
                           Zero LLM calls, so it can run on every change.
    --provider anthropic|gemini
                           adds citation validity and groundedness, which need
                           a real answer.

Usage:
    python eval/run_tasks.py
    python eval/run_tasks.py --provider anthropic --tag post-fix
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "src"))

from chunker import chunk_pages  # noqa: E402
from context_builder import labels_for  # noqa: E402
from document_loader import load_document_from_bytes  # noqa: E402
from embeddings import create_provider  # noqa: E402
from pipeline import make_document_id  # noqa: E402
from retriever import Retriever  # noqa: E402
from llm import LLMClient, create_llm  # noqa: E402
from task_router import route  # noqa: E402
from vector_store import VectorStore  # noqa: E402
import tasks as task_plans  # noqa: E402

GOLD = PROJECT_ROOT / "eval" / "tasks_gold.jsonl"
RESULTS = PROJECT_ROOT / "eval" / "results"


class StructuralLLM(LLMClient):
    """
    Stands in for a model when only structure is being measured.

    It echoes every label that exists plus two that do not, so the citation
    checker is exercised even without a real answer: the invented ones must be
    stripped and the real ones kept.
    """

    name = "structural"
    active_model = "structural"

    def __init__(self):
        self.prompt = None

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        labels = sorted(set(__import__("re").findall(r"\[([AB]?S?\d+)\]", prompt)))
        real = " ".join(f"[{l}]" for l in labels)
        return f"structural answer {real} [S999] [Z1]"


def load_gold():
    return [json.loads(line) for line in GOLD.read_text().splitlines() if line.strip()]


def index(path: Path, model):
    data = path.read_bytes()
    pages = load_document_from_bytes(data, path.name)
    chunks = chunk_pages(pages, 500, 50, document_name=path.name,
                         document_id=make_document_id(data))
    store = VectorStore(dimension=model.dimension)
    store.add(model.embed_texts([c["text"] for c in chunks]), chunks)
    return Retriever(model, store), chunks


def doc_sections(chunks):
    out = []
    for c in chunks:
        s = c.get("section")
        if s and (not out or out[-1] != s):
            out.append(s)
    return out


def grounded(answer: str, facts) -> float:
    """Share of expected facts present. 'a|b' counts if either appears."""
    if not facts:
        return float("nan")
    low = answer.lower()
    hit = sum(1 for f in facts if any(alt.strip().lower() in low for alt in f.split("|")))
    return hit / len(facts)


def evaluate(provider_name: str, tag: str):
    import re

    model = create_provider("huggingface")
    llm = StructuralLLM() if provider_name == "none" else create_llm(provider_name)

    cache, rows = {}, []
    for item in load_gold():
        path = PROJECT_ROOT / item["doc"]
        if not path.exists():
            rows.append({"id": item["id"], "skipped": "document missing"})
            continue
        if item["doc"] not in cache:
            cache[item["doc"]] = index(path, model)
        retriever, chunks = cache[item["doc"]]

        task = route(item["question"])
        row = {
            "id": item["id"], "doc": item["doc"], "note": item["note"],
            "expected_task": item["task"], "routed_task": task.kind,
            "task_ok": task.kind == item["task"],
        }
        if item["task"] == "compare":
            row["expected_parts"] = item.get("parts")
            row["routed_parts"] = task.parts
            row["parts_ok"] = [p.lower() for p in task.parts] == [p.lower() for p in item.get("parts", [])]

        if not row["task_ok"]:
            rows.append(row)
            continue

        started = time.monotonic()
        if task.kind == "summarize":
            result = task_plans.summarize(retriever, llm, item["question"])
            passages = result["context"].passages
            shown = [p.section for p in passages if p.section]
            all_secs = doc_sections(chunks)
            row.update({
                "sections_total": len(all_secs),
                "sections_shown": len(set(shown)),
                "cover_first": (not all_secs) or (all_secs[0] in shown),
                "cover_last": (not all_secs) or (all_secs[-1] in shown),
                "full_coverage": (not all_secs) or len(set(shown)) == len(set(all_secs)),
                # Sampling is observable from the result: fewer sections shown
                # than the document has. No need to peek at the prompt, so this
                # works identically with a real provider.
                "sampled": len(set(shown)) < len(set(all_secs)) if all_secs else False,
                "within_budget": result["context_chars"] <= task_plans.SUMMARY_BUDGET_CHARS,
                "passages": len(passages),
                "context_chars": result["context_chars"],
            })
        elif task.kind == "compare":
            result = task_plans.compare(retriever, llm, item["question"], task.parts[0], task.parts[1])
            passages = result["context"].passages
            # Per-side passage counts, recovered by re-running each side's
            # retrieval shape is unnecessary: _combine puts A's passages first.
            row.update({
                "passages": len(passages),
                "evidence_level": result["evidence_level"],
                "llm_called": result["llm_called"],
                # Per-side status, because _combine folds the two levels into
                # one and a below-floor side still carries passages.
                "sides": result.get("compare_sides"),
                "missing_side": any(s["evidence_level"] == "none"
                                    for s in result.get("compare_sides", [])),
            })
        else:
            from rag import RAGSystem
            result = RAGSystem(retriever=retriever, llm=llm, debug=False).answer(item["question"])
            row["passages"] = len(result["source_citations"])

        # CITATION VALIDITY and A/B ISOLATION.
        #
        # The label space is read out of the evidence block the model was
        # actually given, so this works uniformly for [S#] and for a
        # comparison's [A#]/[B#] without the harness having to know how the
        # plan split its sides.
        formatted = result["context"].formatted
        real_labels = re.findall(r"\[([A-Z]{1,2}\d+)\]", formatted)
        valid = set(real_labels)
        emitted = set(re.findall(r"\[([A-Z]{1,2}\d+)\]", result["answer"]))
        row["citations_emitted"] = len(emitted)
        row["citations_invalid_left"] = len(emitted - valid)

        if task.kind == "compare":
            # _combine lists A's passages then B's, so cited position p is an
            # A-side source exactly when p <= (number of A labels). An emitted
            # A-label whose verified position lands past that boundary would
            # mean a claim about A citing B's evidence.
            n_a = sum(1 for l in real_labels if l.startswith("A"))
            order = [l for l in real_labels]
            leaks = 0
            for position in result["cited_labels"]:
                if 1 <= position <= len(order):
                    label = order[position - 1]
                    on_a_side = position <= n_a
                    if label.startswith("A") != on_a_side:
                        leaks += 1
            row["ab_labels"] = f"{n_a}A+{len(real_labels) - n_a}B"
            row["ab_isolation_leaks"] = leaks

        row["cited_labels"] = result["cited_labels"]
        row["groundedness"] = grounded(result["answer"], item.get("facts", []))
        row["seconds"] = round(time.monotonic() - started, 2)
        row["answer_chars"] = len(result["answer"])
        rows.append(row)

    out = {"tag": tag, "provider": provider_name, "rows": rows}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{tag}.json").write_text(json.dumps(out, indent=2))
    return out


def report(out):
    rows = [r for r in out["rows"] if "skipped" not in r]
    skipped = [r for r in out["rows"] if "skipped" in r]
    print(f"\n=== TASK EVAL [{out['tag']}]  provider={out['provider']} ===")
    print(f"  cases: {len(rows)}  skipped: {len(skipped)}\n")

    ok = sum(r["task_ok"] for r in rows)
    print(f"  TASK DETECTION        {ok}/{len(rows)}")
    for r in rows:
        if not r["task_ok"]:
            print(f"    MISROUTED {r['id']}: expected {r['expected_task']}, got {r['routed_task']}")

    parts = [r for r in rows if "parts_ok" in r]
    if parts:
        print(f"  COMPARE PARSING       {sum(p['parts_ok'] for p in parts)}/{len(parts)}")
        for p in parts:
            if not p["parts_ok"]:
                print(f"    {p['id']}: expected {p['expected_parts']} got {p['routed_parts']}")

    sums = [r for r in rows if r.get("expected_task") == "summarize" and "cover_first" in r]
    if sums:
        print(f"\n  SUMMARY COVERAGE")
        print(f"    {'id':7s} {'secs':>9s} {'psgs':>5s} {'chars':>6s}  first  last  full  budget")
        for r in sums:
            print(f"    {r['id']:7s} {r['sections_shown']:>4}/{r['sections_total']:<4} "
                  f"{r['passages']:>5} {r['context_chars']:>6}  "
                  f"{'Y' if r['cover_first'] else 'N':^5}  {'Y' if r['cover_last'] else 'N':^4}  "
                  f"{'Y' if r['full_coverage'] else '-':^4}  {'Y' if r['within_budget'] else 'N':^6}")
        print(f"    first-section coverage: {sum(r['cover_first'] for r in sums)}/{len(sums)}")
        print(f"    last-section coverage : {sum(r['cover_last'] for r in sums)}/{len(sums)}")
        print(f"    within budget         : {sum(r['within_budget'] for r in sums)}/{len(sums)}")

    cmps = [r for r in rows if r.get("expected_task") == "compare" and "evidence_level" in r]
    if cmps:
        print(f"\n  COMPARE")
        for r in cmps:
            print(f"    {r['id']:7s} passages={r['passages']:<3} labels={r.get('ab_labels','-'):<7} "
                  f"evidence={r['evidence_level']:<5} missing_side={str(r['missing_side']):<5} "
                  f"A/B leaks={r.get('ab_isolation_leaks','-')}")
        leaks = sum(r.get("ab_isolation_leaks", 0) for r in cmps)
        print(f"    A/B CITATION ISOLATION: {leaks} leak(s) across {len(cmps)} comparisons "
              f"({'an A-claim citing B evidence, or the reverse' if leaks else 'none — labels never cross sides'})")

    cited = [r for r in rows if "citations_invalid_left" in r]
    bad = sum(r["citations_invalid_left"] for r in cited)
    print(f"\n  CITATION VALIDITY     {len(cited) - sum(1 for r in cited if r['citations_invalid_left'])}"
          f"/{len(cited)} cases clean  ({bad} invalid labels survived)")

    if out["provider"] == "none":
        print("\n  GROUNDEDNESS          not measured (structural mode uses a stub model;"
              " run with --provider anthropic|gemini)")
    g = [] if out["provider"] == "none" else [
        r["groundedness"] for r in rows if r.get("groundedness") == r.get("groundedness")]
    if g:
        print(f"  GROUNDEDNESS          {sum(g)/len(g):.2f} mean over {len(g)} scored cases")
        for r in rows:
            if r.get("groundedness") == r.get("groundedness") and r["groundedness"] < 1.0:
                print(f"    {r['id']}: {r['groundedness']:.2f}")
    if skipped:
        print(f"\n  skipped: {[r['id'] for r in skipped]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="none", choices=["none", "anthropic", "gemini"])
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    report(evaluate(args.provider, args.tag or f"tasks-{args.provider}"))
