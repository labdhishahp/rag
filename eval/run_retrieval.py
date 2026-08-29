"""
Retrieval + context evaluation. ZERO LLM calls, so it can run on every change.

Answers one question: after retrieval and context expansion, does the evidence
the LLM would receive actually contain the answer — and how much noise came
with it?

Per gold question:
    gold_in_entries    a gold chunk is among the raw top-k similarity hits
    gold_in_context    a gold chunk is in the final context (after expansion)
    context_precision  chars from gold chunks / total context chars
    context_chars      size of the evidence block
    best_similarity    top hit score (drives the evidence gate)
    n_entries / n_expanded / n_passages

Aggregates per document and overall. Gold is specified as TEXT NEEDLES, not
chunk ids, because ids shift whenever chunking changes; needles are resolved
to chunk ids at run time and unresolved needles are reported loudly.

Usage:
    ./.venv/bin/python eval/run_retrieval.py --tag pre-step4
    ./.venv/bin/python eval/run_retrieval.py --tag post-step4
    ./.venv/bin/python eval/run_retrieval.py --compare pre-step4 post-step4
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chunker import chunk_pages  # noqa: E402
from context_builder import build_context  # noqa: E402
from document_loader import load_document_from_bytes  # noqa: E402
from embeddings import EmbeddingModel  # noqa: E402
from pipeline import make_document_id  # noqa: E402
from retriever import Retriever  # noqa: E402
from vector_store import VectorStore  # noqa: E402

GOLD_PATH = PROJECT_ROOT / "eval" / "gold.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
TOP_K = 3


def load_gold() -> list[dict]:
    return [json.loads(line) for line in GOLD_PATH.read_text().splitlines() if line.strip()]


def norm(s: str) -> str:
    return " ".join(s.split()).lower()


def resolve_gold(chunks: list[dict], needles: list[str]) -> list[int]:
    ids = set()
    for needle in needles:
        n = norm(needle)
        ids.update(c["chunk_id"] for c in chunks if n in norm(c["text"]))
    return sorted(ids)


class DocIndex:
    def __init__(self, path: Path, model: EmbeddingModel):
        data = path.read_bytes()
        pages = load_document_from_bytes(data, path.name)
        self.chunks = chunk_pages(
            pages, 500, 50, document_name=path.name, document_id=make_document_id(data)
        )
        self.store = VectorStore(dimension=model.dimension)
        self.store.add(model.embed_texts([c["text"] for c in self.chunks]), self.chunks)
        self.retriever = Retriever(model, self.store)
        self.by_id = {c["chunk_id"]: c for c in self.chunks}


def evaluate(tag: str, model_name: str | None = None, top_k: int = TOP_K,
             query_prefix: str | None = None) -> dict:
    """
    model_name / top_k exist so that "should we change the embedding model or
    top_k?" is answered by this harness, not by opinion. Run once per variant
    with a distinct tag, then --compare.
    """
    gold = load_gold()
    model = EmbeddingModel(model_name, query_prefix=query_prefix) if model_name else EmbeddingModel(query_prefix=query_prefix)
    indexes: dict[str, DocIndex] = {}
    rows = []
    skipped = []

    for item in gold:
        path = PROJECT_ROOT / item["doc"]
        if not path.exists():
            skipped.append(item["id"])
            continue
        if item["doc"] not in indexes:
            indexes[item["doc"]] = DocIndex(path, model)
        idx = indexes[item["doc"]]

        gold_ids = resolve_gold(idx.chunks, item["gold"])
        if item["answerable"] and not gold_ids:
            print(f"  !! {item['id']}: gold needle not found in any chunk: {item['gold']}")

        t0 = time.perf_counter()
        entries = idx.retriever.retrieve(item["question"], top_k=top_k)
        ctx = build_context(entries, idx.store, debug=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        covered = [cid for p in ctx.passages for cid in p.chunk_ids]
        entry_ids = [e["chunk_id"] for e in entries]
        gold_chars = sum(len(idx.by_id[cid]["text"]) for cid in covered if cid in gold_ids)

        rows.append(
            {
                "id": item["id"],
                "doc": item["doc"],
                "type": item["type"],
                "answerable": item["answerable"],
                "gold_ids": gold_ids,
                "entry_ids": entry_ids,
                "expanded_ids": ctx.expanded_chunk_ids,
                "context_ids": covered,
                "gold_in_entries": bool(set(gold_ids) & set(entry_ids)),
                "gold_in_context": bool(set(gold_ids) & set(covered)),
                "context_precision": (gold_chars / ctx.total_chars) if ctx.total_chars else 0.0,
                "context_chars": ctx.total_chars,
                "n_passages": len(ctx.passages),
                "n_expanded": len(ctx.expanded_chunk_ids),
                "best_similarity": entries[0]["similarity"] if entries else 0.0,
                "elapsed_ms": elapsed_ms,
            }
        )

    summary = summarise(rows)
    result = {
        "tag": tag,
        "top_k": top_k,
        "embedding_model": model_name or "default",
        "skipped": skipped,
        "rows": rows,
        "summary": summary,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{tag}.json").write_text(json.dumps(result, indent=2))
    return result


def summarise(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    absent = [r for r in rows if not r["answerable"]]

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    overall = {
        "n_answerable": len(answerable),
        "n_absent": len(absent),
        "gold_in_entries": mean([r["gold_in_entries"] for r in answerable]),
        "gold_in_context": mean([r["gold_in_context"] for r in answerable]),
        "context_precision": mean([r["context_precision"] for r in answerable]),
        "context_chars_answerable": mean([r["context_chars"] for r in answerable]),
        "context_chars_absent": mean([r["context_chars"] for r in absent]),
        "absent_best_similarity_max": max([r["best_similarity"] for r in absent], default=0.0),
        "answerable_best_similarity_min": min([r["best_similarity"] for r in answerable], default=0.0),
        "mean_elapsed_ms": mean([r["elapsed_ms"] for r in rows]),
    }
    per_doc = {}
    for doc in sorted({r["doc"] for r in rows}):
        sub = [r for r in answerable if r["doc"] == doc]
        per_doc[doc] = {
            "n": len(sub),
            "gold_in_entries": mean([r["gold_in_entries"] for r in sub]),
            "gold_in_context": mean([r["gold_in_context"] for r in sub]),
            "context_precision": mean([r["context_precision"] for r in sub]),
            "context_chars": mean([r["context_chars"] for r in sub]),
        }
    return {"overall": overall, "per_doc": per_doc}


def print_report(result: dict) -> None:
    s = result["summary"]["overall"]
    print(f"\n=== RETRIEVAL EVAL [{result['tag']}]  top_k={result['top_k']}  "
          f"model={result.get('embedding_model', 'default')} ===")
    if result["skipped"]:
        print(f"  skipped (document missing): {result['skipped']}")
    print(f"  answerable questions      : {s['n_answerable']}   absent: {s['n_absent']}")
    print(f"  gold in entries (recall@k): {s['gold_in_entries']:.2f}")
    print(f"  gold in context           : {s['gold_in_context']:.2f}   <- what the LLM actually sees")
    print(f"  context precision         : {s['context_precision']:.2f}   (gold chars / context chars)")
    print(f"  context chars, answerable : {s['context_chars_answerable']:.0f}")
    print(f"  context chars, absent     : {s['context_chars_absent']:.0f}")
    print(f"  best-sim: absent max      : {s['absent_best_similarity_max']:.3f}   answerable min: {s['answerable_best_similarity_min']:.3f}")
    print(f"  mean retrieval+build time : {s['mean_elapsed_ms']:.0f} ms")
    print("\n  per document:")
    for doc, d in result["summary"]["per_doc"].items():
        print(f"    {Path(doc).name:<22} n={d['n']:<2} entries={d['gold_in_entries']:.2f} "
              f"context={d['gold_in_context']:.2f} precision={d['context_precision']:.2f} chars={d['context_chars']:.0f}")
    print("\n  misses (answerable, gold NOT in context):")
    for r in result["rows"]:
        if r["answerable"] and not r["gold_in_context"]:
            print(f"    {r['id']}  gold={r['gold_ids']} entries={r['entry_ids']} best={r['best_similarity']:.3f}")


def compare(tag_a: str, tag_b: str) -> None:
    a = json.loads((RESULTS_DIR / f"{tag_a}.json").read_text())
    b = json.loads((RESULTS_DIR / f"{tag_b}.json").read_text())
    sa, sb = a["summary"]["overall"], b["summary"]["overall"]
    print(f"\n=== COMPARE  {tag_a}  ->  {tag_b} ===")
    for key in ["gold_in_entries", "gold_in_context", "context_precision",
                "context_chars_answerable", "context_chars_absent"]:
        va, vb = sa[key], sb[key]
        arrow = "up" if vb > va else ("down" if vb < va else "same")
        fmt = "{:.0f}" if "chars" in key else "{:.2f}"
        print(f"  {key:<26} {fmt.format(va):>8} -> {fmt.format(vb):>8}   {arrow}")
    ra = {r["id"]: r for r in a["rows"]}
    rb = {r["id"]: r for r in b["rows"]}
    print("\n  per-question changes:")
    for qid in sorted(set(ra) & set(rb)):
        x, y = ra[qid], rb[qid]
        if x["gold_in_context"] != y["gold_in_context"] or abs(x["context_chars"] - y["context_chars"]) > 200:
            print(f"    {qid}: in_context {x['gold_in_context']}->{y['gold_in_context']}  "
                  f"chars {x['context_chars']}->{y['context_chars']}  "
                  f"ids {x['context_ids']}->{y['context_ids']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="adhoc")
    ap.add_argument("--model", default=None, help="sentence-transformers model name to try")
    ap.add_argument("--no-query-prefix", action="store_true",
                    help="disable the model's documented query instruction prefix")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"))
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    print_report(evaluate(args.tag, model_name=args.model, top_k=args.top_k,
                          query_prefix="" if args.no_query_prefix else None))


if __name__ == "__main__":
    main()
