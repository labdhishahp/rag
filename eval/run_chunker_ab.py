"""
A/B the production chunker against the EXPERIMENTAL recursive chunker.

Nothing here touches the running application: the recursive chunker lives in
eval/experimental/ and is imported only by this script.

Controlled so that chunking is the only variable:
  - same documents           (eval/gold.jsonl -> data/*.pdf)
  - same questions           (the 28 gold rows)
  - same embedding provider  ONE model instance shared by both arms, so the
                             vectors provably live in the same space
  - same retrieval settings  same VectorStore, same Retriever, same top_k
  - same context settings    same build_context and similarity floors
  - same post-processing      runt merge, section assignment, prev/next links

Primary metric: Recall@3 — for each answerable question, is a gold chunk among
the top 3 similarity hits. Reported alongside gold_in_context (what the LLM
would actually receive) because a retrieval win that expansion throws away is
not a win.

Usage:
    ./.venv/bin/python eval/run_chunker_ab.py
    ./.venv/bin/python eval/run_chunker_ab.py --top-k 3 --provider huggingface
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from chunker import chunk_pages  # noqa: E402
from context_builder import build_context  # noqa: E402
from document_loader import load_document_from_bytes  # noqa: E402
from embeddings import create_provider  # noqa: E402
from pipeline import make_document_id  # noqa: E402
from retriever import Retriever  # noqa: E402
from vector_store import VectorStore  # noqa: E402

from experimental.recursive_chunker import chunk_pages_recursive  # noqa: E402

GOLD_PATH = PROJECT_ROOT / "eval" / "gold.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
# Retrieve deeper than we score, so a miss can be reported as "gold was at
# rank 5" instead of just "absent". Recall@3 always uses the first 3.
PROBE_K = 10


def norm(s: str) -> str:
    return " ".join(s.split()).lower()


def load_gold() -> list[dict]:
    return [json.loads(l) for l in GOLD_PATH.read_text().splitlines() if l.strip()]


def resolve_gold(chunks: list[dict], needles: list[str]) -> list[int]:
    ids = set()
    for needle in needles:
        n = norm(needle)
        ids.update(c["chunk_id"] for c in chunks if n in norm(c["text"]))
    return sorted(ids)


class Arm:
    """One chunking strategy, indexed and queryable."""

    def __init__(self, name: str, chunk_fn, model):
        self.name = name
        self.chunk_fn = chunk_fn
        self.model = model
        self.indexes: dict[str, dict] = {}

    def index(self, doc: str) -> dict:
        if doc in self.indexes:
            return self.indexes[doc]
        path = PROJECT_ROOT / doc
        data = path.read_bytes()
        pages = load_document_from_bytes(data, path.name)
        chunks = self.chunk_fn(
            pages, 500, 50, document_name=path.name, document_id=make_document_id(data)
        )
        # Offset invariant: if this fails the arm is not comparable, because
        # citations and context expansion both rely on it.
        page_text = {}
        for page in pages:
            from chunker import page_text_for_chunking
            page_text[page["page_number"]] = page_text_for_chunking(page)
        bad = [
            c["chunk_id"]
            for c in chunks
            if page_text[c["page_number"]][c["char_start"]:c["char_end"]] != c["text"]
        ]
        store = VectorStore(dimension=self.model.dimension)
        store.add(self.model.embed_texts([c["text"] for c in chunks]), chunks)
        entry = {
            "chunks": chunks,
            "store": store,
            "retriever": Retriever(self.model, store),
            "by_id": {c["chunk_id"]: c for c in chunks},
            "offset_violations": bad,
        }
        self.indexes[doc] = entry
        return entry

    def run(self, gold: list[dict], top_k: int) -> dict:
        rows = {}
        for item in gold:
            path = PROJECT_ROOT / item["doc"]
            if not path.exists():
                continue
            idx = self.index(item["doc"])
            gold_ids = resolve_gold(idx["chunks"], item["gold"])

            entries = idx["retriever"].retrieve(item["question"], top_k=PROBE_K)
            ranked = [e["chunk_id"] for e in entries]
            top3 = ranked[:top_k]

            ctx = build_context(
                entries[:top_k], idx["store"], debug=False,
                hard_floor=self.model.hard_floor, soft_floor=self.model.soft_floor,
            )
            covered = [cid for p in ctx.passages for cid in p.chunk_ids]

            gold_rank = next(
                (i + 1 for i, cid in enumerate(ranked) if cid in set(gold_ids)), None
            )
            rows[item["id"]] = {
                "id": item["id"],
                "doc": item["doc"],
                "type": item["type"],
                "answerable": item["answerable"],
                "n_chunks_doc": len(idx["chunks"]),
                "gold_ids": gold_ids,
                "needle_resolved": bool(gold_ids) or not item["answerable"],
                "top3": top3,
                "recall_at_3": bool(set(gold_ids) & set(top3)),
                "gold_rank": gold_rank,
                "gold_in_context": bool(set(gold_ids) & set(covered)),
                "best_similarity": entries[0]["similarity"] if entries else 0.0,
                "gold_similarity": max(
                    [e["similarity"] for e in entries if e["chunk_id"] in set(gold_ids)],
                    default=None,
                ),
                "context_chars": ctx.total_chars,
            }
        return rows


def chunk_stats(arm: Arm) -> dict:
    out = {}
    for doc, idx in arm.indexes.items():
        lens = [len(c["text"]) for c in idx["chunks"]]
        out[doc] = {
            "n": len(lens),
            "mean_len": sum(lens) / len(lens),
            "min_len": min(lens),
            "max_len": max(lens),
            "offset_violations": len(idx["offset_violations"]),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="huggingface")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    gold = load_gold()
    # ONE model instance, shared. Both arms therefore embed with identical
    # weights in an identical vector space; nothing about the provider differs.
    model = create_provider(args.provider)
    print(f"embedding: {model.model_name}  dim={model.dimension}  "
          f"floors=({model.hard_floor}, {model.soft_floor})")
    print(f"questions: {len(gold)}  top_k={args.top_k}  probe_k={PROBE_K}\n")

    arms = {
        "production": Arm("production", chunk_pages, model),
        "recursive": Arm("recursive", chunk_pages_recursive, model),
    }
    results = {name: arm.run(gold, args.top_k) for name, arm in arms.items()}

    answerable = [g["id"] for g in gold if g["answerable"] and (PROJECT_ROOT / g["doc"]).exists()]
    absent = [g["id"] for g in gold if not g["answerable"] and (PROJECT_ROOT / g["doc"]).exists()]

    def rate(name, key, ids):
        rows = results[name]
        vals = [rows[i][key] for i in ids if i in rows]
        return sum(vals) / len(vals) if vals else 0.0

    print("=== CHUNKING ===")
    for name, arm in arms.items():
        st = chunk_stats(arm)
        total = sum(d["n"] for d in st.values())
        viol = sum(d["offset_violations"] for d in st.values())
        print(f"  {name:<11} total_chunks={total:<5} offset_violations={viol}")
        for doc, d in sorted(st.items()):
            print(f"      {Path(doc).name:<24} n={d['n']:<4} mean={d['mean_len']:.0f} "
                  f"min={d['min_len']:<4} max={d['max_len']}")

    print(f"\n=== RECALL@{args.top_k}  (answerable n={len(answerable)}) ===")
    pr = rate("production", "recall_at_3", answerable)
    rr = rate("recursive", "recall_at_3", answerable)
    print(f"  production  {pr:.3f}  ({sum(results['production'][i]['recall_at_3'] for i in answerable)}/{len(answerable)})")
    print(f"  recursive   {rr:.3f}  ({sum(results['recursive'][i]['recall_at_3'] for i in answerable)}/{len(answerable)})")
    print(f"  delta       {rr - pr:+.3f}")

    print(f"\n=== gold_in_context (what the LLM receives) ===")
    pc = rate("production", "gold_in_context", answerable)
    rc = rate("recursive", "gold_in_context", answerable)
    print(f"  production  {pc:.3f}     recursive  {rc:.3f}     delta {rc - pc:+.3f}")

    # A needle that resolves to no chunk means chunking split the gold text
    # across a boundary — a chunking failure, not a retrieval one.
    print("\n=== unresolved gold needles (gold text split across a boundary) ===")
    for name in arms:
        bad = [i for i in answerable if not results[name][i]["needle_resolved"]]
        print(f"  {name:<11} {len(bad)} {bad}")

    print("\n=== QUESTIONS THAT CHANGED ===")
    changed = [
        i for i in answerable
        if results["production"][i]["recall_at_3"] != results["recursive"][i]["recall_at_3"]
    ]
    if not changed:
        print("  none — Recall@3 identical on every answerable question")
    for i in changed:
        p, r = results["production"][i], results["recursive"][i]
        q = next(g for g in gold if g["id"] == i)
        direction = "GAINED" if r["recall_at_3"] else "LOST"
        print(f"\n  [{direction}] {i}  ({q['type']}, {Path(q['doc']).name})")
        print(f"    Q: {q['question']}")
        print(f"    needle: {q['gold'][0][:88]!r}")
        for name, row in (("production", p), ("recursive", r)):
            gs = "--" if row["gold_similarity"] is None else format(row["gold_similarity"], ".3f")
            print(f"    {name:<11} gold_ids={row['gold_ids']} top3={row['top3']} "
                  f"gold_rank={row['gold_rank']} gold_sim={gs} "
                  f"best_sim={row['best_similarity']:.3f} doc_chunks={row['n_chunks_doc']}")

    print("\n=== RANK MOVEMENT on unchanged questions (gold rank, deeper probe) ===")
    for i in answerable:
        p, r = results["production"][i], results["recursive"][i]
        if p["recall_at_3"] == r["recall_at_3"] and p["gold_rank"] != r["gold_rank"]:
            print(f"  {i:<8} rank {p['gold_rank']} -> {r['gold_rank']}  "
                  f"sim {'--' if p['gold_similarity'] is None else format(p['gold_similarity'], '.3f')}"
                  f" -> {'--' if r['gold_similarity'] is None else format(r['gold_similarity'], '.3f')}")

    print("\n=== ABSENT questions (must stay unretrievable; guards the refusal gate) ===")
    for name in arms:
        sims = [results[name][i]["best_similarity"] for i in absent]
        print(f"  {name:<11} max_best_similarity={max(sims):.3f}  "
              f"(hard_floor={model.hard_floor})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "chunker_ab.json"
    out.write_text(json.dumps({
        "embedding_model": model.model_name,
        "top_k": args.top_k,
        "summary": {
            "recall_at_3": {"production": pr, "recursive": rr},
            "gold_in_context": {"production": pc, "recursive": rc},
        },
        "chunk_stats": {n: chunk_stats(a) for n, a in arms.items()},
        "rows": results,
    }, indent=2))
    print(f"\nwrote {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
