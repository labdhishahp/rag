"""
Follow-up evaluation — does the second turn retrieve the right evidence?
ZERO LLM calls: the first turn's "answer" is stood in by the gold passage text,
which is what a correct answer would have contained anyway.

For each (first, second) pair:
    dependent    the second message leans on the first (needs history)
    detected     query_understanding flagged it as a follow-up
    raw_hit      gold in context when retrieving the second message AS-IS
    aug_hit      gold in context with history augmentation (the Phase 4 path)

Aggregates:
    follow-up detection accuracy       detected == dependent
    dependent: gold-in-context raw vs augmented   (the improvement)
    standalone: false augmentation rate           (must stay near 0)

Usage:  ./.venv/bin/python eval/run_followups.py [--tag post-step6]
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from conversation import Conversation  # noqa: E402
from embeddings import EmbeddingModel  # noqa: E402
from llm import LLMClient  # noqa: E402
from rag import RAGSystem  # noqa: E402
from run_retrieval import DocIndex, norm, resolve_gold  # noqa: E402

PAIRS = PROJECT_ROOT / "eval" / "followups.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


class StandInLLM(LLMClient):
    """Returns the gold text so the recorded first answer is realistic."""

    def __init__(self):
        self.reply = "(stand-in answer)"

    def generate(self, prompt: str) -> str:
        return self.reply


def covered_ids(result) -> set[int]:
    return {cid for p in result["context"].passages for cid in p.chunk_ids}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="followups")
    args = ap.parse_args()

    pairs = [json.loads(l) for l in PAIRS.read_text().splitlines() if l.strip()]
    model = EmbeddingModel()
    indexes: dict[str, DocIndex] = {}
    rows = []

    for item in pairs:
        path = PROJECT_ROOT / item["doc"]
        if not path.exists():
            continue
        if item["doc"] not in indexes:
            indexes[item["doc"]] = DocIndex(path, model)
        idx = indexes[item["doc"]]
        gold = resolve_gold(idx.chunks, item["gold"])
        if not gold:
            print(f"  !! {item['id']}: gold needle not found: {item['gold']}")
            continue

        llm = StandInLLM()
        rag = RAGSystem(retriever=idx.retriever, llm=llm, debug=False)

        # Turn 1 — answer stands in for what a correct answer would say.
        conv = Conversation()
        first = rag.answer(item["first"], conv)
        first["answer"] = " ".join(idx.by_id[g]["text"] for g in gold[:1])[:400]
        RAGSystem.record_turn(conv, first)

        # Turn 2, two ways: raw (no history) vs with history.
        raw = rag.answer(item["second"], None)
        aug = rag.answer(item["second"], conv)

        rows.append(
            {
                "id": item["id"],
                "dependent": item["dependent"],
                "detected": aug["was_follow_up"],
                "raw_hit": bool(set(gold) & covered_ids(raw)),
                "aug_hit": bool(set(gold) & covered_ids(aug)),
                "retrieval_query": aug["retrieval_query"],
            }
        )
        r = rows[-1]
        flag = "OK " if r["detected"] == r["dependent"] else "MISDETECT"
        print(f"  {r['id']}  dep={str(r['dependent']):<5} detected={str(r['detected']):<5} {flag}  "
              f"raw_hit={str(r['raw_hit']):<5} aug_hit={str(r['aug_hit']):<5} | {r['retrieval_query'][:80]}")

    dep = [r for r in rows if r["dependent"]]
    ind = [r for r in rows if not r["dependent"]]
    summary = {
        "n_dependent": len(dep),
        "n_standalone": len(ind),
        "detection_accuracy": sum(r["detected"] == r["dependent"] for r in rows) / len(rows),
        "dependent_gold_in_context_raw": sum(r["raw_hit"] for r in dep) / len(dep) if dep else 0,
        "dependent_gold_in_context_augmented": sum(r["aug_hit"] for r in dep) / len(dep) if dep else 0,
        "standalone_false_augmentation": sum(r["detected"] for r in ind) / len(ind) if ind else 0,
        "standalone_gold_in_context": sum(r["aug_hit"] for r in ind) / len(ind) if ind else 0,
    }
    print(f"\n=== FOLLOW-UP EVAL [{args.tag}] ===")
    for k, v in summary.items():
        print(f"  {k:<40} {v:.2f}" if isinstance(v, float) else f"  {k:<40} {v}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"followups-{args.tag}.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
