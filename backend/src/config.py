"""
Retrieval and context constants — one place, one meaning each.

Why this file exists:
  The similarity floor used to be defined twice (rag.py and context_builder.py)
  with the same value and the same intent. Two copies drift. Everything that
  interprets a similarity score or sizes a context now reads from here.

------------------------------------------------------------------------------
WHERE THE SIMILARITY FLOORS COME FROM
------------------------------------------------------------------------------
They were measured, not chosen by feel, and each pair belongs to ONE embedding
model. Cosine scores are not comparable across models: each spreads "unrelated"
and "relevant" over its own range, so a threshold carried over from another
model is a threshold that means nothing. That is why the floors are keyed by
provider and travel with the document (see embeddings.py).

The measurement, per model: over the 28-question gold set (23 answerable, 5
absent) in eval/gold.jsonl, record the best-hit similarity for each question
and put the hard floor in the gap between the two groups.
Reproduce with: ./.venv/bin/python eval/run_retrieval.py --provider <name>

  BAAI/bge-small-en-v1.5 @384  (huggingface, PRIMARY)
      absent        0.415  0.419  0.452  0.506  | 0.758 (a05)
      answerable    0.622 (min) ................. 0.917 (max)
      gap (0.506, 0.622)  ->  hard 0.55,  soft 0.65

  gemini-embedding-001 @768  (gemini, FALLBACK)
      absent        0.504  0.507  0.524  0.535  | 0.691 (a05)
      answerable    0.657 (min) ................. 0.834 (max)
      gap (0.535, 0.657)  ->  hard 0.60,  soft 0.70

In BOTH models four of five absent questions sit clearly below every answerable
one, and the fifth (a05 — "Acme's stock price" asked of Acme's financial
report) lands inside the answerable range. It is a genuine hard negative that
no similarity threshold can catch under any model measured, which is why the
generation prompt also carries a refusal rule.

Between HARD and SOFT the system still answers, but flags low confidence.

If you add a provider or change a model's dimensionality, measure the new pair
against the gold set. Do not interpolate.
"""

# (hard floor, soft floor) per embedding provider. Not interchangeable.
FLOORS_BY_PROVIDER = {
    "huggingface": (0.55, 0.65),
    "gemini": (0.60, 0.70),
}

# Defaults for callers that have no document in hand (a bare Retriever in a
# test, say). Anything answering a real question should use the floors of the
# provider that embedded the document it is searching.
SIMILARITY_HARD_FLOOR, SIMILARITY_SOFT_FLOOR = FLOORS_BY_PROVIDER["huggingface"]

# Relative expansion margin: expand only around entries within this much of
# the best hit. DISABLED (None) after measurement:
#   With 0.15, "explain the compound interest formula in detail" lost the
#   worked example — the variables chunk (0.663) sat 0.174 below the best hit
#   (0.837) and stopped expanding to its neighbour. The noise case this rule
#   targeted (Simple Interest dragging in Present Value) is already prevented
#   by section-scoped expansion, which knows WHERE a subject changes rather
#   than guessing it from a score gap. Kept as a knob for experiments.
EXPANSION_MARGIN: float | None = None

# Neighbours pulled in on each side of an entry, in reading order.
NEIGHBOUR_WINDOW = 1

# Ceiling on the evidence block sent to the LLM. ~5000 chars is ~1250 tokens.
CONTEXT_BUDGET_CHARS = 5000

# Retrieval fetches this many candidates before selection.
DEFAULT_TOP_K = 3
