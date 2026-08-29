"""
Retrieval and context constants — one place, one meaning each.

Why this file exists:
  The similarity floor used to be defined twice (rag.py and context_builder.py)
  with the same value and the same intent. Two copies drift. Everything that
  interprets a similarity score or sizes a context now reads from here.

------------------------------------------------------------------------------
WHY THE FLOORS DEPEND ON THE EMBEDDING MODEL
------------------------------------------------------------------------------
Cosine scores from different embedding models are NOT comparable. Each model
spreads "unrelated" and "relevant" over its own range, so a threshold copied
between models is a threshold that means nothing. That is why these are picked
per backend, from a MEASURED distribution, and why swapping the model without
re-measuring is the one change most likely to break retrieval silently.

The method, for each model: run eval/run_retrieval.py over the gold set, read
the best-hit similarity for every ABSENT question and every ANSWERABLE one, and
put the hard floor in the gap between them.

  BAAI/bge-small-en-v1.5 (local backend)      eval/results/numpy-swap.json
      absent        0.415  0.419  0.452  0.506  | 0.758 (a05)
      answerable    0.622 (min) ................. 0.917 (max)
      gap (0.506, 0.622)  ->  hard 0.55, soft 0.65

  gemini-embedding-001 @768 (deployed)        eval/results/gemini-768.json
      absent        0.504  0.507  0.524  0.535  | 0.691 (a05)
      answerable    0.657 (min) ................. 0.834 (max)
      gap (0.535, 0.657)  ->  hard 0.60, soft 0.70

In BOTH models, four of five absent questions sit clearly below every answerable
one, and the fifth (a05 — "Acme's stock price" asked of Acme's financial report)
lands inside the answerable range. It is a genuine hard negative: no similarity
threshold can catch it under any model we have measured, so the generation
prompt has to. It stays in the gold set precisely so the refusal path is tested
on a case retrieval cannot gate.

  One measured improvement from the swap: with the Gemini floors, a05 (0.691)
  now falls BELOW the soft floor, so it is at least flagged as low confidence.
  0.70 was chosen over 0.69 for exactly that reason — it flags a05 without
  flagging a single additional answerable question (the next one up is 0.707).

Between HARD and SOFT the system still answers but flags low confidence.

If you change the embedding model or its dimensionality, re-run
eval/run_retrieval.py and re-read these numbers off the absent/answerable
distribution before trusting them. Do not interpolate.
"""

import os

# (hard floor, soft floor) per embedding backend. See the docstring for how
# each pair was derived; they are not interchangeable.
_FLOORS_BY_BACKEND = {
    "gemini": (0.60, 0.70),
    "local": (0.55, 0.65),
}

_BACKEND = os.getenv("EMBEDDING_BACKEND", "gemini").strip().lower()

# Resolved once at import: a process uses one embedding backend for its whole
# life, so there is nothing to recompute per call. An unknown backend falls back
# to the deployed default rather than guessing a threshold for a model nobody
# has measured.
SIMILARITY_HARD_FLOOR, SIMILARITY_SOFT_FLOOR = _FLOORS_BY_BACKEND.get(
    _BACKEND, _FLOORS_BY_BACKEND["gemini"]
)

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
