"""
Retrieval and context constants — one place, one meaning each.

Why this file exists:
  The similarity floor used to be defined twice (rag.py and context_builder.py)
  with the same value and the same intent. Two copies drift. Everything that
  interprets a similarity score or sizes a context now reads from here.

------------------------------------------------------------------------------
WHERE THE SIMILARITY FLOORS COME FROM
------------------------------------------------------------------------------
They were measured, not chosen by feel, and they belong to ONE embedding model
(gemini-embedding-001 at 768 dimensions). Cosine scores are not comparable
across models: each spreads "unrelated" and "relevant" over its own range, so a
threshold carried over from another model is a threshold that means nothing.

The measurement: over a 28-question gold set (23 answerable, 5 absent), record
the best-hit similarity for each question and put the hard floor in the gap.

    absent        0.504  0.507  0.524  0.535  | 0.691 (a05)
    answerable    0.657 (min) ................. 0.834 (max)
    gap (0.535, 0.657)  ->  hard 0.60,  soft 0.70

Four of the five absent questions sit clearly below every answerable one. The
fifth (a05 — "Acme's stock price" asked of Acme's financial report) lands
inside the answerable range: a genuine hard negative that no similarity
threshold can catch, which is why the generation prompt also carries a refusal
rule. The soft floor is 0.70 rather than 0.69 because 0.70 flags a05 as low
confidence without flagging one additional answerable question (the next is
0.707).

Between HARD and SOFT the system still answers, but flags low confidence.

If you change the embedding model or its dimensionality, these two numbers must
be re-measured against a gold set. Do not interpolate. The evaluation harness
that produced them is in git history (removed in the runtime-only cleanup);
recover it with `git log --diff-filter=D -- eval/`.
"""

# Below this, retrieval found nothing relevant. No expansion, and the answer
# path declines WITHOUT calling the LLM (see rag.py).
SIMILARITY_HARD_FLOOR = 0.60

# Below this (but above the hard floor) the best hit is weak: answer, but warn.
SIMILARITY_SOFT_FLOOR = 0.70

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
