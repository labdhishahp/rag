"""
Retrieval and context constants — one place, one meaning each.

Why this file exists:
  The similarity floor used to be defined twice (rag.py and context_builder.py)
  with the same value and the same intent. Two copies drift. Everything that
  interprets a similarity score or sizes a context now reads from here.

How the thresholds were chosen:
  Measured, not guessed, and specific to the embedding model (cosine scores
  are not comparable across models). For BAAI/bge-small-en-v1.5 on the gold
  set (eval/results/exp-bge-small.json), best-hit similarity:

    absent questions        0.415  0.419  0.452  0.506   | 0.758 (a05)
    answerable questions    0.622 (min)  ...  0.917 (max)

  Four of five absent questions sit below 0.51; every answerable question sits
  above 0.62. The hard floor splits that gap. The fifth absent question (a05,
  "Acme's stock price" asked of Acme's financial report) scores 0.758 — a hard
  negative that lands inside the answerable range under EVERY model we tried
  (0.543 with MiniLM, above that model's answerable minimum too). No similarity
  threshold can catch it; the generation prompt must. It stays in the gold set
  precisely so that the refusal path is tested on a case retrieval cannot gate.

  Between HARD and SOFT the system still answers but flags low confidence.

  If you change the embedding model, re-run eval/run_retrieval.py and re-read
  these two numbers off the absent/answerable distribution before trusting them.
"""

# Below this, retrieval found nothing relevant. No expansion; the answer path
# may decline without calling the LLM (Phase 3).
SIMILARITY_HARD_FLOOR = 0.55

# Below this (but above the hard floor) the best hit is weak: answer, but warn.
SIMILARITY_SOFT_FLOOR = 0.65

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
