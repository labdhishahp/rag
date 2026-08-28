"""
Turn similarity hits into a complete, non-redundant, budgeted evidence block.

------------------------------------------------------------------------------
THE PROBLEM THIS SOLVES
------------------------------------------------------------------------------
Similarity search answers one question well: "which passage looks most like the
user's question?" It does NOT answer: "which passages do I need in order to
ANSWER the question?"

Measured on our own test document (Stage 0 baseline):

    Question: "What do each of the variables in the compound interest formula mean?"
    Retrieved: chunks 1, 0, 5
    Answer:    "The document context only defines P as the principal. The
                definitions for the remaining variables are not available."

The definitions WERE in the document. They were in the chunk right next door.
The model answered honestly about what it was given; the retrieval layer simply
never gave it the neighbouring chunk.

Why similarity can never fix this on its own:
    A list like "P = principal, r = rate, n = compounding frequency" does not
    resemble a question. It is a symbol table. No amount of top-k tuning makes
    a symbol table out-rank fluent prose that mentions the topic by name.

------------------------------------------------------------------------------
THE ANALOGY
------------------------------------------------------------------------------
Ask a librarian for the compound interest formula.

  A bad librarian photocopies the single line with the formula on it.
  Correct, and useless — you still cannot use the formula.

  A different bad librarian hands you the whole textbook.
  The answer is in there somewhere, buried.

  A good librarian gives you the formula PLUS THE FACING PAGE, where the
  symbols are defined and an example is worked.

This module is the good librarian. Similarity finds the page; adjacency
supplies the facing page; the budget stops us handing over the whole book.

------------------------------------------------------------------------------
THE TRADEOFF, STATED PLAINLY
------------------------------------------------------------------------------
More context                          Less context
-----------------------------------   -----------------------------------
Explanations become answerable        Higher precision, fewer distractions
Formula + variables arrive together   Lower cost and latency
Fewer false "not in the document"     Avoids "lost in the middle": models
                                      attend well to the start and end of a
                                      long context and less well to the
                                      middle, so padding can BURY the answer

So "just raise top_k to 20" trades one failure mode for a subtler one. Instead:

    retrieve NARROWLY by similarity   (find the entry point)
    expand LOCALLY by adjacency       (complete the thought)
    merge overlaps, then TRIM to a budget

------------------------------------------------------------------------------
WHERE THIS SITS
------------------------------------------------------------------------------
    retriever.py     question -> entry-point chunks        (pure similarity)
    context_builder  entry points -> expanded, merged, budgeted evidence
    prompt_builder   evidence + question -> prompt
    llm.py           prompt -> answer

retriever.py is deliberately left as pure similarity search. Stage 4 will hand
the agent a retrieval tool, and a tool with one clear job is easier to reason
about than one that silently expands its own results.
"""

from dataclasses import dataclass, field

# How many chunks to pull in on each side of a similarity hit.
#
# 1 is deliberate. Each chunk is already ~450 characters, so a window of 1
# means one hit contributes ~1350 characters — the formula, plus what comes
# immediately before and after it. A window of 2 doubles the cost for material
# that is, by construction, less related to the question.
DEFAULT_NEIGHBOUR_WINDOW = 1

# Hard ceiling on the evidence block, in characters.
#
# ~5000 chars is roughly 1250 tokens. For comparison, the Stage 0 baseline sent
# 1484 chars. This is a deliberate ~3x increase: enough to carry a formula, its
# variable list and a worked example, while staying far below the point where
# "lost in the middle" degrades the answer.
DEFAULT_CONTEXT_BUDGET_CHARS = 5000

# Minimum similarity an entry chunk needs before we expand around it.
#
# Why this exists — a bug our own Step 3 test caught:
#   Expansion was unconditional. Asked "what is the parental leave policy?" —
#   which our test document cannot answer — similarity returned near-noise
#   (0.136, 0.084), and expansion dutifully pulled in every neighbour of that
#   noise: 7 of 8 chunks, 3066 characters, for a question with no answer.
#
#   That is worse than useless. A bigger pile of irrelevant text gives the model
#   more opportunity to find something that merely LOOKS relevant, which is
#   exactly how a confident wrong answer gets produced.
#
# The principle: adjacency is only worth following from a foothold that is
# actually relevant. Expanding around noise produces more noise.
#
# Entry chunks below this floor are still included (they are what similarity
# found, and the LLM should judge them), but they do not earn neighbours.
DEFAULT_EXPAND_MIN_SIMILARITY = 0.35


@dataclass
class Passage:
    """
    One contiguous run of text, assembled from one or more adjacent chunks.

    A passage is what the LLM actually reads. Merging adjacent chunks into a
    passage matters because the model should see flowing text, not the same
    sentence three times with different headers on it.
    """

    text: str
    document_name: str | None
    document_id: str | None
    page_number: int
    page_label: str
    section: str | None
    chunk_ids: list[int]
    entry_chunk_ids: list[int]  # which of these chunks similarity actually found
    best_similarity: float
    char_start: int
    char_end: int

    @property
    def is_pure_expansion(self) -> bool:
        """True if no chunk here was a similarity hit — adjacency found it all."""
        return not self.entry_chunk_ids


@dataclass
class ContextResult:
    """Everything the answer layer and the debug output need."""

    passages: list[Passage] = field(default_factory=list)
    formatted: str = ""
    entry_chunk_ids: list[int] = field(default_factory=list)
    expanded_chunk_ids: list[int] = field(default_factory=list)
    dropped_chunk_ids: list[int] = field(default_factory=list)
    total_chars: int = 0
    duplicate_chars_removed: int = 0
    # Entry chunks that scored too low to be worth expanding around.
    not_expanded_chunk_ids: list[int] = field(default_factory=list)

    @property
    def has_evidence(self) -> bool:
        return bool(self.passages)


def build_context(
    entry_chunks: list[dict],
    store,
    neighbour_window: int = DEFAULT_NEIGHBOUR_WINDOW,
    budget_chars: int = DEFAULT_CONTEXT_BUDGET_CHARS,
    expand_min_similarity: float = DEFAULT_EXPAND_MIN_SIMILARITY,
    debug: bool = True,
) -> ContextResult:
    """
    Expand similarity hits into a merged, deduplicated, budgeted evidence block.

    Input:  entry_chunks — the output of Retriever.retrieve() (similarity hits)
            store        — the VectorStore, used to fetch neighbours by ID
    Output: ContextResult

    The five stages, in order:
      1. SELECT   entry chunks, plus their neighbours within the window
      2. BUDGET   drop the least valuable expansions until we fit
      3. ORDER    sort by document position — reading order, not score order
      4. MERGE    join adjacent chunks, removing the duplicated overlap
      5. FORMAT   render with real citations
    """
    if not entry_chunks:
        return ContextResult()

    # -------------------------------------------------------------------
    # STAGE 1: SELECT — entry points, then walk outwards to neighbours
    # -------------------------------------------------------------------
    # selected maps a chunk's identity to how we found it and how much we want
    # to keep it. "priority" is lower = more important:
    #   an entry chunk inherits its search rank (0 = best match)
    #   a neighbour inherits its entry's rank, plus how far away it sits
    selected: dict[tuple[str | None, int], dict] = {}

    for rank, chunk in enumerate(entry_chunks):
        key = (chunk.get("document_id"), chunk["chunk_id"])
        selected[key] = {
            "chunk": chunk,
            "origin": "entry",
            "priority": rank,
            "similarity": chunk.get("similarity", 0.0),
        }

    not_expanded: list[int] = []

    for rank, chunk in enumerate(entry_chunks):
        document_id = chunk.get("document_id")

        # Only follow adjacency from a foothold that is actually relevant.
        # See DEFAULT_EXPAND_MIN_SIMILARITY for why.
        if chunk.get("similarity", 0.0) < expand_min_similarity:
            not_expanded.append(chunk["chunk_id"])
            continue

        # Walk backwards then forwards using the prev/next links Step 1 added.
        for direction in ("prev_chunk_id", "next_chunk_id"):
            current = chunk
            for distance in range(1, neighbour_window + 1):
                neighbour_id = current.get(direction)
                if neighbour_id is None:
                    break  # start or end of the document

                neighbour = store.get_chunk(neighbour_id, document_id)
                if neighbour is None:
                    break

                # Never cross a document boundary. Two documents are not
                # continuous text, so "adjacent" is meaningless across them.
                if neighbour.get("document_id") != document_id:
                    break

                key = (document_id, neighbour_id)
                existing = selected.get(key)
                candidate_priority = rank + distance

                if existing is None:
                    selected[key] = {
                        "chunk": neighbour,
                        "origin": "expanded",
                        "priority": candidate_priority,
                        "similarity": 0.0,
                    }
                elif (
                    existing["origin"] == "expanded"
                    and candidate_priority < existing["priority"]
                ):
                    # Reached from a better-ranked entry; promote it.
                    existing["priority"] = candidate_priority

                current = neighbour

    # -------------------------------------------------------------------
    # STAGE 2: BUDGET — trim expansions, never the similarity hits
    # -------------------------------------------------------------------
    # An entry chunk is why we are answering at all; dropping one would discard
    # the actual match. Expansions are enrichment, so they yield first — worst
    # priority (furthest from the best hit) goes first.
    dropped: list[int] = []
    entries = [v for v in selected.values() if v["origin"] == "entry"]
    expansions = sorted(
        (v for v in selected.values() if v["origin"] == "expanded"),
        key=lambda v: v["priority"],
    )

    def size(items) -> int:
        return sum(len(v["chunk"]["text"]) for v in items)

    kept_expansions: list[dict] = []
    running = size(entries)
    for expansion in expansions:
        cost = len(expansion["chunk"]["text"])
        if running + cost <= budget_chars:
            kept_expansions.append(expansion)
            running += cost
        else:
            dropped.append(expansion["chunk"]["chunk_id"])

    chosen = entries + kept_expansions

    # -------------------------------------------------------------------
    # STAGE 3: ORDER — by position in the document, NOT by score
    # -------------------------------------------------------------------
    # Score order would present the formula, then something from page 4, then
    # the formula's variable list. Reading order lets the model follow the
    # document's own argument.
    chosen.sort(
        key=lambda v: (
            str(v["chunk"].get("document_id") or ""),
            v["chunk"]["page_number"],
            v["chunk"]["chunk_id"],
        )
    )

    # -------------------------------------------------------------------
    # STAGE 4: MERGE — join adjacent chunks and delete duplicated overlap
    # -------------------------------------------------------------------
    passages, duplicate_chars = _merge_into_passages(chosen)

    # -------------------------------------------------------------------
    # STAGE 5: FORMAT
    # -------------------------------------------------------------------
    formatted = format_passages(passages)

    result = ContextResult(
        passages=passages,
        formatted=formatted,
        entry_chunk_ids=[v["chunk"]["chunk_id"] for v in entries],
        expanded_chunk_ids=sorted(
            v["chunk"]["chunk_id"] for v in kept_expansions
        ),
        dropped_chunk_ids=sorted(dropped),
        total_chars=sum(len(p.text) for p in passages),
        duplicate_chars_removed=duplicate_chars,
        not_expanded_chunk_ids=sorted(not_expanded),
    )

    if debug:
        print_context_debug(result)

    return result


def _merge_into_passages(chosen: list[dict]) -> tuple[list[Passage], int]:
    """
    Join runs of adjacent chunks into single passages, removing overlap.

    Why overlap must be removed:
      The chunker repeats trailing sentences at the start of the next chunk so
      that a fact spanning a boundary survives in at least one chunk. That is
      correct for indexing and wasteful for reading — if both chunks reach the
      LLM, the repeated sentences arrive twice, spending budget and inviting the
      model to treat repetition as emphasis.

    How we know where the duplicate is:
      Step 1 gave every chunk char_start/char_end into its page's cleaned text.
      If the next chunk starts BEFORE the previous one ended, the difference is
      exactly the number of duplicated characters, so we skip that many.

    Chunks only merge when they are in the same document AND the same page AND
    consecutive by chunk_id. Same page matters twice over: offsets are per-page
    coordinates, and a passage that spanned pages could not be cited precisely.
    """
    passages: list[Passage] = []
    duplicate_chars = 0

    current: dict | None = None

    for item in chosen:
        chunk = item["chunk"]

        if current is not None and _is_continuation(current, chunk):
            overlap = current["char_end"] - chunk["char_start"]
            if overlap > 0:
                addition = chunk["text"][overlap:]
                duplicate_chars += min(overlap, len(chunk["text"]))
                separator = ""
            else:
                addition = chunk["text"]
                # The chunker split on a space or newline here; that separator
                # is not stored, so use a newline. Only ever 1-2 characters.
                separator = "\n"

            current["text"] = current["text"] + separator + addition
            current["char_end"] = max(current["char_end"], chunk["char_end"])
            current["chunk_ids"].append(chunk["chunk_id"])
            if item["origin"] == "entry":
                current["entry_chunk_ids"].append(chunk["chunk_id"])
            current["best_similarity"] = max(
                current["best_similarity"], item["similarity"]
            )
            # A passage's section is the first one we saw; keep it if the
            # continuation has none.
            current["section"] = current["section"] or chunk.get("section")
            continue

        if current is not None:
            passages.append(_finish_passage(current))

        current = {
            "text": chunk["text"],
            "document_name": chunk.get("document_name"),
            "document_id": chunk.get("document_id"),
            "page_number": chunk["page_number"],
            "page_label": chunk.get("page_label", "page"),
            "section": chunk.get("section"),
            "chunk_ids": [chunk["chunk_id"]],
            "entry_chunk_ids": (
                [chunk["chunk_id"]] if item["origin"] == "entry" else []
            ),
            "best_similarity": item["similarity"],
            "char_start": chunk["char_start"],
            "char_end": chunk["char_end"],
        }

    if current is not None:
        passages.append(_finish_passage(current))

    return passages, duplicate_chars


def _is_continuation(current: dict, chunk: dict) -> bool:
    return (
        current["document_id"] == chunk.get("document_id")
        and current["page_number"] == chunk["page_number"]
        and chunk["chunk_id"] == current["chunk_ids"][-1] + 1
    )


def _finish_passage(state: dict) -> Passage:
    return Passage(
        text=state["text"],
        document_name=state["document_name"],
        document_id=state["document_id"],
        page_number=state["page_number"],
        page_label=state["page_label"],
        section=state["section"],
        chunk_ids=state["chunk_ids"],
        entry_chunk_ids=state["entry_chunk_ids"],
        best_similarity=state["best_similarity"],
        char_start=state["char_start"],
        char_end=state["char_end"],
    )


def citation_for(passage: Passage) -> str:
    """
    A human-verifiable source line, built only from real metadata.

    Never invents a page or a section. If the chunker found no heading, the
    section is simply omitted rather than guessed.
    """
    parts = []
    if passage.document_name:
        parts.append(passage.document_name)
    parts.append(f"{passage.page_label} {passage.page_number}")
    if passage.section:
        parts.append(f"section: {passage.section}")
    return " — ".join(parts)


def format_passages(passages: list[Passage]) -> str:
    """
    Render passages as a numbered evidence block.

    Why numbered labels ([S1], [S2]): they give the model a stable handle for
    each source. This is the groundwork for inline citations later; right now
    they simply keep the sources visually separate so the model does not blur
    two documents together.
    """
    if not passages:
        return "(No document passages were retrieved.)"

    blocks = []
    for number, passage in enumerate(passages, start=1):
        blocks.append(f"[S{number}] {citation_for(passage)}\n{passage.text}")
    return "\n\n".join(blocks)


def print_context_debug(result: ContextResult) -> None:
    """
    Show exactly what similarity found and what adjacency added.

    This is the whole point of Step 3 being inspectable: when an answer is
    wrong, you need to see whether the evidence was missing (a retrieval
    problem) or present and ignored (a prompt problem).
    """
    print("\n=== RETRIEVAL: entry points (similarity search) ===")
    print(f"  chunks {result.entry_chunk_ids}")

    print("\n=== EXPANSION: neighbours added by adjacency ===")
    if result.expanded_chunk_ids:
        print(f"  chunks {result.expanded_chunk_ids}")
    else:
        print("  none")
    if result.not_expanded_chunk_ids:
        print(
            f"  not expanded (similarity below floor): "
            f"chunks {result.not_expanded_chunk_ids}"
        )
    if result.dropped_chunk_ids:
        print(f"  dropped to stay within budget: chunks {result.dropped_chunk_ids}")

    print("\n=== CONTEXT: merged passages, in reading order ===")
    for number, passage in enumerate(result.passages, start=1):
        origin = (
            "expansion only"
            if passage.is_pure_expansion
            else f"entry {passage.entry_chunk_ids}"
        )
        print(
            f"  [S{number}] chunks {passage.chunk_ids} | {citation_for(passage)}"
            f" | {len(passage.text)} chars | {origin}"
            f" | best similarity {passage.best_similarity:.3f}"
        )

    print(
        f"\n  total context: {result.total_chars} chars"
        f" | duplicate overlap removed: {result.duplicate_chars_removed} chars"
    )
