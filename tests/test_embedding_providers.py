"""
The provider/index relationship, which is the part of the fallback that can
silently corrupt answers if it is wrong.

An embedding only means something inside the space that produced it. bge-small
returns 384 numbers and gemini-embedding-001 returns 768; a similarity computed
across the two is nonsense even when the arithmetic runs. So the rule these
tests pin down is:

    fallback may choose a provider when INDEXING a document,
    never when QUERYING one.

Most of this needs no network: it is about which provider is selected and what
happens when the wrong one is used.
"""

import numpy as np
import pytest

from config import FLOORS_BY_PROVIDER
from context_builder import evidence_level_for
from vector_store import VectorStore


# ---- floors travel with the provider ---------------------------------------

def test_each_provider_has_its_own_measured_floors():
    """A threshold copied between models is a threshold that means nothing."""
    assert FLOORS_BY_PROVIDER["huggingface"] == (0.55, 0.65)
    assert FLOORS_BY_PROVIDER["gemini"] == (0.60, 0.70)
    assert FLOORS_BY_PROVIDER["huggingface"] != FLOORS_BY_PROVIDER["gemini"]


def test_same_score_can_mean_different_things_under_different_providers():
    """
    The two floor pairs disagree over two whole bands, so the gate must use the
    floors of the document being queried rather than a global constant.

      0.57  -> bge-small says "weak" (answer, with a caution)
               Gemini says "none"    (refuse without calling the LLM)
      0.67  -> bge-small says "ok"
               Gemini says "weak"

    Using the wrong pair therefore either invents an answer from noise or
    refuses a question it could have answered.
    """
    hf = FLOORS_BY_PROVIDER["huggingface"]
    gm = FLOORS_BY_PROVIDER["gemini"]
    assert evidence_level_for(0.57, *hf) == "weak"
    assert evidence_level_for(0.57, *gm) == "none"
    assert evidence_level_for(0.67, *hf) == "ok"
    assert evidence_level_for(0.67, *gm) == "weak"


def test_evidence_gate_boundaries_are_inclusive_of_the_floor():
    hard, soft = FLOORS_BY_PROVIDER["huggingface"]
    assert evidence_level_for(hard - 0.001, hard, soft) == "none"
    assert evidence_level_for(hard, hard, soft) == "weak"
    assert evidence_level_for(soft, hard, soft) == "ok"


# ---- the vector store refuses mismatched dimensions ------------------------

def test_store_rejects_vectors_from_a_different_embedding_space():
    """
    The last line of defence: even if provider selection were wrong somewhere,
    a 768-d vector cannot be added to a 384-d index.
    """
    store = VectorStore(dimension=384)
    chunks = [{"chunk_id": 0, "document_id": "d", "text": "x"}]
    with pytest.raises(ValueError, match="dimension"):
        store.add(np.zeros((1, 768), dtype=np.float32), chunks)


def test_store_rejects_a_query_vector_of_the_wrong_width():
    """A 768-d query against a 384-d index must fail, not return a ranking."""
    store = VectorStore(dimension=384)
    store.add(
        np.ones((2, 384), dtype=np.float32) / np.sqrt(384),
        [{"chunk_id": i, "document_id": "d", "text": "x", "page_number": 1} for i in range(2)],
    )
    with pytest.raises(ValueError):
        store.search(np.ones(768, dtype=np.float32) / np.sqrt(768), top_k=1)


# ---- provider selection ----------------------------------------------------

def test_query_time_selection_never_falls_back(monkeypatch):
    """
    provider_for_document must return the recorded provider or raise. If it
    quietly substituted another one, every similarity for that document would
    be computed across two different spaces.
    """
    import embeddings

    built = []

    def fake_create(name):
        built.append(name)
        if name == "huggingface":
            raise embeddings.EmbeddingError("primary down")
        return object()

    monkeypatch.setattr(embeddings, "create_provider", fake_create)
    with pytest.raises(embeddings.EmbeddingError):
        embeddings.provider_for_document("huggingface")
    # It tried exactly one provider — the one recorded — and gave up.
    assert built == ["huggingface"]


def test_index_time_selection_does_fall_back(monkeypatch):
    """Choosing a provider for a NEW document is the one place fallback is safe."""
    import embeddings

    built = []

    def fake_create(name):
        built.append(name)
        if name == "huggingface":
            raise embeddings.EmbeddingError("primary down")
        marker = type("P", (), {"name": "gemini"})()
        return marker

    monkeypatch.setattr(embeddings, "create_provider", fake_create)
    provider = embeddings.provider_for_indexing("huggingface", "gemini")
    assert provider.name == "gemini"
    assert built == ["huggingface", "gemini"]


def test_document_with_no_recorded_provider_is_treated_as_gemini(monkeypatch):
    """Documents indexed before providers were recorded predate the HF switch."""
    import embeddings

    seen = []
    monkeypatch.setattr(embeddings, "create_provider", lambda n: seen.append(n))
    embeddings.provider_for_document(None)
    assert seen == ["gemini"]


def test_unknown_provider_name_is_rejected():
    import embeddings

    with pytest.raises(embeddings.EmbeddingError, match="Unknown embedding provider"):
        embeddings.create_provider("not-a-provider")


# ---- the live primary provider ---------------------------------------------

@pytest.mark.embedding
def test_primary_provider_batches_and_normalizes(embedding_model):
    """
    The two properties the pipeline depends on: one vector PER input (a
    truncated batch would silently misalign vectors and chunks), and unit
    length (the store treats a dot product as a cosine similarity).
    """
    texts = [f"chunk number {i} about compound interest" for i in range(5)]
    vectors = embedding_model.embed_texts(texts)
    assert vectors.shape == (5, embedding_model.dimension)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    # Distinct inputs must not collapse to the same vector — the failure mode
    # of an endpoint that aggregates a batch instead of embedding each item.
    assert not np.allclose(vectors[0], vectors[1])


@pytest.mark.embedding
def test_primary_provider_is_bge_small_at_384(embedding_model):
    assert embedding_model.name == "huggingface"
    assert embedding_model.dimension == 384
    assert "bge-small" in embedding_model.model_name
    assert (embedding_model.hard_floor, embedding_model.soft_floor) == (0.55, 0.65)


@pytest.mark.embedding
def test_query_and_document_encodings_differ_but_stay_comparable(embedding_model):
    """A question is encoded to be similar to its ANSWER, not to other questions."""
    doc = embedding_model.embed_texts(["The compound interest formula is A = P(1 + r/n)^(nt)."])[0]
    query = embedding_model.embed_query("What is the compound interest formula?")
    assert query.shape == doc.shape
    assert float(np.dot(query, doc)) > 0.6
