"""Tests for MMR (Maximal Marginal Relevance) re-ranker."""

from raglib.adapters.dummy_embedder import DummyEmbedder
from raglib.core import TechniqueResult
from raglib.schemas import Chunk, Hit
from raglib.techniques.mmr import MMRReRanker


def test_mmr_adapterless():
    """Test MMR re-ranking without embedder (fallback mode)."""
    # Create test hits with similar and distinct content
    hits = [
        Hit(doc_id="d1", score=0.9, chunk=Chunk(id="c1", document_id="d1", text="machine learning algorithms", start_idx=0, end_idx=25)),
        Hit(doc_id="d2", score=0.8, chunk=Chunk(id="c2", document_id="d2", text="machine learning models", start_idx=0, end_idx=22)),  # Similar to d1
        Hit(doc_id="d3", score=0.7, chunk=Chunk(id="c3", document_id="d3", text="natural language processing", start_idx=0, end_idx=27)),  # Different topic
        Hit(doc_id="d4", score=0.6, chunk=Chunk(id="c4", document_id="d4", text="deep neural networks", start_idx=0, end_idx=20)),  # Different topic
        Hit(doc_id="d5", score=0.5, chunk=Chunk(id="c5", document_id="d5", text="machine learning techniques", start_idx=0, end_idx=26)),  # Similar to d1, d2
    ]

    # Use more diversity weight (lower lambda) to encourage diverse selection
    mmr = MMRReRanker(lambda_param=0.3)  # 0.3 relevance, 0.7 diversity
    result = mmr.apply(hits=hits, query="machine learning", top_k=3)

    assert isinstance(result, TechniqueResult)
    assert result.success is True
    assert "hits" in result.payload

    selected_hits = result.payload["hits"]
    assert len(selected_hits) <= 3
    assert len(selected_hits) > 0

    # Should contain diverse items, not all similar "machine learning" docs
    selected_doc_ids = {hit.doc_id for hit in selected_hits}
    # At least one diverse document should be selected
    diverse_docs = {"d3", "d4"}  # NLP and neural networks

    # MMR should select some diverse documents, not just the most similar ones
    assert len(selected_doc_ids & diverse_docs) > 0, "MMR should select diverse documents"


def test_mmr_adapterless_deterministic():
    """Test that MMR produces deterministic results."""
    hits = [
        Hit(doc_id="d1", score=0.9, chunk=Chunk(id="c1", document_id="d1", text="apple fruit", start_idx=0, end_idx=11)),
        Hit(doc_id="d2", score=0.8, chunk=Chunk(id="c2", document_id="d2", text="banana fruit", start_idx=0, end_idx=12)),
        Hit(doc_id="d3", score=0.7, chunk=Chunk(id="c3", document_id="d3", text="orange citrus", start_idx=0, end_idx=13)),
    ]

    mmr = MMRReRanker(lambda_param=0.5)

    result1 = mmr.apply(hits=hits, query="fruit", top_k=2)
    result2 = mmr.apply(hits=hits, query="fruit", top_k=2)

    # Results should be identical
    hits1 = result1.payload["hits"]
    hits2 = result2.payload["hits"]

    assert len(hits1) == len(hits2)
    for h1, h2 in zip(hits1, hits2):
        assert h1.doc_id == h2.doc_id


def test_mmr_with_embedder():
    """Test MMR with provided embedder."""
    hits = [
        Hit(doc_id="d1", score=0.9, chunk=Chunk(id="c1", document_id="d1", text="apple fruit sweet", start_idx=0, end_idx=17)),
        Hit(doc_id="d2", score=0.8, chunk=Chunk(id="c2", document_id="d2", text="banana fruit yellow", start_idx=0, end_idx=19)),
        Hit(doc_id="d3", score=0.7, chunk=Chunk(id="c3", document_id="d3", text="car vehicle transport", start_idx=0, end_idx=21)),
    ]

    embedder = DummyEmbedder(dim=16)
    mmr = MMRReRanker(embedder=embedder, lambda_param=0.6)

    result = mmr.apply(hits=hits, query="fruit", top_k=2)

    assert isinstance(result, TechniqueResult)
    assert result.success is True
    selected_hits = result.payload["hits"]
    assert len(selected_hits) <= 2

    # Should be deterministic with same embedder
    result2 = mmr.apply(hits=hits, query="fruit", top_k=2)
    hits2 = result2.payload["hits"]

    assert len(selected_hits) == len(hits2)
    for h1, h2 in zip(selected_hits, hits2):
        assert h1.doc_id == h2.doc_id


def test_mmr_empty_hits():
    """Test MMR with empty hits list."""
    mmr = MMRReRanker()
    result = mmr.apply(hits=[], query="test", top_k=5)

    assert isinstance(result, TechniqueResult)
    assert result.success is True
    assert result.payload["hits"] == []


def test_mmr_with_precomputed_embeddings():
    """Test MMR when hits already have embeddings."""
    # Create hits with pre-computed embeddings
    dummy_embedder = DummyEmbedder(dim=8)
    embeddings = dummy_embedder.embed(["apple red", "apple green", "orange citrus"])

    hits = [
        Hit(doc_id="d1", score=0.9, chunk=Chunk(
            id="c1", document_id="d1", text="apple red",
            start_idx=0, end_idx=9, embedding=embeddings[0])),
        Hit(doc_id="d2", score=0.8, chunk=Chunk(
            id="c2", document_id="d2", text="apple green",
            start_idx=0, end_idx=11, embedding=embeddings[1])),
        Hit(doc_id="d3", score=0.7, chunk=Chunk(
            id="c3", document_id="d3", text="orange citrus",
            start_idx=0, end_idx=13, embedding=embeddings[2])),
    ]

    # Provide query embedding too
    query_embedding = dummy_embedder.embed(["apple"])[0]

    mmr = MMRReRanker(lambda_param=0.8)
    result = mmr.apply(hits=hits, query="apple", query_embedding=query_embedding, top_k=2)

    assert isinstance(result, TechniqueResult)
    assert result.success is True
    selected_hits = result.payload["hits"]
    assert len(selected_hits) <= 2


def test_mmr_with_kwargs():
    """Test MMR with keyword arguments."""
    hits = [
        Hit(doc_id="d1", score=0.9, chunk=Chunk(id="c1", document_id="d1", text="python programming", start_idx=0, end_idx=18)),
        Hit(doc_id="d2", score=0.8, chunk=Chunk(id="c2", document_id="d2", text="java programming", start_idx=0, end_idx=16)),
    ]

    mmr = MMRReRanker()
    result = mmr.apply(hits=hits, query="programming", top_k=1)

    assert isinstance(result, TechniqueResult)
    assert result.success is True
    assert len(result.payload["hits"]) == 1
