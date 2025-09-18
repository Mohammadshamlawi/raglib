"""Tests for Dual-Encoder technique."""
def create_chunk(id, text, document_id, start_idx=0, end_idx=None):
    """Helper function to create a chunk with required parameters."""
    if end_idx is None:
        end_idx = len(text)
    return Chunk(id=id, text=text, document_id=document_id, start_idx=start_idx, end_idx=end_idx)


from raglib.adapters.base import EmbedderAdapter
from raglib.schemas import Chunk
from raglib.techniques.dual_encoder import DualEncoder


class DummyEmbedder(EmbedderAdapter):
    """Test embedder that returns deterministic embeddings."""

    def __init__(self, seed=0):
        self.seed = seed

    def embed(self, texts):
        embeddings = []
        for i, text in enumerate(texts):
            # Create different embeddings based on seed and text
            base_score = self.seed * 100 + i
            embedding = [
                (len(text) + base_score) / 100.0,
                (text.count('a') + base_score) / 10.0,
                (text.count('e') + base_score) / 10.0,
                (sum(ord(c) for c in text[:3]) + base_score) / 1000.0
            ]
            # Pad to 16 dimensions
            while len(embedding) < 16:
                embedding.append(0.1 * (len(embedding) + 1 + base_score))
            embeddings.append(embedding[:16])
        return embeddings


def test_dual_encoder_initialization():
    """Test dual-encoder initialization."""
    encoder = DualEncoder()
    assert encoder.meta.name == "dual_encoder"
    assert encoder.meta.category == "core_retrieval"
    assert "dense_retrieval" in encoder.meta.tags["type"]


def test_dual_encoder_different_encoders():
    """Test dual-encoder with different query and document encoders."""
    query_encoder = DummyEmbedder()
    doc_encoder = DummyEmbedder()

    encoder = DualEncoder(
        query_encoder=query_encoder,
        doc_encoder=doc_encoder,
        dim=16
    )

    # Add test chunks
    chunks = [
        create_chunk(id="1", text="machine learning", document_id="doc1"),
        create_chunk(id="2", text="deep learning", document_id="doc1"),
    ]

    result = encoder.apply(chunks)
    assert result.success
    assert result.payload["added"] == 2


def test_dual_encoder_search():
    """Test dual-encoder search functionality."""
    query_encoder = DummyEmbedder()
    doc_encoder = DummyEmbedder()

    encoder = DualEncoder(
        query_encoder=query_encoder,
        doc_encoder=doc_encoder,
        dim=16
    )

    # Add test chunks
    chunks = [
        create_chunk(id="1", text="artificial intelligence", document_id="doc1"),
        create_chunk(id="2", text="machine learning algorithms", document_id="doc1"),
        create_chunk(id="3", text="natural language processing", document_id="doc2"),
    ]

    encoder.add_chunks(chunks)

    # Test search
    result = encoder.apply(query="AI technology", top_k=2)

    assert result.success
    assert "hits" in result.payload
    hits = result.payload["hits"]
    assert len(hits) <= 2

    # Verify metadata
    assert "query_encoder" in result.meta
    assert "doc_encoder" in result.meta
    assert result.meta["similarity_metric"] == "cosine"


def test_dual_encoder_similarity_metrics():
    """Test different similarity metrics."""
    encoder_cosine = DualEncoder(similarity_metric="cosine", dim=16)
    encoder_dot = DualEncoder(similarity_metric="dot_product", dim=16)
    encoder_l2 = DualEncoder(similarity_metric="l2", dim=16)

    chunks = [
        create_chunk(id="1", text="test document", document_id="doc1"),
    ]

    # Test each encoder
    for encoder in [encoder_cosine, encoder_dot, encoder_l2]:
        encoder.add_chunks(chunks)
        result = encoder.apply(query="test", top_k=1)
        assert result.success


def test_dual_encoder_shared_encoder():
    """Test dual-encoder with shared encoder (default behavior)."""
    shared_encoder = DummyEmbedder()
    encoder = DualEncoder(query_encoder=shared_encoder, dim=16)

    # Should use the same encoder for both query and documents
    assert encoder.query_encoder is encoder.doc_encoder

    chunks = [
        create_chunk(id="1", text="shared encoder test", document_id="doc1"),
    ]

    encoder.add_chunks(chunks)
    result = encoder.apply(query="test", top_k=1)
    assert result.success


def test_dual_encoder_empty_query():
    """Test dual-encoder with empty query."""
    encoder = DualEncoder()

    result = encoder.apply(query="")
    assert result.success
    assert result.payload["hits"] == []


def test_dual_encoder_no_chunks():
    """Test dual-encoder search with no chunks."""
    encoder = DualEncoder()

    result = encoder.apply(query="test query")
    assert result.success
    assert result.payload["hits"] == []


def test_dual_encoder_metadata_preservation():
    """Test that dual-encoder preserves chunk metadata."""
    encoder = DualEncoder(dim=16)

    chunks = [
        create_chunk(id="chunk1", text="test content", document_id="doc1"),
    ]

    encoder.add_chunks(chunks)
    result = encoder.apply(query="test", top_k=1)

    assert result.success
    hits = result.payload["hits"]
    if hits:
        hit = hits[0]
        assert hit.chunk.id == "chunk1"
        assert hit.chunk.document_id == "doc1"
        assert "encoder" in hit.meta
