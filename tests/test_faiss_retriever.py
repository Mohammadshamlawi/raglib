"""Tests for FAISS Retriever technique."""

from raglib.adapters.base import EmbedderAdapter
from raglib.schemas import Chunk
from raglib.techniques.faiss_retriever import FAISSRetriever


class DummyEmbedder(EmbedderAdapter):
    """Test embedder that returns deterministic embeddings."""

    def embed(self, texts):
        embeddings = []
        for text in texts:
            # Simple deterministic embedding based on text
            embedding = [
                len(text) / 100.0,
                text.count('a') / 10.0,
                text.count('e') / 10.0,
                sum(ord(c) for c in text[:3]) / 1000.0
            ]
            # Pad to 16 dimensions for testing
            while len(embedding) < 16:
                embedding.append(0.1 * (len(embedding) + 1))
            embeddings.append(embedding[:16])
        return embeddings


def create_chunk(chunk_id: str, text: str, doc_id: str) -> Chunk:
    """Helper function to create chunks with proper parameters."""
    return Chunk(
        id=chunk_id,
        text=text,
        document_id=doc_id,
        start_idx=0,
        end_idx=len(text)
    )


def test_faiss_retriever_initialization():
    """Test FAISS retriever initialization."""
    retriever = FAISSRetriever()
    assert retriever.meta.name == "faiss_retriever"
    assert retriever.meta.category == "core_retrieval"
    assert "dense_retrieval" in retriever.meta.tags["type"]


def test_faiss_retriever_with_custom_embedder():
    """Test FAISS retriever with custom embedder."""
    embedder = DummyEmbedder()
    retriever = FAISSRetriever(embedder=embedder, dimension=16)

    # Test adding chunks
    chunks = [
        create_chunk("1", "machine learning algorithms", "doc1"),
        create_chunk("2", "deep neural networks", "doc1"),
        create_chunk("3", "natural language processing", "doc2"),
    ]

    result = retriever.apply(chunks)
    assert result.success
    assert result.payload["added"] == 3


def test_faiss_retriever_search():
    """Test FAISS retriever search functionality."""
    embedder = DummyEmbedder()
    retriever = FAISSRetriever(embedder=embedder, dimension=16)

    # Add test chunks
    chunks = [
        create_chunk("1", "machine learning is great", "doc1"),
        create_chunk("2", "deep learning neural networks", "doc1"),
        create_chunk("3", "natural language processing", "doc2"),
    ]

    retriever.add_chunks(chunks)

    # Test search
    result = retriever.apply(query="machine learning", top_k=2)

    assert result.success
    assert "hits" in result.payload
    hits = result.payload["hits"]
    assert len(hits) <= 2

    # Verify hit structure
    if hits:
        hit = hits[0]
        assert hasattr(hit, "doc_id")
        assert hasattr(hit, "score")
        assert hasattr(hit, "chunk")


def test_faiss_retriever_empty_query():
    """Test FAISS retriever with empty query."""
    retriever = FAISSRetriever()
    
    result = retriever.apply(query="")
    assert result.success
    assert result.payload["hits"] == []


def test_faiss_retriever_no_chunks():
    """Test FAISS retriever search with no chunks added."""
    embedder = DummyEmbedder()
    retriever = FAISSRetriever(embedder=embedder)
    
    result = retriever.apply(query="test query")
    assert result.success
    assert result.payload["hits"] == []


def test_faiss_retriever_different_index_types():
    """Test FAISS retriever with different index types."""
    embedder = DummyEmbedder()
    
    # Test with IndexFlatL2
    retriever_l2 = FAISSRetriever(
        embedder=embedder,
        dimension=16,
        index_type="IndexFlatL2"
    )
    
    chunks = [
        create_chunk("1", "test document one", "doc1"),
        create_chunk("2", "test document two", "doc2"),
    ]
    
    result = retriever_l2.apply(chunks)
    assert result.success
    
    result = retriever_l2.apply(query="test", top_k=1)
    assert result.success


def test_faiss_retriever_fallback_without_faiss():
    """Test FAISS retriever fallback when FAISS is not available."""
    # This test works regardless of whether FAISS is installed
    # because the retriever has fallback logic
    embedder = DummyEmbedder()
    retriever = FAISSRetriever(embedder=embedder, dimension=16)
    
    chunks = [
        create_chunk("1", "artificial intelligence", "doc1"),
        create_chunk("2", "machine learning", "doc1"),
    ]
    
    # Add chunks
    result = retriever.apply(chunks)
    assert result.success
    
    # Search
    result = retriever.apply(query="AI", top_k=1)
    assert result.success
    assert "library_available" in result.meta


def test_faiss_retriever_metadata_preservation():
    """Test that FAISS retriever preserves chunk metadata."""
    embedder = DummyEmbedder()
    retriever = FAISSRetriever(embedder=embedder, dimension=16)
    
    chunks = [
        create_chunk("chunk1", "test content", "doc1"),
    ]
    
    retriever.add_chunks(chunks)
    result = retriever.apply(query="test", top_k=1)
    
    assert result.success
    hits = result.payload["hits"]
    if hits:
        hit = hits[0]
        assert hit.chunk.id == "chunk1"
        assert hit.chunk.document_id == "doc1"
        assert "original_text" in hit.meta


def test_faiss_retriever_top_k_limiting():
    """Test FAISS retriever respects top_k parameter."""
    embedder = DummyEmbedder()
    retriever = FAISSRetriever(embedder=embedder, dimension=16)
    
    chunks = [
        create_chunk(f"chunk{i}", f"document {i}", f"doc{i}")
        for i in range(10)
    ]
    
    retriever.add_chunks(chunks)
    
    # Test different top_k values
    for k in [1, 3, 5]:
        result = retriever.apply(query="document", top_k=k)
        assert result.success
        assert len(result.payload["hits"]) <= k