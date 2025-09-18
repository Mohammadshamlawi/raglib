"""Tests for Multi-Query Retrieval technique."""

import pytest
from raglib.schemas import Chunk, Hit
from raglib.adapters.base import EmbedderAdapter, LLMAdapter
from raglib.techniques.multi_query_retriever import MultiQueryRetriever


def create_chunk(id, text, document_id, start_idx=0, end_idx=None):
    """Helper function to create a chunk with required parameters."""
    if end_idx is None:
        end_idx = len(text)
    return Chunk(
        id=id, text=text, document_id=document_id, 
        start_idx=start_idx, end_idx=end_idx
    )


class DummyEmbedder(EmbedderAdapter):
    """Simple dummy embedder for testing."""
    def __init__(self, dim=16):
        self.dim = dim

    def embed(self, texts):
        """Return simple hash-based embeddings."""
        embeddings = []
        for text in texts:
            # Simple hash-based embedding
            embedding = [float(hash(text + str(i)) % 100) / 100.0 for i in range(self.dim)]
            embeddings.append(embedding)
        return embeddings


class DummyLLM(LLMAdapter):
    """Simple dummy LLM for testing."""
    def __init__(self):
        self.call_count = 0

    def generate(self, prompt, **kwargs):
        """Return simple query variations."""
        self.call_count += 1
        if "machine learning" in prompt.lower():
            return "ML algorithms\nArtificial Intelligence\nDeep learning"
        return "query variant 1\nquery variant 2\nquery variant 3"


def test_multi_query_retriever_initialization():
    """Test multi-query retriever initialization."""
    retriever = MultiQueryRetriever()
    
    assert retriever.embedder is not None
    assert retriever.llm_adapter is None  # Should be None by default
    assert retriever.vectorstore is not None
    assert retriever.fusion_method == "rrf"
    assert retriever.num_queries == 3


def test_multi_query_retriever_with_custom_components():
    """Test multi-query retriever with custom components."""
    embedder = DummyEmbedder(dim=16)
    llm = DummyLLM()
    
    retriever = MultiQueryRetriever(
        embedder=embedder,
        llm_adapter=llm,
        num_queries=5,
        fusion_method="rrf",
        dim=16
    )
    
    chunks = [
        create_chunk(id="1", text="machine learning algorithms", document_id="doc1"),
        create_chunk(id="2", text="deep neural networks", document_id="doc1"),
    ]
    
    retriever.add_chunks(chunks)
    result = retriever.apply(query="machine learning", top_k=2)
    
    assert isinstance(result.payload["hits"], list)
    assert len(result.payload["hits"]) <= 2
    assert all(isinstance(hit, Hit) for hit in result.payload["hits"])


def test_multi_query_retriever_search():
    """Test multi-query retriever search functionality."""
    embedder = DummyEmbedder(dim=16)
    llm = DummyLLM()
    
    retriever = MultiQueryRetriever(
        embedder=embedder,
        llm_adapter=llm,
        num_queries=3,
        dim=16
    )
    
    # Add test chunks
    chunks = [
        create_chunk(id="1", text="artificial intelligence research", document_id="doc1"),
        create_chunk(id="2", text="machine learning algorithms", document_id="doc1"),
        create_chunk(id="3", text="natural language processing", document_id="doc2"),
    ]
    
    retriever.add_chunks(chunks)
    
    # Search with main query
    result = retriever.apply(query="machine learning", top_k=3)
    
    assert isinstance(result.payload["hits"], list)
    assert len(result.payload["hits"]) <= 3
    assert all(isinstance(hit, Hit) for hit in result.payload["hits"])
    assert llm.call_count > 0  # LLM should have been called


def test_multi_query_retriever_different_fusion_strategies():
    """Test different fusion strategies."""
    embedder = DummyEmbedder(dim=16)
    llm = DummyLLM()
    
    # Test RRF strategy
    retriever_rrf = MultiQueryRetriever(
        embedder=embedder,
        llm_adapter=llm,
        fusion_method="rrf",
        dim=16
    )
    
    chunks = [
        create_chunk(id="1", text="machine learning", document_id="doc1"),
        create_chunk(id="2", text="deep learning", document_id="doc2"),
    ]
    
    retriever_rrf.add_chunks(chunks)
    result_rrf = retriever_rrf.apply(query="AI", top_k=2)
    
    assert isinstance(result_rrf.payload["hits"], list)
    assert len(result_rrf.payload["hits"]) <= 2


def test_multi_query_retriever_query_generation():
    """Test query generation functionality."""
    llm = DummyLLM()
    retriever = MultiQueryRetriever(llm_adapter=llm, num_queries=4, dim=16)
    
    # This will indirectly test query generation through apply
    chunks = [create_chunk(id="1", text="test content", document_id="doc1")]
    retriever.add_chunks(chunks)
    
    initial_count = llm.call_count
    retriever.apply(query="test query", top_k=1)
    
    # LLM should have been called to generate queries
    assert llm.call_count > initial_count


def test_multi_query_retriever_empty_query():
    """Test multi-query retriever with empty query."""
    retriever = MultiQueryRetriever(dim=16)
    result = retriever.apply(query="", top_k=5)
    
    assert isinstance(result.payload["hits"], list)
    assert len(result.payload["hits"]) == 0


def test_multi_query_retriever_no_chunks():
    """Test multi-query retriever search with no chunks."""
    retriever = MultiQueryRetriever(dim=16)
    result = retriever.apply(query="test query", top_k=5)
    
    assert isinstance(result.payload["hits"], list)
    assert len(result.payload["hits"]) == 0


def test_multi_query_retriever_metadata_preservation():
    """Test that multi-query retriever preserves metadata."""
    retriever = MultiQueryRetriever(dim=16)
    
    chunks = [
        create_chunk(id="chunk1", text="test content", document_id="doc1"),
    ]
    # Manually set metadata after creation
    chunks[0].metadata = {"category": "test", "score": 0.9}
    
    retriever.add_chunks(chunks)
    result = retriever.apply(query="test", top_k=1)
    
    assert len(result.payload["hits"]) > 0
    assert result.payload["hits"][0].meta is not None
    assert result.payload["hits"][0].meta.get("category") == "test"


def test_multi_query_retriever_top_k_limiting():
    """Test that multi-query retriever respects top_k parameter."""
    retriever = MultiQueryRetriever(dim=16)
    
    chunks = [
        create_chunk(id=f"chunk{i}", text=f"document {i}", document_id=f"doc{i}")
        for i in range(10)
    ]
    
    retriever.add_chunks(chunks)
    
    result = retriever.apply(query="document", top_k=3)
    
    assert len(result.payload["hits"]) <= 3
    assert all(isinstance(hit, Hit) for hit in result.payload["hits"])


def test_multi_query_retriever_rrf_fusion():
    """Test Reciprocal Rank Fusion specifically."""
    from raglib.techniques.multi_query_retriever import _reciprocal_rank_fusion
    
    # Create test hit lists
    hit1 = Hit(doc_id="doc1", score=0.9, chunk=None)
    hit2 = Hit(doc_id="doc2", score=0.8, chunk=None) 
    hit3 = Hit(doc_id="doc1", score=0.7, chunk=None)  # Same doc
    
    hit_lists = [
        [hit1, hit2],
        [hit3, hit2],
    ]
    
    fused_hits = _reciprocal_rank_fusion(hit_lists, k=60)
    
    assert isinstance(fused_hits, list)
    assert len(fused_hits) == 2  # Two unique documents
    assert all(isinstance(hit, Hit) for hit in fused_hits)
    # doc1 should rank higher as it appears in both lists
    assert fused_hits[0].doc_id == "doc1"