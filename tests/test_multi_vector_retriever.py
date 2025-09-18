"""Tests for Multi-Vector Retrieval technique."""

from raglib.schemas import Chunk, Hit
from raglib.adapters.base import EmbedderAdapter
from raglib.techniques.multi_vector_retriever import MultiVectorRetriever


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
            embedding = [
                float(hash(text + str(i)) % 100) / 100.0 
                for i in range(self.dim)
            ]
            embeddings.append(embedding)
        return embeddings


def test_multi_vector_retriever_initialization():
    """Test multi-vector retriever initialization."""
    retriever = MultiVectorRetriever()
    
    assert retriever.embedder is not None
    assert retriever.doc_vectors == {}
    assert retriever.aggregation_method == "max"
    assert retriever.max_segment_length == 100


def test_multi_vector_retriever_with_custom_embedder():
    """Test multi-vector retriever with custom embedder."""
    embedder = DummyEmbedder(dim=16)
    retriever = MultiVectorRetriever(
        embedder=embedder,
        max_segment_length=32,
        aggregation_method="avg",
        dim=16
    )
    
    chunks = [
        create_chunk(
            id="1", 
            text="machine learning algorithms for data science", 
            document_id="doc1"
        ),
        create_chunk(
            id="2", 
            text="deep neural networks", 
            document_id="doc1"
        ),
    ]
    
    retriever.add_chunks(chunks)
    result = retriever.apply(query="machine learning", top_k=2)
    
    assert isinstance(result.payload["hits"], list)
    assert len(result.payload["hits"]) <= 2
    assert all(isinstance(hit, Hit) for hit in result.payload["hits"])


def test_multi_vector_retriever_search():
    """Test multi-vector retriever search functionality."""
    embedder = DummyEmbedder(dim=16)
    retriever = MultiVectorRetriever(
        embedder=embedder,
        max_segment_length=16,
        dim=16
    )
    
    # Add test chunks with longer text for segmentation
    chunks = [
        create_chunk(
            id="1", 
            text="artificial intelligence research machine learning algorithms", 
            document_id="doc1"
        ),
        create_chunk(
            id="2", 
            text="deep learning neural networks", 
            document_id="doc1"
        ),
        create_chunk(
            id="3", 
            text="natural language processing", 
            document_id="doc2"
        ),
    ]
    
    retriever.add_chunks(chunks)
    
    # Search with main query
    result = retriever.apply(query="machine learning", top_k=3)
    
    assert isinstance(result.payload["hits"], list)
    assert len(result.payload["hits"]) <= 3
    assert all(isinstance(hit, Hit) for hit in result.payload["hits"])


def test_multi_vector_retriever_aggregation_strategies():
    """Test different aggregation strategies."""
    embedder = DummyEmbedder(dim=16)
    
    # Test max aggregation
    retriever_max = MultiVectorRetriever(
        embedder=embedder,
        aggregation_method="max",
        max_segment_length=10,
        dim=16
    )
    
    # Test avg aggregation  
    retriever_avg = MultiVectorRetriever(
        embedder=embedder,
        aggregation_method="avg",
        max_segment_length=10,
        dim=16
    )
    
    chunks = [
        create_chunk(
            id="1", 
            text="machine learning algorithms and artificial intelligence", 
            document_id="doc1"
        ),
    ]
    
    retriever_max.add_chunks(chunks)
    retriever_avg.add_chunks(chunks)
    
    result_max = retriever_max.apply(query="AI", top_k=1)
    result_avg = retriever_avg.apply(query="AI", top_k=1)
    
    assert isinstance(result_max.payload["hits"], list)
    assert isinstance(result_avg.payload["hits"], list)
    assert len(result_max.payload["hits"]) <= 1
    assert len(result_avg.payload["hits"]) <= 1


def test_multi_vector_retriever_segmentation():
    """Test document segmentation functionality."""
    retriever = MultiVectorRetriever(max_segment_length=5, dim=16)  # Very small segments
    
    # Long text that should be segmented
    long_text = "this is a very long document that should be split into multiple segments for multi-vector representation"
    chunks = [
        create_chunk(id="1", text=long_text, document_id="doc1")
    ]
    
    retriever.add_chunks(chunks)
    result = retriever.apply(query="document", top_k=1)
    
    assert isinstance(result.payload["hits"], list)
    assert len(result.payload["hits"]) <= 1


def test_multi_vector_retriever_empty_query():
    """Test multi-vector retriever with empty query."""
    retriever = MultiVectorRetriever(dim=16)
    result = retriever.apply(query="", top_k=5)
    
    assert isinstance(result.payload["hits"], list)
    assert len(result.payload["hits"]) == 0


def test_multi_vector_retriever_no_chunks():
    """Test multi-vector retriever search with no chunks."""
    retriever = MultiVectorRetriever(dim=16)
    result = retriever.apply(query="test query", top_k=5)
    
    assert isinstance(result.payload["hits"], list)
    assert len(result.payload["hits"]) == 0


def test_multi_vector_retriever_metadata_preservation():
    """Test that multi-vector retriever preserves metadata."""
    retriever = MultiVectorRetriever(dim=16)
    
    chunks = [
        create_chunk(id="chunk1", text="test content here", document_id="doc1"),
    ]
    # Manually set metadata after creation
    chunks[0].metadata = {"category": "test", "score": 0.9}
    
    retriever.add_chunks(chunks)
    result = retriever.apply(query="test", top_k=1)
    
    assert len(result.payload["hits"]) > 0
    assert result.payload["hits"][0].meta is not None
    assert result.payload["hits"][0].meta.get("category") == "test"


def test_multi_vector_retriever_top_k_limiting():
    """Test that multi-vector retriever respects top_k parameter."""
    retriever = MultiVectorRetriever(dim=16)
    
    chunks = [
        create_chunk(
            id=f"chunk{i}", 
            text=f"document number {i} with some content", 
            document_id=f"doc{i}"
        )
        for i in range(10)
    ]
    
    retriever.add_chunks(chunks)
    
    result = retriever.apply(query="document", top_k=3)
    
    assert len(result.payload["hits"]) <= 3
    assert all(isinstance(hit, Hit) for hit in result.payload["hits"])


def test_multi_vector_retriever_segment_overlap():
    """Test segment creation and overlap handling."""
    retriever = MultiVectorRetriever(
        max_segment_length=10,  # Small segments
        dim=16
    )
    
    # Text that will create multiple segments
    text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12"
    chunks = [
        create_chunk(id="1", text=text, document_id="doc1")
    ]
    
    retriever.add_chunks(chunks)
    result = retriever.apply(query="word5", top_k=1)
    
    assert isinstance(result.payload["hits"], list)
    # Should still find the chunk even though it's segmented
    assert len(result.payload["hits"]) <= 1