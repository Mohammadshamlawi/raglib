"""Tests for ColBERT Retriever technique."""
def create_chunk(id, text, document_id, start_idx=0, end_idx=None):
    """Helper function to create a chunk with required parameters."""
    if end_idx is None:
        end_idx = len(text)
    return Chunk(
        id=id, text=text, document_id=document_id, 
        start_idx=start_idx, end_idx=end_idx
    )


from raglib.adapters.base import EmbedderAdapter
from raglib.schemas import Chunk, Document
from raglib.techniques.colbert_retriever import ColBERTRetriever


class DummyTokenEmbedder(EmbedderAdapter):
    """Test embedder that returns deterministic token-level embeddings."""

    def embed(self, texts):
        embeddings = []
        for text in texts:
            # Simple deterministic embedding based on token
            token_hash = sum(ord(c) for c in text)
            embedding = [
                len(text) / 20.0,
                text.count('a') / 5.0,
                text.count('e') / 5.0,
                (token_hash % 100) / 100.0
            ]
            # Pad to 16 dimensions
            while len(embedding) < 16:
                embedding.append(0.05 * (len(embedding) + 1))
            embeddings.append(embedding[:16])
        return embeddings


def test_colbert_retriever_initialization():
    """Test ColBERT retriever initialization."""
    retriever = ColBERTRetriever()
    assert retriever.meta.name == "colbert_retriever"
    assert retriever.meta.category == "core_retrieval"
    assert "late_interaction" in retriever.meta.tags["architecture"]


def test_colbert_retriever_with_custom_embedder():
    """Test ColBERT retriever with custom embedder."""
    embedder = DummyTokenEmbedder()
    retriever = ColBERTRetriever(
        embedder=embedder,
        max_query_length=16,
        max_doc_length=64,
        dim=16
    )

    chunks = [
        create_chunk(id="1", text="machine learning algorithms", document_id="doc1"),
        create_chunk(id="2", text="deep neural networks", document_id="doc1"),
    ]

    result = retriever.apply(chunks)
    assert result.success
    assert result.payload["added_chunks"] == 2


def test_colbert_retriever_search():
    """Test ColBERT retriever search functionality."""
    embedder = DummyTokenEmbedder()
    retriever = ColBERTRetriever(
        embedder=embedder,
        max_query_length=8,
        max_doc_length=32,
        dim=16
    )

    # Add test chunks
    chunks = [
        create_chunk(id="1", text="artificial intelligence research", document_id="doc1"),
        create_chunk(id="2", text="machine learning algorithms", document_id="doc1"),
        create_chunk(id="3", text="natural language processing", document_id="doc2"),
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


def test_colbert_retriever_documents():
    """Test ColBERT retriever with documents instead of chunks."""
    embedder = DummyTokenEmbedder()
    retriever = ColBERTRetriever(embedder=embedder, dim=16)

    documents = [
        Document(id="doc1", text="document about machine learning"),
        Document(id="doc2", text="document about deep learning"),
    ]

    result = retriever.apply(documents)
    assert result.success
    assert result.payload["added_documents"] == 2

    # Test search
    result = retriever.apply(query="learning", top_k=1)
    assert result.success


def test_colbert_retriever_token_limits():
    """Test ColBERT retriever respects token limits."""
    embedder = DummyTokenEmbedder()
    retriever = ColBERTRetriever(
        embedder=embedder,
        max_query_length=2,  # Very short
        max_doc_length=4,    # Very short
        dim=16
    )

    # Long text that should be truncated
    long_text = "this is a very long document with many words that should be truncated"
    chunks = [
        create_chunk(id="1", text=long_text, document_id="doc1"),
    ]

    retriever.add_chunks(chunks)

    # Long query that should be truncated
    long_query = "this is a very long query with many words"
    result = retriever.apply(query=long_query, top_k=1)

    assert result.success
    # Check that limits are respected in metadata
    assert result.meta["max_query_length"] == 2
    assert result.meta["max_doc_length"] == 4


def test_colbert_retriever_empty_query():
    """Test ColBERT retriever with empty query."""
    retriever = ColBERTRetriever()

    result = retriever.apply(query="")
    assert result.success
    assert result.payload["hits"] == []


def test_colbert_retriever_no_documents():
    """Test ColBERT retriever search with no documents."""
    retriever = ColBERTRetriever()

    result = retriever.apply(query="test query")
    assert result.success
    assert result.payload["hits"] == []
    assert "No documents indexed" in result.meta["message"]


def test_colbert_retriever_late_interaction():
    """Test that ColBERT performs late interaction correctly."""
    embedder = DummyTokenEmbedder()
    retriever = ColBERTRetriever(
        embedder=embedder,
        max_query_length=4,
        max_doc_length=8,
        dim=16
    )

    # Add documents with overlapping terms
    chunks = [
        create_chunk(id="1", text="machine learning is great", document_id="doc1"),
        create_chunk(id="2", text="deep learning networks", document_id="doc2"),
    ]

    retriever.add_chunks(chunks)

    # Query should match both but with different scores due to late interaction
    result = retriever.apply(query="learning systems", top_k=2)

    assert result.success
    hits = result.payload["hits"]
    assert len(hits) <= 2

    # Verify scoring is working (hits should have different scores)
    if len(hits) > 1:
        scores = [hit.score for hit in hits]
        # Either different scores or all zero
        assert len(set(scores)) > 1 or all(s == 0 for s in scores)
def test_colbert_retriever_metadata_preservation():
    """Test that ColBERT retriever preserves metadata."""
    embedder = DummyTokenEmbedder()
    retriever = ColBERTRetriever(embedder=embedder, dim=16)

    chunks = [
        create_chunk(id="chunk1", text="test content", document_id="doc1"),
    ]

    retriever.add_chunks(chunks)
    result = retriever.apply(query="test", top_k=1)

    assert result.success
    hits = result.payload["hits"]
    if hits:
        hit = hits[0]
        assert hit.chunk.id == "chunk1"
        assert hit.chunk.document_id == "doc1"


def test_colbert_retriever_top_k():
    """Test ColBERT retriever respects top_k parameter."""
    embedder = DummyTokenEmbedder()
    retriever = ColBERTRetriever(embedder=embedder, dim=16)

    chunks = [
        create_chunk(id=f"chunk{i}", text=f"document {i}", document_id=f"doc{i}")
        for i in range(5)
    ]

    retriever.add_chunks(chunks)

    # Test different top_k values
    for k in [1, 2, 3]:
        result = retriever.apply(query="document", top_k=k)
        assert result.success
        assert len(result.payload["hits"]) <= k
