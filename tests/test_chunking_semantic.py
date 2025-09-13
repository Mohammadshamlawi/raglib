"""Tests for SemanticChunker technique."""

from raglib.techniques.semantic_chunker import SemanticChunker
from raglib.schemas import Document
from raglib.adapters.dummy_embedder import DummyEmbedder


class TestSemanticChunker:
    """Test suite for SemanticChunker."""
    
    def test_fallback_embedder_deterministic(self):
        """Test that fallback embedder produces deterministic embeddings."""
        chunker = SemanticChunker(
            embedder=None,  # Use fallback
            chunk_size=20,
            overlap=5,
            similarity_threshold=0.9  # High threshold to prevent merging for this test
        )
        
        test_text = "This is a test document for semantic chunking. " * 3  # Repeated text
        document = Document(id="test_doc", text=test_text)
        
        # Run twice to ensure deterministic behavior
        result1 = chunker.apply(document)
        result2 = chunker.apply(document)
        
        chunks1 = result1.payload["chunks"]
        chunks2 = result2.payload["chunks"]
        
        # Same number of chunks
        assert len(chunks1) == len(chunks2)
        
        # Same embeddings and content
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.id == c2.id
            assert c1.text == c2.text
            assert c1.embedding == c2.embedding
            assert c1.embedding is not None
            assert len(c1.embedding) == 16  # Default dim
    
    def test_embeddings_attached(self):
        """Test that chunks have embeddings attached."""
        chunker = SemanticChunker(
            embedder=None,  # Use fallback
            chunk_size=30,
            overlap=5,
            similarity_threshold=0.5
        )
        
        test_text = "The quick brown fox jumps over the lazy dog. The dog sleeps peacefully."
        document = Document(id="test_doc", text=test_text)
        
        result = chunker.apply(document)
        chunks = result.payload["chunks"]
        
        # All chunks should have embeddings
        for chunk in chunks:
            assert chunk.embedding is not None
            assert isinstance(chunk.embedding, list)
            assert len(chunk.embedding) == 16  # Default dimension
            assert all(isinstance(x, float) for x in chunk.embedding)
    
    def test_semantic_merging(self):
        """Test that similar adjacent segments are merged."""
        chunker = SemanticChunker(
            embedder=None,
            chunk_size=15,  # Small chunks to force similarity comparison
            overlap=3,
            similarity_threshold=0.3  # Low threshold to encourage merging
        )
        
        # Create text with repeated similar phrases
        test_text = "cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat cat"
        document = Document(id="similarity_test", text=test_text)
        
        result = chunker.apply(document)
        chunks = result.payload["chunks"]
        
        # Should have fewer chunks than if we used basic fixed chunking due to merging
        # With chunk_size=15, overlap=3, step=12, we'd expect ~5 initial segments
        # But merging should reduce this significantly
        initial_segments = result.meta["initial_segments"]
        final_chunks = len(chunks)
        
        assert final_chunks <= initial_segments
        
        # Check that merged chunks have metadata about merging
        merged_chunks = [c for c in chunks if len(c.meta.get("merged_from", [])) > 1]
        if merged_chunks:  # If any merging occurred
            for chunk in merged_chunks:
                assert "merged_from" in chunk.meta
                assert len(chunk.meta["merged_from"]) > 1
                assert chunk.meta["method"] == "semantic_merge"
    
    def test_explicit_embedder_usage(self):
        """Test that explicit embedder is used when provided."""
        explicit_embedder = DummyEmbedder(dim=32)  # Different dimension
        
        chunker = SemanticChunker(
            embedder=explicit_embedder,
            chunk_size=25,
            overlap=5,
            similarity_threshold=0.7
        )
        
        test_text = "This text will be embedded using the explicit embedder instance."
        document = Document(id="explicit_test", text=test_text)
        
        result = chunker.apply(document)
        chunks = result.payload["chunks"]
        
        # All chunks should have 32-dimensional embeddings
        for chunk in chunks:
            assert chunk.embedding is not None
            assert len(chunk.embedding) == 32  # Explicit embedder dimension
            assert chunk.meta["embedder_type"] == "DummyEmbedder"
    
    def test_consistent_fallback_behavior(self):
        """Test that fallback and explicit DummyEmbedder produce same results."""
        # Chunker with fallback embedder
        chunker_fallback = SemanticChunker(
            embedder=None,
            chunk_size=20,
            overlap=4,
            similarity_threshold=0.6,
            dim=16
        )
        
        # Chunker with explicit DummyEmbedder
        chunker_explicit = SemanticChunker(
            embedder=DummyEmbedder(dim=16),
            chunk_size=20,
            overlap=4,
            similarity_threshold=0.6
        )
        
        test_text = "Consistent behavior test text for semantic chunking functionality."
        document = Document(id="consistency_test", text=test_text)
        
        result_fallback = chunker_fallback.apply(document)
        result_explicit = chunker_explicit.apply(document)
        
        chunks_fallback = result_fallback.payload["chunks"]
        chunks_explicit = result_explicit.payload["chunks"]
        
        # Should produce same results
        assert len(chunks_fallback) == len(chunks_explicit)
        
        for cf, ce in zip(chunks_fallback, chunks_explicit):
            assert cf.text == ce.text
            assert cf.start_idx == ce.start_idx
            assert cf.end_idx == ce.end_idx
            # Embeddings should be the same (DummyEmbedder is deterministic)
            assert cf.embedding == ce.embedding
    
    def test_chunk_ids_deterministic(self):
        """Test that chunk IDs are deterministic."""
        chunker = SemanticChunker(
            embedder=None,
            chunk_size=25,
            overlap=5,
            similarity_threshold=0.8
        )
        
        document = Document(id="deterministic_test", text="Test text for ID consistency checks.")
        
        # Run multiple times
        results = [chunker.apply(document) for _ in range(3)]
        
        # All results should be identical
        base_chunks = results[0].payload["chunks"]
        
        for result in results[1:]:
            chunks = result.payload["chunks"]
            assert len(chunks) == len(base_chunks)
            
            for base_chunk, chunk in zip(base_chunks, chunks):
                assert chunk.id == base_chunk.id
                assert chunk.text == base_chunk.text
                assert chunk.id.startswith("deterministic_test_chunk_")
    
    def test_string_input(self):
        """Test that chunker works with plain string input."""
        chunker = SemanticChunker(chunk_size=30, overlap=5, similarity_threshold=0.7)
        
        test_text = "Plain string input for semantic chunker testing purposes."
        
        result = chunker.apply(test_text)
        
        assert result.success is True
        chunks = result.payload["chunks"]
        
        for chunk in chunks:
            assert chunk.document_id == "unknown"  # Default for string input
            assert chunk.embedding is not None
    
    def test_parameter_override(self):
        """Test that parameters can be overridden in apply()."""
        chunker = SemanticChunker(
            chunk_size=100,  # Large default
            overlap=20,
            similarity_threshold=0.9  # High default
        )
        
        test_text = "Override test with repeated content. Override test with repeated content."
        document = Document(id="override_test", text=test_text)
        
        # Override with values that should cause more chunking/merging
        result = chunker.apply(
            document,
            chunk_size=20,
            overlap=3,
            similarity_threshold=0.3  # Low threshold to encourage merging
        )
        
        chunks = result.payload["chunks"]
        
        # Should have processed with overridden parameters
        assert result.meta["chunk_size"] == 20
        assert result.meta["overlap"] == 3
        assert result.meta["similarity_threshold"] == 0.3
    
    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        chunker = SemanticChunker()
        
        document = Document(id="test", text="Test text")
        
        # Invalid chunk_size
        result = chunker.apply(document, chunk_size=0)
        assert result.success is False
        assert "error" in result.payload
        
        # Invalid overlap
        result = chunker.apply(document, chunk_size=10, overlap=15)
        assert result.success is False
        assert "error" in result.payload
    
    def test_empty_text(self):
        """Test handling of empty text."""
        chunker = SemanticChunker()
        
        document = Document(id="empty", text="")
        
        result = chunker.apply(document)
        
        assert result.success is True
        chunks = result.payload["chunks"]
        assert len(chunks) == 0
    
    def test_very_small_text(self):
        """Test handling of text smaller than chunk size."""
        chunker = SemanticChunker(chunk_size=100, overlap=10, similarity_threshold=0.5)
        
        test_text = "Small"
        document = Document(id="small", text=test_text)
        
        result = chunker.apply(document)
        
        assert result.success is True
        chunks = result.payload["chunks"]
        
        assert len(chunks) == 1
        assert chunks[0].text == test_text
        assert chunks[0].embedding is not None
    
    def test_metadata_preservation(self):
        """Test that document metadata is preserved in chunks."""
        chunker = SemanticChunker(chunk_size=30, overlap=5, similarity_threshold=0.6)
        
        document = Document(
            id="meta_test",
            text="Metadata preservation test for semantic chunking technique implementation.",
            meta={"source": "semantic_test.txt", "domain": "nlp"}
        )
        
        result = chunker.apply(document)
        chunks = result.payload["chunks"]
        
        for chunk in chunks:
            assert "source" in chunk.meta
            assert "domain" in chunk.meta
            assert chunk.meta["source"] == "semantic_test.txt"
            assert chunk.meta["domain"] == "nlp"
            assert chunk.meta["method"] == "semantic_merge"
            assert "merged_from" in chunk.meta
            assert "embedder_type" in chunk.meta
    
    def test_dissimilar_segments_not_merged(self):
        """Test that dissimilar segments are not merged."""
        chunker = SemanticChunker(
            chunk_size=10,
            overlap=2,
            similarity_threshold=0.9  # Very high threshold
        )
        
        # Create text with clearly different content
        test_text = "Dogs are pets. Mathematics involves numbers and equations and formulas."
        document = Document(id="dissimilar_test", text=test_text)
        
        result = chunker.apply(document)
        
        # With high similarity threshold, dissimilar content shouldn't merge much
        initial_segments = result.meta["initial_segments"]
        final_chunks = len(result.payload["chunks"])
        
        # Should have close to the original number of segments
        assert final_chunks >= initial_segments * 0.7  # Allow some merging but not too much
    
    def test_similarity_computation(self):
        """Test the internal similarity computation method."""
        chunker = SemanticChunker()
        
        # Test with known embeddings
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0]
        emb3 = [1.0, 0.0, 0.0]
        
        # Orthogonal vectors should have 0 similarity
        sim_orthogonal = chunker._compute_similarity(emb1, emb2)
        assert abs(sim_orthogonal) < 1e-10
        
        # Identical vectors should have 1.0 similarity
        sim_identical = chunker._compute_similarity(emb1, emb3)
        assert abs(sim_identical - 1.0) < 1e-10
        
        # Different length embeddings should return 0
        sim_different_len = chunker._compute_similarity([1.0, 0.0], [1.0, 0.0, 0.0])
        assert sim_different_len == 0.0
