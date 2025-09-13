"""Tests for FixedSizeChunker technique."""

from raglib.techniques.fixed_size_chunker import FixedSizeChunker
from raglib.schemas import Document, Chunk
from raglib.core import TechniqueResult


class TestFixedSizeChunker:
    """Test suite for FixedSizeChunker."""
    
    def test_basic_chunking(self):
        """Test basic fixed-size chunking functionality."""
        chunker = FixedSizeChunker(chunk_size=10, overlap=2)
        
        # Create test document
        test_text = "This is a test document for chunking."  # 38 characters
        document = Document(id="test_doc", text=test_text)
        
        result = chunker.apply(document)
        
        assert isinstance(result, TechniqueResult)
        assert result.success is True
        assert "chunks" in result.payload
        
        chunks = result.payload["chunks"]
        assert len(chunks) > 1
        
        # Check all chunks are Chunk instances
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.document_id == "test_doc"
            assert len(chunk.text) <= 10
            assert chunk.start_idx >= 0
            assert chunk.end_idx <= len(test_text)
            assert chunk.embedding is None  # Basic chunker doesn't set embeddings
    
    def test_chunk_text_lengths(self):
        """Test that chunks don't exceed specified size."""
        chunker = FixedSizeChunker(chunk_size=20, overlap=5)
        
        test_text = "A" * 100  # 100 character string
        document = Document(id="test_doc", text=test_text)
        
        result = chunker.apply(document)
        chunks = result.payload["chunks"]
        
        # All chunks except possibly the last should be exactly chunk_size
        for i, chunk in enumerate(chunks[:-1]):
            assert len(chunk.text) == 20
            
        # Last chunk can be shorter
        assert len(chunks[-1].text) <= 20
    
    def test_overlap_behavior(self):
        """Test that overlapping works correctly."""
        chunker = FixedSizeChunker(chunk_size=10, overlap=3)
        
        test_text = "0123456789012345678901234567890"  # 31 characters
        document = Document(id="test_doc", text=test_text)
        
        result = chunker.apply(document)
        chunks = result.payload["chunks"]
        
        # Check that overlaps are correct
        # step = 10 - 3 = 7, so chunks start at 0, 7, 14, 21, 28
        expected_starts = [0, 7, 14, 21, 28]
        
        for i, chunk in enumerate(chunks[:-1]):  # Exclude last chunk for simpler testing
            if i < len(expected_starts):
                assert chunk.start_idx == expected_starts[i]
    
    def test_text_reconstruction(self):
        """Test that original text can be reconstructed from chunks accounting for overlap."""
        chunker = FixedSizeChunker(chunk_size=15, overlap=5)
        
        test_text = "The quick brown fox jumps over the lazy dog."
        document = Document(id="test_doc", text=test_text)
        
        result = chunker.apply(document)
        chunks = result.payload["chunks"]
        
        # Verify no gaps in coverage
        assert chunks[0].start_idx == 0
        assert chunks[-1].end_idx == len(test_text)
        
        # Check that each chunk's text matches the document text at those positions
        for chunk in chunks:
            assert chunk.text == test_text[chunk.start_idx:chunk.end_idx]
    
    def test_chunk_ids_deterministic(self):
        """Test that chunk IDs are deterministic and follow expected pattern."""
        chunker = FixedSizeChunker(chunk_size=10, overlap=2)
        
        document = Document(id="my_test_doc", text="This is a test document.")
        
        # Run twice to ensure deterministic behavior
        result1 = chunker.apply(document)
        result2 = chunker.apply(document)
        
        chunks1 = result1.payload["chunks"]
        chunks2 = result2.payload["chunks"]
        
        # Same number of chunks
        assert len(chunks1) == len(chunks2)
        
        # Same IDs
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.id == c2.id
            assert c1.id.startswith("my_test_doc_chunk_")
            
        # IDs are sequential
        for i, chunk in enumerate(chunks1):
            expected_id = f"my_test_doc_chunk_{i}"
            assert chunk.id == expected_id
    
    def test_string_input(self):
        """Test that chunker works with plain string input."""
        chunker = FixedSizeChunker(chunk_size=15, overlap=3)
        
        test_text = "Plain string input test."
        
        result = chunker.apply(test_text)
        
        assert result.success is True
        chunks = result.payload["chunks"]
        
        for chunk in chunks:
            assert chunk.document_id == "unknown"  # Default for string input
            assert chunk.text == test_text[chunk.start_idx:chunk.end_idx]
    
    def test_parameter_override(self):
        """Test that parameters can be overridden in apply()."""
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)  # Large defaults
        
        test_text = "Short text for override test."
        document = Document(id="test", text=test_text)
        
        # Override with smaller chunk size
        result = chunker.apply(document, chunk_size=10, overlap=2)
        
        chunks = result.payload["chunks"]
        
        # Should have multiple chunks due to smaller override chunk_size
        assert len(chunks) > 1
        for chunk in chunks[:-1]:  # All but last should be exactly 10 chars
            assert len(chunk.text) == 10
    
    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        chunker = FixedSizeChunker()
        
        document = Document(id="test", text="Test text")
        
        # Invalid chunk_size
        result = chunker.apply(document, chunk_size=0)
        assert result.success is False
        assert "error" in result.payload
        
        # Invalid overlap
        result = chunker.apply(document, chunk_size=10, overlap=15)
        assert result.success is False
        assert "error" in result.payload
        
        result = chunker.apply(document, chunk_size=10, overlap=-1)
        assert result.success is False
        assert "error" in result.payload
    
    def test_empty_text(self):
        """Test handling of empty text."""
        chunker = FixedSizeChunker(chunk_size=10, overlap=2)
        
        document = Document(id="empty", text="")
        
        result = chunker.apply(document)
        
        assert result.success is True
        chunks = result.payload["chunks"]
        assert len(chunks) == 0
    
    def test_very_small_text(self):
        """Test handling of text smaller than chunk size."""
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        
        test_text = "Short"
        document = Document(id="short", text=test_text)
        
        result = chunker.apply(document)
        
        assert result.success is True
        chunks = result.payload["chunks"]
        
        assert len(chunks) == 1
        assert chunks[0].text == test_text
        assert chunks[0].start_idx == 0
        assert chunks[0].end_idx == len(test_text)
    
    def test_metadata_preservation(self):
        """Test that document metadata is preserved in chunks."""
        chunker = FixedSizeChunker(chunk_size=10, overlap=2)
        
        document = Document(
            id="meta_test",
            text="Test with metadata.",
            meta={"source": "test_file.txt", "author": "tester"}
        )
        
        result = chunker.apply(document)
        chunks = result.payload["chunks"]
        
        for chunk in chunks:
            assert "source" in chunk.meta
            assert "author" in chunk.meta
            assert chunk.meta["source"] == "test_file.txt"
            assert chunk.meta["author"] == "tester"
            assert chunk.meta["method"] == "fixed_size"
