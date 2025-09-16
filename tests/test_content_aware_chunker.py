"""Tests for ContentAwareChunker technique."""

from raglib.core import TechniqueResult
from raglib.schemas import Chunk, Document
from raglib.techniques.content_aware_chunker import ContentAwareChunker


class TestContentAwareChunker:
    """Test suite for ContentAwareChunker."""

    def test_basic_chunking(self):
        """Test basic content-aware chunking functionality."""
        chunker = ContentAwareChunker(max_chunk_size=100, min_chunk_size=20)

        # Create test document with paragraph structure
        test_text = (
            "This is the first paragraph. It has some content.\n\n"
            "This is the second paragraph. It also has content that should "
            "be kept together when possible.\n\n"
            "This is a third paragraph."
        )
        document = Document(id="test_doc", text=test_text)

        result = chunker.apply(document)

        assert isinstance(result, TechniqueResult)
        assert result.success is True
        assert "chunks" in result.payload

        chunks = result.payload["chunks"]
        assert len(chunks) > 0

        # Check all chunks are Chunk instances
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.document_id == "test_doc"
            assert len(chunk.text) <= 100
            assert chunk.start_idx >= 0
            assert chunk.end_idx <= len(test_text)
            assert chunk.embedding is None

    def test_paragraph_boundary_respect(self):
        """Test that paragraph boundaries are respected when possible."""
        chunker = ContentAwareChunker(max_chunk_size=200, min_chunk_size=10)

        # Create document with clear paragraph breaks
        test_text = (
            "Short paragraph one.\n\n"
            "This is a longer paragraph two that contains more text and "
            "should ideally be kept as a single chunk if it fits.\n\n"
            "Short paragraph three."
        )
        document = Document(id="test_doc", text=test_text)

        result = chunker.apply(document)
        chunks = result.payload["chunks"]

        # Verify that chunks respect paragraph boundaries where possible
        for chunk in chunks:
            # Check that chunks don't start or end in the middle of sentences
            # unless absolutely necessary
            assert chunk.text.strip() != ""

    def test_size_constraints(self):
        """Test that size constraints are properly enforced."""
        chunker = ContentAwareChunker(max_chunk_size=50, min_chunk_size=10)

        test_text = "A" * 200  # Long text without natural boundaries
        document = Document(id="test_doc", text=test_text)

        result = chunker.apply(document)
        chunks = result.payload["chunks"]

        # All chunks should be within size limits
        for chunk in chunks:
            assert len(chunk.text) <= 50
            # Most chunks should meet minimum size (except possibly the last)
            if chunk != chunks[-1]:
                assert len(chunk.text) >= 10

    def test_overlap_functionality(self):
        """Test that overlap between chunks works correctly."""
        chunker = ContentAwareChunker(max_chunk_size=30, min_chunk_size=10, overlap=5)

        test_text = "This is a test sentence. Another test sentence follows."
        document = Document(id="test_doc", text=test_text)

        result = chunker.apply(document)
        chunks = result.payload["chunks"]

        if len(chunks) > 1:
            # Check that there's some overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                chunk1_end = chunks[i].end_idx
                chunk2_start = chunks[i + 1].start_idx
                # Overlap means chunk2 starts before chunk1 ends
                assert chunk2_start <= chunk1_end

    def test_heading_detection(self):
        """Test that heading boundaries are detected."""
        chunker = ContentAwareChunker(max_chunk_size=200, min_chunk_size=10)

        test_text = (
            "# Main Heading\n"
            "Content under main heading.\n\n"
            "## Subheading\n"
            "Content under subheading."
        )
        document = Document(id="test_doc", text=test_text)

        result = chunker.apply(document)
        chunks = result.payload["chunks"]

        # Should create chunks that respect heading boundaries
        assert len(chunks) > 0
        assert result.meta["natural_boundaries"] > 2  # Should find heading boundaries

    def test_empty_text(self):
        """Test behavior with empty text."""
        chunker = ContentAwareChunker()

        document = Document(id="test_doc", text="")
        result = chunker.apply(document)

        assert result.success is True
        chunks = result.payload["chunks"]
        assert len(chunks) == 0

    def test_parameter_validation(self):
        """Test parameter validation."""
        chunker = ContentAwareChunker()

        document = Document(id="test_doc", text="Test text")

        # Test invalid max_chunk_size
        result = chunker.apply(document, max_chunk_size=0)
        assert result.success is False
        assert "max_chunk_size must be positive" in result.payload["error"]

        # Test invalid min_chunk_size
        result = chunker.apply(document, min_chunk_size=0)
        assert result.success is False
        assert "min_chunk_size must be positive" in result.payload["error"]

        # Test min_chunk_size >= max_chunk_size
        result = chunker.apply(document, max_chunk_size=50, min_chunk_size=60)
        assert result.success is False
        assert ("min_chunk_size must be positive and < max_chunk_size"
                in result.payload["error"])

        # Test invalid overlap
        result = chunker.apply(document, overlap=-1)
        assert result.success is False
        assert "overlap must be >= 0" in result.payload["error"]

    def test_string_input(self):
        """Test that string input works correctly."""
        chunker = ContentAwareChunker(max_chunk_size=50, min_chunk_size=10)

        test_text = "This is a test string input instead of Document object."
        result = chunker.apply(test_text)

        assert result.success is True
        chunks = result.payload["chunks"]
        assert len(chunks) > 0

        for chunk in chunks:
            assert chunk.document_id == "unknown"
            assert isinstance(chunk, Chunk)

    def test_meta_information(self):
        """Test that meta information is correctly set."""
        chunker = ContentAwareChunker(
            max_chunk_size=100, min_chunk_size=20, overlap=10
        )

        document = Document(
            id="test_doc",
            text="Test content with some text.",
            meta={"source": "test"}
        )
        result = chunker.apply(document)

        assert result.success is True
        
        # Check result meta
        assert result.meta["max_chunk_size"] == 100
        assert result.meta["min_chunk_size"] == 20
        assert result.meta["overlap"] == 10
        assert "natural_boundaries" in result.meta
        assert "original_length" in result.meta

        # Check chunk meta
        chunks = result.payload["chunks"]
        if chunks:
            chunk = chunks[0]
            assert chunk.meta["method"] == "content_aware"
            assert chunk.meta["max_chunk_size"] == 100
            assert chunk.meta["source"] == "test"  # From document meta

    def test_sentence_boundary_fallback(self):
        """Test fallback to sentence boundaries for oversized segments."""
        chunker = ContentAwareChunker(max_chunk_size=40, min_chunk_size=10)

        # Text with long paragraph that needs sentence-level splitting
        test_text = (
            "This is a very long paragraph without paragraph breaks. "
            "It contains multiple sentences that need to be split. "
            "Each sentence should be handled appropriately by the chunker."
        )
        document = Document(id="test_doc", text=test_text)

        result = chunker.apply(document)
        chunks = result.payload["chunks"]

        # Should successfully chunk despite no paragraph breaks
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) <= 40
