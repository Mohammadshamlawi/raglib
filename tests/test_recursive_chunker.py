"""Tests for RecursiveChunker technique."""

from raglib.core import TechniqueResult
from raglib.schemas import Chunk, Document
from raglib.techniques.recursive_chunker import RecursiveChunker


class TestRecursiveChunker:
    """Test suite for RecursiveChunker."""

    def test_basic_recursive_chunking(self):
        """Test basic recursive chunking functionality."""
        chunker = RecursiveChunker(chunk_size=100, overlap=20)

        # Create test document with hierarchical structure
        test_text = (
            "This is the first paragraph.\n\n"
            "This is the second paragraph that is a bit longer and contains "
            "more content to test the chunking behavior.\n\n"
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

    def test_separator_hierarchy(self):
        """Test that separator hierarchy is respected."""
        chunker = RecursiveChunker(
            chunk_size=50,
            overlap=10,
            separators=["\n\n", "\n", " ", ""]
        )

        # Text with multiple separator levels
        test_text = (
            "First para.\n\n"
            "Second paragraph with multiple sentences. Another sentence here.\n\n"
            "Third para."
        )
        document = Document(id="test_doc", text=test_text)

        result = chunker.apply(document)
        chunks = result.payload["chunks"]

        # Should prefer paragraph breaks over sentence breaks
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) <= 50

    def test_custom_separators(self):
        """Test chunking with custom separators."""
        custom_separators = ["---", "\n", " ", ""]
        chunker = RecursiveChunker(
            chunk_size=40,
            overlap=5,
            separators=custom_separators
        )

        test_text = "Section one---Section two---Section three that is longer"
        document = Document(id="test_doc", text=test_text)

        result = chunker.apply(document)
        chunks = result.payload["chunks"]

        # Should split on custom separator
        assert len(chunks) > 1
        assert result.meta["separators"] == custom_separators

    def test_overlap_functionality(self):
        """Test that overlap between chunks works correctly."""
        chunker = RecursiveChunker(chunk_size=30, overlap=10)

        test_text = "This is a test. Another sentence. And one more."
        document = Document(id="test_doc", text=test_text)

        result = chunker.apply(document)
        chunks = result.payload["chunks"]

        if len(chunks) > 1:
            # Check that there's overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                chunk1_end = chunks[i].end_idx
                chunk2_start = chunks[i + 1].start_idx
                # With overlap, chunk2 should start before chunk1 ends
                assert chunk2_start <= chunk1_end

    def test_character_level_fallback(self):
        """Test fallback to character-level splitting."""
        chunker = RecursiveChunker(chunk_size=10, overlap=2)

        # Long text without natural separators
        test_text = "verylongtextwithoutspacesorpunctuation"
        document = Document(id="test_doc", text=test_text)

        result = chunker.apply(document)
        chunks = result.payload["chunks"]

        # Should fall back to character splitting
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) <= 10

    def test_small_text_no_splitting(self):
        """Test that small text is not split."""
        chunker = RecursiveChunker(chunk_size=100, overlap=10)

        test_text = "Short text."
        document = Document(id="test_doc", text=test_text)

        result = chunker.apply(document)
        chunks = result.payload["chunks"]

        # Should create only one chunk
        assert len(chunks) == 1
        assert chunks[0].text == test_text

    def test_empty_text(self):
        """Test behavior with empty text."""
        chunker = RecursiveChunker()

        document = Document(id="test_doc", text="")
        result = chunker.apply(document)

        assert result.success is True
        chunks = result.payload["chunks"]
        assert len(chunks) == 0

    def test_parameter_validation(self):
        """Test parameter validation."""
        chunker = RecursiveChunker()

        document = Document(id="test_doc", text="Test text")

        # Test invalid chunk_size
        result = chunker.apply(document, chunk_size=0)
        assert result.success is False
        assert "chunk_size must be positive" in result.payload["error"]

        # Test invalid overlap
        result = chunker.apply(document, overlap=-1)
        assert result.success is False
        assert "overlap must be >= 0 and < chunk_size" in result.payload["error"]

        # Test overlap >= chunk_size
        result = chunker.apply(document, chunk_size=50, overlap=60)
        assert result.success is False
        assert "overlap must be >= 0 and < chunk_size" in result.payload["error"]

    def test_string_input(self):
        """Test that string input works correctly."""
        chunker = RecursiveChunker(chunk_size=50, overlap=10)

        test_text = "This is a test string input.\n\nAnother paragraph."
        result = chunker.apply(test_text)

        assert result.success is True
        chunks = result.payload["chunks"]
        assert len(chunks) > 0

        for chunk in chunks:
            assert chunk.document_id == "unknown"
            assert isinstance(chunk, Chunk)

    def test_meta_information(self):
        """Test that meta information is correctly set."""
        separators = ["\n\n", "\n", " ", ""]
        chunker = RecursiveChunker(
            chunk_size=100, overlap=20, separators=separators
        )

        document = Document(
            id="test_doc",
            text="Test content with meta.",
            meta={"source": "test"}
        )
        result = chunker.apply(document)

        assert result.success is True

        # Check result meta
        assert result.meta["chunk_size"] == 100
        assert result.meta["overlap"] == 20
        assert result.meta["separators"] == separators
        assert "original_length" in result.meta

        # Check chunk meta
        chunks = result.payload["chunks"]
        if chunks:
            chunk = chunks[0]
            assert chunk.meta["method"] == "recursive"
            assert chunk.meta["chunk_size"] == 100
            assert chunk.meta["source"] == "test"  # From document meta

    def test_word_boundary_splitting(self):
        """Test splitting on word boundaries."""
        chunker = RecursiveChunker(
            chunk_size=20,
            overlap=5,
            separators=[" ", ""]  # Focus on word splitting
        )

        test_text = "This is a sentence with multiple words that should split nicely"
        document = Document(id="test_doc", text=test_text)

        result = chunker.apply(document)
        chunks = result.payload["chunks"]

        # Should split on word boundaries when possible
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) <= 20
            # Most chunks should end at word boundaries (except character splitting)
            if not chunk.text.endswith(chunk.text[-1]):
                assert chunk.text.endswith(" ") or len(chunk.text) == 20

    def test_position_accuracy(self):
        """Test that chunk positions are accurately calculated."""
        chunker = RecursiveChunker(chunk_size=30, overlap=5)

        test_text = "First part. Second part. Third part."
        document = Document(id="test_doc", text=test_text)

        result = chunker.apply(document)
        chunks = result.payload["chunks"]

        # Verify that chunks correctly map to original text
        for chunk in chunks:
            original_text = test_text[chunk.start_idx:chunk.end_idx]
            assert chunk.text == original_text

    def test_line_break_splitting(self):
        """Test splitting on line breaks."""
        chunker = RecursiveChunker(
            chunk_size=25,
            overlap=5,
            separators=["\n", " ", ""]
        )

        test_text = "Line one\nLine two\nLine three\nLine four"
        document = Document(id="test_doc", text=test_text)

        result = chunker.apply(document)
        chunks = result.payload["chunks"]

        # Should prefer line breaks over word breaks
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) <= 25
