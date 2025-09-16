"""Tests for SentenceWindowChunker technique."""

from raglib.core import TechniqueResult
from raglib.schemas import Chunk, Document
from raglib.techniques.sentence_window_chunker import SentenceWindowChunker


class TestSentenceWindowChunker:
    """Test suite for SentenceWindowChunker."""

    def test_basic_sentence_windowing(self):
        """Test basic sentence windowing functionality."""
        chunker = SentenceWindowChunker(window_size=2, overlap_sentences=1)

        test_text = "First sentence. Second sentence. Third sentence. Fourth sentence."
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
            assert chunk.start_idx >= 0
            assert chunk.end_idx <= len(test_text)

    def test_sentence_boundaries(self):
        """Test that chunks respect sentence boundaries."""
        chunker = SentenceWindowChunker(window_size=1, overlap_sentences=0)

        test_text = "Sentence one! Sentence two? Sentence three."
        document = Document(id="test_doc", text=test_text)

        result = chunker.apply(document)
        chunks = result.payload["chunks"]

        # Should have 3 chunks, one for each sentence
        assert len(chunks) == 3

        # Check that each chunk contains one sentence
        expected_texts = ["Sentence one", "Sentence two", "Sentence three"]
        for i, chunk in enumerate(chunks):
            assert expected_texts[i] in chunk.text

    def test_window_size_logic(self):
        """Test window size creates expected number of sentences per chunk."""
        chunker = SentenceWindowChunker(window_size=3, overlap_sentences=0)

        # 6 sentences total
        test_text = "One. Two. Three. Four. Five. Six."
        document = Document(id="test_doc", text=test_text)

        result = chunker.apply(document)
        chunks = result.payload["chunks"]

        # Should have 2 chunks with 3 sentences each (no overlap)
        assert len(chunks) == 2

        # Verify sentence counts in metadata
        for chunk in chunks:
            # Each chunk should have been created from multiple sentences
            assert chunk.meta["sentence_count"] <= 3

    def test_overlap_behavior(self):
        """Test that sentence overlap works correctly."""
        chunker = SentenceWindowChunker(window_size=2, overlap_sentences=1)

        test_text = "First. Second. Third. Fourth."
        document = Document(id="test_doc", text=test_text)

        result = chunker.apply(document)
        chunks = result.payload["chunks"]

        # With window_size=2, overlap=1, step=1, we should get chunks like:
        # Chunk 0: "First. Second."
        # Chunk 1: "Second. Third."
        # Chunk 2: "Third. Fourth."
        # So 3 chunks total
        assert len(chunks) == 3

    def test_character_indices(self):
        """Test that start_idx and end_idx are correct character positions."""
        chunker = SentenceWindowChunker(window_size=1, overlap_sentences=0)

        test_text = "Hello world. How are you?"
        document = Document(id="test_doc", text=test_text)

        result = chunker.apply(document)
        chunks = result.payload["chunks"]

        # Verify that chunk text matches original text at those positions
        for chunk in chunks:
            actual_text = test_text[chunk.start_idx:chunk.end_idx].strip()
            chunk_text = chunk.text.strip()
            # Allow for whitespace differences at sentence boundaries
            assert chunk_text in actual_text or actual_text in chunk_text

    def test_chunk_ids_deterministic(self):
        """Test that chunk IDs are deterministic."""
        chunker = SentenceWindowChunker(window_size=2, overlap_sentences=0)

        document = Document(id="my_test_doc", text="One. Two. Three. Four.")

        # Run twice to ensure deterministic behavior
        result1 = chunker.apply(document)
        result2 = chunker.apply(document)

        chunks1 = result1.payload["chunks"]
        chunks2 = result2.payload["chunks"]

        # Same number of chunks
        assert len(chunks1) == len(chunks2)

        # Same IDs and content
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.id == c2.id
            assert c1.text == c2.text
            assert c1.id.startswith("my_test_doc_chunk_")

    def test_string_input(self):
        """Test that chunker works with plain string input."""
        chunker = SentenceWindowChunker(window_size=2, overlap_sentences=1)

        test_text = "First sentence. Second sentence. Third sentence."

        result = chunker.apply(test_text)

        assert result.success is True
        chunks = result.payload["chunks"]

        for chunk in chunks:
            assert chunk.document_id == "unknown"  # Default for string input

    def test_parameter_override(self):
        """Test that parameters can be overridden in apply()."""
        chunker = SentenceWindowChunker(window_size=5, overlap_sentences=2)  # Large defaults

        test_text = "One. Two. Three. Four."
        document = Document(id="test", text=test_text)

        # Override with smaller window
        result = chunker.apply(document, window_size=1, overlap_sentences=0)

        chunks = result.payload["chunks"]

        # Should have 4 chunks (one per sentence) due to window_size=1
        assert len(chunks) == 4

    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        chunker = SentenceWindowChunker()

        document = Document(id="test", text="Test sentence.")

        # Invalid window_size
        result = chunker.apply(document, window_size=0)
        assert result.success is False
        assert "error" in result.payload

        # Invalid overlap
        result = chunker.apply(document, window_size=2, overlap_sentences=3)
        assert result.success is False
        assert "error" in result.payload

        result = chunker.apply(document, window_size=2, overlap_sentences=-1)
        assert result.success is False
        assert "error" in result.payload

    def test_empty_text(self):
        """Test handling of empty text."""
        chunker = SentenceWindowChunker(window_size=2, overlap_sentences=1)

        document = Document(id="empty", text="")

        result = chunker.apply(document)

        assert result.success is True
        chunks = result.payload["chunks"]
        assert len(chunks) == 0

    def test_no_sentence_boundaries(self):
        """Test handling of text with no clear sentence boundaries."""
        chunker = SentenceWindowChunker(window_size=2, overlap_sentences=0)

        test_text = "This is text with no sentence endings"
        document = Document(id="no_sentences", text=test_text)

        result = chunker.apply(document)

        assert result.success is True
        chunks = result.payload["chunks"]

        # Should still create at least one chunk
        assert len(chunks) >= 1
        assert chunks[0].text.strip() == test_text.strip()

    def test_very_short_document(self):
        """Test handling of very short documents."""
        chunker = SentenceWindowChunker(window_size=3, overlap_sentences=1)

        test_text = "Short!"
        document = Document(id="short", text=test_text)

        result = chunker.apply(document)

        assert result.success is True
        chunks = result.payload["chunks"]

        assert len(chunks) == 1
        assert chunks[0].text.strip() == "Short!"

    def test_metadata_preservation(self):
        """Test that document metadata is preserved in chunks."""
        chunker = SentenceWindowChunker(window_size=2, overlap_sentences=1)

        document = Document(
            id="meta_test",
            text="First sentence. Second sentence. Third sentence.",
            meta={"source": "test_file.txt", "type": "article"}
        )

        result = chunker.apply(document)
        chunks = result.payload["chunks"]

        for chunk in chunks:
            assert "source" in chunk.meta
            assert "type" in chunk.meta
            assert chunk.meta["source"] == "test_file.txt"
            assert chunk.meta["type"] == "article"
            assert chunk.meta["method"] == "sentence_window"

    def test_newline_sentence_boundaries(self):
        """Test that newlines are treated as sentence boundaries."""
        chunker = SentenceWindowChunker(window_size=1, overlap_sentences=0)

        test_text = "First line\nSecond line\nThird line"
        document = Document(id="newlines", text=test_text)

        result = chunker.apply(document)
        chunks = result.payload["chunks"]

        # Should create separate chunks for newline-separated text
        assert len(chunks) >= 2

        # Each chunk should contain text from the original
        for chunk in chunks:
            chunk_text = chunk.text.strip()
            assert chunk_text in test_text
