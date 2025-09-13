"""SentenceWindowChunker: Sentence-based windowing chunking technique."""

import re
from typing import List, Union
from ..core import RAGTechnique, TechniqueMeta, TechniqueResult
from ..schemas import Document, Chunk
from ..registry import TechniqueRegistry


@TechniqueRegistry.register
class SentenceWindowChunker(RAGTechnique):
    """Sentence-based windowing chunking technique.
    
    Splits text into sentences and creates chunks by sliding a window
    of sentences with optional overlap.
    """
    
    meta = TechniqueMeta(
        name="sentence_window_chunker",
        category="chunking",
        description="Sentence-based windowing with configurable window size and overlap"
    )
    
    # Simple sentence boundary detection pattern
    SENTENCE_PATTERN = re.compile(r'[.!?]+\s+|[.!?]+$|\n+', re.MULTILINE)
    
    def __init__(self, window_size: int = 3, overlap_sentences: int = 1):
        """Initialize SentenceWindowChunker.
        
        Args:
            window_size: Number of sentences per chunk window
            overlap_sentences: Number of sentences to overlap between windows
        """
        super().__init__(self.meta)
        self.window_size = window_size
        self.overlap_sentences = overlap_sentences
        
    def _split_sentences(self, text: str) -> List[tuple]:
        """Split text into sentences with position information.
        
        Returns:
            List of (sentence_text, start_pos, end_pos) tuples
        """
        sentences = []
        last_end = 0
        
        for match in self.SENTENCE_PATTERN.finditer(text):
            sentence_start = last_end
            sentence_end = match.end()
            
            # Extract sentence text, removing trailing whitespace/punctuation from the content
            sentence_text = text[sentence_start:match.start()].strip()
            
            # Only add non-empty sentences
            if sentence_text:
                sentences.append((sentence_text, sentence_start, sentence_end))
            
            last_end = sentence_end
        
        # Handle remaining text after last sentence boundary
        if last_end < len(text):
            remaining_text = text[last_end:].strip()
            if remaining_text:
                sentences.append((remaining_text, last_end, len(text)))
        
        return sentences
        
    def apply(self, document: Union[Document, str], *args, **kwargs) -> TechniqueResult:
        """Apply sentence window chunking to document.
        
        Args:
            document: Document object or plain text string
            window_size: Override default window size (optional)
            overlap_sentences: Override default overlap (optional)
            
        Returns:
            TechniqueResult with chunks in payload
        """
        # Extract parameters
        window_size = kwargs.get('window_size', self.window_size)
        overlap_sentences = kwargs.get('overlap_sentences', self.overlap_sentences)
        
        # Validate parameters
        if window_size <= 0:
            return TechniqueResult(
                success=False,
                payload={"error": "window_size must be positive"},
                meta={"window_size": window_size, "overlap_sentences": overlap_sentences}
            )
            
        if overlap_sentences < 0 or overlap_sentences >= window_size:
            return TechniqueResult(
                success=False,
                payload={"error": "overlap_sentences must be >= 0 and < window_size"},
                meta={"window_size": window_size, "overlap_sentences": overlap_sentences}
            )
        
        # Extract text and document_id
        if isinstance(document, Document):
            text = document.text
            document_id = document.id
            doc_meta = document.meta
        else:
            text = str(document)
            document_id = "unknown"
            doc_meta = {}
        
        # Split text into sentences
        sentences = self._split_sentences(text)
        
        if not sentences:
            return TechniqueResult(
                success=True,
                payload={"chunks": []},
                meta={
                    "total_chunks": 0,
                    "window_size": window_size,
                    "overlap_sentences": overlap_sentences,
                    "total_sentences": 0
                }
            )
        
        chunks = []
        chunk_index = 0
        step = window_size - overlap_sentences
        
        sentence_idx = 0
        while sentence_idx < len(sentences):
            # Determine window end
            window_end = min(sentence_idx + window_size, len(sentences))
            
            # Get sentences in current window
            window_sentences = sentences[sentence_idx:window_end]
            
            # Calculate chunk boundaries
            chunk_start = window_sentences[0][1]  # Start of first sentence
            chunk_end = window_sentences[-1][2]   # End of last sentence
            
            # Combine sentence texts
            chunk_text = text[chunk_start:chunk_end].strip()
            
            # Create chunk
            chunk_id = f"{document_id}_chunk_{chunk_index}"
            chunk = Chunk(
                id=chunk_id,
                document_id=document_id,
                text=chunk_text,
                start_idx=chunk_start,
                end_idx=chunk_end,
                embedding=None,  # No embeddings for basic sentence chunker
                meta={
                    "window_size": window_size,
                    "overlap_sentences": overlap_sentences,
                    "chunk_index": chunk_index,
                    "sentence_count": len(window_sentences),
                    "method": "sentence_window",
                    **doc_meta
                }
            )
            
            chunks.append(chunk)
            
            # Move to next window
            chunk_index += 1
            sentence_idx += step
            
            # If we processed the last possible window, break
            if window_end == len(sentences):
                break
        
        return TechniqueResult(
            success=True,
            payload={"chunks": chunks},
            meta={
                "total_chunks": len(chunks),
                "window_size": window_size,
                "overlap_sentences": overlap_sentences,
                "total_sentences": len(sentences),
                "original_length": len(text)
            }
        )
