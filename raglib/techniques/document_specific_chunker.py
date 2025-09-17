"""DocumentSpecificChunker: Document-specific chunking based on document type."""

import re
from typing import Union

from ..core import RAGTechnique, TechniqueMeta, TechniqueResult
from ..registry import TechniqueRegistry
from ..schemas import Chunk, Document


@TechniqueRegistry.register
class DocumentSpecificChunker(RAGTechnique):
    """Document-specific chunking technique that adapts to document type.

    Analyzes document structure and content to apply appropriate chunking
    strategies based on detected document type (markdown, code, prose, etc.).
    """

    meta = TechniqueMeta(
        name="document_specific_chunker",
        category="chunking",
        description="Document-specific chunking that adapts to document type"
    )

    def __init__(
        self,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        overlap: int = 50
    ):
        """Initialize DocumentSpecificChunker.

        Args:
            max_chunk_size: Maximum size of each chunk in characters
            min_chunk_size: Minimum size of each chunk in characters
            overlap: Number of characters to overlap between adjacent chunks
        """
        super().__init__(self.meta)
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap

    def _detect_document_type(self, text: str) -> str:
        """Detect the document type based on content patterns."""
        # Check for markdown patterns
        markdown_patterns = [
            r'^#{1,6}\s+',  # Headers
            r'\*\*.*?\*\*',  # Bold
            r'\*.*?\*',  # Italic
            r'```.*?```',  # Code blocks
            r'^\s*[-*+]\s+',  # Lists
            r'^\s*\d+\.\s+',  # Numbered lists
        ]

        markdown_score = 0
        for pattern in markdown_patterns:
            if re.search(pattern, text, re.MULTILINE | re.DOTALL):
                markdown_score += 1

        # Check for code patterns
        code_patterns = [
            r'def\s+\w+\s*\(',  # Python functions
            r'function\s+\w+\s*\(',  # JavaScript functions
            r'class\s+\w+\s*[:{]',  # Class definitions
            r'import\s+\w+',  # Import statements
            r'#include\s*<',  # C/C++ includes
            r'public\s+class\s+\w+',  # Java classes
        ]

        code_score = 0
        for pattern in code_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                code_score += 1

        # Check for structured document patterns
        structured_patterns = [
            r'^\s*\d+\.\d+',  # Section numbers
            r'Table\s+\d+',  # Table references
            r'Figure\s+\d+',  # Figure references
            r'Chapter\s+\d+',  # Chapter references
            r'Section\s+\d+',  # Section references
        ]

        structured_score = 0
        for pattern in structured_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                structured_score += 1

        # Determine document type
        if markdown_score >= 2:
            return "markdown"
        elif code_score >= 2:
            return "code"
        elif structured_score >= 2:
            return "structured"
        else:
            return "prose"

    def _chunk_markdown(self, text: str, document_id: str) -> list[dict]:
        """Chunk markdown document respecting markdown structure."""
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_size = 0
        chunk_start = 0
        line_start = 0

        for i, line in enumerate(lines):
            line_with_newline = line + '\n' if i < len(lines) - 1 else line
            line_size = len(line_with_newline)

            # Check if this line starts a new section (header)
            is_header = re.match(r'^#{1,6}\s+', line)

            # If we hit a header and have content, finalize current chunk
            if is_header and current_chunk and current_size >= self.min_chunk_size:
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    chunks.append({
                        'text': chunk_text,
                        'start_idx': chunk_start,
                        'end_idx': line_start - 1,
                        'document_id': document_id
                    })

                # Start new chunk with overlap
                overlap_lines = self._get_overlap_lines(current_chunk)
                current_chunk = overlap_lines + [line]
                current_size = sum(len(line) + 1 for line in overlap_lines) + line_size
                chunk_start = max(
                    0, line_start - sum(len(line) + 1 for line in overlap_lines)
                )
            else:
                current_chunk.append(line)
                current_size += line_size

            # If chunk exceeds max size, finalize it
            if current_size >= self.max_chunk_size:
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    chunks.append({
                        'text': chunk_text,
                        'start_idx': chunk_start,
                        'end_idx': line_start + line_size,
                        'document_id': document_id
                    })

                # Start new chunk with overlap
                overlap_lines = self._get_overlap_lines(current_chunk)
                current_chunk = overlap_lines
                current_size = sum(len(line) + 1 for line in overlap_lines)
                chunk_start = line_start + line_size - current_size

            line_start += line_size

        # Handle remaining content
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append({
                    'text': chunk_text,
                    'start_idx': chunk_start,
                    'end_idx': len(text),
                    'document_id': document_id
                })

        return chunks

    def _chunk_code(self, text: str, document_id: str) -> list[dict]:
        """Chunk code document respecting code structure."""
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_size = 0
        chunk_start = 0
        line_start = 0
        brace_depth = 0

        for i, line in enumerate(lines):
            line_with_newline = line + '\n' if i < len(lines) - 1 else line
            line_size = len(line_with_newline)

            # Track brace depth for scope awareness
            brace_depth += line.count('{') - line.count('}')

            # Check for function/class definitions
            is_definition = re.match(
                r'^\s*(def|function|class|public\s+class|private\s+class)\s+',
                line,
                re.IGNORECASE
            )

            # If we're at scope boundary and have content, consider chunking
            should_chunk = (
                (is_definition or brace_depth == 0) and
                current_chunk and
                current_size >= self.min_chunk_size
            )

            if should_chunk:
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    chunks.append({
                        'text': chunk_text,
                        'start_idx': chunk_start,
                        'end_idx': line_start - 1,
                        'document_id': document_id
                    })

                # Start new chunk with overlap
                overlap_lines = self._get_overlap_lines(current_chunk)
                current_chunk = overlap_lines + [line]
                current_size = sum(len(line) + 1 for line in overlap_lines) + line_size
                chunk_start = max(
                    0, line_start - sum(len(line) + 1 for line in overlap_lines)
                )
            else:
                current_chunk.append(line)
                current_size += line_size

            # Force chunk if too large
            if current_size >= self.max_chunk_size:
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    chunks.append({
                        'text': chunk_text,
                        'start_idx': chunk_start,
                        'end_idx': line_start + line_size,
                        'document_id': document_id
                    })

                # Start new chunk
                current_chunk = []
                current_size = 0
                chunk_start = line_start + line_size

            line_start += line_size

        # Handle remaining content
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append({
                    'text': chunk_text,
                    'start_idx': chunk_start,
                    'end_idx': len(text),
                    'document_id': document_id
                })

        return chunks

    def _chunk_structured(self, text: str, document_id: str) -> list[dict]:
        """Chunk structured document respecting section boundaries."""
        chunks = []
        
        # Find section boundaries
        section_pattern = r'^(\d+\.(?:\d+\.)*|\w+\.)\s+[A-Z]'
        sections = []
        lines = text.split('\n')
        
        current_section = {'start': 0, 'lines': []}
        
        for i, line in enumerate(lines):
            if re.match(section_pattern, line.strip()):
                # Finalize previous section
                if current_section['lines']:
                    current_section['text'] = '\n'.join(current_section['lines'])
                    sections.append(current_section)
                
                # Start new section
                current_section = {'start': i, 'lines': [line]}
            else:
                current_section['lines'].append(line)
        
        # Add last section
        if current_section['lines']:
            current_section['text'] = '\n'.join(current_section['lines'])
            sections.append(current_section)
        
        # Create chunks from sections
        for section in sections:
            section_text = section['text'].strip()
            if len(section_text) <= self.max_chunk_size:
                if len(section_text) >= self.min_chunk_size:
                    chunks.append({
                        'text': section_text,
                        'start_idx': sum(
                            len(lines[j]) + 1 for j in range(section['start'])
                        ),
                        'end_idx': sum(
                            len(lines[j]) + 1
                            for j in range(section['start'] + len(section['lines']))
                        ),
                        'document_id': document_id
                    })
            else:
                # Split large sections using prose chunking
                section_chunks = self._chunk_prose(section_text, document_id)
                chunks.extend(section_chunks)
        
        return chunks

    def _chunk_prose(self, text: str, document_id: str) -> list[dict]:
        """Chunk prose document using sentence and paragraph boundaries."""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        current_chunk = ""
        chunk_start = 0
        
        text_pos = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If adding this paragraph exceeds max size
            if len(current_chunk) + len(para) + 2 > self.max_chunk_size:
                # Finalize current chunk if it meets minimum size
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'start_idx': chunk_start,
                        'end_idx': text_pos,
                        'document_id': document_id
                    })
                    
                    # Start new chunk with overlap
                    overlap_text = (current_chunk[-self.overlap:]
                                    if self.overlap > 0 else "")
                    current_chunk = overlap_text + para
                    chunk_start = text_pos - len(overlap_text)
                else:
                    # Current chunk too small, add paragraph anyway
                    current_chunk += "\n\n" + para if current_chunk else para
            else:
                # Add paragraph to current chunk
                current_chunk += "\n\n" + para if current_chunk else para
            
            text_pos += len(para) + 2  # +2 for paragraph separator
        
        # Handle remaining content
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append({
                'text': current_chunk.strip(),
                'start_idx': chunk_start,
                'end_idx': len(text),
                'document_id': document_id
            })
        
        return chunks

    def _get_overlap_lines(self, lines: list[str]) -> list[str]:
        """Get overlap lines from the end of current chunk."""
        if not lines or self.overlap <= 0:
            return []
        
        overlap_size = 0
        overlap_lines = []
        
        for line in reversed(lines):
            if overlap_size + len(line) + 1 <= self.overlap:
                overlap_lines.insert(0, line)
                overlap_size += len(line) + 1
            else:
                break
        
        return overlap_lines

    def apply(self, document: Union[Document, str], *args, **kwargs) -> TechniqueResult:
        """Apply document-specific chunking to document.

        Args:
            document: Document object or plain text string
            max_chunk_size: Override default max chunk size (optional)
            min_chunk_size: Override default min chunk size (optional)
            overlap: Override default overlap (optional)

        Returns:
            TechniqueResult with chunks in payload
        """
        # Extract parameters
        max_chunk_size = kwargs.get('max_chunk_size', self.max_chunk_size)
        min_chunk_size = kwargs.get('min_chunk_size', self.min_chunk_size)
        overlap = kwargs.get('overlap', self.overlap)

        # Validate parameters
        if max_chunk_size <= 0:
            return TechniqueResult(
                success=False,
                payload={"error": "max_chunk_size must be positive"},
                meta={
                    "max_chunk_size": max_chunk_size,
                    "min_chunk_size": min_chunk_size,
                    "overlap": overlap
                }
            )

        if min_chunk_size <= 0 or min_chunk_size >= max_chunk_size:
            return TechniqueResult(
                success=False,
                payload={
                    "error": "min_chunk_size must be positive and < max_chunk_size"
                },
                meta={
                    "max_chunk_size": max_chunk_size,
                    "min_chunk_size": min_chunk_size,
                    "overlap": overlap
                }
            )

        if overlap < 0 or overlap >= max_chunk_size:
            return TechniqueResult(
                success=False,
                payload={"error": "overlap must be >= 0 and < max_chunk_size"},
                meta={
                    "max_chunk_size": max_chunk_size,
                    "min_chunk_size": min_chunk_size,
                    "overlap": overlap
                }
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

        # Detect document type
        doc_type = self._detect_document_type(text)

        # Apply appropriate chunking strategy
        if doc_type == "markdown":
            chunk_data = self._chunk_markdown(text, document_id)
        elif doc_type == "code":
            chunk_data = self._chunk_code(text, document_id)
        elif doc_type == "structured":
            chunk_data = self._chunk_structured(text, document_id)
        else:  # prose
            chunk_data = self._chunk_prose(text, document_id)

        # Convert to Chunk objects
        chunks = []
        for chunk_index, chunk_info in enumerate(chunk_data):
            chunk_id = f"{document_id}_chunk_{chunk_index}"
            chunk = Chunk(
                id=chunk_id,
                document_id=document_id,
                text=chunk_info['text'],
                start_idx=chunk_info['start_idx'],
                end_idx=chunk_info['end_idx'],
                embedding=None,
                meta={
                    "max_chunk_size": max_chunk_size,
                    "min_chunk_size": min_chunk_size,
                    "overlap": overlap,
                    "chunk_index": chunk_index,
                    "method": "document_specific",
                    "detected_type": doc_type,
                    **doc_meta
                }
            )

            chunks.append(chunk)

        return TechniqueResult(
            success=True,
            payload={"chunks": chunks},
            meta={
                "total_chunks": len(chunks),
                "max_chunk_size": max_chunk_size,
                "min_chunk_size": min_chunk_size,
                "overlap": overlap,
                "detected_type": doc_type,
                "original_length": len(text)
            }
        )
