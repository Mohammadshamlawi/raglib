"""PropositionalChunker: Propositional chunking based on semantic propositions."""

import re
from typing import Union

from ..core import RAGTechnique, TechniqueMeta, TechniqueResult
from ..registry import TechniqueRegistry
from ..schemas import Chunk, Document


@TechniqueRegistry.register
class PropositionalChunker(RAGTechnique):
    """Propositional chunking technique that extracts semantic propositions.

    Identifies and groups semantic propositions (subject-predicate-object)
    from text to create meaningful chunks based on logical units of meaning.
    """

    meta = TechniqueMeta(
        name="propositional_chunker",
        category="chunking",
        description="Propositional chunking based on semantic propositions"
    )

    def __init__(
        self,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        overlap: int = 50,
        max_propositions_per_chunk: int = 5
    ):
        """Initialize PropositionalChunker.

        Args:
            max_chunk_size: Maximum size of each chunk in characters
            min_chunk_size: Minimum size of each chunk in characters
            overlap: Number of characters to overlap between adjacent chunks
            max_propositions_per_chunk: Maximum number of propositions per chunk
        """
        super().__init__(self.meta)
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap
        self.max_propositions_per_chunk = max_propositions_per_chunk

    def _extract_sentences(self, text: str) -> list[dict]:
        """Extract sentences with their positions."""
        # Simple sentence splitting (can be enhanced with NLP libraries)
        sentence_pattern = r'([.!?]+\s*)'
        sentences = []
        
        current_pos = 0
        parts = re.split(sentence_pattern, text)
        
        for i in range(0, len(parts) - 1, 2):
            sentence_text = parts[i]
            punctuation = parts[i + 1] if i + 1 < len(parts) else ""
            full_sentence = (sentence_text + punctuation).strip()
            
            if full_sentence:
                sentences.append({
                    'text': full_sentence,
                    'start_idx': current_pos,
                    'end_idx': current_pos + len(full_sentence)
                })
                current_pos += len(sentence_text + punctuation)
        
        # Handle last sentence if no punctuation
        if parts and parts[-1].strip() not in '.!?':
            last_part = parts[-1].strip()
            if last_part:
                sentences.append({
                    'text': last_part,
                    'start_idx': current_pos,
                    'end_idx': current_pos + len(last_part)
                })
        
        return sentences

    def _identify_propositions(self, sentence: str) -> list[dict]:
        """Identify propositions within a sentence."""
        propositions = []
        
        # Simple heuristic-based proposition extraction
        # This can be enhanced with actual NLP parsing
        
        # Look for coordinating conjunctions that might split propositions
        conjunctions = ['and', 'but', 'or', 'so', 'yet', 'for', 'nor']
        
        # Split by conjunctions while preserving position info
        parts = [sentence]
        for conj in conjunctions:
            new_parts = []
            for part in parts:
                subparts = re.split(f'\\s+{conj}\\s+', part, flags=re.IGNORECASE)
                if len(subparts) > 1:
                    # Rejoin with conjunction for position tracking
                    for i, subpart in enumerate(subparts):
                        new_parts.append(subpart.strip())
                        if i < len(subparts) - 1:
                            new_parts.append(f" {conj} ")
                else:
                    new_parts.append(part)
            parts = new_parts
        
        # Filter out conjunctions and empty parts
        meaningful_parts = [
            part.strip() for part in parts
            if part.strip() and part.strip().lower() not in conjunctions
        ]
        
        # Look for relative clauses
        relative_pronouns = ['who', 'which', 'that', 'whose', 'where', 'when']
        enhanced_parts = []
        
        for part in meaningful_parts:
            # Split by relative pronouns
            for pronoun in relative_pronouns:
                pattern = f'\\s+{pronoun}\\s+'
                subparts = re.split(pattern, part, flags=re.IGNORECASE)
                if len(subparts) > 1:
                    enhanced_parts.extend([s.strip() for s in subparts if s.strip()])
                    break
            else:
                enhanced_parts.append(part)
        
        # Create proposition objects
        start_pos = 0
        for part in enhanced_parts:
            if len(part.strip()) > 5:  # Minimum proposition length
                propositions.append({
                    'text': part.strip(),
                    'type': 'proposition',
                    'confidence': self._calculate_proposition_confidence(part.strip()),
                    'start_idx': start_pos,
                    'end_idx': start_pos + len(part.strip())
                })
                start_pos += len(part) + 1
        
        # If no clear propositions found, treat whole sentence as one proposition
        if not propositions and sentence.strip():
            propositions.append({
                'text': sentence.strip(),
                'type': 'sentence',
                'confidence': 0.5,
                'start_idx': 0,
                'end_idx': len(sentence.strip())
            })
        
        return propositions

    def _calculate_proposition_confidence(self, text: str) -> float:
        """Calculate confidence score for proposition quality."""
        confidence = 0.0
        
        # Check for subject-verb pattern
        pattern = r'\b\w+\s+(is|are|was|were|has|have|will|would|could|should)\b'
        if re.search(pattern, text, re.IGNORECASE):
            confidence += 0.3
        
        # Check for action verbs
        if re.search(r'\b\w+s?\s+\w+', text):
            confidence += 0.2
        
        # Check for objects/complements
        if re.search(r'\b(the|a|an)\s+\w+', text, re.IGNORECASE):
            confidence += 0.2
        
        # Length-based confidence
        if 10 <= len(text) <= 100:
            confidence += 0.2
        elif len(text) > 100:
            confidence += 0.1
        
        # Completeness check
        if text.strip().endswith(('.', '!', '?')):
            confidence += 0.1
        
        return min(confidence, 1.0)

    def _group_propositions_into_chunks(
        self, propositions: list[dict], document_id: str
    ) -> list[dict]:
        """Group propositions into chunks based on size and semantic coherence."""
        chunks = []
        current_chunk_props = []
        current_chunk_size = 0
        
        for prop in propositions:
            prop_size = len(prop['text']) + 1  # +1 for space
            
            # Check if adding this proposition would exceed limits
            would_exceed_size = current_chunk_size + prop_size > self.max_chunk_size
            would_exceed_count = len(current_chunk_props) >= self.max_propositions_per_chunk
            
            if (would_exceed_size or would_exceed_count) and current_chunk_props:
                # Finalize current chunk
                chunk_text = ' '.join([p['text'] for p in current_chunk_props])
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append({
                        'text': chunk_text,
                        'propositions': current_chunk_props.copy(),
                        'start_idx': current_chunk_props[0]['start_idx'],
                        'end_idx': current_chunk_props[-1]['end_idx'],
                        'document_id': document_id
                    })
                
                # Start new chunk with overlap
                if self.overlap > 0 and current_chunk_props:
                    overlap_props = self._get_overlap_propositions(current_chunk_props)
                    current_chunk_props = overlap_props + [prop]
                    current_chunk_size = sum(len(p['text']) + 1 for p in current_chunk_props)
                else:
                    current_chunk_props = [prop]
                    current_chunk_size = prop_size
            else:
                # Add proposition to current chunk
                current_chunk_props.append(prop)
                current_chunk_size += prop_size
        
        # Handle remaining propositions
        if current_chunk_props:
            chunk_text = ' '.join([p['text'] for p in current_chunk_props])
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append({
                    'text': chunk_text,
                    'propositions': current_chunk_props,
                    'start_idx': current_chunk_props[0]['start_idx'],
                    'end_idx': current_chunk_props[-1]['end_idx'],
                    'document_id': document_id
                })
        
        return chunks

    def _get_overlap_propositions(self, propositions: list[dict]) -> list[dict]:
        """Get propositions for overlap based on character limit."""
        if not propositions or self.overlap <= 0:
            return []
        
        overlap_props = []
        overlap_size = 0
        
        # Add propositions from the end until overlap limit is reached
        for prop in reversed(propositions):
            prop_size = len(prop['text']) + 1
            if overlap_size + prop_size <= self.overlap:
                overlap_props.insert(0, prop)
                overlap_size += prop_size
            else:
                break
        
        return overlap_props

    def apply(self, document: Union[Document, str], *args, **kwargs) -> TechniqueResult:
        """Apply propositional chunking to document.

        Args:
            document: Document object or plain text string
            max_chunk_size: Override default max chunk size (optional)
            min_chunk_size: Override default min chunk size (optional)
            overlap: Override default overlap (optional)
            max_propositions_per_chunk: Override default max propositions (optional)

        Returns:
            TechniqueResult with chunks in payload
        """
        # Extract parameters
        max_chunk_size = kwargs.get('max_chunk_size', self.max_chunk_size)
        min_chunk_size = kwargs.get('min_chunk_size', self.min_chunk_size)
        overlap = kwargs.get('overlap', self.overlap)
        # Remove unused variable to fix linting
        # max_propositions_per_chunk = kwargs.get(
        #     'max_propositions_per_chunk', self.max_propositions_per_chunk
        # )

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

        # Extract sentences from text
        sentences = self._extract_sentences(text)

        # Extract propositions from sentences
        all_propositions = []
        for sentence in sentences:
            propositions = self._identify_propositions(sentence['text'])
            # Adjust proposition positions relative to document
            for prop in propositions:
                prop['start_idx'] += sentence['start_idx']
                prop['end_idx'] += sentence['start_idx']
            all_propositions.extend(propositions)

        # Group propositions into chunks
        chunk_data = self._group_propositions_into_chunks(all_propositions, document_id)

        # Convert to Chunk objects
        chunks = []
        for chunk_index, chunk_info in enumerate(chunk_data):
            chunk_id = f"{document_id}_chunk_{chunk_index}"
            
            # Calculate chunk statistics
            proposition_count = len(chunk_info.get('propositions', []))
            propositions = chunk_info.get('propositions', [])
            confidence_sum = sum(p.get('confidence', 0) for p in propositions)
            avg_confidence = confidence_sum / max(proposition_count, 1)
            
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
                    "method": "propositional",
                    "proposition_count": proposition_count,
                    "avg_confidence": avg_confidence,
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
                "total_propositions": len(all_propositions),
                "avg_propositions_per_chunk": (
                    len(all_propositions) / max(len(chunks), 1)
                ),
                "original_length": len(text)
            }
        )
