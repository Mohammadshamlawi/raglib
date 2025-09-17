#!/usr/bin/env python3
"""
Comprehensive Chunking Techniques Showcase

This example demonstrates all 8 chunking techniques available in RAGLib,
showing how they handle different types of documents and their unique characteristics.
"""

import sys
from pathlib import Path

# Add raglib to path for development
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from raglib.techniques import (
    FixedSizeChunker,
    SemanticChunker,
    SentenceWindowChunker,
    ContentAwareChunker,
    DocumentSpecificChunker,
    RecursiveChunker,
    PropositionalChunker,
    ParentDocumentChunker
)
from raglib.schemas import Document


def main():
    """Demonstrate all chunking techniques with different document types."""
    print("🔬 RAGLib Chunking Techniques Showcase")
    print("=" * 50)
    
    # Prepare test documents with different structures
    documents = prepare_test_documents()
    
    # Initialize all chunking techniques
    chunkers = initialize_chunkers()
    
    # Test each technique on each document
    for doc in documents:
        print(f"\n📄 Document: {doc.id} ({doc.meta.get('type', 'unknown')} type)")
        print(f"   Length: {len(doc.text)} characters")
        print(f"   Preview: {doc.text[:100]}...")
        print("-" * 50)
        
        for name, chunker in chunkers.items():
            try:
                result = chunker.apply(doc)
                
                if result.success:
                    if name == "parent_document_chunker":
                        # Special handling for parent-document chunker
                        payload = result.payload
                        child_chunks = payload.get("child_chunks", [])
                        parent_chunks = payload.get("parent_chunks", [])
                        
                        print(f"✅ {name}:")
                        print(f"   Child chunks: {len(child_chunks)}")
                        print(f"   Parent chunks: {len(parent_chunks)}")
                        
                        if child_chunks:
                            avg_child_length = sum(len(c.text) for c in child_chunks) / len(child_chunks)
                            print(f"   Avg child length: {avg_child_length:.0f} chars")
                        
                        if parent_chunks:
                            avg_parent_length = sum(len(c.text) for c in parent_chunks) / len(parent_chunks)
                            print(f"   Avg parent length: {avg_parent_length:.0f} chars")
                    else:
                        # Standard chunking techniques
                        chunks = result.payload.get("chunks", [])
                        if chunks:
                            avg_length = sum(len(c.text) for c in chunks) / len(chunks)
                            min_length = min(len(c.text) for c in chunks)
                            max_length = max(len(c.text) for c in chunks)
                            
                            print(f"✅ {name}:")
                            print(f"   Chunks: {len(chunks)}")
                            print(f"   Avg length: {avg_length:.0f} chars")
                            print(f"   Range: {min_length}-{max_length} chars")
                            
                            # Show boundary analysis
                            boundary_info = analyze_boundaries(chunks)
                            print(f"   Boundaries: {boundary_info}")
                        else:
                            print(f"✅ {name}: No chunks created")
                else:
                    print(f"❌ {name}: {result.error}")
                    
            except Exception as e:
                print(f"💥 {name}: Exception - {e}")
        
        print()
    
    # Demonstrate technique-specific features
    demonstrate_special_features()


def prepare_test_documents():
    """Create test documents with different structures."""
    return [
        Document(
            id="academic_paper",
            text="""
# Advances in Natural Language Processing

## Abstract

This paper presents recent advances in natural language processing, focusing on transformer architectures and their applications.

## 1. Introduction

Natural language processing (NLP) has evolved significantly over the past decade. The introduction of attention mechanisms and transformer models has revolutionized the field.

### 1.1 Background

Traditional NLP approaches relied heavily on rule-based systems and statistical methods. However, these approaches had significant limitations in handling complex linguistic phenomena.

### 1.2 Motivation

The need for more sophisticated language understanding capabilities has driven research toward neural approaches.

## 2. Methodology

Our approach combines several key innovations:

1. Multi-head attention mechanisms
2. Positional encoding schemes
3. Layer normalization techniques

### 2.1 Architecture Details

The transformer architecture consists of encoder and decoder layers, each containing self-attention and feed-forward sub-layers.

## 3. Results

Experimental results demonstrate significant improvements over baseline methods.

## 4. Conclusion

This work establishes new state-of-the-art results in several NLP benchmarks.
            """,
            meta={"type": "academic", "domain": "nlp", "structure": "hierarchical"}
        ),
        
        Document(
            id="technical_manual",
            text="""
API Documentation

Authentication
All requests require API authentication using Bearer tokens.

GET /users
Retrieve user information.

Parameters:
- id (required): User identifier
- fields (optional): Comma-separated list of fields

Response:
{
  "id": "user123",
  "name": "John Doe",
  "email": "john@example.com"
}

POST /users
Create a new user account.

Request Body:
{
  "name": "Jane Smith",
  "email": "jane@example.com",
  "password": "secure_password"
}

Error Codes:
- 400: Bad Request
- 401: Unauthorized
- 404: Not Found
- 500: Internal Server Error
            """,
            meta={"type": "technical", "format": "api_docs"}
        ),
        
        Document(
            id="narrative_story",
            text="""
The old lighthouse stood majestically on the cliff, its weathered walls telling stories of countless storms weathered and ships guided safely to harbor. Sarah had always been drawn to this place, ever since her grandmother first brought her here as a child.

The beacon had been automated decades ago, but the lighthouse keeper's quarters remained, now serving as a small museum. Sarah climbed the spiral staircase, each step echoing in the cylindrical chamber. At the top, the view was breathtaking—endless ocean stretching to the horizon, waves crashing against the rocks below.

She opened her grandmother's journal, its pages yellowed with age. The entries spoke of stormy nights, of watching for ships in distress, of the profound responsibility that came with being a guardian of the sea. Each page revealed a piece of family history she had never known.

As the sun began to set, painting the sky in brilliant oranges and purples, Sarah understood why her grandmother had loved this place so much. It represented something eternal, something that connected past and present, earth and sea, duty and love.
            """,
            meta={"type": "narrative", "genre": "literary"}
        ),
        
        Document(
            id="scientific_report",
            text="""
Experiment Report: Protein Folding Analysis
Date: 2024-01-15
Researcher: Dr. Emily Chen

Objective:
Analyze protein folding patterns under varying temperature conditions.

Materials:
- Wild-type protein samples
- Mutant variants (T45A, R67Q)
- Circular dichroism spectrometer
- Temperature control system

Method:
1. Prepare protein solutions (0.1 mg/mL)
2. Equilibrate at target temperatures (20-80°C)
3. Record CD spectra (190-260 nm)
4. Calculate secondary structure percentages
5. Determine melting temperatures

Results:
Wild-type: Tm = 68.2°C, 45% α-helix, 20% β-sheet
Mutant T45A: Tm = 52.1°C, 38% α-helix, 25% β-sheet
Mutant R67Q: Tm = 71.8°C, 52% α-helix, 18% β-sheet

Statistical analysis shows significant differences (p < 0.001).

Discussion:
The T45A mutation destabilizes the protein structure, likely due to loss of hydrophobic interactions. The R67Q substitution enhances stability by eliminating unfavorable electrostatic repulsions.

Conclusion:
Amino acid composition significantly affects thermal stability. Future work should investigate dynamic behavior using NMR spectroscopy.
            """,
            meta={"type": "scientific", "field": "biochemistry"}
        )
    ]


def initialize_chunkers():
    """Initialize all chunking techniques with appropriate parameters."""
    return {
        "fixed_size": FixedSizeChunker(chunk_size=200, overlap=50),
        "semantic": SemanticChunker(chunk_size=300, similarity_threshold=0.7),
        "sentence_window": SentenceWindowChunker(window_size=3, overlap_sentences=1),
        "content_aware": ContentAwareChunker(max_chunk_size=250, overlap=40),
        "document_specific": DocumentSpecificChunker(chunk_size=300, overlap=30),
        "recursive": RecursiveChunker(
            chunk_size=200, 
            overlap=25,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        ),
        "propositional": PropositionalChunker(chunk_size=180, overlap=20),
        "parent_document": ParentDocumentChunker(
            child_chunk_size=100,
            parent_chunk_size=400,
            overlap=20
        )
    }


def analyze_boundaries(chunks):
    """Analyze how well chunks respect natural text boundaries."""
    if not chunks:
        return "No chunks"
    
    sentence_ends = sum(1 for c in chunks if c.text.strip().endswith(('.', '!', '?')))
    paragraph_breaks = sum(1 for c in chunks if '\n\n' in c.text[-10:])
    
    total = len(chunks)
    
    return f"Sentences: {sentence_ends}/{total}, Paragraphs: {paragraph_breaks}/{total}"


def demonstrate_special_features():
    """Demonstrate unique features of specific chunking techniques."""
    print("🔧 Special Features Demonstration")
    print("=" * 40)
    
    # Recursive chunker with custom separators
    print("\n📐 Recursive Chunker - Custom Separator Hierarchy")
    recursive_chunker = RecursiveChunker(
        chunk_size=100,
        separators=["###", "##", "#", "\n\n", ".", " "]  # Markdown hierarchy
    )
    
    markdown_doc = Document(
        id="markdown",
        text="""
# Main Title

## Section 1

Some content here.

### Subsection 1.1

More detailed content.

## Section 2

Different content.

### Subsection 2.1

Final content.
        """
    )
    
    result = recursive_chunker.apply(markdown_doc)
    if result.success:
        chunks = result.payload["chunks"]
        print(f"   Created {len(chunks)} chunks respecting Markdown hierarchy")
        for i, chunk in enumerate(chunks):
            preview = chunk.text.strip().replace('\n', ' ')[:60]
            print(f"   Chunk {i+1}: {preview}...")
    
    # Parent-document chunker demonstration
    print("\n👨‍👧 Parent-Document Chunker - Context Preservation")
    parent_chunker = ParentDocumentChunker(
        child_chunk_size=50,
        parent_chunk_size=200,
        overlap=10
    )
    
    long_doc = Document(
        id="context_test",
        text="""
The concept of artificial intelligence has evolved significantly since its inception in the 1950s. Early pioneers like Alan Turing and John McCarthy laid the groundwork for what would become one of the most transformative technologies of the modern era.

Machine learning, a subset of AI, focuses on algorithms that can learn and improve from experience without being explicitly programmed. This field has seen remarkable advances with the development of neural networks and deep learning architectures.

Natural language processing represents another crucial branch of AI, enabling computers to understand, interpret, and generate human language in a valuable way. Recent breakthroughs in transformer models have revolutionized this field.
        """
    )
    
    result = parent_chunker.apply(long_doc)
    if result.success:
        payload = result.payload
        child_chunks = payload.get("child_chunks", [])
        parent_chunks = payload.get("parent_chunks", [])
        
        print(f"   Child chunks: {len(child_chunks)} (for precise retrieval)")
        print(f"   Parent chunks: {len(parent_chunks)} (for context)")
        
        if child_chunks and parent_chunks:
            print(f"   Example mapping: Child chunk maps to parent for full context")
    
    print("\n✨ All demonstrations complete!")


if __name__ == "__main__":
    main()