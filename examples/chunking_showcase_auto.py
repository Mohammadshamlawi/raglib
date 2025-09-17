#!/usr/bin/env python3
"""
Auto-generated Chunking Showcase Script

This script demonstrates all available chunking techniques with examples.
Generated automatically by the RAGLib documentation updater.

Generated on: 2025-09-17 22:09:34
"""

import sys
from pathlib import Path

# Add raglib to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from raglib.techniques import (
        ContentAwareChunker, DocumentSpecificChunker, FixedSizeChunker, ParentDocumentChunker, PropositionalChunker, RecursiveChunker, SemanticChunker, SentenceWindowChunker
    )
    from raglib.schemas import Document
except ImportError as e:
    print(f"Failed to import raglib: {e}")
    sys.exit(1)


def main():
    """Demonstrate all chunking techniques."""
    print("🔬 RAGLib Chunking Techniques Showcase")
    print("=" * 50)
    
    # Test document
    document = Document(
        id="showcase_doc",
        text="""
# Machine Learning Fundamentals

## Introduction

Machine learning is a subset of artificial intelligence that focuses on 
algorithms that can learn and improve from experience.

### Supervised Learning

In supervised learning, algorithms learn from labeled training data.
Common examples include:

- Classification problems
- Regression analysis
- Pattern recognition

### Unsupervised Learning

Unsupervised learning finds hidden patterns in data without labels.
This includes clustering and dimensionality reduction techniques.

## Applications

Machine learning has numerous real-world applications:

1. Computer vision and image recognition
2. Natural language processing
3. Recommendation systems
4. Autonomous vehicles

## Conclusion

Understanding these fundamentals provides a solid foundation for
exploring more advanced machine learning concepts.
        """,
        meta={"type": "educational", "topic": "machine_learning"}
    )
    
    # Initialize all techniques
    techniques = [
        ("ContentAwareChunker", ContentAwareChunker(max_chunk_size=200)),
        ("DocumentSpecificChunker", DocumentSpecificChunker(chunk_size=200)),
        ("FixedSizeChunker", FixedSizeChunker(chunk_size=200)),
        ("ParentDocumentChunker", ParentDocumentChunker(child_chunk_size=100, parent_chunk_size=300)),
        ("PropositionalChunker", PropositionalChunker(chunk_size=200)),
        ("RecursiveChunker", RecursiveChunker(chunk_size=200)),
        ("SemanticChunker", SemanticChunker(chunk_size=200)),
        ("SentenceWindowChunker", SentenceWindowChunker(window_size=3)),
    ]
    
    print(f"Testing {len(techniques)} chunking techniques:")
    print()
    
    for name, technique in techniques:
        try:
            result = technique.apply(document)
            
            if result.success:
                if "Parent" in name:
                    payload = result.payload
                    child_count = len(payload.get("child_chunks", []))
                    parent_count = len(payload.get("parent_chunks", []))
                    print(f"✅ {name}: {child_count} child, {parent_count} parent chunks")
                else:
                    chunks = result.payload.get("chunks", [])
                    if chunks:
                        avg_length = sum(len(c.text) for c in chunks) / len(chunks)
                        print(f"✅ {name}: {len(chunks)} chunks (avg: {avg_length:.0f} chars)")
                        
                        # Show first chunk preview
                        if self.config.verbose and chunks:
                            preview = chunks[0].text[:100].replace('\n', ' ').strip()
                            print(f"   Preview: {preview}...")
                    else:
                        print(f"✅ {name}: No chunks created")
            else:
                print(f"❌ {name}: {result.error}")
                
        except Exception as e:
            print(f"💥 {name}: {e}")
    
    print("\n🎉 Showcase complete!")


if __name__ == "__main__":
    main()
