#!/usr/bin/env python3
"""Test script for new chunking techniques."""

from raglib.schemas import Document
from raglib.techniques.content_aware_chunker import (
    ContentAwareChunker,
)
from raglib.techniques.document_specific_chunker import (
    DocumentSpecificChunker,
)
from raglib.techniques.propositional_chunker import (
    PropositionalChunker,
)
from raglib.techniques.recursive_chunker import RecursiveChunker

def test_chunking_techniques():
    """Test all new chunking techniques."""
    
    # Test document
    test_text = """
    # Introduction
    
    This is a test document with multiple paragraphs. It contains various
    types of content to test different chunking strategies.
    
    ## Content Structure
    
    The document has headings, paragraphs, and different sentence structures.
    Some sentences are short. Others are much longer and contain multiple
    clauses that could be split into propositions.
    
    ### Code Example
    
    def example_function():
        return "This is code content"
    
    ## Conclusion
    
    This concludes our test document.
    """
    
    document = Document(id="test_doc", text=test_text)
    
    # Test ContentAwareChunker
    print("Testing ContentAwareChunker:")
    chunker1 = ContentAwareChunker(max_chunk_size=200, min_chunk_size=50)
    result1 = chunker1.apply(document)
    print(f"  Created {len(result1.payload['chunks'])} chunks")
    print(f"  Detected {result1.meta['natural_boundaries']} natural boundaries")
    
    # Test DocumentSpecificChunker
    print("\nTesting DocumentSpecificChunker:")
    chunker2 = DocumentSpecificChunker(max_chunk_size=200, min_chunk_size=50)
    result2 = chunker2.apply(document)
    print(f"  Created {len(result2.payload['chunks'])} chunks")
    print(f"  Detected document type: {result2.meta['detected_type']}")
    
    # Test RecursiveChunker
    print("\nTesting RecursiveChunker:")
    chunker3 = RecursiveChunker(chunk_size=150, overlap=30)
    result3 = chunker3.apply(document)
    print(f"  Created {len(result3.payload['chunks'])} chunks")
    print(f"  Used separators: {result3.meta['separators']}")
    
    # Test PropositionalChunker
    print("\nTesting PropositionalChunker:")
    chunker4 = PropositionalChunker(max_chunk_size=200, min_chunk_size=50)
    result4 = chunker4.apply(document)
    print(f"  Created {len(result4.payload['chunks'])} chunks")
    print(f"  Found {result4.meta['total_propositions']} propositions")
    print(f"  Avg propositions per chunk: "
          f"{result4.meta['avg_propositions_per_chunk']:.1f}")
    
    print("\n✅ All chunking techniques working successfully!")

if __name__ == "__main__":
    test_chunking_techniques()