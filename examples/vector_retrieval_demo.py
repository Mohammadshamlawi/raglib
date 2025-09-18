#!/usr/bin/env python3
"""
Vector Retrieval Techniques Demo for RAGLib.

This script provides a quick demonstration of the vector retrieval techniques
available in RAGLib with practical examples.

Usage:
    python vector_retrieval_demo.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from raglib.adapters import DummyEmbedder
from raglib.registry import TechniqueRegistry
from raglib.schemas import Chunk, Document


def main():
    """Main demo execution."""
    print("RAGLib Vector Retrieval Techniques Demo")
    print("=" * 50)
    
    # Sample documents
    documents = [
        Document(
            id="ai_intro",
            text="Artificial intelligence involves creating systems that can perform tasks "
                 "that typically require human intelligence, such as reasoning, learning, "
                 "and problem-solving."
        ),
        Document(
            id="ml_basics",
            text="Machine learning is a subset of AI that enables computers to learn "
                 "and improve from experience without being explicitly programmed."
        ),
        Document(
            id="neural_nets",
            text="Neural networks are computing systems inspired by biological neural "
                 "networks. They consist of interconnected nodes that process information."
        ),
        Document(
            id="deep_learning",
            text="Deep learning uses neural networks with multiple layers to model "
                 "and understand complex patterns in data like images, text, and speech."
        ),
        Document(
            id="nlp_field",
            text="Natural language processing focuses on the interaction between "
                 "computers and human language, enabling machines to understand text."
        )
    ]
    
    # Convert to chunks
    chunks = [
        Chunk(
            id=doc.id,
            text=doc.text,
            start_idx=0,
            end_idx=len(doc.text),
            doc_id=doc.id
        )
        for doc in documents
    ]
    
    # Initialize embedder
    embedder = DummyEmbedder(dim=384)
    
    print(f"Corpus: {len(documents)} documents")
    print("Query: 'neural networks for pattern recognition'")
    print()
    
    # Demo each vector retrieval technique
    vector_techniques = [
        "faiss_retriever",
        "dual_encoder",
        "colbert_retriever",
        "multi_vector_retriever"
    ]
    
    query = "neural networks for pattern recognition"
    
    for technique_name in vector_techniques:
        print(f"--- {technique_name.upper().replace('_', ' ')} ---")
        
        try:
            # Get technique class
            TechniqueClass = TechniqueRegistry.get(technique_name)
            
            # Initialize with appropriate parameters
            if technique_name == "faiss_retriever":
                technique = TechniqueClass(embedder=embedder, index_type="flat")
            elif technique_name == "dual_encoder":
                technique = TechniqueClass(
                    query_embedder=embedder,
                    doc_embedder=embedder,
                    similarity="cosine"
                )
            elif technique_name == "colbert_retriever":
                technique = TechniqueClass(embedder=embedder, max_tokens=32)
            elif technique_name == "multi_vector_retriever":
                technique = TechniqueClass(
                    embedder=embedder,
                    segment_size=50,
                    aggregation_method="max"
                )
            else:
                technique = TechniqueClass(embedder=embedder)
            
            # Add chunks and retrieve
            technique.add_chunks(chunks)
            result = technique.apply(query=query, top_k=3)
            
            if result.success:
                hits = result.payload["hits"]
                print(f"Retrieved {len(hits)} results:")
                
                for i, hit in enumerate(hits, 1):
                    chunk = next(c for c in chunks if c.id == hit.doc_id)
                    print(f"  {i}. {chunk.text[:80]}... (score: {hit.score:.3f})")
            else:
                print(f"Error: {result.error}")
                
        except Exception as e:
            print(f"Demo error: {e}")
        
        print()
    
    # Special demo for Multi-Query (requires LLM, so we show what it would do)
    print("--- MULTI QUERY RETRIEVER ---")
    print("Note: Multi-Query Retrieval expands the original query into multiple")
    print("variations and fuses results using reciprocal rank fusion.")
    print("Example expanded queries for 'neural networks for pattern recognition':")
    print("  1. 'deep learning models for pattern detection'")
    print("  2. 'artificial neural networks in pattern analysis'") 
    print("  3. 'machine learning algorithms for recognizing patterns'")
    print("Results would be fused from all query variations.")
    print()
    
    print("=" * 50)
    print("Demo complete! All techniques use the same interface:")
    print("1. Initialize technique with parameters")
    print("2. Add chunks with technique.add_chunks(chunks)")
    print("3. Query with technique.apply(query=query, top_k=k)")
    print()
    print("Try different techniques in your applications!")


if __name__ == "__main__":
    main()