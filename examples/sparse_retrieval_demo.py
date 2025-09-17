#!/usr/bin/env python3
"""
Sparse Retrieval Quick Start Example for RAGLib.

This script demonstrates how to use all sparse retrieval techniques
available in RAGLib: BM25, TF-IDF, Lexical Matching, SPLADE, and 
Lexical Transformer.
"""

import sys
from pathlib import Path

# Add raglib to path for development
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from raglib.registry import TechniqueRegistry
    from raglib.schemas import Document
except ImportError as e:
    print(f"Failed to import raglib: {e}")
    print("Make sure raglib is installed: pip install -e .")
    sys.exit(1)


def create_sample_documents():
    """Create a sample document collection for testing."""
    return [
        Document(
            id="ml1",
            text="Machine learning algorithms automatically learn patterns from data. "
                 "Neural networks use backpropagation for training and can handle "
                 "complex non-linear relationships in large datasets."
        ),
        Document(
            id="nlp1", 
            text="Natural language processing enables computers to understand and "
                 "generate human language. Tokenization, stemming, and embedding "
                 "are fundamental preprocessing steps."
        ),
        Document(
            id="ir1",
            text="Information retrieval systems find relevant documents from large "
                 "collections. Classical methods include TF-IDF and BM25, while "
                 "modern approaches use dense vector representations."
        ),
        Document(
            id="ai1",
            text="Artificial intelligence encompasses machine learning, natural "
                 "language processing, computer vision, and robotics. Deep learning "
                 "has revolutionized many AI applications."
        ),
        Document(
            id="data1",
            text="Data science combines statistics, programming, and domain expertise "
                 "to extract insights from data. Data cleaning and feature engineering "
                 "are crucial preprocessing steps."
        )
    ]


def demonstrate_technique(technique_name: str, documents: list[Document], query: str):
    """Demonstrate a single sparse retrieval technique."""
    print(f"\n{'='*60}")
    print(f"🔍 {technique_name.upper()} Demonstration")
    print(f"{'='*60}")
    
    try:
        # Get technique class from registry
        TechniqueClass = TechniqueRegistry.get(technique_name)
        
        # Initialize technique with documents
        technique = TechniqueClass(docs=documents)
        
        # Apply technique to query
        result = technique.apply(query=query, top_k=3)
        
        if result.success:
            hits = result.payload['hits']
            print(f"Query: '{query}'")
            print(f"Found {len(hits)} relevant documents:")
            
            for i, hit in enumerate(hits, 1):
                doc = next(d for d in documents if d.id == hit.doc_id)
                print(f"\n{i}. Document '{hit.doc_id}' (Score: {hit.score:.4f})")
                print(f"   Text: {doc.text[:100]}...")
            
            # Show technique metadata
            if hasattr(result, 'meta') and result.meta:
                print(f"\nTechnique metadata:")
                for key, value in result.meta.items():
                    if key != 'hits':  # Don't print hits again
                        print(f"   {key}: {value}")
        else:
            print(f"❌ Error: {result.error}")
            
    except Exception as e:
        print(f"❌ Failed to test {technique_name}: {e}")


def main():
    """Run the sparse retrieval demonstration."""
    print("🎯 RAGLib Sparse Retrieval Quick Start")
    print("=" * 50)
    
    # Create sample documents
    documents = create_sample_documents()
    print(f"📚 Created {len(documents)} sample documents")
    
    # Test query
    query = "machine learning algorithms"
    print(f"🔎 Test query: '{query}'")
    
    # Get all sparse retrieval techniques
    sparse_techniques = TechniqueRegistry.find_by_category('sparse_retrieval')
    print(f"🛠️  Available sparse techniques: {', '.join(sparse_techniques.keys())}")
    
    # Demonstrate each technique
    for technique_name in sorted(sparse_techniques.keys()):
        demonstrate_technique(technique_name, documents, query)
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("📊 TECHNIQUE COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Technique':<20} {'Top Score':<12} {'Results':<10} {'Best Match'}")
    print("-" * 70)
    
    for technique_name in sorted(sparse_techniques.keys()):
        try:
            TechniqueClass = sparse_techniques[technique_name]
            technique = TechniqueClass(docs=documents)
            result = technique.apply(query=query, top_k=3)
            
            if result.success and result.payload['hits']:
                hits = result.payload['hits']
                top_score = hits[0].score
                num_results = len(hits)
                best_doc = hits[0].doc_id
                print(f"{technique_name:<20} {top_score:<12.4f} {num_results:<10} {best_doc}")
            else:
                print(f"{technique_name:<20} {'N/A':<12} {'0':<10} {'None'}")
        except Exception:
            print(f"{technique_name:<20} {'ERROR':<12} {'N/A':<10} {'N/A'}")
    
    print(f"\n✅ Sparse retrieval demonstration complete!")
    print("\n💡 Tips for choosing techniques:")
    print("   • BM25: Best general-purpose sparse retrieval, good baseline")
    print("   • TF-IDF: Classic approach, good for statistical analysis")
    print("   • Lexical Matcher: Flexible matching modes, good for exact matching")
    print("   • SPLADE: Hybrid approach with term expansion for better recall")
    print("   • Lexical Transformer: Advanced technique with attention weighting")


if __name__ == "__main__":
    main()