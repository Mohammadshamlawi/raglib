#!/usr/bin/env python3
"""
Sparse Retrieval Techniques Benchmark for RAGLib.

This script demonstrates and benchmarks all sparse retrieval techniques 
available in RAGLib: BM25, TF-IDF, Lexical Matching, SPLADE, and 
Lexical Transformer.

Usage:
    python sparse_retrieval_benchmark.py              # Run all sparse techniques
    python sparse_retrieval_benchmark.py --quick      # Quick mode
    python sparse_retrieval_benchmark.py --technique bm25  # Test single technique
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from raglib.registry import TechniqueRegistry
from raglib.schemas import Document


def create_benchmark_corpus() -> list[Document]:
    """Create a comprehensive corpus for testing sparse retrieval techniques."""
    return [
        Document(
            id="tech1",
            text="Machine learning algorithms learn patterns from large datasets. "
                 "Neural networks use backpropagation for training. Deep learning "
                 "models require substantial computational resources and data."
        ),
        Document(
            id="tech2", 
            text="Natural language processing enables computers to understand human "
                 "language. Tokenization splits text into words. Stemming reduces "
                 "words to their root forms for better matching."
        ),
        Document(
            id="tech3",
            text="Information retrieval systems find relevant documents from large "
                 "collections. BM25 and TF-IDF are classic sparse retrieval methods. "
                 "Modern approaches combine lexical and semantic matching."
        ),
        Document(
            id="sci1",
            text="Photosynthesis converts sunlight into chemical energy in plants. "
                 "Chlorophyll absorbs light energy. Carbon dioxide and water "
                 "combine to produce glucose and oxygen."
        ),
        Document(
            id="sci2",
            text="DNA contains genetic information in all living organisms. "
                 "Genes encode proteins that determine traits. Mutations can "
                 "alter genetic sequences and lead to evolution."
        ),
        Document(
            id="hist1",
            text="The Renaissance period marked a cultural rebirth in Europe. "
                 "Art, science, and literature flourished. Leonardo da Vinci "
                 "exemplified the Renaissance ideal of the universal genius."
        ),
        Document(
            id="hist2",
            text="World War II was a global conflict from 1939 to 1945. "
                 "The war involved major powers and resulted in significant "
                 "technological and social changes worldwide."
        ),
        Document(
            id="cook1",
            text="Pasta cooking requires boiling salted water. Different pasta "
                 "shapes have varying cooking times. Al dente texture is preferred "
                 "for optimal taste and nutritional value."
        ),
        Document(
            id="cook2",
            text="Bread baking involves mixing flour, water, yeast, and salt. "
                 "Kneading develops gluten structure. Rising time depends on "
                 "temperature and yeast activity."
        ),
        Document(
            id="travel1",
            text="Paris is famous for the Eiffel Tower and Louvre Museum. "
                 "French cuisine and culture attract millions of tourists. "
                 "The Seine River flows through the heart of the city."
        )
    ]


def run_sparse_technique_benchmark(technique_name: str, corpus: list[Document]) -> dict:
    """Run benchmark for a single sparse retrieval technique."""
    print(f"\n{'='*60}")
    print(f"Testing: {technique_name.upper()}")
    print(f"{'='*60}")
    
    try:
        # Get technique class
        TechniqueClass = TechniqueRegistry.get(technique_name)
        technique = TechniqueClass(docs=corpus)
        
        # Test queries covering different domains
        test_queries = [
            "machine learning algorithms",
            "natural language processing", 
            "information retrieval systems",
            "photosynthesis in plants",
            "DNA and genetics",
            "Renaissance art and culture",
            "World War II history",
            "pasta cooking methods",
            "bread baking process",
            "Paris tourism attractions"
        ]
        
        results = []
        total_time = 0
        
        for query in test_queries:
            start_time = time.time()
            result = technique.apply(query=query, top_k=3)
            query_time = time.time() - start_time
            total_time += query_time
            
            if result.success:
                hits = result.payload['hits']
                results.append({
                    'query': query,
                    'num_results': len(hits),
                    'top_score': hits[0].score if hits else 0.0,
                    'top_doc_id': hits[0].doc_id if hits else None,
                    'query_time': query_time
                })
                print(f"Query: '{query[:30]}...' -> {len(hits)} results "
                      f"(top: {hits[0].score:.3f} from {hits[0].doc_id})" 
                      if hits else f"Query: '{query[:30]}...' -> No results")
            else:
                print(f"Query: '{query[:30]}...' -> ERROR: {result.error}")
                results.append({
                    'query': query,
                    'error': result.error,
                    'query_time': query_time
                })
        
        avg_time = total_time / len(test_queries)
        successful_queries = sum(1 for r in results if 'error' not in r)
        
        print(f"\n📊 Summary for {technique_name}:")
        print(f"   Successful queries: {successful_queries}/{len(test_queries)}")
        print(f"   Average query time: {avg_time:.3f}s")
        print(f"   Total time: {total_time:.3f}s")
        
        return {
            'technique': technique_name,
            'successful_queries': successful_queries,
            'total_queries': len(test_queries),
            'avg_query_time': avg_time,
            'total_time': total_time,
            'query_results': results
        }
        
    except Exception as e:
        print(f"❌ Error testing {technique_name}: {e}")
        return {
            'technique': technique_name,
            'error': str(e)
        }


def compare_techniques(all_results: list[dict]):
    """Generate comparison summary of all techniques."""
    print(f"\n{'='*80}")
    print("SPARSE RETRIEVAL TECHNIQUES COMPARISON")
    print(f"{'='*80}")
    
    print(f"{'Technique':<20} {'Success Rate':<12} {'Avg Time (s)':<12} {'Status'}")
    print("-" * 60)
    
    for result in all_results:
        if 'error' in result:
            print(f"{result['technique']:<20} {'N/A':<12} {'N/A':<12} ❌ Error")
        else:
            success_rate = result['successful_queries'] / result['total_queries']
            print(f"{result['technique']:<20} {success_rate:.1%}:<12 "
                  f"{result['avg_query_time']:.3f}s:<12 ✅ OK")


def main():
    """Run the sparse retrieval benchmark."""
    parser = argparse.ArgumentParser(description="Sparse Retrieval Benchmark")
    parser.add_argument("--technique", help="Test specific technique only")
    parser.add_argument("--quick", action="store_true", 
                       help="Quick mode with fewer queries")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    print("🔍 RAGLib Sparse Retrieval Benchmark")
    print("=" * 50)
    
    # Create corpus
    corpus = create_benchmark_corpus()
    print(f"📚 Corpus: {len(corpus)} documents")
    
    # Determine which techniques to test
    sparse_techniques = TechniqueRegistry.find_by_category('sparse_retrieval')
    available_sparse = list(sparse_techniques.keys())
    
    if args.technique:
        if args.technique not in available_sparse:
            print(f"❌ Technique '{args.technique}' not found")
            print(f"Available: {', '.join(available_sparse)}")
            return 1
        techniques_to_test = [args.technique]
    else:
        techniques_to_test = sorted(available_sparse)
    
    print(f"🧪 Testing: {', '.join(techniques_to_test)}")
    
    # Run benchmarks
    all_results = []
    for technique in techniques_to_test:
        result = run_sparse_technique_benchmark(technique, corpus)
        all_results.append(result)
    
    # Generate comparison
    compare_techniques(all_results)
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                'benchmark_type': 'sparse_retrieval',
                'corpus_size': len(corpus),
                'techniques_tested': techniques_to_test,
                'results': all_results
            }, f, indent=2)
        print(f"💾 Results saved to: {output_path}")
    
    successful_techniques = sum(1 for r in all_results if 'error' not in r)
    print(f"\n✅ Completed {successful_techniques}/{len(techniques_to_test)} "
          f"technique benchmarks")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())