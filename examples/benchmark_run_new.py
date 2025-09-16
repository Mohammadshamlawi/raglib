#!/usr/bin/env python3
"""
Advanced benchmark runner for RAGLib techniques.

This script demonstrates how to:
1. Use the BenchmarkHarness for evaluation
2. Run benchmarks on QA datasets 
3. Compare multiple techniques
4. Generate detailed results and reports

Usage:
    python benchmark_run.py --quick                    # Quick mode with limited examples
    python benchmark_run.py --techniques bm25,dummy    # Test specific techniques
    python benchmark_run.py --output results/bench.json # Custom output location
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from raglib.benchmark import BenchmarkHarness, load_qa_dataset
from raglib.registry import TechniqueRegistry
from raglib.schemas import Document


def create_sample_corpus() -> list[Document]:
    """Create a sample document corpus for benchmarking."""
    docs = [
        Document(
            id="1", 
            text="France is a country in Western Europe. Paris is the capital and largest city of France. The Eiffel Tower is located in Paris and is one of the most famous landmarks in the world."
        ),
        Document(
            id="2",
            text="Tea is a popular beverage made from the leaves of the tea plant. To prepare tea, you need to boil water, add tea leaves or tea bags, and let it steep for several minutes. Different types of tea require different steeping times."
        ),
        Document(
            id="3", 
            text="Photosynthesis is the process by which plants convert sunlight into energy. During photosynthesis, plants use carbon dioxide from the air and water from the soil to create glucose and oxygen. Chlorophyll in plant leaves captures the sunlight needed for this process."
        ),
        Document(
            id="4",
            text="Coffee is another popular beverage made from coffee beans. The beans are roasted and ground before brewing. Coffee contains caffeine which provides energy and alertness. Many people drink coffee in the morning."
        ),
        Document(
            id="5",
            text="Python is a high-level programming language known for its simplicity and readability. It is widely used for web development, data analysis, artificial intelligence, and automation. Python has a large ecosystem of libraries and frameworks."
        ),
        Document(
            id="6",
            text="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from data without being explicitly programmed. Common machine learning techniques include supervised learning, unsupervised learning, and reinforcement learning."
        )
    ]
    return docs


def run_single_benchmark(
    technique_name: str, 
    corpus: list[Document],
    dataset_path: Path,
    output_dir: Path,
    quick_mode: bool = False
) -> dict:
    """Run benchmark for a single technique."""
    print(f"\n{'='*50}")
    print(f"Benchmarking: {technique_name}")
    print(f"{'='*50}")
    
    try:
        # Get technique from registry
        technique = TechniqueRegistry.get(technique_name)
        
        # Set up benchmark harness
        harness = BenchmarkHarness(quick_mode=quick_mode, verbose=True)
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"benchmark_{technique_name}_{timestamp}.json"
        
        # Run benchmark
        results = harness.run_benchmark(
            technique=technique,
            dataset_path=dataset_path, 
            corpus_docs=corpus,
            output_path=output_file,
            top_k=3
        )
        
        print(f"\n📊 Results for {technique_name}:")
        print(f"   Exact Match: {results['metrics']['exact_match']:.3f}")
        print(f"   F1 Score:    {results['metrics']['f1']:.3f}")
        print(f"   Overlap:     {results['metrics']['overlap']:.3f}")
        print(f"   Runtime:     {results['runtime_seconds']:.2f}s")
        
        return results
        
    except Exception as e:
        print(f"❌ Error benchmarking {technique_name}: {e}")
        return None


def compare_techniques(results_list: list) -> None:
    """Generate comparison summary of multiple technique results."""
    if not results_list:
        print("No results to compare.")
        return
    
    print(f"\n{'='*60}")
    print("BENCHMARK COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    # Table header
    print(f"{'Technique':<20} {'EM':<8} {'F1':<8} {'Overlap':<8} {'Time(s)':<8}")
    print("-" * 60)
    
    # Sort by F1 score (descending)
    valid_results = [r for r in results_list if r is not None]
    valid_results.sort(key=lambda x: x['metrics']['f1'], reverse=True)
    
    for result in valid_results:
        name = result['technique']
        em = result['metrics']['exact_match']
        f1 = result['metrics']['f1'] 
        overlap = result['metrics']['overlap']
        time_s = result['runtime_seconds']
        
        print(f"{name:<20} {em:<8.3f} {f1:<8.3f} {overlap:<8.3f} {time_s:<8.2f}")
    
    if valid_results:
        print(f"\n🏆 Best performing technique: {valid_results[0]['technique']} (F1: {valid_results[0]['metrics']['f1']:.3f})")


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="Run RAGLib technique benchmarks")
    parser.add_argument("--quick", action="store_true", 
                       help="Quick mode - run on limited examples")
    parser.add_argument("--techniques", type=str, 
                       help="Comma-separated list of techniques to test (default: all available)")
    parser.add_argument("--dataset", type=str, 
                       help="Path to QA dataset (default: tests/data/tiny_qa_dataset.jsonl)")
    parser.add_argument("--output", type=str,
                       help="Output directory for results (default: results/)")
    parser.add_argument("--list-techniques", action="store_true",
                       help="List all available techniques and exit")
    
    args = parser.parse_args()
    
    # List available techniques
    available_techniques = TechniqueRegistry.list()
    
    if args.list_techniques:
        print("Available techniques:")
        for technique in sorted(available_techniques):
            print(f"  - {technique}")
        return 0
    
    print("🚀 RAGLib Benchmark Runner")
    print("=" * 40)
    
    # Set up paths
    project_root = Path(__file__).parent.parent
    dataset_path = Path(args.dataset) if args.dataset else project_root / "tests" / "data" / "tiny_qa_dataset.jsonl"
    output_dir = Path(args.output) if args.output else project_root / "results"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Check dataset exists
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please provide a valid dataset path with --dataset")
        return 1
    
    # Determine techniques to test
    if args.techniques:
        techniques_to_test = [t.strip() for t in args.techniques.split(",")]
        # Validate techniques exist
        for technique in techniques_to_test:
            if technique not in available_techniques:
                print(f"❌ Unknown technique: {technique}")
                print(f"Available: {', '.join(sorted(available_techniques))}")
                return 1
    else:
        # Test a subset of reliable techniques for demo
        techniques_to_test = [
            t for t in ["bm25", "dense_retriever"] 
            if t in available_techniques
        ]
    
    if not techniques_to_test:
        print("❌ No techniques available to test")
        return 1
    
    print(f"📋 Testing techniques: {', '.join(techniques_to_test)}")
    print(f"📊 Dataset: {dataset_path}")
    print(f"📁 Output: {output_dir}")
    print(f"⚡ Quick mode: {args.quick}")
    
    # Create corpus
    corpus = create_sample_corpus()
    print(f"📚 Corpus size: {len(corpus)} documents")
    
    # Run benchmarks
    all_results = []
    
    for technique_name in techniques_to_test:
        result = run_single_benchmark(
            technique_name=technique_name,
            corpus=corpus,
            dataset_path=dataset_path, 
            output_dir=output_dir,
            quick_mode=args.quick
        )
        all_results.append(result)
    
    # Generate comparison
    compare_techniques(all_results)
    
    # Summary
    successful_runs = sum(1 for r in all_results if r is not None)
    print(f"\n✅ Completed {successful_runs}/{len(techniques_to_test)} benchmarks")
    print(f"📁 Results saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
