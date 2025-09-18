# Benchmarking RAGLib Techniques

RAGLib includes a comprehensive benchmarking system that allows you to evaluate and compare different RAG techniques on question-answering datasets. This guide shows how to use the benchmarking infrastructure to measure technique performance.

## Overview

The benchmarking system provides:

- **BenchmarkHarness**: Core evaluation engine that runs techniques on QA datasets
- **Dataset loaders**: Support for JSONL and CSV formats  
- **Metrics computation**: Automatic calculation of exact match, F1, and overlap scores
- **Result output**: JSON format for analysis and comparison
- **CLI tools**: Ready-to-use scripts for running benchmarks

## Quick Start

### Running a Basic Benchmark

```python
from raglib.benchmark import BenchmarkHarness
from raglib.registry import TechniqueRegistry
from raglib.schemas import Document

# Create your document corpus
corpus = [
    Document(id="1", text="Paris is the capital of France"),
    Document(id="2", text="To make tea, boil water and steep leaves"),
    Document(id="3", text="Photosynthesis converts sunlight to energy")
]

# Get a technique to test
technique = TechniqueRegistry.get("bm25")

# Run benchmark
harness = BenchmarkHarness(quick_mode=True, verbose=True)
results = harness.run_benchmark(
    technique=technique,
    dataset_path="tests/data/tiny_qa_dataset.jsonl",
    corpus_docs=corpus,
    output_path="results/benchmark_results.json"
)

print(f"F1 Score: {results['metrics']['f1']:.3f}")
```

### Using the CLI Tool

```bash
# Run quick benchmark on default techniques
python examples/benchmark_run.py --quick

# Test specific techniques
python examples/benchmark_run.py --techniques bm25,dense_retriever

# Use custom dataset and output location
python examples/benchmark_run.py \
    --dataset my_qa_dataset.jsonl \
    --output results/my_benchmark

# List all available techniques
python examples/benchmark_run.py --list-techniques
```

## Dataset Format

### JSONL Format (Recommended)

Each line contains a JSON object with question-answer pairs:

```jsonl
{"question": "What is the capital of France?", "answer": "Paris", "context": "France is in Europe..."}
{"question": "How do you make tea?", "answer": "Boil water and steep leaves", "context": "Tea preparation..."}
{"question": "What is photosynthesis?", "answer": "Process converting sunlight to energy"}
```

Required fields:
- `question`: The question to ask
- `answer`: Expected answer for evaluation

Optional fields:
- `context`: Additional context information
- Any other fields will be stored in metadata

### CSV Format

CSV files with headers for question, answer, and optional fields:

```csv
question,answer,context
"What is the capital of France?","Paris","France is in Europe..."
"How do you make tea?","Boil water and steep leaves","Tea preparation involves..."
```

## Sparse Retrieval Benchmarking

RAGLib includes a comprehensive set of sparse retrieval techniques that can be benchmarked and compared. These techniques use lexical matching and term-based scoring rather than dense embeddings.

### Available Sparse Techniques

| Technique | Description | Use Case |
|-----------|-------------|----------|
| **BM25** | Classic probabilistic ranking function | Baseline sparse retrieval, exact term matching |
| **TF-IDF** | Term frequency-inverse document frequency | Statistical term weighting, cosine similarity |
| **Lexical Matcher** | Configurable lexical matching modes | Flexible text matching (exact, substring, token overlap) |
| **SPLADE** | Sparse lexical and dense expansion | Hybrid approach with term expansion |
| **Lexical Transformer** | Transformer-aware lexical retrieval | Attention weighting and positional encoding |

### Running Sparse Retrieval Benchmark

Use the dedicated sparse retrieval benchmark script:

```bash
# Benchmark all sparse retrieval techniques
python examples/sparse_retrieval_benchmark.py

# Test specific technique
python examples/sparse_retrieval_benchmark.py --technique bm25

# Quick mode with fewer test queries  
python examples/sparse_retrieval_benchmark.py --quick

# Save results to file
python examples/sparse_retrieval_benchmark.py --output results/sparse_benchmark.json
```

**Example Output:**
```
Sparse Retrieval Comparison:
BM25:
  - Results: 5, Top score: 2.845
  - Best match: tech3
TF-IDF:
  - Results: 5, Top score: 0.712
  - Best match: tech3
```

## Vector Retrieval Benchmarking

RAGLib includes advanced vector/dense retrieval techniques that use semantic embeddings for finding relevant information. These techniques provide more nuanced understanding of query-document relationships.

### Available Vector Techniques

| Technique | Description | Use Case |
|-----------|-------------|----------|
| **FAISS Retriever** | High-performance vector search with FAISS | Large-scale similarity search, production systems |
| **Dual Encoder** | Asymmetric query/document encoding | Different encoders for queries vs documents |
| **ColBERT Retriever** | Token-level late interaction | Fine-grained matching, high precision |
| **Multi-Query Retriever** | Query expansion with result fusion | Improved recall through query diversity |
| **Multi-Vector Retriever** | Document segmentation representation | Long documents, granular matching |
| **Dense Retriever** | Basic embedding-based retrieval | Baseline vector retrieval |

### Running Vector Retrieval Benchmark

Use the dedicated vector retrieval benchmark script:

```bash
# Benchmark all vector retrieval techniques
python examples/vector_retrieval_benchmark.py

# Test specific technique
python examples/vector_retrieval_benchmark.py --technique faiss_retriever

# Quick mode with fewer test queries
python examples/vector_retrieval_benchmark.py --quick

# Save results to custom file
python examples/vector_retrieval_benchmark.py --output my_results.json
```

**Example Output:**
```
Performance Comparison:
Technique            Precision  Recall   F1       NDCG     Avg Time
----------------------------------------------------------------------
faiss_retriever      0.667      0.600    0.632    0.745    0.012s
dual_encoder         0.733      0.667    0.698    0.789    0.015s
colbert_retriever    0.800      0.700    0.747    0.823    0.028s
multi_vector_retriever 0.667    0.633    0.649    0.756    0.021s
```

### Enhanced Benchmark Comparison

Compare all retrieval types (sparse and vector) side-by-side:

```bash
# Compare all retrieval techniques
python examples/enhanced_benchmark.py

# Quick comparison mode
python examples/enhanced_benchmark.py --quick
```

### Technique Comparison Example

```python
from raglib.registry import TechniqueRegistry
from raglib.schemas import Document

# Create test corpus
corpus = [
    Document(id="1", text="Machine learning algorithms learn from data"),
    Document(id="2", text="Natural language processing handles text"),
    Document(id="3", text="Information retrieval finds relevant documents")
]

# Test multiple sparse techniques
sparse_techniques = TechniqueRegistry.find_by_category('sparse_retrieval')
query = "machine learning"

for name, TechniqueClass in sparse_techniques.items():
    technique = TechniqueClass(docs=corpus)
    result = technique.apply(query=query, top_k=3)
    
    if result.success:
        hits = result.payload['hits']
        print(f"{name}: {len(hits)} results, top score: {hits[0].score:.3f}")
```

## Metrics

The benchmark system computes three main metrics:

### Exact Match (EM)
Binary score indicating whether the expected answer appears exactly in the retrieved text (case-insensitive).

```python
# Example: Expected="Paris", Retrieved="Paris is the capital" → EM=1.0
# Example: Expected="Paris", Retrieved="London is the capital" → EM=0.0
```

### F1 Score
Token-level F1 score measuring precision and recall of word overlap between expected and retrieved text.

```python
# Example: Expected="machine learning", Retrieved="machine learning algorithms" 
# Precision=2/3, Recall=2/2 → F1=0.8
```

### Overlap Score
Ratio of expected answer tokens found in retrieved text.

```python
# Example: Expected="deep neural networks", Retrieved="neural networks are powerful"
# Overlap = 2/3 = 0.67 (found "neural networks" out of "deep neural networks")
```

## BenchmarkHarness API

### Initialization

```python
harness = BenchmarkHarness(
    quick_mode=False,    # If True, limit to first 3 examples for fast testing
    verbose=True         # Print progress information
)
```

### Running Benchmarks

```python
results = harness.run_benchmark(
    technique=technique,           # RAGTechnique instance to evaluate
    dataset_path="data.jsonl",    # Path to QA dataset
    corpus_docs=documents,        # List of Document objects to search
    output_path="results.json",   # Optional: save results to file
    top_k=5                       # Number of documents to retrieve
)
```

### Result Structure

The benchmark returns a comprehensive result dictionary:

```python
{
    "timestamp": "2024-01-15T10:30:00",
    "technique": "BM25Simple", 
    "dataset_path": "/path/to/dataset.jsonl",
    "corpus_size": 100,
    "num_questions": 25,
    "quick_mode": false,
    "top_k": 5,
    "runtime_seconds": 2.34,
    "metrics": {
        "exact_match": 0.72,
        "f1": 0.85,
        "overlap": 0.78
    },
    "individual_results": [
        {
            "question": "What is the capital of France?",
            "expected_answer": "Paris", 
            "retrieved_count": 3,
            "metrics": {"exact_match": 1.0, "f1": 0.9, "overlap": 0.8},
            "metadata": {}
        }
        // ... more individual results
    ]
}
```

## Advanced Usage

### Custom Evaluation Metrics

You can extend the BenchmarkHarness to compute custom metrics:

```python
class CustomBenchmarkHarness(BenchmarkHarness):
    def _compute_metrics(self, qa_item, result):
        # Get standard metrics
        metrics = super()._compute_metrics(qa_item, result)
        
        # Add custom metrics
        metrics["custom_score"] = self._compute_custom_score(qa_item, result)
        
        return metrics
    
    def _compute_custom_score(self, qa_item, result):
        # Your custom scoring logic
        return 0.85
```

### Batch Evaluation

Compare multiple techniques systematically:

```python
techniques_to_test = ["bm25", "dense_retriever", "hyde"]
results_list = []

for technique_name in techniques_to_test:
    technique = TechniqueRegistry.get(technique_name)
    result = harness.run_benchmark(
        technique=technique,
        dataset_path=dataset_path,
        corpus_docs=corpus
    )
    results_list.append(result)

# Find best performing technique
best_technique = max(results_list, key=lambda r: r['metrics']['f1'])
print(f"Best technique: {best_technique['technique']} (F1: {best_technique['metrics']['f1']:.3f})")
```

### Error Handling

The benchmark system handles errors gracefully:

```python
# Individual question errors are captured in results
result = harness.run_benchmark(technique, dataset_path, corpus)

for individual_result in result["individual_results"]:
    if "error" in individual_result:
        print(f"Error processing question: {individual_result['error']}")
```

## Performance Considerations

### Quick Mode
For rapid iteration and testing, use quick mode:

```python
# Only processes first 3 examples
harness = BenchmarkHarness(quick_mode=True)
```

### Large Datasets
For large datasets, consider:

- Processing in batches
- Using multiprocessing for technique comparisons  
- Sampling representative subsets
- Caching intermediate results

### Memory Management
Monitor memory usage when working with large corpora:

```python
# Process documents in chunks for very large corpora
chunk_size = 1000
for i in range(0, len(large_corpus), chunk_size):
    chunk = large_corpus[i:i+chunk_size]
    # Process chunk...
```

## Integration with Development Workflow

### Automated Testing

Include benchmark runs in your CI/CD pipeline:

```yaml
# .github/workflows/ci.yml
- name: Run benchmark smoke test
  run: python examples/benchmark_run.py --quick
  env:
    RUN_BENCH: "true"
```

### Regression Detection

Track performance over time:

```python
# Compare against baseline results
baseline_f1 = 0.75
current_f1 = results['metrics']['f1']

if current_f1 < baseline_f1 - 0.05:
    print("⚠️  Performance regression detected!")
```

### A/B Testing New Techniques

```python
# Compare new technique against current best
baseline_technique = TechniqueRegistry.get("current_best")
new_technique = TechniqueRegistry.get("experimental")

baseline_results = harness.run_benchmark(baseline_technique, dataset, corpus)
new_results = harness.run_benchmark(new_technique, dataset, corpus)

improvement = new_results['metrics']['f1'] - baseline_results['metrics']['f1']
print(f"F1 improvement: {improvement:+.3f}")
```

## Best Practices

1. **Use representative datasets**: Ensure your QA dataset reflects real use cases
2. **Multiple metrics**: Don't rely on a single metric; consider all three
3. **Cross-validation**: Test on multiple datasets for robust evaluation  
4. **Statistical significance**: Run multiple trials for statistical validity
5. **Document assumptions**: Note any preprocessing or filtering applied
6. **Version control results**: Track benchmark results over time
7. **Reproducible runs**: Set random seeds and document environment

## Troubleshooting

### Common Issues

**Dataset loading errors**:
```bash
# Check file format and encoding
python -c "from raglib.benchmark import load_qa_dataset; print(load_qa_dataset('data.jsonl')[:2])"
```

**Memory errors with large corpora**:
```python
# Use generators or chunking for large datasets
def chunk_corpus(corpus, chunk_size=1000):
    for i in range(0, len(corpus), chunk_size):
        yield corpus[i:i+chunk_size]
```

**Technique not found**:
```python
# List available techniques
print("Available:", TechniqueRegistry.list())
```

**Low scores across all techniques**:
- Check dataset quality and answer format
- Verify corpus contains relevant information
- Review technique parameters and configuration

For more detailed examples, see `examples/benchmark_run.py` and the test files in `tests/test_benchmark.py`.
