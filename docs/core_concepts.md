# Core Concepts 🧠

Understanding RAGLib's core concepts is essential for effectively building and using RAG systems. This guide covers the fundamental principles and design patterns.

## The RAGTechnique Interface 🔧

The heart of RAGLib is the unified `RAGTechnique` interface. Every technique, regardless of complexity, implements the same basic contract:

```python
class RAGTechnique:
    def apply(self, *args, **kwargs) -> TechniqueResult:
        """Apply the technique to input data."""
        pass
```

### Why a Unified Interface?

- **Composability**: Techniques can be easily chained together
- **Interchangeability**: Swap techniques without changing pipeline code
- **Testability**: Consistent interface makes testing straightforward
- **Extensibility**: New techniques integrate seamlessly

### The apply() Method

The `apply()` method is intentionally flexible:

```python
# Different techniques, same interface
chunker_result = chunker.apply(documents)
retrieval_result = retriever.apply(query, documents=chunks)
generation_result = generator.apply(query, context=retrieved_docs)
```

## TechniqueResult: Consistent Outputs 📊

Every technique returns a `TechniqueResult` object:

```python
@dataclass
class TechniqueResult:
    success: bool
    payload: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Success Handling

```python
result = technique.apply(input_data)

if result.success:
    # Process successful result
    output_data = result.payload
    processing_time = result.metadata.get('processing_time', 0)
else:
    # Handle error
    print(f"Technique failed: {result.error}")
    # Access diagnostic information
    debug_info = result.metadata
```

### Payload Structure

The payload format depends on the technique category:

=== "Chunking"
    ```python
    {
        "chunks": [
            {"text": "chunk content", "metadata": {...}},
            {"text": "another chunk", "metadata": {...}}
        ],
        "chunk_count": 2
    }
    ```

=== "Retrieval"
    ```python
    {
        "documents": [
            {"content": "doc1", "score": 0.95, "metadata": {...}},
            {"content": "doc2", "score": 0.87, "metadata": {...}}
        ],
        "query": "original query"
    }
    ```

=== "Generation"
    ```python
    {
        "generated_text": "The answer is...",
        "context_used": ["doc1", "doc2"],
        "confidence_score": 0.92
    }
    ```

## Technique Categories 📁

RAGLib organizes techniques into five main categories:

### Chunking Techniques

**Purpose**: Break down large documents into manageable pieces

**Common Parameters**:
- `chunk_size`: Maximum size per chunk
- `overlap`: Overlap between consecutive chunks
- `separator`: Text used to split documents

**Examples**:
```python
from raglib.techniques import FixedSizeChunker, SemanticChunker

# Fixed-size chunking
fixed_chunker = FixedSizeChunker(chunk_size=512, overlap=50)

# Semantic-aware chunking  
semantic_chunker = SemanticChunker(similarity_threshold=0.8)
```

### Retrieval Techniques

**Purpose**: Find relevant information from document collections

**Common Parameters**:
- `top_k`: Number of results to return
- `similarity_threshold`: Minimum similarity score
- `query_expansion`: Whether to expand queries

**Examples**:
```python
from raglib.techniques import DenseRetriever, BM25

# Dense retrieval using embeddings
dense = DenseRetriever(top_k=10, similarity_threshold=0.7)

# Sparse retrieval using BM25
sparse = BM25(top_k=10)
```

### Reranking Techniques

**Purpose**: Improve the quality of initial retrieval results

**Common Parameters**:
- `model_name`: Reranking model to use
- `max_candidates`: Maximum documents to rerank
- `diversity_lambda`: Balance between relevance and diversity

**Examples**:
```python
from raglib.techniques import CrossEncoderReRanker, MMRReRanker

# Cross-encoder reranking
reranker = CrossEncoderReRanker(model_name="cross-encoder/ms-marco-TinyBERT-L-2")

# Maximal Marginal Relevance
mmr = MMRRerank(diversity_lambda=0.3)
```

### Generation Techniques

**Purpose**: Create human-readable responses from retrieved context

**Common Parameters**:
- `model_name`: Language model to use
- `temperature`: Randomness in generation
- `max_tokens`: Maximum response length

**Examples**:
```python
from raglib.techniques import HyDE

# HyDE for query expansion and improved retrieval
hyde = HyDE(
    llm_adapter=custom_llm,  # Your LLM adapter
    temperature=0.7
)
```

## Metadata and Configuration 📋

### TechniqueMeta

Each technique is described by a `TechniqueMeta` object:

```python
from raglib.schemas import TechniqueMeta

meta = TechniqueMeta(
    name="dense_retriever",
    category="retrieval",
    description="Dense retrieval using sentence embeddings",
    version="1.2.0",
    dependencies=["sentence-transformers", "faiss"],
    parameters={
        "top_k": {
            "type": "int", 
            "default": 10, 
            "description": "Number of results to return"
        },
        "similarity_threshold": {
            "type": "float",
            "default": 0.0,
            "description": "Minimum similarity score"
        }
    }
)
```

### Parameter Validation

RAGLib techniques can validate their parameters:

```python
class MyTechnique(RAGTechnique):
    def __init__(self, top_k: int = 10):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if top_k > 1000:
            raise ValueError("top_k too large, maximum is 1000")
        
        super().__init__(self.get_meta())
        self.top_k = top_k
```

## Registry System 🗂️

### Technique Discovery

The registry enables dynamic technique discovery:

```python
from raglib.registry import TechniqueRegistry

# List all available techniques
all_techniques = TechniqueRegistry.list_techniques()

# Find techniques by category
chunking_techniques = TechniqueRegistry.list_by_category("chunking")

# Get technique metadata
meta = TechniqueRegistry.get_meta("dense_retriever")
print(f"Description: {meta.description}")
print(f"Dependencies: {meta.dependencies}")
```

### Custom Registration

Register your own techniques:

```python
from raglib.registry import TechniqueRegistry
from raglib.core import RAGTechnique

class MyCustomTechnique(RAGTechnique):
    # Implementation here
    pass

# Register the technique
TechniqueRegistry.register("my_custom", MyCustomTechnique)

# Now it's available for use
MyTechnique = TechniqueRegistry.get("my_custom")
```

## Error Handling Patterns 🚨

### Graceful Degradation

```python
def safe_apply(technique, input_data, fallback_technique=None):
    result = technique.apply(input_data)
    
    if result.success:
        return result
    elif fallback_technique:
        print(f"Primary technique failed: {result.error}")
        return fallback_technique.apply(input_data)
    else:
        raise RuntimeError(f"Technique failed: {result.error}")
```

### Error Context

Always include helpful context in errors:

```python
class MyTechnique(RAGTechnique):
    def apply(self, documents):
        try:
            # Process documents
            processed = self._process(documents)
            return TechniqueResult(
                success=True,
                payload=processed,
                metadata={"processed_count": len(processed)}
            )
        except ValueError as e:
            return TechniqueResult(
                success=False,
                error=f"Invalid input format: {str(e)}",
                metadata={
                    "input_type": type(documents).__name__,
                    "input_length": len(documents) if hasattr(documents, '__len__') else None
                }
            )
```

## Design Principles 🎯

### 1. Composability

Techniques should work well together:

```python
# Easy to chain techniques
chunks = chunker.apply(documents)
if chunks.success:
    retrieved = retriever.apply(query, chunks.payload["chunks"])
    if retrieved.success:
        answer = generator.apply(query, retrieved.payload["documents"])
```

### 2. Configurability

Make techniques highly configurable:

```python
class ConfigurableTechnique(RAGTechnique):
    def __init__(self, **config):
        # Flexible configuration
        self.config = {
            "default_param1": "default_value",
            "default_param2": 42,
            **config  # Override with user config
        }
```

### 3. Observability

Include rich metadata for debugging:

```python
def apply(self, input_data):
    start_time = time.time()
    
    # Process data
    result_data = self._process(input_data)
    
    return TechniqueResult(
        success=True,
        payload=result_data,
        metadata={
            "processing_time": time.time() - start_time,
            "input_size": len(input_data),
            "output_size": len(result_data),
            "technique_version": self.meta.version
        }
    )
```

### 4. Testability

Design for easy testing:

```python
def test_technique():
    technique = MyTechnique(param1="test_value")
    
    # Test with known input
    result = technique.apply(["test document"])
    
    assert result.success
    assert "expected_output" in result.payload
    assert result.metadata["processing_time"] > 0
```

---

🎓 **Next Steps**: Learn about pipeline architecture to understand how to chain techniques together effectively.
