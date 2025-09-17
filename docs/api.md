# API Reference 📖

This page provides comprehensive API documentation for RAGLib's core components and interfaces.

## Core Components 🏗️

### RAGTechnique

Base class for all RAG techniques in RAGLib. Every technique must inherit from this class and implement the `apply` method.

```python
from raglib.core import RAGTechnique, TechniqueResult
from raglib.schemas import TechniqueMeta

class RAGTechnique:
    """Base class for all RAG techniques."""
    
    def __init__(self, meta: TechniqueMeta):
        """Initialize technique with metadata."""
        self.meta = meta
    
    def apply(self, *args, **kwargs) -> TechniqueResult:
        """Apply the technique to input data.
        
        This method must be implemented by all subclasses.
        
        Returns:
            TechniqueResult: Result object with success status and data
        """
        raise NotImplementedError("Subclasses must implement apply()")
```

**Example Usage:**

```python
class MyCustomTechnique(RAGTechnique):
    def __init__(self, param1: str, param2: int = 10):
        super().__init__(TechniqueMeta(
            name="my_custom_technique",
            category="retrieval",
            description="My custom retrieval technique"
        ))
        self.param1 = param1
        self.param2 = param2
    
    def apply(self, query: str, mode: str = "retrieve") -> TechniqueResult:
        # Implementation here
        return TechniqueResult(
            success=True,
            payload={"results": ["result1", "result2"]},
            metadata={"query": query, "mode": mode}
        )
```

### TechniqueResult

Standard result object returned by all technique `apply` methods.

```python
@dataclass
class TechniqueResult:
    """Result object returned by technique apply methods."""
    success: bool
    payload: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Example:**

```python
# Successful result
result = TechniqueResult(
    success=True,
    payload={"chunks": ["chunk1", "chunk2"]},
    metadata={"processing_time": 0.5}
)

# Error result
result = TechniqueResult(
    success=False,
    error="Invalid input format",
    metadata={"attempted_input": input_data}
)
```

### TechniqueMeta

Metadata describing a technique's properties and capabilities.

```python
@dataclass
class TechniqueMeta:
    """Metadata for RAG techniques."""
    name: str
    category: str
    description: str
    version: str = "1.0.0"
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
```

**Example:**

```python
from raglib.schemas import TechniqueMeta

meta = TechniqueMeta(
    name="dense_retriever",
    category="retrieval", 
    description="Dense retrieval using embeddings",
    version="1.2.0",
    dependencies=["faiss", "transformers"],
    parameters={
        "top_k": {"type": "int", "default": 10},
        "similarity_threshold": {"type": "float", "default": 0.7}
    }
## Registry System 🗃️

### TechniqueRegistry

::: raglib.registry.TechniqueRegistry
    options:
      show_root_heading: true
      show_source: true
      members_order: source

Central registry for technique discovery and management.

**Example Usage:**

```python
from raglib.registry import TechniqueRegistry

# List all available techniques
techniques = TechniqueRegistry.list_techniques()

# Get a specific technique
DenseRetriever = TechniqueRegistry.get("dense_retriever")

# Register a new technique
TechniqueRegistry.register("my_technique", MyCustomTechnique)

# Find techniques by category
chunking_techniques = TechniqueRegistry.list_by_category("chunking")

# Check if technique exists
if TechniqueRegistry.exists("some_technique"):
    technique_class = TechniqueRegistry.get("some_technique")
```

## Pipeline System 🔄

### Pipeline

::: raglib.pipelines.Pipeline
    options:
      show_root_heading: true
      show_source: true
      members_order: source

Orchestration system for chaining multiple techniques together.

**Example Usage:**

```python
from raglib.pipelines import Pipeline
from raglib.techniques import FixedSizeChunker, DenseRetriever

# Create pipeline
pipeline = Pipeline([
    ("chunker", FixedSizeChunker(chunk_size=512)),
    ("retriever", DenseRetriever(top_k=10))
])

# Execute pipeline
result = pipeline.run(
    documents=["Document 1", "Document 2"],
    query="What is RAG?"
)

# Access intermediate results
chunking_result = pipeline.get_step_result("chunker")
retrieval_result = pipeline.get_step_result("retriever")
```

Result object returned by all technique applications.

```python
@dataclass
class TechniqueResult:
    success: bool
    payload: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Attributes:**
- `success`: Whether the operation succeeded
- `payload`: Main result data
- `error`: Error message if operation failed
- `metadata`: Additional metadata about the operation

## Registry

### TechniqueRegistry

Central registry for technique discovery and access.

```python
class TechniqueRegistry:
    """Registry for RAGTechnique classes."""
    
    @classmethod
    def register(cls, technique_class: Type[RAGTechnique]) -> Type[RAGTechnique]:
        """Register a technique class."""
        pass
    
    @classmethod
    def get(cls, name: str) -> Type[RAGTechnique]:
        """Get a technique class by name."""
        pass
    
    @classmethod
    def list(cls) -> Dict[str, Type[RAGTechnique]]:
        """List all registered techniques."""
        pass
    
    @classmethod
    def find_by_category(cls, category: str) -> Dict[str, Type[RAGTechnique]]:
        """Find techniques by category."""
        pass
```

**Methods:**
- `register(technique_class)`: Class decorator to register techniques
- `get(name)`: Retrieve technique class by name
- `list()`: Get dictionary of all registered techniques
- `find_by_category(category)`: Filter techniques by category

## Adapters

### Base Adapter Classes

#### Embedder

Base class for text embedding adapters.

```python
class Embedder:
    """Base class for text embedders."""
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts into vectors."""
        raise NotImplementedError
    
    def embed_single(self, text: str) -> List[float]:
        """Embed a single text into a vector."""
        raise NotImplementedError
    
    @property
    def dimension(self) -> int:
        """Vector dimension."""
        raise NotImplementedError
```

#### VectorStore

Base class for vector storage adapters.

```python
class VectorStore:
    """Base class for vector stores."""
    
    def add_vectors(self, vectors: List[List[float]], 
                   metadata: List[Dict] = None) -> List[str]:
        """Add vectors to the store."""
        raise NotImplementedError
    
    def search(self, query_vector: List[float], 
              top_k: int = 10) -> List[VectorSearchResult]:
        """Search for similar vectors."""
        raise NotImplementedError
    
    def delete(self, ids: List[str]) -> bool:
        """Delete vectors by ID."""
        raise NotImplementedError
```

### Concrete Adapters

#### DummyEmbedder

Fallback embedder that generates random vectors.

```python
class DummyEmbedder(Embedder):
    def __init__(self, dimension: int = 384, seed: int = 42):
        """Initialize with vector dimension and random seed."""
        pass
```

**Parameters:**
- `dimension`: Vector dimension (default: 384)
- `seed`: Random seed for reproducibility (default: 42)

#### InMemoryVectorStore

In-memory vector store for development and testing.

```python
class InMemoryVectorStore(VectorStore):
    def __init__(self, similarity_metric: str = "cosine"):
        """Initialize with similarity metric."""
        pass
```

**Parameters:**
- `similarity_metric`: Similarity metric ("cosine", "euclidean", "dot_product")

## Techniques

### Chunking Techniques

#### FixedSizeChunker

Splits text into fixed-size chunks with optional overlap.

```python
@TechniqueRegistry.register
class FixedSizeChunker(RAGTechnique):
    def __init__(self, chunk_size: int = 512, overlap: int = 0, 
                 separator: str = " "):
        pass
    
    def apply(self, documents: List[str]) -> TechniqueResult:
        """Split documents into chunks."""
        pass
```

**Parameters:**
- `chunk_size`: Maximum chunk size in characters
- `overlap`: Overlap between consecutive chunks
- `separator`: Text separator for splitting

**Returns:**
- `TechniqueResult` with payload containing `chunks` list

#### SemanticChunker

Splits text based on semantic similarity.

```python
@TechniqueRegistry.register
class SemanticChunker(RAGTechnique):
    def __init__(self, embedder: Embedder, 
                 similarity_threshold: float = 0.8):
        pass
```

**Parameters:**
- `embedder`: Embedder adapter for computing similarities
- `similarity_threshold`: Threshold for chunk boundaries

### Retrieval Techniques

#### DenseRetriever

Retrieves documents using dense vector similarity.

```python
@TechniqueRegistry.register
class DenseRetriever(RAGTechnique):
    def __init__(self, embedder: Embedder, vectorstore: VectorStore):
        pass
    
    def apply(self, query_or_docs, mode: str = "retrieve", 
             top_k: int = 5) -> TechniqueResult:
        """Index documents or retrieve relevant chunks."""
        pass
```

**Parameters:**
- `embedder`: Embedder adapter
- `vectorstore`: Vector store adapter

**Modes:**
- `index`: Index documents for later retrieval
- `retrieve`: Retrieve relevant documents for a query

#### BM25

BM25 retrieval technique with in-memory indexing.

```python
@TechniqueRegistry.register
class BM25(RAGTechnique):
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        pass
```

**Parameters:**
- `k1`: BM25 k1 parameter (term frequency saturation)
- `b`: BM25 b parameter (length normalization)

#### TF-IDF

TF-IDF retrieval technique with cosine similarity scoring.

```python
@TechniqueRegistry.register
class TfIdf(RAGTechnique):
    def __init__(self, normalize: bool = True, smoothing: bool = True):
        pass
```

**Parameters:**
- `normalize`: Whether to apply L2 normalization to vectors
- `smoothing`: Whether to apply IDF smoothing (+1)

#### LexicalMatcher

Lexical matching retrieval with configurable matching modes.

```python
@TechniqueRegistry.register
class LexicalMatcher(RAGTechnique):
    def __init__(self, mode: str = "token_overlap", threshold: float = 0.1):
        pass
```

**Parameters:**
- `mode`: Matching mode ("exact", "substring", "token_overlap", "weighted_overlap")
- `threshold`: Minimum score threshold for results

**Modes:**
- `exact`: Exact phrase matching
- `substring`: Substring matching within documents
- `token_overlap`: Token-level overlap scoring
- `weighted_overlap`: Weighted token overlap with term frequency

#### SPLADE

SPLADE sparse-dense hybrid retrieval with term expansion.

```python
@TechniqueRegistry.register
class Splade(RAGTechnique):
    def __init__(self, expansion_factor: float = 0.3, 
                 sparsity_threshold: float = 0.01):
        pass
```

**Parameters:**
- `expansion_factor`: Weight for expanded terms (0.0 to 1.0)
- `sparsity_threshold`: Minimum importance for term inclusion

#### LexicalTransformer

Transformer-aware lexical retrieval with attention weighting.

```python
@TechniqueRegistry.register
class LexicalTransformer(RAGTechnique):
    def __init__(self, attention_weight: float = 0.7, 
                 position_weight: float = 0.3):
        pass
```

**Parameters:**
- `attention_weight`: Weight for attention-based scoring
- `position_weight`: Weight for positional encoding effects

### Reranking Techniques

#### CrossEncoderReRanker

Reranks candidates using a cross-encoder model.

```python
@TechniqueRegistry.register
class CrossEncoderReRanker(RAGTechnique):
    def __init__(self, model: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2"):
        pass
    
    def apply(self, query: str, candidates: List[str], 
             top_k: int = None) -> TechniqueResult:
        """Rerank candidates for the given query."""
        pass
```

**Parameters:**
- `model`: Hugging Face model identifier
- `top_k`: Number of top candidates to return

#### MMR (Maximal Marginal Relevance)

Reranks to optimize relevance and diversity trade-off.

```python
@TechniqueRegistry.register
class MMR(RAGTechnique):
    def __init__(self, lambda_param: float = 0.5, 
                 embedder: Embedder = None):
        pass
```

**Parameters:**
- `lambda_param`: Relevance vs. diversity trade-off (0=diversity, 1=relevance)
- `embedder`: Embedder for computing similarities

### Query Expansion Techniques

#### HyDE

Hypothetical Document Embeddings for improved retrieval.

```python
@TechniqueRegistry.register
class HyDE(RAGTechnique):
    def __init__(self, llm_adapter=None, temperature: float = 0.7):
        pass
    
    def apply(self, query: str, top_k: int = 5) -> TechniqueResult:
        """Generate hypothetical document for improved retrieval."""
        pass
```

**Parameters:**
- `llm_adapter`: LLM adapter for generating hypothetical documents
- `temperature`: Sampling temperature
- `max_tokens`: Maximum tokens to generate
- `prompt_template`: Custom prompt template

## Architectures

### FiDPipeline

Fusion-in-Decoder architecture for end-to-end RAG.

```python
class FiDPipeline:
    def __init__(self, chunker_name: str, retriever_name: str, 
                 generator_name: str, **component_configs):
        pass
    
    def apply(self, documents: List[str], query: str) -> TechniqueResult:
        """Run end-to-end RAG pipeline."""
        pass
```

**Parameters:**
- `chunker_name`: Name of registered chunking technique
- `retriever_name`: Name of registered retrieval technique
- `generator_name`: Name of registered generation technique
- `component_configs`: Configuration for individual components

## Utilities

### Configuration Management

Functions for loading and managing configurations.

```python
def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    pass

def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """Validate configuration against schema."""
    pass
```

### Evaluation Metrics

Common evaluation metrics for RAG systems.

```python
def exact_match(predicted: str, expected: str) -> float:
    """Compute exact match score."""
    pass

def f1_score(predicted: str, expected: str) -> float:
    """Compute F1 score."""
    pass

def bleu_score(predicted: str, expected: str) -> float:
    """Compute BLEU score."""
    pass
```

## Error Handling

### Exception Classes

Custom exceptions used by RAGLib.

```python
class RAGLibError(Exception):
    """Base exception for RAGLib."""
    pass

class TechniqueNotFoundError(RAGLibError):
    """Raised when a technique is not found in registry."""
    pass

class AdapterError(RAGLibError):
    """Raised for adapter-related errors."""
    pass

class ConfigurationError(RAGLibError):
    """Raised for configuration-related errors."""
    pass
```

## Type Definitions

### Common Types

```python
from typing import List, Dict, Any, Optional, Union

# Vector types
Vector = List[float]
VectorBatch = List[Vector]

# Document types
Document = str
DocumentBatch = List[Document]

# Search results
@dataclass
class VectorSearchResult:
    id: str
    score: float
    metadata: Dict[str, Any]

# Chunk types
@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
```

For implementation details and examples, see the [Getting Started](getting_started.md) guide and [Techniques Catalog](techniques.md).
