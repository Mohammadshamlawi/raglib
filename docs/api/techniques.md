# Techniques API 🛠️

This page provides detailed API documentation for all RAGLib techniques organized by category.

## Quick Reference 📋

| Technique | Category | Purpose | Key Parameters |
|-----------|----------|---------|----------------|
| FixedSizeChunker | Chunking | Split into fixed-size chunks | `chunk_size`, `overlap` |
| SemanticChunker | Chunking | Semantically-aware chunking | `similarity_threshold` |
| SentenceWindowChunker | Chunking | Sentence-based windowing | `window_size`, `step_size` |
| DenseRetriever | Retrieval | Embedding-based retrieval | `top_k`, `similarity_threshold` |
| BM25 | Retrieval | Keyword-based retrieval | `top_k`, `k1`, `b` |
| CrossEncoderReRanker | Reranking | Neural reranking | `model_name`, `top_k` |
| MMRReRanker | Reranking | Diversity-aware reranking | `diversity_lambda` |
| HyDE | Generation | Hypothetical document embeddings | `model_name`, `temperature` |

## Usage Patterns 🚀

### Basic Technique Usage

```python
from raglib.techniques import FixedSizeChunker

# Initialize technique
chunker = FixedSizeChunker(chunk_size=512, overlap=50)

# Apply technique
result = chunker.apply(documents)

# Check result
if result.success:
    chunks = result.payload["chunks"]
    print(f"Created {len(chunks)} chunks")
else:
    print(f"Error: {result.error}")
```

### Chaining Techniques

```python
from raglib.techniques import FixedSizeChunker, DenseRetriever, LLMGenerator

# Setup pipeline components
chunker = FixedSizeChunker(chunk_size=512)
retriever = DenseRetriever(top_k=5)
generator = LLMGenerator(model_name="gpt-3.5-turbo")

# Execute pipeline
chunks = chunker.apply(documents)
if chunks.success:
    retrieved = retriever.apply(query, chunks.payload["chunks"])
    if retrieved.success:
        answer = generator.apply(query, retrieved.payload["documents"])
```

### Error Handling

All techniques return consistent `TechniqueResult` objects:

```python
result = technique.apply(input_data)

if result.success:
    output = result.payload
    metadata = result.metadata
else:
    print(f"Technique failed: {result.error}")
    error_context = result.metadata
```

## Technique Categories 📁

### Chunking Techniques 📄

Break documents into processable segments:

- **FixedSizeChunker**: Split text into chunks of fixed character length
- **SemanticChunker**: Create chunks based on semantic similarity  
- **SentenceWindowChunker**: Use sentence boundaries with overlapping windows

### Retrieval Techniques 🔍

Find relevant information from document collections:

- **DenseRetriever**: Vector-based semantic retrieval using embeddings
- **BM25**: Traditional keyword-based sparse retrieval with in-memory indexing

### Reranking Techniques 🎯

Improve initial retrieval results:

- **CrossEncoderReRanker**: Deep learning-based relevance scoring
- **MMRReRanker**: Balance relevance with diversity using Maximal Marginal Relevance

### Generation Techniques ✍️

Create responses from retrieved context:

- **LLMGenerator**: Generate answers using large language models
- **TemplateResponse**: Rule-based template filling
- **HyDE**: Hypothetical document embeddings for enhanced retrieval

### Utility Techniques 🔧

Helper techniques for testing and development:

- **EchoTechnique**: Echo input with optional modifications
- **NullTechnique**: No-operation technique for pipeline testing

## Advanced Usage 🎯

### Custom Technique Development

```python
from raglib.core import RAGTechnique, TechniqueResult
from raglib.schemas import TechniqueMeta

class MyTechnique(RAGTechnique):
    def __init__(self, custom_param: str):
        meta = TechniqueMeta(
            name="my_technique",
            category="custom",
            description="My custom technique"
        )
        super().__init__(meta)
        self.custom_param = custom_param
    
    def apply(self, input_data) -> TechniqueResult:
        try:
            # Your logic here
            result = self._process(input_data)
            return TechniqueResult(
                success=True,
                payload=result,
                metadata={"processing_info": "success"}
            )
        except Exception as e:
            return TechniqueResult(
                success=False,
                error=str(e)
            )
```

### Registry Integration

```python
from raglib.registry import TechniqueRegistry

# Register your technique
TechniqueRegistry.register("my_technique", MyTechnique)

# Use via registry
technique_class = TechniqueRegistry.get("my_technique")
technique = technique_class(custom_param="value")
```

---

💡 **For detailed parameter documentation and examples, see the individual technique source files or use the interactive help:**

```python
from raglib.techniques import FixedSizeChunker
help(FixedSizeChunker)
```
