# Techniques Catalog 📚

RAGLib provides a comprehensive collection of RAG techniques, each implementing the unified `RAGTechnique.apply()` interface. This catalog showcases all available techniques organized by category.

## Updating the Techniques Index

!!! tip "Techniques Index Update"
    The complete techniques catalog is generated from the RAGLib registry. To update the catalog:
    
    === "Linux/macOS"
        **Using Make**
        ```bash
        make docs-generate    # Generate techniques index
        make docs            # Build docs (includes generation)  
        make docs-serve      # Build and serve locally
        ```
        
        **Manual Generation**
        ```bash
        python tools/generate_techniques_index.py
        ```
    
    === "Windows"
        **Using PowerShell**
        ```powershell
        .\build.ps1 docs-generate    # Generate techniques index
        .\build.ps1 docs            # Build docs (includes generation)
        .\build.ps1 docs-serve      # Build and serve locally
        ```
        
        **Using Batch Script**
        ```batch
        .\build.bat docs-generate    # Generate techniques index
        .\build.bat docs            # Build docs (includes generation)
        .\build.bat docs-serve      # Build and serve locally
        ```
        
        **Manual Generation**
        ```batch
        python tools\generate_techniques_index.py
        ```
    
    The content appears in `docs/techniques_generated.md` and is included in the documentation build.

!!! warning "Do Not Edit Index Files"
    The file `docs/techniques_generated.md` is created by the generation script. Manual edits will be overwritten when the index is regenerated.

## Technique Categories 🗂️

RAG techniques are organized into the following categories:

=== ":material-file-document: Chunking"
    Split documents into processable segments
    
    - **Purpose**: Prepare documents for embedding and retrieval
    - **When to use**: When working with large documents that need to be broken down
    - **Output**: Structured chunks with metadata

=== ":material-magnify: Retrieval"
    Find relevant information from knowledge bases
    
    - **Purpose**: Locate relevant information based on queries
    - **When to use**: Core component of most RAG pipelines
    - **Output**: Ranked list of relevant documents/chunks

=== ":material-target: Reranking"
    Improve retrieval quality through reordering
    
    - **Purpose**: Refine initial retrieval results
    - **When to use**: When you need higher precision in retrieved results
    - **Output**: Reordered list of documents with improved relevance scores

=== ":material-pencil: Generation"
    Produce final answers using retrieved context
    
    - **Purpose**: Generate human-readable responses
    - **When to use**: Final step in RAG pipelines to create answers
    - **Output**: Generated text based on retrieved context

=== ":material-orchestra: Orchestration"
    Coordinate multiple techniques in workflows
    
    - **Purpose**: Chain multiple techniques together
    - **When to use**: Building complex RAG pipelines
    - **Output**: Results from multi-step processing

## Available Techniques 🛠️

The following sections contain the complete catalog of techniques available in RAGLib. This content is generated from the technique registry.

--8<-- "docs/techniques_generated.md"

---

## Adding New Techniques

To add a new technique to RAGLib:

### Chunking Techniques

#### FixedSizeChunker
- **Description**: Split documents into fixed-size chunks with optional overlap
- **Category**: chunking
- **Dependencies**: None
- **Parameters**: `chunk_size`, `overlap`, `separator`

#### SemanticChunker  
- **Description**: Creates semantically coherent chunks based on content similarity
- **Category**: chunking
- **Dependencies**: sentence-transformers
- **Parameters**: `similarity_threshold`, `max_chunk_size`

#### SentenceWindowChunker
- **Description**: Creates chunks with sentence boundaries and context windows
- **Category**: chunking
- **Dependencies**: None
- **Parameters**: `window_size`, `stride`

### Retrieval Techniques

#### DenseRetriever
- **Description**: Dense retrieval using semantic embeddings
- **Category**: retrieval  
- **Dependencies**: faiss
- **Parameters**: `top_k`, `similarity_threshold`

#### BM25SimpleRetriever
- **Description**: Simple BM25-based sparse retrieval
- **Category**: retrieval
- **Dependencies**: None
- **Parameters**: `top_k`, `k1`, `b`

### Reranking Techniques

#### CrossEncoderRerank
- **Description**: Rerank documents using a cross-encoder model
- **Category**: reranking
- **Dependencies**: transformers, torch
- **Parameters**: `model_name`, `top_k`

#### MMR (Maximal Marginal Relevance)
- **Description**: Diversify results using Maximal Marginal Relevance
- **Category**: reranking
- **Dependencies**: None
- **Parameters**: `diversity_lambda`, `top_k`

!!! tip "Technique Discovery"
    Use the `TechniqueRegistry.list_techniques()` method to programmatically discover available techniques at runtime.

## Using Techniques 🚀

### Basic Usage Pattern

All techniques follow the same consistent pattern:

```python
from raglib.registry import TechniqueRegistry

# Discover available techniques
available = TechniqueRegistry.list_techniques()
print(f"Available techniques: {list(available.keys())}")

# Get technique class
TechniqueClass = TechniqueRegistry.get("technique_name")

# Initialize with configuration
technique = TechniqueClass(param1=value1, param2=value2)

# Apply technique
result = technique.apply(input_data, mode="some_mode")

# Check results
if result.success:
    output = result.payload
    print(f"Success: {output}")
else:
    print(f"Error: {result.error}")
```

### Finding Techniques by Category

```python
from raglib.registry import TechniqueRegistry

# Get all chunking techniques
chunking_techniques = TechniqueRegistry.list_by_category("chunking")

# Get all retrieval techniques  
retrieval_techniques = TechniqueRegistry.list_by_category("retrieval")

# Print technique information
for name, meta in chunking_techniques.items():
    print(f"{name}: {meta.description}")
```

# Find all chunking techniques
chunking_techniques = TechniqueRegistry.find_by_category("chunking")

# List available techniques
for name, technique_class in chunking_techniques.items():
    meta = technique_class.meta
    print(f"{name}: {meta.description}")
```

### Technique Metadata

Each technique provides metadata through its `meta` attribute:

```python
technique_class = TechniqueRegistry.get("fixed_size_chunker")
meta = technique_class.meta

print(f"Name: {meta.name}")
print(f"Category: {meta.category}")
print(f"Description: {meta.description}")
print(f"Version: {meta.version}")
print(f"Dependencies: {meta.dependencies}")
```

## Chunking Techniques

Chunking techniques split documents into smaller, processable segments.

### Common Parameters

- `chunk_size`: Maximum size of each chunk (in characters or tokens)
- `overlap`: Number of characters/tokens to overlap between chunks
- `separator`: Text used to split documents

### Usage Example

```python
from raglib.techniques import FixedSizeChunker

chunker = FixedSizeChunker(chunk_size=512, overlap=50)
documents = ["Long document text here..."]

result = chunker.apply(documents)
chunks = result.payload["chunks"]
```

## Retrieval Techniques

Retrieval techniques find relevant information from knowledge bases.

### Common Parameters

- `top_k`: Number of results to return
- `similarity_threshold`: Minimum similarity score for results
- `embedder`: Embedder adapter for encoding queries
- `vectorstore`: Vector store adapter for similarity search

### Usage Example

```python
from raglib.techniques import DenseRetriever
from raglib.adapters import InMemoryVectorStore, DummyEmbedder

retriever = DenseRetriever(
    embedder=DummyEmbedder(),
    vectorstore=InMemoryVectorStore()
)

# Index documents
chunks = ["chunk1", "chunk2", "chunk3"]
retriever.apply(chunks, mode="index")

# Retrieve relevant chunks
query = "search query"
result = retriever.apply(query, mode="retrieve", top_k=3)
relevant_chunks = result.payload["chunks"]
```

## Reranking Techniques

Reranking techniques improve the quality of retrieved results.

### Common Parameters

- `model`: Model used for reranking
- `top_k`: Number of results to return after reranking
- `cross_encoder`: Whether to use cross-encoder architecture

### Usage Example

```python
from raglib.techniques import CrossEncoderRerank

reranker = CrossEncoderRerank(model="cross-encoder/ms-marco-TinyBERT-L-2-v2")

query = "search query"
candidates = ["candidate1", "candidate2", "candidate3"]

result = reranker.apply(query=query, candidates=candidates, top_k=2)
reranked_candidates = result.payload["reranked_chunks"]
```

## Generation Techniques

Generation techniques produce final answers using retrieved context.

### Common Parameters

- `model`: Language model to use for generation
- `temperature`: Sampling temperature for generation
- `max_tokens`: Maximum number of tokens to generate
- `prompt_template`: Template for formatting prompts

### Usage Example

```python
from raglib.techniques import LLMGenerator

generator = LLMGenerator(model="gpt-3.5-turbo", temperature=0.1)

query = "What is RAG?"
context = ["RAG stands for Retrieval-Augmented Generation..."]

result = generator.apply(query=query, context=context)
answer = result.payload["answer"]
```

## Orchestration Techniques

Orchestration techniques coordinate multiple techniques in complex workflows.

### Usage Example

```python
from raglib.architectures import FiDPipeline

pipeline = FiDPipeline(
    chunker_name="fixed_size_chunker",
    retriever_name="dense_retriever", 
    generator_name="llm_generator"
)

documents = ["document1", "document2"]
query = "question"

result = pipeline.apply(documents=documents, query=query)
answer = result.payload["answer"]
```

## Custom Techniques

### Creating a New Technique

To create a custom technique:

1. **Create a new file** in `raglib/techniques/your_technique.py`
2. **Implement the technique** following the `RAGTechnique` interface
3. **Register the technique** using the `@TechniqueRegistry.register` decorator
4. **Add tests** in `tests/test_your_technique.py`

Example:

```python
from raglib.core import RAGTechnique, TechniqueMeta, TechniqueResult
from raglib.registry import TechniqueRegistry

@TechniqueRegistry.register
class MyCustomTechnique(RAGTechnique):
    meta = TechniqueMeta(
        name="my_custom_technique",
        category="custom",
        description="A custom technique for special processing",
        version="1.0.0",
        dependencies=[]
    )

    def __init__(self, param1=None):
        super().__init__(self.meta)
        self.param1 = param1

    def apply(self, input_data, **kwargs):
        try:
            # Your processing logic here
            processed_data = self._process(input_data)
            
            return TechniqueResult(
                success=True,
                payload={"result": processed_data}
            )
        except Exception as e:
            return TechniqueResult(
                success=False,
                error=str(e)
            )

    def _process(self, data):
        # Implementation details
        return data
```

### Best Practices

1. **One technique per file**: Keep each technique in its own file
2. **Clear metadata**: Provide comprehensive `TechniqueMeta` information
3. **Error handling**: Always return `TechniqueResult` with proper error handling
4. **Documentation**: Include docstrings and type hints
5. **Testing**: Add comprehensive tests for your technique
6. **Dependencies**: Use optional dependencies for external libraries

For detailed contribution guidelines, see our [Contributing Guide](https://github.com/your-org/raglib/blob/main/CONTRIBUTING.md).
