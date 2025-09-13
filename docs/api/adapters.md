# Adapters API 🔌

RAGLib adapters provide interfaces to external services and libraries. This page documents all available adapters and their APIs.

## Overview 📖

Adapters abstract away the complexities of different external dependencies, providing a consistent interface for RAGLib techniques to use. This allows techniques to be library-agnostic while still leveraging powerful external tools.

## Base Adapter Classes 🏗️

### BaseAdapter

Base class for all RAGLib adapters.

```python
class BaseAdapter:
    """Base class for all RAGLib adapters."""
    
    def __init__(self):
        """Initialize the adapter."""
        pass
    
    def cleanup(self):
        """Clean up adapter resources."""
        pass
```

### BaseEmbedder

Base class for embedding models.

```python
from typing import List
import numpy as np

class BaseEmbedder(BaseAdapter):
    """Base class for embedding models."""
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Convert texts to embeddings.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            numpy.ndarray: Array of embeddings with shape (len(texts), embedding_dim)
        """
        raise NotImplementedError
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query.
        
        Args:
            query: Query text to embed
            
        Returns:
            numpy.ndarray: Query embedding
        """
        raise NotImplementedError
```

**Example Implementation:**

```python
from raglib.adapters.base import BaseEmbedder
from typing import List
import numpy as np

class MyEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        # Initialize your embedding model here
    
    def embed(self, texts: List[str]) -> np.ndarray:
        # Convert texts to embeddings
        embeddings = []
        for text in texts:
            # Your embedding logic here
            embedding = self._compute_embedding(text)
            embeddings.append(embedding)
        return np.array(embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        # Embed a single query
        return self.embed([query])[0]
```

### BaseVectorStore

Base class for vector storage systems.

```python
from typing import List, Tuple, Optional
import numpy as np

class BaseVectorStore(BaseAdapter):
    """Base class for vector storage systems."""
    
    def add(self, vectors: np.ndarray, metadata: List[dict]) -> None:
        """Add vectors to the store.
        
        Args:
            vectors: Array of vectors to add
            metadata: List of metadata dicts for each vector
        """
        raise NotImplementedError
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[float, dict]]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            
        Returns:
            List of (similarity_score, metadata) tuples
        """
        raise NotImplementedError
    
    def delete(self, ids: List[str]) -> None:
        """Delete vectors by ID.
        
        Args:
            ids: List of vector IDs to delete
        """
        raise NotImplementedError
```

**Example Implementation:**

```python
from raglib.adapters.base import BaseVectorStore
from typing import List, Tuple, Optional
import numpy as np

class MyVectorStore(BaseVectorStore):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self.vectors = []
        self.metadata = []
    
    def add(self, vectors: np.ndarray, metadata: List[dict]) -> None:
        self.vectors.extend(vectors.tolist())
        self.metadata.extend(metadata)
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[float, dict]]:
        # Compute similarities and return top-k
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = self._cosine_similarity(query_vector, np.array(vector))
            similarities.append((similarity, self.metadata[i]))
        
        # Sort by similarity (descending) and return top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:k]
```

## Embedding Adapters 🤖

### DummyEmbedder

Simple dummy embedder for testing and prototyping.

```python
class DummyEmbedder(BaseEmbedder):
    """Simple dummy embedder for testing."""
    
    def __init__(self, dimension: int = 768):
        super().__init__()
        self.dimension = dimension
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate random embeddings."""
        return np.random.randn(len(texts), self.dimension)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate random query embedding."""
        return np.random.randn(self.dimension)
```

**Example Usage:**

```python
from raglib.adapters import DummyEmbedder

embedder = DummyEmbedder(dimension=768)

# Embed texts
texts = ["Hello world", "RAG is great"]
embeddings = embedder.embed(texts)
print(f"Embeddings shape: {embeddings.shape}")  # (2, 768)

# Embed single query
query_embedding = embedder.embed_query("What is RAG?")
print(f"Query embedding shape: {query_embedding.shape}")  # (768,)
```

### SentenceTransformerEmbedder

```python
# Example implementation (not included in base RAGLib)
from sentence_transformers import SentenceTransformer
from raglib.adapters.base import BaseEmbedder

class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__()
        self.model = SentenceTransformer(model_name)
    
    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts)
    
    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query])[0]
```

### OpenAIEmbedder

```python
# Example implementation (not included in base RAGLib)
from openai import OpenAI
from raglib.adapters.base import BaseEmbedder

class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model: str = "text-embedding-ada-002", api_key: str = None):
        super().__init__()
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def embed(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return np.array([embedding.embedding for embedding in response.data])
    
    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]
```

## Vector Store Adapters 🗄️

### InMemoryVectorStore

Simple in-memory vector store for development and testing.

```python
class InMemoryVectorStore(BaseVectorStore):
    """Simple in-memory vector store."""
    
    def __init__(self):
        super().__init__()
        self.vectors = []
        self.metadata = []
    
    def add(self, vectors: np.ndarray, metadata: List[dict]) -> None:
        """Add vectors to memory."""
        self.vectors.extend(vectors.tolist())
        self.metadata.extend(metadata)
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[float, dict]]:
        """Search using cosine similarity."""
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = self._cosine_similarity(query_vector, np.array(vector))
            similarities.append((similarity, self.metadata[i]))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:k]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

**Example Usage:**

```python
from raglib.adapters import InMemoryVectorStore
import numpy as np

# Create vector store
vectorstore = InMemoryVectorStore()

# Add vectors with metadata
vectors = np.random.randn(100, 768)  # 100 random 768-dim vectors
metadata = [{"id": i, "text": f"Document {i}"} for i in range(100)]

vectorstore.add(vectors, metadata)

# Search for similar vectors
query_vector = np.random.randn(768)
results = vectorstore.search(query_vector, k=5)

for similarity, meta in results:
    print(f"Similarity: {similarity:.3f}, Document: {meta['text']}")
```

### FAISSVectorStore

```python
# Example implementation (not included in base RAGLib)
import faiss
from raglib.adapters.base import BaseVectorStore

class FAISSVectorStore(BaseVectorStore):
    def __init__(self, dimension: int, index_type: str = "IndexFlatIP"):
        super().__init__()
        if index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "IndexIVFFlat":
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        self.metadata = []
    
    def add(self, vectors: np.ndarray, metadata: List[dict]) -> None:
        # Normalize vectors for cosine similarity
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self.index.add(vectors.astype(np.float32))
        self.metadata.extend(metadata)
        
        # Train index if needed
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            self.index.train(vectors.astype(np.float32))
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[float, dict]]:
        # Normalize query vector
        query_vector = query_vector / np.linalg.norm(query_vector)
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        
        scores, indices = self.index.search(query_vector, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):  # Valid index
                results.append((float(score), self.metadata[idx]))
        
        return results
```

## LLM Adapters 🤖

### BaseLLM

```python
from raglib.adapters.base import BaseAdapter
from typing import List, Optional

class BaseLLM(BaseAdapter):
    """Base class for Language Model adapters."""
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 100,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """Generate text from prompt."""
        raise NotImplementedError
    
    def chat(
        self,
        messages: List[dict],
        max_tokens: int = 100,
        temperature: float = 0.7
    ) -> str:
        """Chat-based generation."""
        raise NotImplementedError
```

### OpenAILLM

```python
# Example implementation (not included in base RAGLib)
from openai import OpenAI
from raglib.adapters.base import BaseLLM

class OpenAILLM(BaseLLM):
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: str = None):
        super().__init__()
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7, stop_sequences: Optional[List[str]] = None) -> str:
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_sequences
        )
        return response.choices[0].text.strip()
    
    def chat(self, messages: List[dict], max_tokens: int = 100, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
```

## Creating Custom Adapters 🛠️

### Step 1: Choose Base Class

Choose the appropriate base class for your adapter:

- `BaseEmbedder` for embedding models
- `BaseVectorStore` for vector databases
- `BaseLLM` for language models
- `BaseAdapter` for other types

### Step 2: Implement Required Methods

```python
from raglib.adapters.base import BaseEmbedder

class MyCustomEmbedder(BaseEmbedder):
    def __init__(self, model_path: str):
        super().__init__()
        # Initialize your model
        self.model = self._load_model(model_path)
    
    def embed(self, texts: List[str]) -> np.ndarray:
        # Your embedding logic
        return self.model.encode(texts)
    
    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]
    
    def _load_model(self, model_path: str):
        # Load your embedding model
        pass
```

### Step 3: Add Error Handling

```python
def embed(self, texts: List[str]) -> np.ndarray:
    try:
        if not texts:
            raise ValueError("Input texts cannot be empty")
        
        embeddings = self.model.encode(texts)
        
        if embeddings is None or len(embeddings) == 0:
            raise RuntimeError("Model failed to generate embeddings")
        
        return embeddings
        
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {str(e)}")
```

### Step 4: Add Configuration

```python
class ConfigurableEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, **config):
        super().__init__()
        
        # Default configuration
        self.config = {
            "batch_size": 32,
            "normalize": True,
            "device": "cpu",
            **config  # Override with user config
        }
        
        self.model = self._load_model(model_name)
    
    def embed(self, texts: List[str]) -> np.ndarray:
        # Use configuration
        batch_size = self.config["batch_size"]
        normalize = self.config["normalize"]
        
        # Process in batches
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch)
            
            if normalize:
                batch_embeddings = self._normalize(batch_embeddings)
            
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
```

## Best Practices 🎯

### 1. Resource Management

```python
class ResourceManagedAdapter(BaseAdapter):
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def cleanup(self):
        # Clean up resources
        if hasattr(self, 'model'):
            del self.model
```

### 2. Configuration Validation

```python
def __init__(self, **config):
    self.config = self._validate_config(config)

def _validate_config(self, config):
    defaults = {"batch_size": 32, "timeout": 30}
    
    # Merge with defaults
    config = {**defaults, **config}
    
    # Validate
    if config["batch_size"] <= 0:
        raise ValueError("batch_size must be positive")
    
    if config["timeout"] <= 0:
        raise ValueError("timeout must be positive")
    
    return config
```

### 3. Progress Tracking

```python
from tqdm import tqdm

def embed(self, texts: List[str]) -> np.ndarray:
    embeddings = []
    batch_size = self.config["batch_size"]
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]
        batch_embeddings = self.model.encode(batch)
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)
```

---

💡 **Need to create a custom adapter?** Check out the [Contributing Guide](https://github.com/your-org/raglib/blob/main/CONTRIBUTING.md) for information on submitting your adapter to RAGLib.
