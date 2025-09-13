# Techniques Index

---

**Total Techniques:** 13
**Categories:** 6

### Chunking

#### demo_fixed_chunker

**Simple fixed-size character chunker for demos and tests.**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `DemoFixedChunker` |
| Module | `raglib.techniques.demo_fixed_chunker` |
| Dependencies | None |

---

#### fixed_size_chunker

**Fixed-size text chunking with overlap support**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `FixedSizeChunker` |
| Module | `raglib.techniques.fixed_size_chunker` |
| Dependencies | None |

---

#### semantic_chunker

**Semantic similarity-based chunking with configurable embedder**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `SemanticChunker` |
| Module | `raglib.techniques.semantic_chunker` |
| Dependencies | None |

---

#### sentence_window_chunker

**Sentence-based windowing with configurable window size and overlap**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `SentenceWindowChunker` |
| Module | `raglib.techniques.sentence_window_chunker` |
| Dependencies | None |

---

### Reranking

#### crossencoder_reranker

**Cross-encoder re-ranking using pairwise (query, document) scoring**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `CrossEncoderReRanker` |
| Module | `raglib.techniques.crossencoder_rerank` |
| Dependencies | None |

---

#### mmr_reranker

**Maximal Marginal Relevance re-ranking for balancing relevance and diversity**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `MMRReRanker` |
| Module | `raglib.techniques.mmr` |
| Dependencies | None |

---

### Generation

#### llm_generator

**LLM text generation with deterministic fallback**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `LLMGenerator` |
| Module | `raglib.techniques.llm_generator` |
| Dependencies | None |

---

### Core-Retrieval

#### bm25_retriever

**Production-friendly BM25 retriever (wrapper over BM25Simple).**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `BM25Retriever` |
| Module | `raglib.techniques.bm25_production` |
| Dependencies | None |

---

#### bm25_simple

**Toy BM25 retriever (pure python, dependency-free).**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `BM25Simple` |
| Module | `raglib.techniques.bm25_simple` |
| Dependencies | None |

---

#### dense_retriever

**Production-friendly dense retriever with optional adapters fallback.**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `DenseRetriever` |
| Module | `raglib.techniques.dense_retriever` |
| Dependencies | None |

---

#### inmemory_dense

**In-memory dense retriever using adapters for embedder and vectorstore.**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `InMemoryDenseRetriever` |
| Module | `raglib.techniques.dummy_dense` |
| Dependencies | None |

---

### Retrieval Enhancement

#### hyde_generator

**Generate hypothetical documents to improve retrieval**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `HyDE` |
| Module | `raglib.techniques.hyde` |
| Dependencies | None |

---

### Utility

#### echo_technique

**Echoes the input payload back as TechniqueResult.**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `EchoTechnique` |
| Module | `raglib.techniques.echo_technique` |
| Dependencies | None |

---
