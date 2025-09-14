# Techniques Index

---

**Total Techniques:** 9
**Categories:** 4

### Chunking

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
