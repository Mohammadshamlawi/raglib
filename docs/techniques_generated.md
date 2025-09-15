# Techniques Index

---

**Total Techniques:** 8
**Categories:** 5

### Chunking

#### Fixed-Size Chunking

**Fixed-size text chunking with overlap support**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `FixedSizeChunker` |
| Module | `raglib.techniques.fixed_size_chunker` |
| Dependencies | None |

---

#### Semantic Chunking

**Semantic similarity-based chunking with configurable embedder**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `SemanticChunker` |
| Module | `raglib.techniques.semantic_chunker` |
| Dependencies | None |

---

#### Sentence Window Retrieval

**Sentence-based windowing with configurable window size and overlap**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `SentenceWindowChunker` |
| Module | `raglib.techniques.sentence_window_chunker` |
| Dependencies | None |

---

### Reranking

#### Cross-Encoder Re-Ranking

**Cross-encoder re-ranking using pairwise (query, document) scoring**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `CrossEncoderReRanker` |
| Module | `raglib.techniques.crossencoder_rerank` |
| Dependencies | None |

---

#### Maximal Marginal Relevance (MMR)

**Maximal Marginal Relevance re-ranking for balancing relevance and diversity**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `MMRReRanker` |
| Module | `raglib.techniques.mmr` |
| Dependencies | None |

---

### Core-Retrieval

#### Dense Retrieval / Vector Search (e.g., DPR)

**Production-friendly dense retriever with optional adapters fallback.**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `DenseRetriever` |
| Module | `raglib.techniques.dense_retriever` |
| Dependencies | None |

---

### Retrieval Enhancement

#### Hypothetical Document Embeddings (HyDE)

**Generate hypothetical documents to improve retrieval**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `HyDE` |
| Module | `raglib.techniques.hyde` |
| Dependencies | None |

---

### Sparse-Retrieval

#### BM25 (Best Matching 25)

**BM25 ranking function for text retrieval with in-memory indexing**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `BM25` |
| Module | `raglib.techniques.bm25` |
| Dependencies | None |

---
