# Techniques Index

---

**Total Techniques:** 22
**Categories:** 5

### Chunking

#### content_aware_chunker

**Content-aware chunking that respects text structure and natural boundaries**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `ContentAwareChunker` |
| Module | `raglib.techniques.content_aware_chunker` |
| Dependencies | None |

---

#### document_specific_chunker

**Document-specific chunking that adapts to document type**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `DocumentSpecificChunker` |
| Module | `raglib.techniques.document_specific_chunker` |
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

#### parent_document_chunker

**Parent document retrieval with small-to-large chunk mapping**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `ParentDocumentChunker` |
| Module | `raglib.techniques.parent_document_chunker` |
| Dependencies | None |

---

#### propositional_chunker

**Propositional chunking based on semantic propositions**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `PropositionalChunker` |
| Module | `raglib.techniques.propositional_chunker` |
| Dependencies | None |

---

#### recursive_chunker

**Recursive chunking with hierarchical text splitting**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `RecursiveChunker` |
| Module | `raglib.techniques.recursive_chunker` |
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

#### crossencoder_rerank

**Cross-encoder re-ranking using pairwise (query, document) scoring**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `CrossEncoderReRanker` |
| Module | `raglib.techniques.crossencoder_rerank` |
| Dependencies | None |

---

#### mmr

**Maximal Marginal Relevance re-ranking for balancing relevance and diversity**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `MMRReRanker` |
| Module | `raglib.techniques.mmr` |
| Dependencies | None |

---

### Core Retrieval

#### colbert_retriever

**ColBERT-style late interaction dense retrieval**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `ColBERTRetriever` |
| Module | `raglib.techniques.colbert_retriever` |
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

#### dual_encoder

**Dual-encoder dense retrieval with separate query/doc encoding**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `DualEncoder` |
| Module | `raglib.techniques.dual_encoder` |
| Dependencies | None |

---

#### faiss_retriever

**FAISS-based dense retriever with efficient similarity search**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `FAISSRetriever` |
| Module | `raglib.techniques.faiss_retriever` |
| Dependencies | None |

---

#### multi_vector_retriever

**Multi-vector dense retrieval with document segmentation**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `MultiVectorRetriever` |
| Module | `raglib.techniques.multi_vector_retriever` |
| Dependencies | None |

---

### Retrieval Enhancement

#### hyde

**Generate hypothetical documents to improve retrieval**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `HyDE` |
| Module | `raglib.techniques.hyde` |
| Dependencies | None |

---

#### multi_query_retriever

**Multi-query retrieval with result fusion**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `MultiQueryRetriever` |
| Module | `raglib.techniques.multi_query_retriever` |
| Dependencies | None |

---

### Sparse Retrieval

#### bm25

**BM25 ranking function for text retrieval with in-memory indexing**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `BM25` |
| Module | `raglib.techniques.bm25` |
| Dependencies | None |

---

#### lexical_matcher

**Lexical matching retrieval with configurable matching modes**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `LexicalMatcher` |
| Module | `raglib.techniques.lexical_matcher` |
| Dependencies | None |

---

#### lexical_transformer

**Transformer-aware lexical retrieval with attention weighting**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `LexicalTransformer` |
| Module | `raglib.techniques.lexical_transformer` |
| Dependencies | None |

---

#### splade

**SPLADE sparse-dense hybrid retrieval with term expansion**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `Splade` |
| Module | `raglib.techniques.splade` |
| Dependencies | None |

---

#### tfidf

**TF-IDF retrieval with cosine similarity scoring**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `TfIdf` |
| Module | `raglib.techniques.tfidf` |
| Dependencies | None |

---
