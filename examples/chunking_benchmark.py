#!/usr/bin/env python3
"""Comprehensive benchmark script for all RAGLib chunking techniques.

This script evaluates all available chunking techniques on a variety of
text types to demonstrate their performance characteristics.
"""

import json
import sys
import time
from pathlib import Path

# Add raglib to path for development
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from raglib.registry import TechniqueRegistry
    from raglib.schemas import Document
    # Import techniques to ensure they are registered
    import raglib.techniques
except ImportError as e:
    print(f"Failed to import raglib: {e}")
    print("Make sure raglib is installed: pip install -e .")
    sys.exit(1)


# Comprehensive test documents with various structures
BENCHMARK_DOCUMENTS = [
    Document(
        id="academic_paper",
        text="""
# Abstract

Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for enhancing language models with external knowledge. This paper presents a comprehensive evaluation of chunking strategies.

## Introduction

The effectiveness of RAG systems heavily depends on how documents are segmented into retrievable chunks. Traditional fixed-size chunking often breaks semantic boundaries, while content-aware approaches can preserve meaning.

### Background

Previous work has focused on simple chunking strategies. However, real-world documents contain complex structures including headers, paragraphs, lists, and code blocks.

## Methodology

We evaluate eight different chunking techniques:

1. Fixed-size chunking with overlap
2. Semantic similarity-based chunking  
3. Sentence window chunking
4. Content-aware boundary detection
5. Document-specific chunking
6. Recursive hierarchical chunking
7. Propositional chunking
8. Parent-document chunking

### Experimental Setup

Each technique was evaluated on documents of varying complexity and structure. Metrics include chunk coherence, boundary preservation, and retrieval effectiveness.

## Results

Our experiments show that content-aware and recursive chunking methods significantly outperform fixed-size approaches in maintaining semantic coherence.

## Conclusion

The choice of chunking strategy should depend on document type and downstream task requirements. No single approach works best for all scenarios.
        """,
        meta={"type": "academic", "structure": "hierarchical"}
    ),
    
    Document(
        id="technical_documentation",
        text="""
# API Documentation

## Authentication

All API requests require authentication using an API key. Include the key in the Authorization header:

```
Authorization: Bearer your-api-key
```

## Endpoints

### GET /documents
Retrieve a list of documents.

**Parameters:**
- limit (optional): Number of documents to return (default: 10)
- offset (optional): Number of documents to skip (default: 0)

**Response:**
```json
{
  "documents": [...],
  "total": 100,
  "limit": 10,
  "offset": 0
}
```

### POST /documents
Create a new document.

**Request Body:**
```json
{
  "title": "Document Title",
  "content": "Document content here...",
  "metadata": {
    "author": "John Doe",
    "tags": ["important", "draft"]
  }
}
```

**Response:**
```json
{
  "id": "doc-123",
  "title": "Document Title",
  "created_at": "2024-01-15T10:30:00Z"
}
```

### Error Handling

The API returns standard HTTP status codes:

- 200: Success
- 400: Bad Request
- 401: Unauthorized
- 404: Not Found
- 500: Internal Server Error

Error responses include a descriptive message:
```json
{
  "error": "Invalid API key",
  "code": "UNAUTHORIZED"
}
```
        """,
        meta={"type": "technical", "structure": "api_docs"}
    ),
    
    Document(
        id="narrative_text",
        text="""
The sun was setting over the mountains as Sarah approached the old cabin. She had been walking for hours through the dense forest, following the map her grandfather had left her. The path was overgrown with brambles and fallen logs, making progress slow and difficult.

As she reached the clearing, memories flooded back from her childhood visits. The wooden structure looked exactly as she remembered it - weathered gray boards, a stone chimney, and the distinctive red door that her grandmother had painted years ago.

Inside, dust motes danced in the fading light that streamed through the windows. The furniture was covered in white sheets, giving the room a ghostly appearance. Sarah carefully removed the sheet from an old writing desk and discovered a leather-bound journal.

The journal contained her grandmother's handwriting, documenting daily life in the mountains. Each entry revealed small details about the changing seasons, wildlife observations, and philosophical reflections on the simple life they had chosen.

Sarah spent the evening reading by candlelight, feeling connected to the generations who had found solace in this remote place. The journal's final entry spoke of hidden treasures - not gold or jewels, but the intangible wealth of peace, solitude, and harmony with nature.

When morning came, she understood why her grandfather had wanted her to find this place. It wasn't about the cabin itself, but about discovering the values and wisdom that had shaped her family's character.
        """,
        meta={"type": "narrative", "structure": "story"}
    ),
    
    Document(
        id="scientific_data",
        text="""
Experiment Report: Protein Folding Analysis

Date: 2024-01-15
Researcher: Dr. Emily Chen
Lab: Computational Biology Unit

Objective: Analyze the folding patterns of three protein variants under different temperature conditions.

Materials and Methods:
- Protein samples: Wild-type, Mutant A (T45A), Mutant B (R67Q)
- Temperature range: 20°C to 80°C (increments of 5°C)
- Measurement technique: Circular dichroism spectroscopy
- Buffer conditions: 50mM phosphate buffer, pH 7.4

Experimental Procedure:
1. Prepare protein solutions at 0.1 mg/mL concentration
2. Equilibrate samples at target temperature for 10 minutes
3. Record CD spectra from 190-260 nm wavelength
4. Calculate secondary structure percentages
5. Determine melting temperature (Tm) values

Results:
Wild-type protein maintained stable secondary structure up to 65°C with Tm = 68.2°C.
Mutant A showed reduced thermal stability with Tm = 52.1°C.
Mutant B exhibited improved stability with Tm = 71.8°C.

Secondary structure analysis revealed:
- Wild-type: 45% α-helix, 20% β-sheet, 35% random coil
- Mutant A: 38% α-helix, 25% β-sheet, 37% random coil  
- Mutant B: 52% α-helix, 18% β-sheet, 30% random coil

Statistical Analysis:
Data represents mean ± standard deviation of triplicate measurements.
ANOVA analysis showed significant differences between variants (p < 0.001).
Post-hoc Tukey tests confirmed all pairwise comparisons were significant.

Discussion:
The T45A mutation destabilized the protein structure, likely due to loss of hydrophobic interactions in the core region. Conversely, the R67Q substitution enhanced stability by eliminating unfavorable electrostatic repulsions.

Conclusions:
1. Amino acid composition significantly affects protein thermal stability
2. Computational predictions aligned with experimental observations
3. Future work should investigate dynamic behavior using NMR spectroscopy
        """,
        meta={"type": "scientific", "structure": "report"}
    )
]


def main():
    """Run comprehensive chunking benchmark."""
    print("🔬 RAGLib Chunking Techniques Comprehensive Benchmark")
    print("=" * 60)
    
    # Get all chunking techniques
    chunking_techniques = TechniqueRegistry.list_by_category("chunking")
    
    if not chunking_techniques:
        print("❌ No chunking techniques found!")
        return 1
    
    print(f"📊 Testing {len(chunking_techniques)} chunking techniques")
    print(f"📝 On {len(BENCHMARK_DOCUMENTS)} different document types")
    print()
    
    # Run benchmark on each technique
    results = {}
    
    for technique_name, technique_class in chunking_techniques.items():
        print(f"🔍 Testing: {technique_name}")
        
        try:
            # Test with default parameters
            technique = technique_class()
            
            technique_results = {
                "name": technique_name,
                "class": technique_class.__name__,
                "meta": {
                    "description": technique.meta.description,
                    "version": technique.meta.version
                },
                "document_results": {}
            }
            
            # Test on each document type
            for doc in BENCHMARK_DOCUMENTS:
                print(f"  📄 Processing: {doc.id}")
                
                start_time = time.time()
                result = technique.apply(doc)
                end_time = time.time()
                
                if result.success:
                    chunks = result.payload.get("chunks", [])
                    
                    # Calculate chunk statistics
                    chunk_lengths = [len(chunk.text) for chunk in chunks]
                    avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
                    
                    # Analyze chunk boundaries
                    boundary_analysis = analyze_chunk_boundaries(chunks, doc.text)
                    
                    doc_result = {
                        "success": True,
                        "num_chunks": len(chunks),
                        "avg_chunk_length": avg_length,
                        "min_chunk_length": min(chunk_lengths) if chunk_lengths else 0,
                        "max_chunk_length": max(chunk_lengths) if chunk_lengths else 0,
                        "processing_time": end_time - start_time,
                        "coverage_ratio": calculate_coverage_ratio(chunks, doc.text),
                        "boundary_analysis": boundary_analysis,
                        "chunks_sample": [
                            {
                                "id": chunk.id,
                                "length": len(chunk.text),
                                "preview": chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
                            }
                            for chunk in chunks[:3]  # Show first 3 chunks
                        ]
                    }
                else:
                    doc_result = {
                        "success": False,
                        "error": result.error,
                        "processing_time": end_time - start_time
                    }
                
                technique_results["document_results"][doc.id] = doc_result
            
            results[technique_name] = technique_results
            print(f"  ✅ Completed: {technique_name}")
            
        except Exception as e:
            print(f"  ❌ Failed: {technique_name} - {e}")
            results[technique_name] = {
                "name": technique_name,
                "error": str(e)
            }
        
        print()
    
    # Display summary results
    display_benchmark_summary(results)
    
    # Save detailed results
    output_file = Path(__file__).parent / "chunking_benchmark_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Detailed results saved to: {output_file}")
    
    return 0


def analyze_chunk_boundaries(chunks, original_text):
    """Analyze how well chunks respect natural text boundaries."""
    if not chunks:
        return {"sentence_breaks": 0, "paragraph_breaks": 0, "section_breaks": 0}
    
    sentence_boundaries = 0
    paragraph_boundaries = 0  
    section_boundaries = 0
    
    for chunk in chunks:
        chunk_text = chunk.text.strip()
        
        # Check if chunk ends with sentence boundary
        if chunk_text.endswith(('.', '!', '?')):
            sentence_boundaries += 1
        
        # Check if chunk ends with paragraph boundary  
        if chunk_text.endswith('\n\n') or '\n\n' in chunk_text[-10:]:
            paragraph_boundaries += 1
            
        # Check if chunk ends with section boundary
        if any(marker in chunk_text[-20:] for marker in ['##', '###', '---']):
            section_boundaries += 1
    
    total_chunks = len(chunks)
    
    return {
        "sentence_breaks": sentence_boundaries / total_chunks if total_chunks > 0 else 0,
        "paragraph_breaks": paragraph_boundaries / total_chunks if total_chunks > 0 else 0,
        "section_breaks": section_boundaries / total_chunks if total_chunks > 0 else 0
    }


def calculate_coverage_ratio(chunks, original_text):
    """Calculate what percentage of original text is covered by chunks."""
    if not chunks:
        return 0.0
    
    total_chunk_chars = sum(len(chunk.text) for chunk in chunks)
    original_chars = len(original_text)
    
    return total_chunk_chars / original_chars if original_chars > 0 else 0.0


def display_benchmark_summary(results):
    """Display a summary of benchmark results."""
    print("📊 BENCHMARK SUMMARY")
    print("=" * 60)
    
    for technique_name, technique_results in results.items():
        if "error" in technique_results:
            print(f"❌ {technique_name}: {technique_results['error']}")
            continue
            
        print(f"\n🔧 {technique_name}")
        print(f"   Description: {technique_results['meta']['description']}")
        
        # Calculate averages across documents
        doc_results = technique_results["document_results"]
        successful_results = [r for r in doc_results.values() if r.get("success", False)]
        
        if successful_results:
            avg_chunks = sum(r["num_chunks"] for r in successful_results) / len(successful_results)
            avg_length = sum(r["avg_chunk_length"] for r in successful_results) / len(successful_results)
            avg_time = sum(r["processing_time"] for r in successful_results) / len(successful_results)
            avg_coverage = sum(r["coverage_ratio"] for r in successful_results) / len(successful_results)
            
            print(f"   📈 Avg chunks per document: {avg_chunks:.1f}")
            print(f"   📏 Avg chunk length: {avg_length:.0f} chars")
            print(f"   ⏱️  Avg processing time: {avg_time:.3f}s")
            print(f"   📋 Avg coverage ratio: {avg_coverage:.2f}")
            
            # Document-specific performance
            for doc_id, doc_result in doc_results.items():
                if doc_result.get("success", False):
                    print(f"      📄 {doc_id}: {doc_result['num_chunks']} chunks, "
                          f"{doc_result['avg_chunk_length']:.0f} avg chars")
        else:
            print("   ❌ No successful results")


if __name__ == "__main__":
    sys.exit(main())