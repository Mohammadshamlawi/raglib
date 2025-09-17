#!/usr/bin/env python3
"""
RAGLib Documentation and Benchmarking Auto-Updater

This script automatically updates all documentation, benchmarking files, and website
content when new techniques are added to RAGLib. It ensures consistency across
all documentation and keeps everything in sync.

Usage:
    python scripts/auto_update_docs.py [--dry-run] [--verbose]
    
Features:
    - Auto-discovers all registered techniques
    - Updates documentation files
    - Regenerates technique catalogs
    - Updates README and getting started guides
    - Creates/updates benchmarking examples
    - Builds documentation website
    - Validates all changes
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import subprocess
import time
from dataclasses import dataclass

# Add raglib to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from raglib.registry import TechniqueRegistry
    from raglib.core import TechniqueMeta
    # Import all techniques to ensure registration
    import raglib.techniques
except ImportError as e:
    print(f"❌ Failed to import raglib: {e}")
    print("Make sure you're running from the project root and raglib is installed")
    sys.exit(1)


@dataclass
class UpdateConfig:
    """Configuration for the documentation update process."""
    project_root: Path
    docs_dir: Path
    examples_dir: Path
    tools_dir: Path
    dry_run: bool = False
    verbose: bool = False
    force_rebuild: bool = False


class DocumentationUpdater:
    """Main class for updating RAGLib documentation and benchmarking files."""
    
    def __init__(self, config: UpdateConfig):
        self.config = config
        self.techniques_data = None
        self.changes_made = []
        
    def run_full_update(self) -> bool:
        """Run the complete documentation update process."""
        try:
            print("🚀 RAGLib Documentation Auto-Updater")
            print("=" * 50)
            
            # Step 1: Discover and analyze techniques
            self.log("📊 Discovering registered techniques...")
            self.techniques_data = self._discover_techniques()
            self._print_technique_summary()
            
            # Step 2: Update core documentation files
            self.log("📝 Updating core documentation...")
            self._update_techniques_md()
            self._update_getting_started_md()
            self._update_readme_md()
            
            # Step 3: Generate auto-generated documentation
            self.log("🔄 Regenerating auto-generated content...")
            self._regenerate_techniques_index()
            
            # Step 4: Update/create benchmarking tools
            self.log("⚡ Updating benchmarking tools...")
            self._update_benchmarking_examples()
            
            # Step 5: Build documentation website
            self.log("🌐 Building documentation website...")
            self._build_documentation_site()
            
            # Step 6: Validate changes
            self.log("✅ Validating changes...")
            validation_success = self._validate_changes()
            
            # Summary
            self._print_update_summary()
            
            return validation_success
            
        except Exception as e:
            print(f"❌ Update failed: {e}")
            if self.config.verbose:
                import traceback
                traceback.print_exc()
            return False
    
    def _discover_techniques(self) -> Dict[str, Any]:
        """Discover all registered techniques and organize by category."""
        techniques = TechniqueRegistry.list()
        
        if not techniques:
            raise RuntimeError("No techniques found in registry")
        
        # Organize by category
        categories = {}
        technique_details = {}
        
        for name, technique_class in techniques.items():
            meta = technique_class.meta
            if isinstance(meta, TechniqueMeta):
                category = meta.category
                description = meta.description
                version = getattr(meta, 'version', '1.0.0')  # Default version
                dependencies = getattr(meta, 'dependencies', [])
            else:
                # Handle legacy metadata
                category = getattr(meta, 'category', 'unknown')
                description = getattr(meta, 'description', 'No description available')
                version = getattr(meta, 'version', '1.0.0')
                dependencies = getattr(meta, 'dependencies', [])
            
            if category not in categories:
                categories[category] = []
            
            technique_info = {
                'name': name,
                'class_name': technique_class.__name__,
                'category': category,
                'description': description,
                'version': version,
                'dependencies': dependencies,
                'module': technique_class.__module__,
                'class_obj': technique_class
            }
            
            categories[category].append(technique_info)
            technique_details[name] = technique_info
        
        return {
            'by_category': categories,
            'by_name': technique_details,
            'total_count': len(techniques),
            'category_count': len(categories),
            'categories': list(categories.keys())
        }
    
    def _print_technique_summary(self):
        """Print a summary of discovered techniques."""
        data = self.techniques_data
        
        print(f"   📈 Found {data['total_count']} techniques in {data['category_count']} categories")
        
        for category, techniques in data['by_category'].items():
            print(f"   📂 {category}: {len(techniques)} techniques")
            if self.config.verbose:
                for tech in techniques:
                    print(f"      - {tech['name']} ({tech['class_name']})")
        print()
    
    def _update_techniques_md(self):
        """Update docs/techniques.md with current technique information."""
        techniques_file = self.config.docs_dir / "techniques.md"
        
        if not techniques_file.exists():
            self.log(f"   ⚠️  {techniques_file} not found, skipping...")
            return
        
        content = techniques_file.read_text(encoding='utf-8')
        
        # Update the chunking techniques section
        chunking_techniques = self.techniques_data['by_category'].get('chunking', [])
        if chunking_techniques:
            chunking_section = self._generate_chunking_section(chunking_techniques)
            
            # Find and replace the chunking techniques section
            pattern = r'(### Chunking Techniques\s*\n)(.*?)(\n### [^#]|\n## [^#]|\n*$)'
            
            if re.search(pattern, content, re.DOTALL):
                new_content = re.sub(
                    pattern,
                    r'\1' + chunking_section + r'\3',
                    content,
                    flags=re.DOTALL
                )
                
                if new_content != content:
                    if not self.config.dry_run:
                        techniques_file.write_text(new_content, encoding='utf-8')
                    self.changes_made.append(f"Updated chunking techniques in {techniques_file.name}")
                    self.log(f"   ✅ Updated chunking techniques section")
                else:
                    self.log(f"   ℹ️  Chunking techniques section already up to date")
            else:
                self.log(f"   ⚠️  Could not find chunking techniques section to update")
    
    def _generate_chunking_section(self, chunking_techniques: List[Dict]) -> str:
        """Generate the chunking techniques section content."""
        lines = []
        
        for tech in sorted(chunking_techniques, key=lambda x: x['name']):
            # Get parameter information by inspecting the class
            params = self._get_technique_parameters(tech['class_obj'])
            
            lines.append(f"#### {tech['class_name']}")
            lines.append(f"- **Description**: {tech['description']}")
            lines.append(f"- **Category**: {tech['category']}")
            
            deps_str = ', '.join(tech['dependencies']) if tech['dependencies'] else 'None'
            lines.append(f"- **Dependencies**: {deps_str}")
            
            if params:
                param_str = ', '.join(f"`{p}`" for p in params)
                lines.append(f"- **Parameters**: {param_str}")
            
            lines.append("")
        
        return '\n'.join(lines)
    
    def _get_technique_parameters(self, technique_class) -> List[str]:
        """Extract parameter names from technique constructor."""
        try:
            import inspect
            sig = inspect.signature(technique_class.__init__)
            params = []
            
            for name, param in sig.parameters.items():
                if name != 'self':
                    params.append(name)
            
            return params
        except Exception:
            return []
    
    def _update_getting_started_md(self):
        """Update docs/getting_started.md with examples of new techniques."""
        getting_started_file = self.config.docs_dir / "getting_started.md"
        
        if not getting_started_file.exists():
            self.log(f"   ⚠️  {getting_started_file} not found, skipping...")
            return
        
        content = getting_started_file.read_text(encoding='utf-8')
        
        # Check if advanced chunking section exists
        if "### 2. Advanced Chunking Techniques" in content:
            self.log(f"   ℹ️  Advanced chunking section already exists")
        else:
            # Add advanced chunking section after basic example
            chunking_section = self._generate_advanced_chunking_section()
            
            # Find insertion point after basic example
            basic_pattern = r'(```\n\n)(### 2\. Using the CLI)'
            if re.search(basic_pattern, content):
                new_content = re.sub(
                    basic_pattern,
                    r'\1' + chunking_section + '\n\n' + r'\2',
                    content
                )
                
                if not self.config.dry_run:
                    getting_started_file.write_text(new_content, encoding='utf-8')
                self.changes_made.append(f"Added advanced chunking section to {getting_started_file.name}")
                self.log(f"   ✅ Added advanced chunking techniques section")
    
    def _generate_advanced_chunking_section(self) -> str:
        """Generate the advanced chunking techniques section."""
        chunking_techniques = self.techniques_data['by_category'].get('chunking', [])
        
        # Filter to get the new/advanced techniques
        advanced_techniques = [
            tech for tech in chunking_techniques
            if tech['name'] in ['content_aware_chunker', 'recursive_chunker', 
                              'document_specific_chunker', 'propositional_chunker',
                              'parent_document_chunker']
        ]
        
        if not advanced_techniques:
            return ""
        
        section = '''### 2. Advanced Chunking Techniques

RAGLib includes several sophisticated chunking techniques optimized for different document types:

```python
from raglib.techniques import (
    ContentAwareChunker,
    RecursiveChunker,
    DocumentSpecificChunker,
    PropositionalChunker,
    ParentDocumentChunker
)
from raglib.schemas import Document

# Academic paper with hierarchical structure
document = Document(
    id="research_paper",
    text="""
# Abstract

This paper presents a novel approach to information retrieval.

## Introduction

Information retrieval has evolved significantly with neural networks.

### Background

Previous work focused on traditional approaches.

## Methodology

We propose a hybrid approach combining neural and symbolic methods.
    """,
    meta={"type": "academic", "domain": "computer_science"}
)

# Content-aware chunking respects document structure
content_chunker = ContentAwareChunker(max_chunk_size=300, overlap=50)
result = content_chunker.apply(document)
content_chunks = result.payload["chunks"]

print(f"Content-aware chunking: {len(content_chunks)} chunks")

# Recursive chunking with hierarchical splitting
recursive_chunker = RecursiveChunker(
    chunk_size=250,
    overlap=30,
    separators=["\n\n", "\n", ". ", " "]
)
result = recursive_chunker.apply(document)
recursive_chunks = result.payload["chunks"]

print(f"Recursive chunking: {len(recursive_chunks)} chunks")

# Compare different approaches
techniques = [
    ("Content-Aware", ContentAwareChunker()),
    ("Recursive", RecursiveChunker()),
    ("Document-Specific", DocumentSpecificChunker()),
    ("Propositional", PropositionalChunker())
]

for name, chunker in techniques:
    result = chunker.apply(document)
    if result.success:
        chunks = result.payload["chunks"]
        print(f"{name}: {len(chunks)} chunks")
```

### 3. Comparing Chunking Strategies

```python
from raglib.registry import TechniqueRegistry

# Get all chunking techniques
chunking_techniques = TechniqueRegistry.find_by_category("chunking")

# Test document
test_doc = Document(
    id="test",
    text="Your test document here...",
    meta={"source": "test"}
)

# Compare different chunking approaches
print("Chunking Strategy Comparison:")
print("-" * 40)

for name, technique_class in chunking_techniques.items():
    try:
        chunker = technique_class()
        result = chunker.apply(test_doc)
        
        if result.success:
            chunks = result.payload["chunks"]
            avg_length = sum(len(c.text) for c in chunks) / len(chunks)
            
            print(f"{name}:")
            print(f"  - Chunks: {len(chunks)}")
            print(f"  - Avg length: {avg_length:.1f} chars")
            
    except Exception as e:
        print(f"{name}: Error - {e}")
```'''
        
        return section
    
    def _update_readme_md(self):
        """Update README.md with current technique listings."""
        readme_file = self.config.project_root / "README.md"
        
        if not readme_file.exists():
            self.log(f"   ⚠️  {readme_file} not found, skipping...")
            return
        
        content = readme_file.read_text(encoding='utf-8')
        
        # Update document processing section
        chunking_techniques = self.techniques_data['by_category'].get('chunking', [])
        if chunking_techniques:
            new_section = self._generate_readme_chunking_section(chunking_techniques)
            
            # Find and replace the document processing section
            pattern = r'(### 🔨 Document Processing\s*\n)(- \*\*.*?\n)*'
            
            if re.search(pattern, content):
                new_content = re.sub(
                    pattern,
                    r'\1' + new_section,
                    content
                )
                
                if new_content != content:
                    if not self.config.dry_run:
                        readme_file.write_text(new_content, encoding='utf-8')
                    self.changes_made.append(f"Updated document processing section in {readme_file.name}")
                    self.log(f"   ✅ Updated README document processing section")
                else:
                    self.log(f"   ℹ️  README already up to date")
    
    def _generate_readme_chunking_section(self, chunking_techniques: List[Dict]) -> str:
        """Generate the README chunking section."""
        lines = []
        
        for tech in sorted(chunking_techniques, key=lambda x: x['name']):
            class_name = tech['class_name']
            description = tech['description']
            
            # Format for README
            if 'fixed' in tech['name'].lower():
                lines.append(f"- **Fixed Size Chunking**: {description}")
            elif 'semantic' in tech['name'].lower():
                lines.append(f"- **Semantic Chunking**: {description}")
            elif 'sentence' in tech['name'].lower():
                lines.append(f"- **Sentence Window Chunking**: {description}")
            elif 'content' in tech['name'].lower():
                lines.append(f"- **Content-Aware Chunking**: {description}")
            elif 'document' in tech['name'].lower():
                lines.append(f"- **Document-Specific Chunking**: {description}")
            elif 'recursive' in tech['name'].lower():
                lines.append(f"- **Recursive Chunking**: {description}")
            elif 'propositional' in tech['name'].lower():
                lines.append(f"- **Propositional Chunking**: {description}")
            elif 'parent' in tech['name'].lower():
                lines.append(f"- **Parent-Document Chunking**: {description}")
            else:
                # Generic format
                formatted_name = class_name.replace('Chunker', ' Chunking')
                lines.append(f"- **{formatted_name}**: {description}")
        
        return '\n'.join(lines) + '\n'
    
    def _regenerate_techniques_index(self):
        """Regenerate the auto-generated techniques index."""
        tools_script = self.config.tools_dir / "generate_techniques_index.py"
        
        if not tools_script.exists():
            self.log(f"   ⚠️  {tools_script} not found, skipping...")
            return
        
        try:
            if not self.config.dry_run:
                result = subprocess.run([
                    sys.executable, str(tools_script)
                ], capture_output=True, text=True, cwd=self.config.project_root)
                
                if result.returncode == 0:
                    self.changes_made.append("Regenerated techniques index")
                    self.log(f"   ✅ Regenerated techniques index")
                    if self.config.verbose:
                        self.log(f"      Output: {result.stdout.strip()}")
                else:
                    self.log(f"   ❌ Failed to regenerate techniques index: {result.stderr}")
            else:
                self.log(f"   🔍 Would regenerate techniques index")
                
        except Exception as e:
            self.log(f"   ❌ Error regenerating techniques index: {e}")
    
    def _update_benchmarking_examples(self):
        """Update or create benchmarking example files."""
        # Update chunking benchmark
        self._create_chunking_benchmark()
        
        # Update chunking showcase
        self._create_chunking_showcase()
    
    def _create_chunking_benchmark(self):
        """Create or update the comprehensive chunking benchmark."""
        benchmark_file = self.config.examples_dir / "chunking_benchmark_auto.py"
        
        chunking_techniques = self.techniques_data['by_category'].get('chunking', [])
        if not chunking_techniques:
            self.log(f"   ⚠️  No chunking techniques found, skipping benchmark creation")
            return
        
        benchmark_content = self._generate_benchmark_script(chunking_techniques)
        
        if not self.config.dry_run:
            benchmark_file.write_text(benchmark_content, encoding='utf-8')
        
        self.changes_made.append(f"Created/updated {benchmark_file.name}")
        self.log(f"   ✅ Created chunking benchmark script")
    
    def _generate_benchmark_script(self, chunking_techniques: List[Dict]) -> str:
        """Generate the benchmark script content."""
        # Get technique names for imports
        technique_classes = [tech['class_name'] for tech in chunking_techniques]
        
        script = f'''#!/usr/bin/env python3
"""
Auto-generated Chunking Benchmark Script

This script was automatically generated by the RAGLib documentation updater.
It benchmarks all {len(chunking_techniques)} available chunking techniques.

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""

import json
import sys
import time
from pathlib import Path

# Add raglib to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from raglib.techniques import (
        {', '.join(technique_classes)}
    )
    from raglib.schemas import Document
    from raglib.registry import TechniqueRegistry
except ImportError as e:
    print(f"Failed to import raglib: {{e}}")
    sys.exit(1)


def main():
    """Run comprehensive benchmark on all chunking techniques."""
    print("🔬 Auto-Generated Chunking Benchmark")
    print("=" * 50)
    
    # Get all chunking techniques
    techniques = TechniqueRegistry.find_by_category("chunking")
    print(f"📊 Testing {{len(techniques)}} chunking techniques")
    
    # Test document
    test_doc = Document(
        id="test_document",
        text="""
# Introduction

This is a test document with various structures. It contains multiple paragraphs,
different section headings, and various text patterns to test chunking strategies.

## Section 1

Here we have some content that spans multiple sentences. The goal is to see
how different chunking techniques handle document structure and boundaries.

### Subsection 1.1

More detailed content with technical information. This section contains
specific details that should ideally stay together for context preservation.

## Section 2

Different content with various formatting. Lists, code blocks, and other
structural elements challenge chunking algorithms differently.

- Item 1: First list item
- Item 2: Second list item  
- Item 3: Third list item

### Code Example

```python
def example_function():
    return "Hello, World!"
```

## Conclusion

This document provides a comprehensive test case for evaluating chunking
strategies across different text structures and patterns.
        """,
        meta={{"type": "test", "structure": "hierarchical"}}
    )
    
    results = {{}}
    
    for name, technique_class in techniques.items():
        print(f"\\n🔍 Testing: {{name}}")
        
        try:
            # Initialize with default parameters
            technique = technique_class()
            
            start_time = time.time()
            result = technique.apply(test_doc)
            end_time = time.time()
            
            if result.success:
                # Handle different result formats
                if name == "parent_document_chunker":
                    payload = result.payload
                    child_chunks = payload.get("child_chunks", [])
                    parent_chunks = payload.get("parent_chunks", [])
                    
                    results[name] = {{
                        "success": True,
                        "child_chunks": len(child_chunks),
                        "parent_chunks": len(parent_chunks),
                        "processing_time": end_time - start_time,
                        "technique_class": technique_class.__name__
                    }}
                    
                    print(f"   ✅ {{len(child_chunks)}} child chunks, {{len(parent_chunks)}} parent chunks")
                else:
                    chunks = result.payload.get("chunks", [])
                    avg_length = sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0
                    
                    results[name] = {{
                        "success": True,
                        "num_chunks": len(chunks),
                        "avg_chunk_length": avg_length,
                        "processing_time": end_time - start_time,
                        "technique_class": technique_class.__name__
                    }}
                    
                    print(f"   ✅ {{len(chunks)}} chunks, avg length: {{avg_length:.0f}} chars")
            else:
                results[name] = {{
                    "success": False,
                    "error": result.error,
                    "processing_time": end_time - start_time
                }}
                print(f"   ❌ Failed: {{result.error}}")
                
        except Exception as e:
            results[name] = {{
                "success": False,
                "error": str(e),
                "processing_time": 0
            }}
            print(f"   💥 Exception: {{e}}")
    
    # Save results
    output_file = Path(__file__).parent / "chunking_benchmark_results_auto.json"
    with open(output_file, 'w') as f:
        json.dump({{
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total_techniques": len(techniques),
            "results": results
        }}, f, indent=2)
    
    print(f"\\n💾 Results saved to: {{output_file}}")
    
    # Summary
    successful = sum(1 for r in results.values() if r.get("success", False))
    print(f"\\n📊 Summary: {{successful}}/{{len(results)}} techniques successful")


if __name__ == "__main__":
    main()
'''
        
        return script
    
    def _create_chunking_showcase(self):
        """Create or update the chunking showcase script."""
        showcase_file = self.config.examples_dir / "chunking_showcase_auto.py"
        
        chunking_techniques = self.techniques_data['by_category'].get('chunking', [])
        if not chunking_techniques:
            return
        
        showcase_content = self._generate_showcase_script(chunking_techniques)
        
        if not self.config.dry_run:
            showcase_file.write_text(showcase_content, encoding='utf-8')
        
        self.changes_made.append(f"Created/updated {showcase_file.name}")
        self.log(f"   ✅ Created chunking showcase script")
    
    def _generate_showcase_script(self, chunking_techniques: List[Dict]) -> str:
        """Generate the showcase script content."""
        technique_classes = [tech['class_name'] for tech in chunking_techniques]
        
        script = f'''#!/usr/bin/env python3
"""
Auto-generated Chunking Showcase Script

This script demonstrates all available chunking techniques with examples.
Generated automatically by the RAGLib documentation updater.

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""

import sys
from pathlib import Path

# Add raglib to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from raglib.techniques import (
        {', '.join(technique_classes)}
    )
    from raglib.schemas import Document
except ImportError as e:
    print(f"Failed to import raglib: {{e}}")
    sys.exit(1)


def main():
    """Demonstrate all chunking techniques."""
    print("🔬 RAGLib Chunking Techniques Showcase")
    print("=" * 50)
    
    # Test document
    document = Document(
        id="showcase_doc",
        text="""
# Machine Learning Fundamentals

## Introduction

Machine learning is a subset of artificial intelligence that focuses on 
algorithms that can learn and improve from experience.

### Supervised Learning

In supervised learning, algorithms learn from labeled training data.
Common examples include:

- Classification problems
- Regression analysis
- Pattern recognition

### Unsupervised Learning

Unsupervised learning finds hidden patterns in data without labels.
This includes clustering and dimensionality reduction techniques.

## Applications

Machine learning has numerous real-world applications:

1. Computer vision and image recognition
2. Natural language processing
3. Recommendation systems
4. Autonomous vehicles

## Conclusion

Understanding these fundamentals provides a solid foundation for
exploring more advanced machine learning concepts.
        """,
        meta={{"type": "educational", "topic": "machine_learning"}}
    )
    
    # Initialize all techniques
    techniques = [
'''
        
        # Add technique initializations
        for tech in chunking_techniques:
            class_name = tech['class_name']
            if 'parent' in tech['name'].lower():
                script += f'        ("{class_name}", {class_name}(child_chunk_size=100, parent_chunk_size=300)),\n'
            elif 'content' in tech['name'].lower():
                script += f'        ("{class_name}", {class_name}(max_chunk_size=200)),\n'
            elif 'semantic' in tech['name'].lower():
                script += f'        ("{class_name}", {class_name}(chunk_size=200)),\n'
            elif 'sentence' in tech['name'].lower():
                script += f'        ("{class_name}", {class_name}(window_size=3)),\n'
            else:
                script += f'        ("{class_name}", {class_name}(chunk_size=200)),\n'
        
        script += '''    ]
    
    print(f"Testing {len(techniques)} chunking techniques:")
    print()
    
    for name, technique in techniques:
        try:
            result = technique.apply(document)
            
            if result.success:
                if "Parent" in name:
                    payload = result.payload
                    child_count = len(payload.get("child_chunks", []))
                    parent_count = len(payload.get("parent_chunks", []))
                    print(f"✅ {name}: {child_count} child, {parent_count} parent chunks")
                else:
                    chunks = result.payload.get("chunks", [])
                    if chunks:
                        avg_length = sum(len(c.text) for c in chunks) / len(chunks)
                        print(f"✅ {name}: {len(chunks)} chunks (avg: {avg_length:.0f} chars)")
                        
                        # Show first chunk preview
                        if self.config.verbose and chunks:
                            preview = chunks[0].text[:100].replace('\\n', ' ').strip()
                            print(f"   Preview: {preview}...")
                    else:
                        print(f"✅ {name}: No chunks created")
            else:
                print(f"❌ {name}: {result.error}")
                
        except Exception as e:
            print(f"💥 {name}: {e}")
    
    print("\\n🎉 Showcase complete!")


if __name__ == "__main__":
    main()
'''
        
        return script
    
    def _build_documentation_site(self):
        """Build the documentation website using mkdocs."""
        mkdocs_config = self.config.project_root / "mkdocs.yml"
        
        if not mkdocs_config.exists():
            self.log(f"   ⚠️  mkdocs.yml not found, skipping site build")
            return
        
        try:
            if not self.config.dry_run:
                result = subprocess.run([
                    "mkdocs", "build"
                ], capture_output=True, text=True, cwd=self.config.project_root)
                
                if result.returncode == 0:
                    self.changes_made.append("Built documentation website")
                    self.log(f"   ✅ Documentation website built successfully")
                    if self.config.verbose and result.stdout:
                        self.log(f"      {result.stdout.strip()}")
                else:
                    self.log(f"   ❌ Failed to build documentation: {result.stderr}")
            else:
                self.log(f"   🔍 Would build documentation website")
                
        except FileNotFoundError:
            self.log(f"   ⚠️  mkdocs not found, install with: pip install mkdocs-material")
        except Exception as e:
            self.log(f"   ❌ Error building documentation: {e}")
    
    def _validate_changes(self) -> bool:
        """Validate that all changes were successful."""
        if self.config.dry_run:
            self.log(f"   🔍 Dry run mode - no validation needed")
            return True
        
        validation_errors = []
        
        # Check that techniques index was generated
        techniques_generated = self.config.docs_dir / "techniques_generated.md"
        if not techniques_generated.exists():
            validation_errors.append("techniques_generated.md not found")
        
        # Check that documentation builds without errors
        site_dir = self.config.project_root / "site"
        if not site_dir.exists():
            validation_errors.append("Documentation site not built")
        
        # Run a quick test to ensure techniques are discoverable
        try:
            techniques = TechniqueRegistry.list()
            if not techniques:
                validation_errors.append("No techniques found in registry")
        except Exception as e:
            validation_errors.append(f"Registry error: {e}")
        
        if validation_errors:
            self.log(f"   ❌ Validation failed:")
            for error in validation_errors:
                self.log(f"      - {error}")
            return False
        else:
            self.log(f"   ✅ All validations passed")
            return True
    
    def _print_update_summary(self):
        """Print a summary of all changes made."""
        print("\\n📋 Update Summary")
        print("=" * 30)
        
        if not self.changes_made:
            print("   ℹ️  No changes were needed - everything up to date!")
        else:
            print(f"   ✅ Made {len(self.changes_made)} changes:")
            for change in self.changes_made:
                print(f"      - {change}")
        
        if self.config.dry_run:
            print("\\n   🔍 This was a dry run - no files were actually modified")
        
        print(f"\\n📊 Current state:")
        print(f"   - {self.techniques_data['total_count']} total techniques")
        print(f"   - {self.techniques_data['category_count']} categories")
        print(f"   - {len(self.techniques_data['by_category'].get('chunking', []))} chunking techniques")
    
    def log(self, message: str):
        """Log a message with optional verbose output."""
        if self.config.verbose or not message.startswith("   "):
            print(message)


def main():
    """Main entry point for the documentation updater."""
    parser = argparse.ArgumentParser(
        description="Auto-update RAGLib documentation and benchmarking files"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what would be changed without making modifications"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Show detailed output during processing"
    )
    parser.add_argument(
        "--force-rebuild", 
        action="store_true", 
        help="Force rebuild of all documentation even if up to date"
    )
    
    args = parser.parse_args()
    
    # Setup configuration
    project_root = Path(__file__).parent.parent
    config = UpdateConfig(
        project_root=project_root,
        docs_dir=project_root / "docs",
        examples_dir=project_root / "examples", 
        tools_dir=project_root / "tools",
        dry_run=args.dry_run,
        verbose=args.verbose,
        force_rebuild=args.force_rebuild
    )
    
    # Run the update
    updater = DocumentationUpdater(config)
    success = updater.run_full_update()
    
    if success:
        print("\\n🎉 Documentation update completed successfully!")
        sys.exit(0)
    else:
        print("\\n❌ Documentation update failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()