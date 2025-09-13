# RAGLib Documentation Automation

This document describes the documentation system for RAGLib, particularly the techniques index generation.

## Overview

RAGLib generates documentation for all registered techniques using the `tools/generate_techniques_index.py` script. This ensures that the documentation stays up-to-date with the available techniques in the codebase.

## How It Works

### 1. Technique Registration

Techniques are automatically discovered through the `TechniqueRegistry` system:

```python
@TechniqueRegistry.register
class MyTechnique(RAGTechnique):
    meta = TechniqueMeta(
        name="my_technique",
        description="My custom RAG technique", 
        category="retrieval",
        version="1.0.0",
        dependencies=["faiss", "transformers"]
    )
```

### 2. Index Generation

The `tools/generate_techniques_index.py` script:

1. Imports all technique modules to trigger registration
2. Queries the `TechniqueRegistry` for all registered techniques  
3. Groups techniques by category
4. Extracts metadata (name, description, version, dependencies)
5. Generates formatted Markdown with tables and usage examples
6. Writes the output to `docs/techniques_generated.md`

### 3. Documentation Integration

The generated file is included in the main documentation using MkDocs snippets:

```markdown
--8<-- "techniques_generated.md"
```

## Generated Content Format

The script generates content with the following structure:

### Header
- Total technique and category counts

### Technique Entries  
Each technique includes:
- **Name**: The registry key for the technique
- **Description**: From the TechniqueMeta
- **Property Table**: Version, class, module, dependencies
- **Usage Examples**: Extracted from docstrings (if available)

### Example Output

```markdown
#### semantic_chunker

**Semantic similarity-based chunking with configurable embedder**

| Property | Value |
|----------|-------|
| Version | `1.0.0` |
| Class | `SemanticChunker` |
| Module | `raglib.techniques.semantic_chunker` |
| Dependencies | None |

---
```

## Build Process Integration

### Generation

The documentation build process generates the techniques index:

**Linux/macOS (Make):**
```bash
make docs           # Runs docs-generate first
make docs-serve     # Runs docs-generate first  
```

**Windows (PowerShell):**
```powershell
.\build.ps1 docs           # Runs docs-generate first
.\build.ps1 docs-serve     # Runs docs-generate first
```

**Windows (Batch):**
```batch
.\build.bat docs           # Runs docs-generate first  
.\build.bat docs-serve     # Runs docs-generate first
```

### Manual Generation

To regenerate just the techniques index:

**Linux/macOS:**
```bash
make docs-generate
# or
python tools/generate_techniques_index.py
```

**Windows:**
```batch
.\build.bat docs-generate
# or  
python tools\generate_techniques_index.py
```

## Adding New Techniques

When you add a new technique:

1. **Create the technique class** with proper `@TechniqueRegistry.register` decorator
2. **Add TechniqueMeta** with complete metadata
3. **Import the module** in `raglib/techniques/__init__.py` 
4. **Regenerate the index** using the build scripts
5. **Rebuild documentation** to see the changes

### Example New Technique

```python
# raglib/techniques/my_new_technique.py

from raglib.core import RAGTechnique, TechniqueMeta, TechniqueRegistry
from raglib.schemas import RagResult

@TechniqueRegistry.register  
class MyNewTechnique(RAGTechnique):
    """My new RAG technique.
    
    This technique does something innovative.
    
    Example:
        technique = MyNewTechnique()
        result = technique.apply(query="test", corpus=docs)
    """
    
    meta = TechniqueMeta(
        name="my_new_technique",
        description="My innovative RAG technique",
        category="retrieval", 
        version="1.0.0",
        dependencies=["some-package"]
    )
    
    def apply(self, **kwargs) -> RagResult:
        # Implementation here
        pass
```

## Troubleshooting

### Common Issues

**Script fails to find techniques:**
- Ensure all technique modules are imported
- Check that `@TechniqueRegistry.register` decorators are present
- Verify `TechniqueMeta` objects are properly configured

**Generated content not appearing:**
- Check that MkDocs has the `pymdownx.snippets` extension enabled
- Verify the `--8<-- "techniques_generated.md"` inclusion syntax
- Ensure `docs/techniques_generated.md` exists and has content

**Build script not working:**
- On Windows, use `.\build.ps1` or `.\build.bat` (note the `.\`)
- Ensure Python virtual environment is activated
- Check that required dependencies (mkdocs, raglib) are installed

### Debug Mode

Run the script with Python directly to see detailed output:

```bash
python tools/generate_techniques_index.py
```

This will show:
- Number of techniques found
- Categories discovered  
- Any import errors
- File generation status

## Files Modified

The automation system involves these files:

- **`tools/generate_techniques_index.py`** - Main generation script
- **`docs/techniques_generated.md`** - Generated output
- **`docs/techniques.md`** - Main techniques page (includes content)
- **`Makefile`** - Linux/macOS build automation
- **`build.ps1`** - Windows PowerShell build automation  
- **`build.bat`** - Windows batch build automation
- **`mkdocs.yml`** - MkDocs configuration with snippets extension

## Continuous Integration

The automation integrates with CI/CD:

```yaml
# .github/workflows/docs.yml
- name: Generate techniques index
  run: python tools/generate_techniques_index.py
  
- name: Build documentation  
  run: mkdocs build --strict
```

This ensures documentation is always up-to-date in deployments.
