#!/usr/bin/env python3
"""Generate techniques index for documentation.

This script reads registered techniques from TechniqueRegistry and creates
a markdown file with the complete catalog of available techniques.
"""

import sys
from pathlib import Path

# Add raglib to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from raglib.core import TechniqueMeta
    from raglib.registry import TechniqueRegistry
except ImportError as e:
    print(f"Failed to import raglib: {e}")
    print("Make sure raglib is installed or run from the project root.")
    sys.exit(1)


def generate_techniques_index():
    """Generate the techniques index markdown file."""
    techniques = TechniqueRegistry.list()
    
    if not techniques:
        print("Warning: No techniques found in registry")
    
    # Group techniques by category
    categories = {}
    for name, technique_class in techniques.items():
        meta = technique_class.meta
        if isinstance(meta, TechniqueMeta):
            category = meta.category
            if category not in categories:
                categories[category] = []
            categories[category].append((name, technique_class))
        else:
            # Handle legacy metadata
            category = getattr(meta, 'category', 'unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append((name, technique_class))
    
    # Generate markdown content
    content = []
    
    # Add header
    content.append("# Techniques Index")
    content.append("")
    content.append("---")
    content.append("")
    
    if not categories:
        content.append("No techniques are currently registered in the registry.")
        content.append("")
        content.append("To add techniques, ensure they are properly registered with:")
        content.append("```python")
        content.append("@TechniqueRegistry.register")
        content.append("class YourTechnique(RAGTechnique):")
        content.append("    meta = TechniqueMeta(...)")
        content.append("```")
    else:
        # Add overview statistics
        total_techniques = len(techniques)
        total_categories = len(categories)
        content.append(f"**Total Techniques:** {total_techniques}")
        content.append(f"**Categories:** {total_categories}")
        content.append("")
        
        # Sort categories for consistent output
        category_order = [
            "chunking", "retrieval", "reranking",
            "generation", "orchestration"
        ]
        sorted_categories = []
        
        # Add known categories in order
        for cat in category_order:
            if cat in categories:
                sorted_categories.append(cat)
        
        # Add any unknown categories at the end
        for cat in sorted(categories.keys()):
            if cat not in sorted_categories:
                sorted_categories.append(cat)
        
        # Generate content for each category
        for category in sorted_categories:
            technique_list = categories[category]
            technique_list.sort(key=lambda x: x[0])  # Sort by name
            
            # Category header
            category_display = category.replace("_", " ").title()
            content.append(f"### {category_display}")
            content.append("")
            
            # Technique list
            for name, technique_class in technique_list:
                meta = technique_class.meta

                # Get description
                if isinstance(meta, TechniqueMeta):
                    description = meta.description
                    version = getattr(meta, 'version', '1.0.0')
                    dependencies = getattr(meta, 'dependencies', [])
                else:
                    description = getattr(
                        meta, 'description', 'No description available'
                    )
                    version = getattr(meta, 'version', '1.0.0')
                    dependencies = getattr(meta, 'dependencies', [])

                # Format technique entry
                content.append(f"#### {name}")
                content.append("")
                content.append(f"**{description}**")
                content.append("")

                # Create info table
                content.append("| Property | Value |")
                content.append("|----------|-------|")
                content.append(f"| Version | `{version}` |")
                content.append(f"| Class | `{technique_class.__name__}` |")
                content.append(f"| Module | `{technique_class.__module__}` |")

                if dependencies:
                    deps = ", ".join(f"`{dep}`" for dep in dependencies)
                    content.append(f"| Dependencies | {deps} |")
                else:
                    content.append("| Dependencies | None |")

                content.append("")

                # Add usage example if available
                doc = technique_class.__doc__
                if doc and "Example:" in doc:
                    # Extract example from docstring
                    lines = doc.split('\n')
                    in_example = False
                    example_lines = []
                    for line in lines:
                        if "Example:" in line:
                            in_example = True
                            continue
                        if in_example:
                            if line.strip() and not line.startswith("    "):
                                break
                            example_lines.append(line)

                    if example_lines:
                        content.append("**Usage Example:**")
                        content.append("```python")
                        content.extend(example_lines)
                        content.append("```")
                        content.append("")

                content.append("---")
                content.append("")    # Write to file
    docs_dir = project_root / "docs"
    output_file = docs_dir / "techniques_generated.md"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        print(f"Generated techniques index: {output_file}")
        print(f"Found {len(techniques)} techniques in {len(categories)} categories")
        return True
    except Exception as e:
        print(f"Failed to write techniques index: {e}")
        return False


def main():
    """Main entry point."""
    print("Generating techniques index...")
    
    # Import technique modules to ensure registration
    try:
        # Import all technique modules to trigger registration
        from pathlib import Path
        
        import raglib.techniques  # noqa: F401
        techniques_dir = Path(__file__).parent.parent / "raglib" / "techniques"
        
        if techniques_dir.exists():
            for technique_file in techniques_dir.glob("*.py"):
                if technique_file.name.startswith("__"):
                    continue
                module_name = f"raglib.techniques.{technique_file.stem}"
                try:
                    __import__(module_name)
                except ImportError as e:
                    print(f"Warning: Could not import {module_name}: {e}")
                    continue
    except ImportError as e:
        print(f"Warning: Could not import techniques: {e}")
    
    success = generate_techniques_index()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
