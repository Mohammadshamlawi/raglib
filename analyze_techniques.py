#!/usr/bin/env python3
"""Analyze current techniques against allowed list."""
import os
import re
from pathlib import Path


def parse_allowed_techniques(file_path):
    """Parse RAG_techniques.txt to get canonical technique names."""
    allowed = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('- '):
                # Remove the "- " prefix
                technique = line[2:].strip()
                allowed.add(technique)
    return allowed


def normalize_name(technique_name):
    """Convert technique name to potential filename."""
    # Handle parenthetical descriptions
    name = re.sub(r'\s*\([^)]+\)', '', technique_name)
    # Convert to lowercase and replace spaces/special chars with underscores
    name = re.sub(r'[^\w]+', '_', name.lower())
    # Remove leading/trailing underscores and collapse multiple underscores
    name = re.sub(r'_+', '_', name).strip('_')
    return name


def get_current_techniques():
    """Get current technique files in raglib/techniques/."""
    techniques_dir = Path("raglib/techniques")
    current = []
    for py_file in techniques_dir.glob("*.py"):
        if py_file.name != "__init__.py":
            current.append(py_file.stem)
    return current


def main():
    # Parse allowed techniques
    allowed_file = Path("../RAG_techniques.txt")
    if not allowed_file.exists():
        print(f"Error: {allowed_file} not found")
        return
    
    allowed = parse_allowed_techniques(allowed_file)
    print(f"Found {len(allowed)} allowed techniques:")
    for tech in sorted(allowed):
        print(f"  - {tech}")
    
    print()
    
    # Get current implementations
    current = get_current_techniques()
    print(f"Current technique files ({len(current)}):")
    for tech in sorted(current):
        print(f"  - {tech}.py")
    
    print()
    
    # Analyze mappings
    print("Analysis:")
    print("=========")
    
    # Create mapping from normalized names to original names
    allowed_normalized = {}
    for tech in allowed:
        norm = normalize_name(tech)
        if norm in allowed_normalized:
            print(f"WARNING: Normalized name collision: '{norm}' for both '{tech}' and '{allowed_normalized[norm]}'")
        allowed_normalized[norm] = tech
    
    # Check current files against allowed
    to_remove = []
    to_keep = []
    to_rename = []
    
    for current_file in current:
        if current_file in ['template_technique', 'demo_fixed_chunker', 'echo_technique', 'null_technique', 'dummy_dense']:
            # These are clearly test/demo files
            to_remove.append(current_file)
        elif current_file in allowed_normalized:
            to_keep.append((current_file, allowed_normalized[current_file]))
        else:
            # Try to find a match by fuzzy matching
            found_match = False
            for norm_name, orig_name in allowed_normalized.items():
                if current_file in norm_name or norm_name in current_file:
                    to_rename.append((current_file, norm_name, orig_name))
                    found_match = True
                    break
            if not found_match:
                to_remove.append(current_file)
    
    print(f"\nFiles to REMOVE ({len(to_remove)}):")
    for f in to_remove:
        print(f"  - {f}.py")
    
    print(f"\nFiles to KEEP ({len(to_keep)}):")
    for f, orig in to_keep:
        print(f"  - {f}.py -> '{orig}'")
    
    print(f"\nFiles to potentially RENAME ({len(to_rename)}):")
    for f, norm, orig in to_rename:
        print(f"  - {f}.py -> {norm}.py ('{orig}')")
    
    # Check for missing implementations
    missing = []
    implemented_normalized = set()
    for f, _ in to_keep:
        implemented_normalized.add(f)
    for f, norm, _ in to_rename:
        implemented_normalized.add(norm)
    
    for norm_name, orig_name in allowed_normalized.items():
        if norm_name not in implemented_normalized:
            missing.append((norm_name, orig_name))
    
    print(f"\nMISSING implementations ({len(missing)}):")
    for norm, orig in missing:
        print(f"  - {norm}.py for '{orig}'")


if __name__ == "__main__":
    main()