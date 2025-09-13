#!/usr/bin/env python3
"""CLI for raglib - entry point for command-line interface."""

import argparse
import subprocess
import sys
from pathlib import Path


def quick_start():
    """Run the quick start example."""
    examples_dir = Path(__file__).parent.parent / "examples"
    quick_start_script = examples_dir / "quick_start.py"
    
    if not quick_start_script.exists():
        print(f"Quick start script not found at {quick_start_script}")
        return 1
    
    try:
        subprocess.run([sys.executable, str(quick_start_script)], check=True)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Quick start failed: {e}")
        return 1


def run_example(example_name: str):
    """Run a specific example by name."""
    examples_dir = Path(__file__).parent.parent / "examples"
    example_script = examples_dir / f"{example_name}.py"
    
    if not example_script.exists():
        print(f"Example script not found at {example_script}")
        available = list(examples_dir.glob("*.py")) if examples_dir.exists() else []
        if available:
            print("Available examples:")
            for script in available:
                print(f"  - {script.stem}")
        return 1
    
    try:
        subprocess.run([sys.executable, str(example_script)], check=True)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Example '{example_name}' failed: {e}")
        return 1


def docs_build():
    """Build the documentation."""
    docs_dir = Path(__file__).parent.parent / "docs"
    
    if not docs_dir.exists():
        print(f"Documentation directory not found at {docs_dir}")
        return 1
    
    # First generate the techniques index
    tools_dir = Path(__file__).parent.parent / "tools"
    index_script = tools_dir / "generate_techniques_index.py"
    
    if index_script.exists():
        try:
            subprocess.run([sys.executable, str(index_script)], check=True)
            print("Generated techniques index")
        except subprocess.CalledProcessError as e:
            print(f"Failed to generate techniques index: {e}")
            return 1
    
    # Then build docs
    try:
        subprocess.run(["mkdocs", "build"], cwd=docs_dir, check=True)
        print("Documentation built successfully")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Documentation build failed: {e}")
        print("Make sure mkdocs is installed: pip install mkdocs mkdocs-material")
        return 1
    except FileNotFoundError:
        print("mkdocs not found. Install with: pip install mkdocs mkdocs-material")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="raglib-cli",
        description="Command-line interface for raglib",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Quick start command
    subparsers.add_parser("quick-start", help="Run the quick start example")
    
    # Run example command
    example_parser = subparsers.add_parser("run-example", help="Run a specific example")
    example_parser.add_argument("name", help="Name of the example to run")
    
    # Docs build command
    subparsers.add_parser("docs-build", help="Build the documentation")
    
    args = parser.parse_args()
    
    if args.command == "quick-start":
        return quick_start()
    elif args.command == "run-example":
        return run_example(args.name)
    elif args.command == "docs-build":
        return docs_build()
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
