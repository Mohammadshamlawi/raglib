#!/usr/bin/env python3
"""
RAGLib Documentation Management System

Comprehensive script for managing all RAGLib documentation, benchmarking,
and website generation tasks.

Commands:
    update          - Update all documentation automatically
    generate        - Regenerate techniques index
    benchmark       - Run comprehensive benchmarking
    build           - Build documentation website
    serve           - Build and serve documentation locally
    validate        - Validate all documentation
    clean           - Clean generated files
    full            - Complete rebuild (clean + update + build)
"""

import argparse
import subprocess
import sys
from pathlib import Path
import shutil
import time

class DocumentationManager:
    """Manages all documentation-related tasks for RAGLib."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.scripts_dir = project_root / "scripts"
        self.docs_dir = project_root / "docs"
        self.examples_dir = project_root / "examples"
        self.tools_dir = project_root / "tools"
        self.site_dir = project_root / "site"
    
    def update_docs(self, dry_run: bool = False, verbose: bool = False) -> bool:
        """Update all documentation using the auto-updater."""
        print("📝 Updating Documentation")
        print("-" * 30)
        
        auto_updater = self.scripts_dir / "auto_update_docs.py"
        if not auto_updater.exists():
            print("❌ Auto-updater script not found!")
            return False
        
        cmd = [sys.executable, str(auto_updater)]
        if dry_run:
            cmd.append("--dry-run")
        if verbose:
            cmd.append("--verbose")
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root)
            return result.returncode == 0
        except Exception as e:
            print(f"❌ Failed to update documentation: {e}")
            return False
    
    def generate_techniques_index(self) -> bool:
        """Generate the techniques index."""
        print("🔄 Generating Techniques Index")
        print("-" * 35)
        
        generator_script = self.tools_dir / "generate_techniques_index.py"
        if not generator_script.exists():
            print("❌ Techniques index generator not found!")
            return False
        
        try:
            result = subprocess.run([
                sys.executable, str(generator_script)
            ], cwd=self.project_root)
            
            if result.returncode == 0:
                print("✅ Techniques index generated successfully")
                return True
            else:
                print("❌ Failed to generate techniques index")
                return False
        except Exception as e:
            print(f"❌ Error generating techniques index: {e}")
            return False
    
    def run_benchmarks(self) -> bool:
        """Run comprehensive benchmarking."""
        print("⚡ Running Benchmarks")
        print("-" * 20)
        
        # Run chunking benchmark
        chunking_benchmark = self.examples_dir / "chunking_benchmark.py"
        if chunking_benchmark.exists():
            print("   🔬 Running chunking benchmark...")
            try:
                result = subprocess.run([
                    sys.executable, str(chunking_benchmark)
                ], cwd=self.project_root, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("   ✅ Chunking benchmark completed")
                else:
                    print(f"   ⚠️  Chunking benchmark had issues: {result.stderr}")
            except Exception as e:
                print(f"   ❌ Chunking benchmark failed: {e}")
        
        # Run auto-generated benchmark if it exists
        auto_benchmark = self.examples_dir / "chunking_benchmark_auto.py"
        if auto_benchmark.exists():
            print("   🤖 Running auto-generated benchmark...")
            try:
                result = subprocess.run([
                    sys.executable, str(auto_benchmark)
                ], cwd=self.project_root, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("   ✅ Auto-generated benchmark completed")
                else:
                    print(f"   ⚠️  Auto-generated benchmark had issues")
            except Exception as e:
                print(f"   ❌ Auto-generated benchmark failed: {e}")
        
        print("   📊 Benchmark summary available in examples/ directory")
        return True
    
    def build_docs(self, clean: bool = False) -> bool:
        """Build the documentation website."""
        print("🌐 Building Documentation Website")
        print("-" * 35)
        
        if clean and self.site_dir.exists():
            print("   🧹 Cleaning previous build...")
            shutil.rmtree(self.site_dir)
        
        mkdocs_config = self.project_root / "mkdocs.yml"
        if not mkdocs_config.exists():
            print("❌ mkdocs.yml not found!")
            return False
        
        try:
            result = subprocess.run([
                "mkdocs", "build"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Documentation website built successfully")
                if self.site_dir.exists():
                    # Count generated files
                    html_files = list(self.site_dir.rglob("*.html"))
                    print(f"   📄 Generated {len(html_files)} HTML pages")
                return True
            else:
                print(f"❌ Failed to build documentation: {result.stderr}")
                return False
        except FileNotFoundError:
            print("❌ mkdocs not found. Install with: pip install mkdocs-material")
            return False
        except Exception as e:
            print(f"❌ Error building documentation: {e}")
            return False
    
    def serve_docs(self, port: int = 8000) -> bool:
        """Build and serve documentation locally."""
        print(f"🚀 Serving Documentation on http://localhost:{port}")
        print("-" * 50)
        
        # Build first
        if not self.build_docs():
            return False
        
        try:
            print(f"   🌐 Starting server on port {port}...")
            print("   ⚠️  Press Ctrl+C to stop the server")
            
            result = subprocess.run([
                "mkdocs", "serve", "--dev-addr", f"localhost:{port}"
            ], cwd=self.project_root)
            
            return result.returncode == 0
        except KeyboardInterrupt:
            print("\n   ⚠️  Server stopped by user")
            return True
        except FileNotFoundError:
            print("❌ mkdocs not found. Install with: pip install mkdocs-material")
            return False
        except Exception as e:
            print(f"❌ Error serving documentation: {e}")
            return False
    
    def validate_docs(self) -> bool:
        """Validate all documentation."""
        print("✅ Validating Documentation")
        print("-" * 28)
        
        issues = []
        
        # Check core files exist
        core_files = [
            self.docs_dir / "index.md",
            self.docs_dir / "techniques.md",
            self.docs_dir / "getting_started.md",
            self.docs_dir / "techniques_generated.md",
            self.project_root / "README.md",
            self.project_root / "mkdocs.yml"
        ]
        
        for file_path in core_files:
            if not file_path.exists():
                issues.append(f"Missing file: {file_path.name}")
            else:
                print(f"   ✅ {file_path.name}")
        
        # Check that techniques are registered
        try:
            sys.path.insert(0, str(self.project_root))
            from raglib.registry import TechniqueRegistry
            import raglib.techniques
            
            techniques = TechniqueRegistry.list()
            if not techniques:
                issues.append("No techniques found in registry")
            else:
                print(f"   ✅ {len(techniques)} techniques registered")
        except ImportError as e:
            issues.append(f"Failed to import raglib: {e}")
        
        # Check documentation builds
        if not self.site_dir.exists():
            issues.append("Documentation site not built")
        else:
            print(f"   ✅ Documentation site exists")
        
        if issues:
            print(f"\n   ❌ Found {len(issues)} issues:")
            for issue in issues:
                print(f"      - {issue}")
            return False
        else:
            print("\n   🎉 All validations passed!")
            return True
    
    def clean_generated(self) -> bool:
        """Clean all generated files."""
        print("🧹 Cleaning Generated Files")
        print("-" * 28)
        
        # Directories to clean
        dirs_to_clean = [
            self.site_dir,
            self.project_root / ".pytest_cache",
            self.project_root / "__pycache__",
        ]
        
        # Files to clean  
        files_to_clean = [
            self.docs_dir / "techniques_generated.md",
            self.examples_dir / "chunking_benchmark_auto.py",
            self.examples_dir / "chunking_showcase_auto.py",
            self.examples_dir / "chunking_benchmark_results.json",
            self.examples_dir / "chunking_benchmark_results_auto.json",
        ]
        
        cleaned_count = 0
        
        for dir_path in dirs_to_clean:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"   🗑️  Removed directory: {dir_path.name}")
                cleaned_count += 1
        
        for file_path in files_to_clean:
            if file_path.exists():
                file_path.unlink()
                print(f"   🗑️  Removed file: {file_path.name}")
                cleaned_count += 1
        
        if cleaned_count == 0:
            print("   ℹ️  No files to clean")
        else:
            print(f"   ✅ Cleaned {cleaned_count} items")
        
        return True
    
    def full_rebuild(self, verbose: bool = False) -> bool:
        """Complete rebuild of all documentation."""
        print("🔄 Full Documentation Rebuild")
        print("=" * 35)
        start_time = time.time()
        
        steps = [
            ("Clean", lambda: self.clean_generated()),
            ("Update", lambda: self.update_docs(verbose=verbose)),
            ("Generate", lambda: self.generate_techniques_index()),
            ("Build", lambda: self.build_docs(clean=True)),
            ("Validate", lambda: self.validate_docs()),
        ]
        
        for step_name, step_func in steps:
            print(f"\n🔸 Step: {step_name}")
            if not step_func():
                print(f"❌ Full rebuild failed at step: {step_name}")
                return False
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n🎉 Full rebuild completed successfully!")
        print(f"   ⏱️  Total time: {duration:.1f} seconds")
        return True


def main():
    """Main entry point for the documentation manager."""
    parser = argparse.ArgumentParser(
        description="RAGLib Documentation Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python manage_docs.py update              # Update all documentation
    python manage_docs.py build               # Build website
    python manage_docs.py serve               # Serve locally
    python manage_docs.py full --verbose      # Complete rebuild
        """
    )
    
    parser.add_argument(
        "command",
        choices=["update", "generate", "benchmark", "build", "serve", "validate", "clean", "full"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes (update only)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for serve command (default: 8000)"
    )
    
    args = parser.parse_args()
    
    # Setup
    project_root = Path(__file__).parent.parent
    manager = DocumentationManager(project_root)
    
    print("📚 RAGLib Documentation Manager")
    print("=" * 40)
    
    # Execute command
    success = False
    
    if args.command == "update":
        success = manager.update_docs(dry_run=args.dry_run, verbose=args.verbose)
    elif args.command == "generate":
        success = manager.generate_techniques_index()
    elif args.command == "benchmark":
        success = manager.run_benchmarks()
    elif args.command == "build":
        success = manager.build_docs()
    elif args.command == "serve":
        success = manager.serve_docs(port=args.port)
    elif args.command == "validate":
        success = manager.validate_docs()
    elif args.command == "clean":
        success = manager.clean_generated()
    elif args.command == "full":
        success = manager.full_rebuild(verbose=args.verbose)
    
    if success:
        print(f"\n✅ Command '{args.command}' completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Command '{args.command}' failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()