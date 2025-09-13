#!/usr/bin/env python3
"""
Release management script for RAGLib.

This script helps manage releases by:
1. Updating version in pyproject.toml
2. Creating and pushing git tags
3. Triggering the automated release workflow

Usage:
    python release.py patch    # 1.0.0 -> 1.0.1
    python release.py minor    # 1.0.0 -> 1.1.0
    python release.py major    # 1.0.0 -> 2.0.0
    python release.py 1.2.3    # Set specific version
"""

import re
import subprocess
import sys
from pathlib import Path

try:
    import toml
except ImportError:
    print("❌ Error: toml package not found. Install with: pip install toml")
    sys.exit(1)


def get_current_version() -> str:
    """Get current version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found")
    
    with open(pyproject_path) as f:
        data = toml.load(f)
    
    return data["project"]["version"]


def set_version(new_version: str) -> None:
    """Update version in pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    
    with open(pyproject_path) as f:
        content = f.read()
    
    # Update version line
    pattern = r'version = "[^"]+"'
    replacement = f'version = "{new_version}"'
    new_content = re.sub(pattern, replacement, content)
    
    with open(pyproject_path, "w") as f:
        f.write(new_content)
    
    print(f"✅ Updated pyproject.toml version to {new_version}")


def bump_version(current: str, bump_type: str) -> str:
    """Bump version based on type."""
    parts = current.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {current}")
    
    major, minor, patch = map(int, parts)
    
    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")


def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run shell command."""
    print(f"🔄 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if check and result.returncode != 0:
        print(f"❌ Command failed: {cmd}")
        print(f"Error: {result.stderr}")
        sys.exit(1)
    
    return result


def is_git_clean() -> bool:
    """Check if git working directory is clean."""
    result = run_command("git status --porcelain", check=False)
    return result.stdout.strip() == ""


def create_and_push_tag(version: str, push: bool = True) -> None:
    """Create and optionally push git tag."""
    tag_name = f"v{version}"
    
    # Create tag
    run_command(f'git tag -a {tag_name} -m "Release {tag_name}"')
    print(f"✅ Created tag: {tag_name}")
    
    if push:
        # Push tag
        run_command(f"git push origin {tag_name}")
        print(f"✅ Pushed tag: {tag_name}")
        print("🚀 GitHub Actions will now handle the release automatically!")
        print("📋 Monitor progress at: "
              "https://github.com/Mohammadshamlawi/raglib/actions")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    
    bump_arg = sys.argv[1]
    
    # Check if git working directory is clean
    if not is_git_clean():
        print("❌ Git working directory is not clean. "
              "Please commit or stash changes first.")
        sys.exit(1)
    
    try:
        current_version = get_current_version()
        print(f"📦 Current version: {current_version}")
        
        # Determine new version
        if bump_arg in ["major", "minor", "patch"]:
            new_version = bump_version(current_version, bump_arg)
        else:
            # Assume it's a specific version
            new_version = bump_arg
            # Validate version format
            if not re.match(r"^\d+\.\d+\.\d+", new_version):
                print(f"❌ Invalid version format: {new_version}")
                print("Expected format: x.y.z (e.g., 1.2.3)")
                sys.exit(1)
        
        print(f"🎯 New version: {new_version}")
        
        # Confirm with user
        response = input(f"Continue with release {new_version}? [y/N]: ")
        if response.lower() not in ["y", "yes"]:
            print("❌ Release cancelled.")
            sys.exit(0)
        
        # Update version
        set_version(new_version)
        
        # Commit version change
        run_command("git add pyproject.toml")
        run_command(f'git commit -m "Bump version to {new_version}"')
        print("✅ Committed version change")
        
        # Push changes
        run_command("git push origin main")
        print("✅ Pushed version change")
        
        # Create and push tag
        create_and_push_tag(new_version)
        
        print("\n" + "="*50)
        print(f"🎉 Release {new_version} initiated successfully!")
        print("\n📋 Next steps:")
        print("1. Monitor GitHub Actions workflow")
        print("2. Check PyPI publication")
        print("3. Verify documentation updates")
        print("\n🔗 Useful links:")
        print("• Actions: https://github.com/Mohammadshamlawi/raglib/actions")
        print(f"• PyPI: https://pypi.org/project/rag-techlib/{new_version}/")
        print(f"• Docs: https://rag-techlib.readthedocs.io/en/v{new_version}/")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()