"""Releasedef run_command(
    cmd: list[str], check: bool = True, capture_output: bool = True
) -> subprocess.CompletedProcess:preparation and validation tool."""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(
    cmd: list[str], check: bool = True, capture_output: bool = True
) -> subprocess.CompletedProcess:
    """Run a shell command and return result."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=capture_output, text=True)


def validate_tests(dry_run: bool = False) -> bool:
    """Validate that all tests pass."""
    print("\n=== Validating Tests ===")

    if dry_run:
        print("DRY RUN: Would run pytest")
        return True

    try:
        run_command(["pytest", "--tb=short", "-q"])
        print("✅ All tests passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed: {e}")
        return False


def validate_code_quality(dry_run: bool = False) -> bool:
    """Validate code quality with linting tools."""
    print("\n=== Validating Code Quality ===")

    if dry_run:
        print("DRY RUN: Would run black, ruff, and mypy")
        return True

    tools = [
        (["black", "--check", "raglib", "tests", "examples"], "Black formatting"),
        (["ruff", "raglib", "tests", "examples"], "Ruff linting"),
        (["mypy", "raglib"], "MyPy type checking")
    ]

    all_passed = True
    for cmd, description in tools:
        try:
            run_command(cmd)
            print(f"✅ {description} passed")
        except subprocess.CalledProcessError:
            print(f"❌ {description} failed")
            all_passed = False
        except FileNotFoundError:
            print(f"⚠️  {description} tool not found, skipping")

    return all_passed


def get_current_version() -> str:
    """Get current version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")

    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found")

    with open(pyproject_path, encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("version = "):
                # Extract version from 'version = "0.1.0"'
                version = line.split("=")[1].strip().strip('"')
                return version

    raise ValueError("Version not found in pyproject.toml")


def update_changelog(version: str, dry_run: bool = False) -> bool:
    """Update CHANGELOG.md with new version header."""
    print(f"\n=== Updating CHANGELOG for version {version} ===")

    changelog_path = Path("CHANGELOG.md")

    if not changelog_path.exists():
        print("❌ CHANGELOG.md not found")
        return False

    if dry_run:
        print("DRY RUN: Would update CHANGELOG.md")
        return True

    # Read current changelog
    with open(changelog_path, encoding="utf-8") as f:
        content = f.read()

    # Check if version already exists
    version_header = f"## [{version}]"
    if version_header in content:
        print(f"⚠️  Version {version} already exists in CHANGELOG.md")
        return True

    # Find unreleased section
    unreleased_line = "## [Unreleased]"
    if unreleased_line not in content:
        print("❌ [Unreleased] section not found in CHANGELOG.md")
        return False

    # Add new version section
    today = datetime.now().strftime("%Y-%m-%d")
    new_section = f"{version_header} - {today}"

    # Replace [Unreleased] with version and add new [Unreleased]
    updated_content = content.replace(
        unreleased_line,
        f"{unreleased_line}\n\n{new_section}"
    )

    # Write back
    with open(changelog_path, "w", encoding="utf-8") as f:
        f.write(updated_content)

    print(f"✅ Updated CHANGELOG.md with version {version}")
    return True


def build_distributions(dry_run: bool = False) -> bool:
    """Build source and wheel distributions."""
    print("\n=== Building Distributions ===")

    if dry_run:
        print("DRY RUN: Would build source and wheel distributions")
        return True

    # Clean previous builds
    dist_path = Path("dist")
    if dist_path.exists():
        import shutil
        shutil.rmtree(dist_path)

    try:
        run_command([sys.executable, "-m", "build"])
        print("✅ Built distributions successfully")

        # List built files
        if dist_path.exists():
            files = list(dist_path.glob("*"))
            print(f"Built files: {[f.name for f in files]}")

        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        return False


def create_draft_release_notes(version: str, dry_run: bool = False) -> bool:
    """Create draft release notes file."""
    print("\n=== Creating Draft Release Notes ===")

    if dry_run:
        print("DRY RUN: Would create draft release notes")
        return True

    release_notes_path = Path(f"release_notes_{version}.md")

    # Extract changelog section for this version
    changelog_path = Path("CHANGELOG.md")
    if not changelog_path.exists():
        print("❌ CHANGELOG.md not found")
        return False

    with open(changelog_path, encoding="utf-8") as f:
        content = f.read()

    # Find version section
    version_header = f"## [{version}]"
    if version_header not in content:
        print(f"❌ Version {version} not found in CHANGELOG.md")
        return False

    # Extract section between this version and next version/section
    lines = content.split("\n")
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if line.startswith(version_header):
            start_idx = i
        elif start_idx is not None and line.startswith("## [") and i > start_idx:
            end_idx = i
            break

    if start_idx is None:
        print(f"❌ Could not find version {version} in CHANGELOG.md")
        return False

    # Get content for this version
    if end_idx is None:
        version_content = "\n".join(lines[start_idx:])
    else:
        version_content = "\n".join(lines[start_idx:end_idx])

    # Create release notes
    release_notes = f"""# RAGLib {version}

{version_content}

## Installation

```bash
pip install rag-techlib=={version}
```

## Links

- [PyPI](https://pypi.org/project/raglib/{version}/)
- [Documentation](https://raglib.readthedocs.io)
- [GitHub](https://github.com/your-org/raglib/releases/tag/v{version})
"""

    with open(release_notes_path, "w", encoding="utf-8") as f:
        f.write(release_notes)

    print(f"✅ Created draft release notes: {release_notes_path}")
    return True


def validate_git_state() -> bool:
    """Validate git repository state."""
    print("\n=== Validating Git State ===")

    try:
        # Check if we're in a git repo
        run_command(["git", "status"], capture_output=True)

        # Check for uncommitted changes
        result = run_command(["git", "status", "--porcelain"], capture_output=True)
        if result.stdout.strip():
            print("⚠️  Uncommitted changes found:")
            print(result.stdout)
            return False

        print("✅ Git repository is clean")
        return True

    except subprocess.CalledProcessError:
        print("❌ Not in a git repository or git not available")
        return False


def main():
    """Main entry point for release preparation."""
    parser = argparse.ArgumentParser(description="Prepare RAGLib release")
    parser.add_argument("--dry-run", action="store_true",
                       help="Run validations without making changes")
    parser.add_argument("--version", help="Override version detection")
    parser.add_argument("--skip-tests", action="store_true",
                       help="Skip test validation")
    parser.add_argument("--skip-quality", action="store_true",
                       help="Skip code quality validation")

    args = parser.parse_args()

    print("🚀 RAGLib Release Preparation Tool")
    print("=" * 40)

    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")

    # Get version
    try:
        version = args.version or get_current_version()
        print(f"📦 Preparing release for version: {version}")
    except Exception as e:
        print(f"❌ Failed to get version: {e}")
        return 1

    # Validation steps
    success = True

    # Validate git state
    if not validate_git_state():
        print("⚠️  Git validation failed, continuing anyway...")

    # Run tests
    if not args.skip_tests:
        if not validate_tests(args.dry_run):
            print("❌ Test validation failed")
            success = False

    # Run code quality checks
    if not args.skip_quality:
        if not validate_code_quality(args.dry_run):
            print("❌ Code quality validation failed")
            success = False

    if not success:
        print("\n❌ Pre-release validation failed!")
        return 1

    # Update changelog
    if not update_changelog(version, args.dry_run):
        print("❌ Changelog update failed")
        return 1

    # Build distributions
    if not build_distributions(args.dry_run):
        print("❌ Distribution build failed")
        return 1

    # Create release notes
    if not create_draft_release_notes(version, args.dry_run):
        print("❌ Release notes creation failed")
        return 1

    print("\n🎉 Release preparation completed successfully!")
    print(f"📦 Version: {version}")

    if not args.dry_run:
        print("\nNext steps:")
        print("1. Review the generated release notes")
        print("2. Commit changelog updates")
        print("3. Tag the release: git tag v{version}")
        print("4. Push: git push origin main --tags")
        print("5. Upload to PyPI: twine upload dist/*")

    return 0


if __name__ == "__main__":
    sys.exit(main())
