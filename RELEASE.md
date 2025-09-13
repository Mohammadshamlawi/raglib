# Release Process

This document outlines the step-by-step process for releasing new versions of RAGLib to PyPI.

## Pre-Release Checklist

Before creating a release, ensure the following are completed:

### 1. Code Quality
- [ ] All tests pass on all supported Python versions
- [ ] Code coverage is >95%
- [ ] All linting checks pass (ruff, black, mypy)
- [ ] Documentation builds without errors
- [ ] Examples work correctly

### 2. Documentation
- [ ] CHANGELOG.md is updated with all changes
- [ ] Version number is updated in pyproject.toml
- [ ] Documentation reflects new features/changes
- [ ] Techniques index is regenerated and current
- [ ] README.md is updated if needed

### 3. Testing
- [ ] Full test suite passes locally
- [ ] CI/CD pipeline passes on GitHub
- [ ] Cross-platform compatibility verified
- [ ] Performance regression tests pass

## Release Steps

### Step 1: Prepare the Release

1. **Create a release branch:**
   ```bash
   git checkout -b release/v1.2.3
   ```

2. **Update version in pyproject.toml:**
   ```toml
   [project]
   name = "raglib"
   version = "1.2.3"  # Update this
   ```

3. **Update CHANGELOG.md:**
   - Move items from `[Unreleased]` to new version section
   - Add release date
   - Create new empty `[Unreleased]` section

   ```markdown
   ## [Unreleased]

   ### Added

   ### Changed

   ## [1.2.3] - 2024-XX-XX

   ### Added
   - New semantic chunking technique
   - Enhanced documentation system
   
   ### Fixed
   - Memory leak in dense retrieval
   ```

4. **Run final checks:**
   ```bash
   # Run all quality checks
   make all-checks  # Linux/macOS
   .\build.ps1 all-checks  # Windows

   # Run full test suite
   make test
   .\build.ps1 test

   # Build documentation
   make docs
   .\build.ps1 docs

   # Test package build
   python -m build
   ```

### Step 2: Build the Package

1. **Clean previous builds:**
   ```bash
   # Linux/macOS
   make clean

   # Windows
   rm -rf build/ dist/ *.egg-info/

   # Or manually delete directories
   ```

2. **Build wheel and source distribution:**
   ```bash
   # Install build tools if not already installed
   pip install build twine

   # Build the package
   python -m build
   ```

3. **Verify the build:**
   ```bash
   # Check the built package
   ls dist/
   # Should contain: raglib-1.2.3-py3-none-any.whl and raglib-1.2.3.tar.gz

   # Test the wheel
   pip install dist/raglib-1.2.3-py3-none-any.whl
   python -c "import raglib; print(raglib.__version__)"
   ```

### Step 3: Test the Package

1. **Create a test environment:**
   ```bash
   python -m venv test_env
   source test_env/bin/activate  # Linux/macOS
   test_env\Scripts\activate     # Windows
   ```

2. **Install from built wheel:**
   ```bash
   pip install dist/raglib-1.2.3-py3-none-any.whl
   ```

3. **Run integration tests:**
   ```bash
   python -c "
   from raglib.techniques.bm25_simple import BM25Simple
   from raglib.schemas import Document
   
   docs = [Document(id='1', text='test document')]
   technique = BM25Simple()
   result = technique.apply(query='test', corpus=docs)
   print('✅ Package works correctly')
   "
   ```

4. **Test CLI:**
   ```bash
   raglib-cli --help
   raglib-cli quick-start
   ```

### Step 4: Upload to PyPI

1. **Test upload to PyPI Test:**
   ```bash
   # Upload to test PyPI first
   python -m twine upload --repository testpypi dist/*

   # Test install from test PyPI
   pip install --index-url https://test.pypi.org/simple/ raglib==1.2.3
   ```

2. **Upload to production PyPI:**
   ```bash
   # Upload to production PyPI
   python -m twine upload dist/*
   ```

3. **Verify the upload:**
   - Visit https://pypi.org/project/raglib/
   - Check that new version is available
   - Test installation: `pip install raglib==1.2.3`

### Step 5: Create GitHub Release

1. **Commit and push the release:**
   ```bash
   git add .
   git commit -m "Release v1.2.3"
   git push origin release/v1.2.3
   ```

2. **Create a pull request:**
   - Create PR from `release/v1.2.3` to `main`
   - Title: "Release v1.2.3"
   - Include changelog in description
   - Wait for CI to pass and get approval

3. **Merge and tag:**
   ```bash
   # After PR is merged
   git checkout main
   git pull origin main

   # Create and push tag
   git tag -a v1.2.3 -m "Release v1.2.3"
   git push origin v1.2.3
   ```

4. **Create GitHub release:**
   - Go to https://github.com/your-org/raglib/releases
   - Click "Draft a new release"
   - Select tag `v1.2.3`
   - Title: "RAGLib v1.2.3"
   - Description: Copy from CHANGELOG.md
   - Attach built wheel and source distribution
   - Click "Publish release"

### Step 6: Post-Release Tasks

1. **Update development version:**
   ```toml
   # In pyproject.toml
   version = "1.2.4-dev"  # Next development version
   ```

2. **Announce the release:**
   - Update README.md if needed
   - Post in community channels
   - Update documentation
   - Send announcements to mailing lists/forums

3. **Monitor for issues:**
   - Watch for bug reports
   - Monitor download statistics
   - Check compatibility reports

## Release Types

### Patch Release (x.y.Z)
- Bug fixes only
- Backward compatible
- No new features
- Quick turnaround (days)

### Minor Release (x.Y.0)
- New features
- Backward compatible
- New techniques or adapters
- Regular cadence (weeks/months)

### Major Release (X.0.0)
- Breaking changes
- API modifications
- Major new features
- Rare (months/years)

## Automation

### GitHub Actions Release Workflow

Consider automating releases with GitHub Actions:

```yaml
# .github/workflows/release.yml
name: Release

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

### Release Scripts

Create helper scripts for releases:

```bash
# scripts/release.sh
#!/bin/bash
set -e

VERSION=$1
if [ -z "$VERSION" ]; then
    echo "Usage: ./scripts/release.sh 1.2.3"
    exit 1
fi

echo "Preparing release $VERSION..."

# Update version
sed -i "s/version = .*/version = \"$VERSION\"/" pyproject.toml

# Run tests
make test

# Build package
make build

# Create git tag
git add .
git commit -m "Release v$VERSION"
git tag -a "v$VERSION" -m "Release v$VERSION"

echo "Release $VERSION prepared. Push with:"
echo "git push origin main --tags"
```

## Troubleshooting

### Common Issues

**Upload fails with authentication error:**
- Ensure PyPI API token is correct
- Check token permissions
- Verify username is `__token__`

**Version already exists:**
- Cannot overwrite existing versions on PyPI
- Increment version number
- Delete from test PyPI if testing

**Package build fails:**
- Check pyproject.toml syntax
- Ensure all files are included in MANIFEST.in
- Verify dependencies are correct

**Tests fail after install:**
- Check optional dependencies
- Verify package structure
- Test in clean environment

### Recovery Procedures

**If release fails after PyPI upload:**
1. Do not delete from PyPI (not allowed)
2. Create a patch release (increment patch version)
3. Fix issues and re-release
4. Document known issues in GitHub

**If tag/release created prematurely:**
1. Delete GitHub release (if not yet published)
2. Delete git tag locally and remotely
3. Fix issues and restart process

## Security Considerations

- **Use API tokens** instead of passwords
- **Store tokens securely** in environment variables or GitHub secrets
- **Enable 2FA** on PyPI account
- **Review uploaded packages** for sensitive information
- **Sign releases** with GPG if required by organization

## Documentation Updates

After each release:
- [ ] Update installation instructions
- [ ] Update version badges
- [ ] Refresh examples with new version
- [ ] Update dependency requirements
- [ ] Archive old version docs if needed
