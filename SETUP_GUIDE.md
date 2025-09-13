# Quick Setup Guide: GitHub Actions for PyPI and Read The Docs

## 🚀 What's Been Created

I've created a comprehensive GitHub Actions setup for your RAGLib project that automatically:

1. **Publishes to PyPI** when you create releases
2. **Updates Read The Docs** documentation automatically
3. **Deploys to GitHub Pages** as a backup
4. **Validates builds** and runs tests before publishing
5. **Creates GitHub releases** automatically with proper permissions

## 📁 New Files Created

```
.github/
├── workflows/
│   ├── publish.yml      # Main release workflow (PyPI + Docs + GitHub Releases)
│   ├── docs-rtd.yml     # Read The Docs specific updates
│   ├── docs.yml         # GitHub Pages deployment (existing, enhanced)
│   └── ci.yml           # Continuous Integration (existing)
├── WORKFLOWS.md         # Detailed documentation
release.py               # Helper script for version management
```

## 🔧 Required Setup (One-time)

### 1. PyPI Secrets

Add these to your GitHub repository secrets (Settings → Secrets and variables → Actions):

```
PYPI_API_TOKEN         # Your PyPI API token for production releases
TEST_PYPI_API_TOKEN    # Your Test PyPI token for prereleases (optional)
```

**Get PyPI tokens:**
1. Go to [pypi.org](https://pypi.org) → Account settings → API tokens
2. Create token with "Entire account" scope
3. Copy the token (starts with `pypi-`)

### 2. Read The Docs Setup

**Option A: Webhook (Recommended)**
1. Go to [readthedocs.org](https://readthedocs.org) and import your project
2. Project Settings → Integrations → Add GitHub webhook
3. Copy webhook URL and secret, extract webhook ID from URL
4. Add to GitHub secrets:
   - `RTD_WEBHOOK_TOKEN`: The webhook secret
   - `RTD_WEBHOOK_ID`: The webhook ID from URL

**Option B: API Token**
1. Your RTD account → Settings → API Tokens
2. Create token and add as `RTD_API_TOKEN` GitHub secret

### 3. GitHub Permissions (Automatic)

The workflows now have proper permissions configured:
- ✅ `contents: write` - Create GitHub releases
- ✅ `id-token: write` - Trusted PyPI publishing
- ✅ `pages: write` - Deploy to GitHub Pages

## 🎯 How to Release

### Method 1: Proper Version Update + Tag (Recommended)
```bash
# 1. Update version in pyproject.toml first
# Edit: version = "1.2.1"

# 2. Commit version change
git add pyproject.toml
git commit -m "Bump version to 1.2.1"
git push

# 3. Create and push tag
git tag v1.2.1
git push origin v1.2.1

# 4. Workflow automatically handles:
#    - PyPI publishing
#    - GitHub release creation
#    - Documentation updates
```

### Method 2: Helper Script (Easiest)
```bash
# This handles everything automatically
python release.py patch    # 1.0.0 → 1.0.1
python release.py minor    # 1.0.0 → 1.1.0  
python release.py major    # 1.0.0 → 2.0.0
python release.py 1.2.3    # Set specific version
```

### Method 3: Manual Workflow
```bash
# Go to Actions → "Publish to PyPI and Update Docs" → Run workflow
# Choose your options and environment
```

## ⚠️ **Important: Version Synchronization**

**Always ensure the version in `pyproject.toml` matches your git tag!**

❌ **Wrong:**
- `pyproject.toml`: `version = "0.1.1"`
- Git tag: `v1.2.0`
- Result: PyPI gets version 0.1.1, but tag says 1.2.0

✅ **Correct:**
- `pyproject.toml`: `version = "1.2.0"`
- Git tag: `v1.2.0` 
- Result: Everything matches perfectly

## 📚 Documentation Updates

Documentation automatically updates when:
- 📝 You push changes to `docs/` folder
- 🏷️ You create a new release
- 🔧 You manually trigger the docs workflow

Available at:
- 📖 [Read The Docs](https://rag-techlib.readthedocs.io)
- 🏠 [GitHub Pages](https://rag-techlib.github.io)

## 🔍 Version Handling

- **Stable releases** (v1.0.0): → PyPI + RTD + GitHub Release
- **Prereleases** (v1.0.0-alpha1): → Test PyPI + RTD + GitHub Prerelease
- **Development** (v1.0.0-dev): → Test PyPI + RTD + GitHub Prerelease

## 📊 Monitoring

Check workflow status at:
- **GitHub Actions**: Repository → Actions tab
- **GitHub Releases**: Repository → Releases section
- **PyPI**: [pypi.org/project/rag-techlib](https://pypi.org/project/rag-techlib)
- **Read The Docs**: [rag-techlib.readthedocs.io](https://rag-techlib.readthedocs.io)

## 🛠️ Quick Test

1. **Test documentation build locally:**
   ```bash
   pip install -e ".[docs]"
   mkdocs build
   ```

2. **Test package build:**
   ```bash
   pip install build
   python -m build
   ```

3. **Create a test release:**
   ```bash
   python release.py 0.1.2-test
   # This will create a prerelease that goes to Test PyPI
   ```

## 🔧 Troubleshooting

**PyPI publication fails:**
- Check if `PYPI_API_TOKEN` is set correctly
- Ensure version doesn't already exist on PyPI

**Read The Docs not updating:**
- Verify webhook/API token setup
- Check RTD build logs for errors
- Validate `.readthedocs.yaml` configuration

**Documentation build fails:**
- Check `mkdocs.yml` syntax
- Ensure all dependencies in `docs/requirements.txt`

## 📋 Next Steps

1. **Set up secrets** in GitHub repository settings
2. **Import project** to Read The Docs
3. **Test with a prerelease** using the helper script
4. **Create your first real release** 

Your automation is now ready! The next time you create a release, everything will be published automatically. 🎉