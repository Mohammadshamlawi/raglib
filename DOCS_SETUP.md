# Setting Up Read the Docs for RAGLib

This guide will help you set up documentation hosting on Read the Docs for your RAGLib project.

## Prerequisites

- [x] GitHub repository with your code
- [x] MkDocs configuration (`mkdocs.yml`)
- [x] Read the Docs configuration (`.readthedocs.yaml`)
- [x] Documentation source files in `/docs/`
- [x] Documentation dependencies specification

## Step 1: Create Read the Docs Account

1. Go to [Read the Docs](https://readthedocs.org/)
2. Sign up with your GitHub account
3. Grant Read the Docs access to your repositories

## Step 2: Import Your Project

1. Once logged in, click "Import a Project"
2. Select your `raglib` repository from the list
3. Or manually import using the repository URL: `https://github.com/Mohammadshamlawi/raglib`

## Step 3: Configure Project Settings

### Basic Settings
- **Name**: `rag-techlib` (to match your PyPI package name)
- **Repository URL**: `https://github.com/Mohammadshamlawi/raglib`
- **Repository type**: Git
- **Default branch**: `main`
- **Default version**: `latest`
- **Programming language**: Python

### Advanced Settings
- **Documentation type**: MkDocs
- **Requirements file**: `docs/requirements.txt`
- **Python configuration file**: `pyproject.toml`
- **Use system packages**: ✓ (checked)

## Step 4: Configure Webhooks (Automatic)

Read the Docs will automatically:
- Set up GitHub webhooks
- Build documentation on every push to main
- Build documentation for pull requests (if enabled)

## Step 5: Build Documentation

1. After importing, Read the Docs will automatically trigger a build
2. Monitor the build process in the "Builds" tab
3. If successful, your documentation will be available at: `https://rag-techlib.readthedocs.io`

## Step 6: Custom Domain (Optional)

If you want to use a custom domain:
1. Go to Admin → Domains
2. Add your custom domain
3. Set up DNS CNAME record pointing to `readthedocs.io`

## Troubleshooting

### Common Build Issues

#### Dependencies Not Found
- Ensure `docs/requirements.txt` includes all necessary packages
- Check that `.readthedocs.yaml` points to the correct requirements file

#### MkDocs Configuration Error
- Validate your `mkdocs.yml` syntax
- Test locally with `mkdocs build`
- Check file paths in navigation

#### Python Package Installation Failed
- Ensure `pyproject.toml` has correct `[project.optional-dependencies]` for docs
- Check that all dependencies have compatible versions

### Build Configuration Debug

If builds fail, check these files:

1. **`.readthedocs.yaml`**: Main configuration
2. **`docs/requirements.txt`**: Documentation dependencies
3. **`pyproject.toml`**: Package configuration with docs extra

### Testing Locally

Before pushing, always test locally:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
mkdocs build

# Serve locally (optional)
mkdocs serve
```

## Current Configuration Files

### `.readthedocs.yaml`
```yaml
version: 2
build:
  os: ubuntu-22.04
  tools:
    python: "3.10"
mkdocs:
  configuration: mkdocs.yml
  fail_on_warning: false
python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - docs
    - requirements: docs/requirements.txt
formats:
  - pdf
  - epub
```

### `docs/requirements.txt`
```
mkdocs>=1.5.0
mkdocs-material>=9.0.0
mkdocs-autorefs>=0.5.0
mkdocstrings[python]>=0.22.0
pymdown-extensions>=10.0.0
griffe>=0.32.0
Pillow>=9.0.0
cairosvg>=2.5.0
```

### `pyproject.toml` (docs section)
```toml
[project.optional-dependencies]
docs = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.0.0",
    "mkdocs-autorefs>=0.5.0",
    "mkdocstrings[python]>=0.22.0",
    "pymdown-extensions>=10.0.0",
    "griffe>=0.32.0",
    "Pillow>=9.0.0",
    "cairosvg>=2.5.0",
]
```

## Expected URLs

After successful setup:
- **Main docs**: https://rag-techlib.readthedocs.io
- **Latest version**: https://rag-techlib.readthedocs.io/en/latest/
- **Specific version**: https://rag-techlib.readthedocs.io/en/stable/

## GitHub Pages Backup

A GitHub Actions workflow has been set up as a backup:
- **File**: `.github/workflows/docs.yml`
- **Backup URL**: https://mohammadshamlawi.github.io/raglib/

## Maintenance

### Updating Documentation
1. Edit files in `/docs/` directory
2. Push changes to `main` branch
3. Read the Docs automatically rebuilds

### Version Management
- Read the Docs can build documentation for different versions/tags
- Configure in Admin → Versions
- Useful for maintaining docs for different package versions

## Next Steps

1. **Import your project** on Read the Docs
2. **Monitor the first build** for any errors
3. **Configure custom domain** if desired
4. **Set up version management** for releases
5. **Enable PR builds** for documentation review

Your documentation should be live within a few minutes of successful import and build!