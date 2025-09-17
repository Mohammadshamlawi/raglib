# RAGLib Scripts Directory

This directory contains automation scripts for managing RAGLib documentation and development tasks.

## Scripts Overview

### 📚 Documentation Management

- **`manage_docs.py`** - Comprehensive documentation management system
- **`manage_docs.bat`** - Windows batch wrapper for the documentation manager
- **`auto_update_docs.py`** - Automated documentation updater (used by manage_docs.py)
- **`update_docs.py`** - Simple runner for the auto-updater

### 🚀 Quick Start

#### Windows (PowerShell/Command Prompt)
```powershell
# Update all documentation
.\manage_docs.bat update

# Build and serve documentation locally
.\manage_docs.bat serve

# Complete rebuild of everything
.\manage_docs.bat full --verbose
```

#### Cross-Platform (Python)
```bash
# Update all documentation
python manage_docs.py update

# Build and serve documentation locally  
python manage_docs.py serve

# Complete rebuild of everything
python manage_docs.py full --verbose
```

## Available Commands

| Command | Description |
|---------|-------------|
| `update` | Update all documentation automatically |
| `generate` | Regenerate techniques index |
| `benchmark` | Run comprehensive benchmarking |
| `build` | Build documentation website |
| `serve` | Build and serve documentation locally |
| `validate` | Validate all documentation |
| `clean` | Clean generated files |
| `full` | Complete rebuild (clean + update + build) |

## Common Workflows

### Adding New Techniques
When you add new techniques to RAGLib:

```bash
# Automatically update all documentation
python manage_docs.py update

# Build and verify the website
python manage_docs.py build
python manage_docs.py validate
```

### Development Workflow
```bash
# Make changes to techniques or documentation...

# Quick validation
python manage_docs.py update --dry-run --verbose

# Apply changes and rebuild everything
python manage_docs.py full --verbose

# Serve locally to test
python manage_docs.py serve
```

### Release Preparation
```bash
# Complete rebuild with benchmarks
python manage_docs.py clean
python manage_docs.py full --verbose
python manage_docs.py benchmark
python manage_docs.py validate
```

## Auto-Updater Features

The `auto_update_docs.py` script automatically:

- ✅ Discovers all registered techniques by category
- ✅ Updates `docs/techniques.md` with current technique lists
- ✅ Updates `README.md` document processing section
- ✅ Generates auto-benchmarking scripts
- ✅ Generates auto-showcase scripts  
- ✅ Maintains consistency across all documentation

### Dry Run Mode
Always test changes first:
```bash
python manage_docs.py update --dry-run --verbose
```

This shows what would be changed without making any modifications.

## Integration

### CI/CD Pipeline
Add to your CI/CD pipeline:
```yaml
- name: Update Documentation
  run: python scripts/manage_docs.py update

- name: Build Documentation
  run: python scripts/manage_docs.py build

- name: Validate Documentation
  run: python scripts/manage_docs.py validate
```

### Pre-commit Hook
Add to `.git/hooks/pre-commit`:
```bash
#!/bin/sh
cd scripts/
python manage_docs.py update --dry-run
```

## Dependencies

The scripts require:
- Python 3.7+
- `mkdocs-material` (for building documentation)
- RAGLib installed in development mode

Install dependencies:
```bash
pip install mkdocs-material
pip install -e .  # Install RAGLib in development mode
```

## Troubleshooting

### Common Issues

**"mkdocs not found"**
```bash
pip install mkdocs-material
```

**"No techniques found in registry"**
- Ensure RAGLib is installed: `pip install -e .`
- Check that technique modules import correctly

**"Auto-updater script not found"**
- Run from the project root directory
- Ensure `scripts/auto_update_docs.py` exists

### Getting Help

```bash
# Show all available commands
python manage_docs.py --help

# Get help for specific command
python manage_docs.py update --help
```

## File Structure

```
scripts/
├── manage_docs.py           # Main documentation manager
├── manage_docs.bat          # Windows batch wrapper
├── auto_update_docs.py      # Automated documentation updater
├── update_docs.py           # Simple auto-updater runner
└── README.md               # This file
```

## Contributing

When modifying the scripts:

1. Test with `--dry-run` first
2. Validate changes with `python manage_docs.py validate`
3. Update this README if adding new features
4. Test on both Windows and Unix-like systems