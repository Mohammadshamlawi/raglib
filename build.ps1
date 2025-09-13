# RAGLib Documentation Build Script
# Windows equivalent of Makefile targets

param(
    [string]$Target = "help"
)

function Show-Help {
    Write-Host "RAGLib Documentation Build Script" -ForegroundColor Green
    Write-Host "=================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Available targets:"
    Write-Host "  docs-generate    Generate techniques index from registry"
    Write-Host "  docs            Build documentation (includes index generation)"
    Write-Host "  docs-serve      Build and serve documentation locally"
    Write-Host "  test            Run tests"
    Write-Host "  coverage        Run tests with coverage"
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\build.ps1 docs-generate"
    Write-Host "  .\build.ps1 docs"
    Write-Host "  .\build.ps1 docs-serve"
}

function Generate-DocsIndex {
    Write-Host "Generating techniques index..." -ForegroundColor Yellow
    python tools/generate_techniques_index.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Techniques index generated successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to generate techniques index" -ForegroundColor Red
        exit 1
    }
}

function Build-Docs {
    Generate-DocsIndex
    Write-Host "Building documentation..." -ForegroundColor Yellow
    mkdocs build
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Documentation built successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to build documentation" -ForegroundColor Red
        exit 1
    }
}

function Serve-Docs {
    Generate-DocsIndex
    Write-Host "Building and serving documentation..." -ForegroundColor Yellow
    mkdocs serve
}

function Run-Tests {
    Write-Host "Running tests..." -ForegroundColor Yellow
    pytest
}

function Run-Coverage {
    Write-Host "Running tests with coverage..." -ForegroundColor Yellow
    pytest --cov=raglib --cov-report=html --cov-report=term-missing
}

# Main execution
switch ($Target) {
    "help" { Show-Help }
    "docs-generate" { Generate-DocsIndex }
    "docs" { Build-Docs }
    "docs-serve" { Serve-Docs }
    "test" { Run-Tests }
    "coverage" { Run-Coverage }
    default { 
        Write-Host "Unknown target: $Target" -ForegroundColor Red
        Show-Help
        exit 1
    }
}
