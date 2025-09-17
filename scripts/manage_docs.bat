@echo off
REM RAGLib Documentation Management Batch Script
REM Usage: manage_docs.bat [command] [options]
REM
REM Commands:
REM   update          - Update all documentation automatically
REM   generate        - Regenerate techniques index
REM   benchmark       - Run comprehensive benchmarking
REM   build           - Build documentation website
REM   serve           - Build and serve documentation locally
REM   validate        - Validate all documentation
REM   clean           - Clean generated files
REM   full            - Complete rebuild (clean + update + build)
REM
REM Examples:
REM   manage_docs.bat update
REM   manage_docs.bat build
REM   manage_docs.bat serve
REM   manage_docs.bat full --verbose

setlocal

REM Change to the script directory
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not available in PATH
    echo Please install Python and ensure it's in your PATH
    exit /b 1
)

REM Run the documentation manager
python manage_docs.py %*

REM Exit with the same code as the Python script
exit /b %errorlevel%