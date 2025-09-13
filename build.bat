@echo off
REM RAGLib Build Script for Windows
REM Usage: build.bat [target]

if "%1"=="docs-generate" goto :docs-generate
if "%1"=="docs" goto :docs
if "%1"=="docs-serve" goto :docs-serve
if "%1"=="test" goto :test
if "%1"=="coverage" goto :coverage
goto :help

:help
echo RAGLib Build Script
echo ===================
echo.
echo Available targets:
echo   docs-generate    Generate techniques index from registry
echo   docs            Build documentation (includes index generation)
echo   docs-serve      Build and serve documentation locally
echo   test            Run tests
echo   coverage        Run tests with coverage
echo.
echo Usage: build.bat [target]
goto :end

:docs-generate
echo Generating techniques index...
python tools/generate_techniques_index.py
if %errorlevel% neq 0 exit /b %errorlevel%
echo ✓ Techniques index generated successfully
goto :end

:docs
echo Generating techniques index...
python tools/generate_techniques_index.py
if %errorlevel% neq 0 exit /b %errorlevel%
echo Building documentation...
mkdocs build
if %errorlevel% neq 0 exit /b %errorlevel%
echo ✓ Documentation built successfully
goto :end

:docs-serve
echo Generating techniques index...
python tools/generate_techniques_index.py
if %errorlevel% neq 0 exit /b %errorlevel%
echo Building and serving documentation...
mkdocs serve
goto :end

:test
echo Running tests...
pytest
goto :end

:coverage
echo Running tests with coverage...
pytest --cov=raglib --cov-report=html --cov-report=term-missing
goto :end

:end
