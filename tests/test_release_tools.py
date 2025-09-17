"""Tests for release preparation tools."""

import subprocess

# Import the functions from prepare_release
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from prepare_release import (
    build_distributions,
    create_draft_release_notes,
    get_current_version,
    update_changelog,
    validate_code_quality,
    validate_tests,
)


class TestReleaseTools:
    """Test release preparation tools."""

    def test_get_current_version(self, tmp_path):
        """Test getting version from pyproject.toml."""
        # Create test pyproject.toml
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_content = '''[project]
name = "test-package"
version = "1.2.3"
description = "Test package"
'''
        pyproject_file.write_text(pyproject_content)

        # Change to temp directory for test
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmp_path)
            version = get_current_version()
            assert version == "1.2.3"
        finally:
            os.chdir(original_cwd)

    def test_get_current_version_file_not_found(self, tmp_path):
        """Test handling when pyproject.toml doesn't exist."""
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmp_path)  # Directory without pyproject.toml
            with pytest.raises(FileNotFoundError):
                get_current_version()
        finally:
            os.chdir(original_cwd)

    def test_get_current_version_no_version(self, tmp_path):
        """Test handling when version is not found in pyproject.toml."""
        # Create pyproject.toml without version
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_content = '''[project]
name = "test-package"
description = "Test package"
'''
        pyproject_file.write_text(pyproject_content)

        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmp_path)
            with pytest.raises(ValueError):
                get_current_version()
        finally:
            os.chdir(original_cwd)

    def test_update_changelog_dry_run(self, tmp_path):
        """Test changelog update in dry run mode."""
        # Create test changelog
        changelog_file = tmp_path / "CHANGELOG.md"
        changelog_content = '''# Changelog

## [Unreleased]

### Added
- New feature

### Fixed
- Bug fix

## [1.0.0] - 2024-01-01
- Initial release
'''
        changelog_file.write_text(changelog_content)

        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmp_path)
            result = update_changelog("1.1.0", dry_run=True)
            assert result is True

            # File should not be modified in dry run
            assert changelog_file.read_text() == changelog_content
        finally:
            os.chdir(original_cwd)

    def test_update_changelog_real_update(self, tmp_path):
        """Test actual changelog update."""
        # Create test changelog
        changelog_file = tmp_path / "CHANGELOG.md"
        changelog_content = '''# Changelog

## [Unreleased]

### Added
- New feature

## [1.0.0] - 2024-01-01
- Initial release
'''
        changelog_file.write_text(changelog_content)

        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmp_path)
            result = update_changelog("1.1.0", dry_run=False)
            assert result is True

            # File should be modified
            updated_content = changelog_file.read_text()
            assert "## [1.1.0]" in updated_content
            assert "## [Unreleased]" in updated_content
        finally:
            os.chdir(original_cwd)

    def test_update_changelog_no_file(self, tmp_path):
        """Test changelog update when file doesn't exist."""
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmp_path)
            result = update_changelog("1.1.0", dry_run=False)
            assert result is False
        finally:
            os.chdir(original_cwd)

    def test_update_changelog_no_unreleased_section(self, tmp_path):
        """Test changelog update when unreleased section is missing."""
        changelog_file = tmp_path / "CHANGELOG.md"
        changelog_content = '''# Changelog

## [1.0.0] - 2024-01-01
- Initial release
'''
        changelog_file.write_text(changelog_content)

        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmp_path)
            result = update_changelog("1.1.0", dry_run=False)
            assert result is False
        finally:
            os.chdir(original_cwd)

    @patch('prepare_release.run_command')
    def test_validate_tests_dry_run(self, mock_run_command):
        """Test test validation in dry run mode."""
        result = validate_tests(dry_run=True)
        assert result is True
        mock_run_command.assert_not_called()

    @patch('prepare_release.run_command')
    def test_validate_tests_success(self, mock_run_command):
        """Test successful test validation."""
        mock_run_command.return_value = Mock()
        result = validate_tests(dry_run=False)
        assert result is True
        mock_run_command.assert_called_once_with(["pytest", "--tb=short", "-q"])

    @patch('prepare_release.run_command')
    def test_validate_tests_failure(self, mock_run_command):
        """Test test validation failure."""
        mock_run_command.side_effect = subprocess.CalledProcessError(1, "pytest")
        result = validate_tests(dry_run=False)
        assert result is False

    @patch('prepare_release.run_command')
    def test_validate_code_quality_dry_run(self, mock_run_command):
        """Test code quality validation in dry run mode."""
        result = validate_code_quality(dry_run=True)
        assert result is True
        mock_run_command.assert_not_called()

    @patch('prepare_release.run_command')
    def test_validate_code_quality_success(self, mock_run_command):
        """Test successful code quality validation."""
        mock_run_command.return_value = Mock()
        result = validate_code_quality(dry_run=False)
        assert result is True
        # Should call black, ruff, and mypy
        assert mock_run_command.call_count == 3

    @patch('prepare_release.run_command')
    def test_validate_code_quality_partial_failure(self, mock_run_command):
        """Test code quality validation with some tools failing."""
        # First call succeeds, second fails, third succeeds
        mock_run_command.side_effect = [
            Mock(),  # black succeeds
            subprocess.CalledProcessError(1, "ruff"),  # ruff fails
            Mock()   # mypy succeeds
        ]
        result = validate_code_quality(dry_run=False)
        assert result is False

    @patch('prepare_release.run_command')
    def test_build_distributions_dry_run(self, mock_run_command):
        """Test distribution build in dry run mode."""
        result = build_distributions(dry_run=True)
        assert result is True
        mock_run_command.assert_not_called()

    @patch('prepare_release.run_command')
    def test_build_distributions_success(self, mock_run_command, tmp_path):
        """Test successful distribution build."""
        mock_run_command.return_value = Mock()

        # Create dist directory to simulate build output
        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()
        (dist_dir / "package-1.0.0.tar.gz").touch()
        (dist_dir / "package-1.0.0-py3-none-any.whl").touch()

        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmp_path)
            result = build_distributions(dry_run=False)
            assert result is True
        finally:
            os.chdir(original_cwd)

    def test_create_draft_release_notes_dry_run(self, tmp_path):
        """Test draft release notes creation in dry run mode."""
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmp_path)
            result = create_draft_release_notes("1.1.0", dry_run=True)
            assert result is True
        finally:
            os.chdir(original_cwd)

    def test_create_draft_release_notes_success(self, tmp_path):
        """Test successful draft release notes creation."""
        # Create changelog with version
        changelog_file = tmp_path / "CHANGELOG.md"
        changelog_content = '''# Changelog

## [Unreleased]

### Added
- New feature

## [1.1.0] - 2024-01-15

### Added
- Feature A
- Feature B

### Fixed
- Bug fix

## [1.0.0] - 2024-01-01
- Initial release
'''
        changelog_file.write_text(changelog_content)

        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmp_path)
            result = create_draft_release_notes("1.1.0", dry_run=False)
            assert result is True

            # Check that release notes file was created
            release_notes_file = tmp_path / "release_notes_1.1.0.md"
            assert release_notes_file.exists()

            content = release_notes_file.read_text()
            assert "RAGLib 1.1.0" in content
            assert "Feature A" in content
            assert "Feature B" in content
            assert "pip install rag-techlib==1.1.0" in content
        finally:
            os.chdir(original_cwd)

    def test_create_draft_release_notes_no_changelog(self, tmp_path):
        """Test release notes creation when changelog doesn't exist."""
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmp_path)
            result = create_draft_release_notes("1.1.0", dry_run=False)
            assert result is False
        finally:
            os.chdir(original_cwd)

    def test_create_draft_release_notes_version_not_found(self, tmp_path):
        """Test release notes creation when version is not in changelog."""
        # Create changelog without the requested version
        changelog_file = tmp_path / "CHANGELOG.md"
        changelog_content = '''# Changelog

## [1.0.0] - 2024-01-01
- Initial release
'''
        changelog_file.write_text(changelog_content)

        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmp_path)
            result = create_draft_release_notes("1.1.0", dry_run=False)
            assert result is False
        finally:
            os.chdir(original_cwd)
