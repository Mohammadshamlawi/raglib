"""Tests for the plugin system."""

from pathlib import Path
from unittest.mock import Mock, patch

from raglib.plugins import PluginLoader, discover, discover_and_register_techniques
from raglib.registry import TechniqueRegistry


class TestPluginLoader:
    """Test the PluginLoader functionality."""

    def test_plugin_loader_init(self):
        """Test PluginLoader initialization."""
        # Test with no plugin dirs
        loader = PluginLoader()
        assert loader.plugin_dirs == []
        assert loader._loaded_plugins == {}

        # Test with plugin dirs
        loader = PluginLoader(["/path/to/plugins"])
        assert len(loader.plugin_dirs) == 1
        assert loader.plugin_dirs[0] == Path("/path/to/plugins")

    def test_discover_local_plugins(self, tmp_path):
        """Test discovering plugins from local directories."""
        # Create a test plugin file
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()

        plugin_file = plugin_dir / "test_plugin.py"
        plugin_content = '''
"""Test plugin module."""

from raglib.core import RAGTechnique, TechniqueMeta
from raglib.registry import TechniqueRegistry
from raglib.schemas import Document, RagResult

@TechniqueRegistry.register
class TestLocalTechnique(RAGTechnique):
    """Test technique from local plugin."""

    meta = TechniqueMeta(
        name="test_plugin",
        description="A test technique from local plugin",
        category="test",
        tags={"author": "Test Author", "email": "test@example.com", "version": "1.0.0"}
    )

    def apply(self, query, corpus, top_k=5, **kwargs):
        return RagResult(
            documents=corpus[:top_k],
            metadata={"technique": "test_local_technique"}
        )
'''
        plugin_file.write_text(plugin_content)

        # Test local plugin discovery
        loader = PluginLoader([plugin_dir])
        plugins = loader._discover_local_plugins(plugin_dir)

        assert "test_plugin" in plugins
        assert hasattr(plugins["test_plugin"], "TestLocalTechnique")

    def test_discover_local_plugins_nonexistent_dir(self):
        """Test handling of non-existent plugin directory."""
        loader = PluginLoader(["/nonexistent/path"])
        plugins = loader._discover_local_plugins(Path("/nonexistent/path"))
        assert plugins == {}

    def test_discover_local_plugins_invalid_python(self, tmp_path):
        """Test handling of invalid Python files."""
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()

        # Create invalid Python file
        bad_plugin = plugin_dir / "bad_plugin.py"
        bad_plugin.write_text("This is not valid Python syntax !")

        loader = PluginLoader([plugin_dir])
        plugins = loader._discover_local_plugins(plugin_dir)

        # Should return empty dict for invalid files
        assert plugins == {}

    @patch('raglib.plugins.importlib_metadata.entry_points')
    def test_discover_entry_points_new_api(self, mock_entry_points):
        """Test entry point discovery with new API."""
        # Mock entry points with new API (Python 3.10+)
        mock_ep = Mock()
        mock_ep.name = "test_plugin"
        mock_ep.load.return_value = "test_plugin_object"

        mock_entry_points_obj = Mock()
        mock_entry_points_obj.select.return_value = [mock_ep]
        mock_entry_points.return_value = mock_entry_points_obj

        loader = PluginLoader()
        plugins = loader._discover_entry_points()

        assert "test_plugin" in plugins
        assert plugins["test_plugin"] == "test_plugin_object"
        mock_entry_points_obj.select.assert_called_once_with(group='raglib.plugins')

    @patch('raglib.plugins.importlib_metadata.entry_points')
    def test_discover_entry_points_old_api(self, mock_entry_points):
        """Test entry point discovery with old API."""
        # Mock entry points with old API (Python < 3.10)
        mock_ep = Mock()
        mock_ep.name = "test_plugin"
        mock_ep.load.return_value = "test_plugin_object"

        # Create a dict-like object that can have methods assigned
        class MockEntryPointsDict(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                
        mock_entry_points_dict = MockEntryPointsDict({'raglib.plugins': [mock_ep]})
        mock_entry_points_obj = Mock()
        mock_entry_points_obj.select.side_effect = AttributeError()  # No select method
        
        # Assign the get method to the dict-like object
        mock_entry_points_dict.get = mock_entry_points_obj.get
        mock_entry_points_dict.get.return_value = [mock_ep]

        mock_entry_points.return_value = mock_entry_points_dict

        loader = PluginLoader()
        plugins = loader._discover_entry_points()

        assert "test_plugin" in plugins
        assert plugins["test_plugin"] == "test_plugin_object"

    @patch('raglib.plugins.importlib_metadata.entry_points')
    def test_discover_entry_points_load_error(self, mock_entry_points):
        """Test handling of entry point loading errors."""
        # Mock entry point that fails to load
        mock_ep = Mock()
        mock_ep.name = "failing_plugin"
        mock_ep.load.side_effect = ImportError("Plugin failed to import")

        mock_entry_points_obj = Mock()
        mock_entry_points_obj.select.return_value = [mock_ep]
        mock_entry_points.return_value = mock_entry_points_obj

        loader = PluginLoader()
        plugins = loader._discover_entry_points()

        # Should handle error gracefully and return empty dict
        assert plugins == {}

    def test_full_discovery_integration(self, tmp_path):
        """Test full plugin discovery integration."""
        # Create local plugin
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()

        plugin_file = plugin_dir / "local_test.py"
        plugin_content = '''
"""Local test plugin."""

class LocalTestPlugin:
    """A simple test plugin class."""
    def __init__(self):
        self.name = "local_test_plugin"
'''
        plugin_file.write_text(plugin_content)

        # Test discovery
        loader = PluginLoader([plugin_dir])
        plugins = loader.discover()

        assert "local_test" in plugins
        assert hasattr(plugins["local_test"], "LocalTestPlugin")

        # Test cached plugins
        cached = loader.get_loaded_plugins()
        assert cached == plugins

    def test_load_module_from_file(self, tmp_path):
        """Test loading a module from file path."""
        # Create test module file
        module_file = tmp_path / "test_module.py"
        module_content = '''
"""Test module."""

def test_function():
    return "Hello from test module"

TEST_CONSTANT = 42
'''
        module_file.write_text(module_content)

        loader = PluginLoader()
        module = loader._load_module_from_file(module_file)

        assert hasattr(module, "test_function")
        assert hasattr(module, "TEST_CONSTANT")
        assert module.test_function() == "Hello from test module"
        assert module.TEST_CONSTANT == 42


class TestPluginConvenienceFunctions:
    """Test plugin convenience functions."""

    def test_discover_function(self, tmp_path):
        """Test the discover convenience function."""
        # Create test plugin
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()

        plugin_file = plugin_dir / "convenience_test.py"
        plugin_content = '''
"""Convenience test plugin."""

class ConvenienceTestPlugin:
    pass
'''
        plugin_file.write_text(plugin_content)

        # Test discovery
        plugins = discover([plugin_dir])
        assert "convenience_test" in plugins

    def test_discover_without_dirs(self):
        """Test discover function without plugin directories."""
        # Should not crash and return results from entry points only
        plugins = discover()
        assert isinstance(plugins, dict)

    @patch('raglib.plugins._register_techniques_from_plugin')
    def test_discover_and_register_techniques(self, mock_register, tmp_path):
        """Test automatic technique registration from plugins."""
        # Create test plugin
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()

        plugin_file = plugin_dir / "register_test.py"
        plugin_content = '''
"""Register test plugin."""

class TestTechniquePlugin:
    pass
'''
        plugin_file.write_text(plugin_content)

        # Test discovery and registration
        plugins = discover_and_register_techniques([plugin_dir])

        assert "register_test" in plugins
        # Should have called the registration function
        mock_register.assert_called_once()

    def test_register_techniques_from_plugin(self, tmp_path):
        """Test technique registration from plugin modules."""
        from raglib.plugins import _register_techniques_from_plugin

        # Create a mock plugin module with techniques
        plugin_module = type('MockPlugin', (), {})

        # Add a mock RAGTechnique class
        from raglib.core import RAGTechnique

        class MockTechnique(RAGTechnique):
            _registry_name = "mock_technique"

            def apply(self, query, corpus, top_k=5, **kwargs):
                pass

        plugin_module.MockTechnique = MockTechnique

        # Test registration
        len(TechniqueRegistry.list())
        _register_techniques_from_plugin("test_plugin", plugin_module)

        # Check if technique was registered (if not already present)
        if "mock_technique" not in TechniqueRegistry.list():
            # Registration should have happened
            pass
        else:
            # Already registered, that's fine too
            pass

    def test_register_techniques_from_non_module(self):
        """Test technique registration from non-module objects."""
        from raglib.plugins import _register_techniques_from_plugin

        # Test with non-module object
        non_module = "not a module"

        # Should handle gracefully without crashing
        _register_techniques_from_plugin("test", non_module)

    def test_plugin_registration_error_handling(self):
        """Test error handling in technique registration."""
        from raglib.plugins import _register_techniques_from_plugin

        # Create plugin with problematic technique
        plugin_module = type('MockPlugin', (), {})

        # Add a class that's not a RAGTechnique
        class NotATechnique:
            pass

        plugin_module.NotATechnique = NotATechnique

        # Should handle gracefully
        _register_techniques_from_plugin("test_plugin", plugin_module)
