"""Plugin discovery and loading system for RAGLib."""

import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Any, Union

try:
    import importlib.metadata as importlib_metadata
except ImportError:
    # Python < 3.8
    import importlib_metadata

from .registry import TechniqueRegistry

logger = logging.getLogger(__name__)


class PluginLoader:
    """Loads and discovers plugins from entry points and local directories."""

    def __init__(self, plugin_dirs: list[Union[str, Path]] = None):
        """
        Initialize plugin loader.

        Args:
            plugin_dirs: Optional list of directories to scan for local plugins
        """
        self.plugin_dirs = [Path(d) for d in (plugin_dirs or [])]
        self._loaded_plugins = {}

    def discover(self) -> dict[str, Any]:
        """
        Discover and load all plugins.

        Returns:
            Dictionary mapping plugin names to their loaded objects
        """
        plugins = {}

        # Load entry point plugins
        entry_point_plugins = self._discover_entry_points()
        plugins.update(entry_point_plugins)

        # Load local directory plugins
        for plugin_dir in self.plugin_dirs:
            local_plugins = self._discover_local_plugins(plugin_dir)
            plugins.update(local_plugins)

        # Store loaded plugins
        self._loaded_plugins = plugins

        return plugins

    def _discover_entry_points(self) -> dict[str, Any]:
        """Discover plugins registered via entry points."""
        plugins = {}

        try:
            # Get entry points for raglib.plugins group
            entry_points = importlib_metadata.entry_points()

            # Handle different importlib.metadata versions
            if hasattr(entry_points, 'select'):
                # New API (Python 3.10+)
                raglib_entries = entry_points.select(group='raglib.plugins')
            else:
                # Old API
                raglib_entries = entry_points.get('raglib.plugins', [])

            for entry_point in raglib_entries:
                try:
                    logger.info(f"Loading entry point plugin: {entry_point.name}")
                    plugin_obj = entry_point.load()
                    plugins[entry_point.name] = plugin_obj
                    logger.info(f"Successfully loaded plugin: {entry_point.name}")

                except Exception as e:
                    logger.error(
                        f"Failed to load entry point plugin {entry_point.name}: {e}"
                    )

        except Exception as e:
            logger.error(f"Error discovering entry point plugins: {e}")

        return plugins

    def _discover_local_plugins(self, plugin_dir: Path) -> dict[str, Any]:
        """Discover plugins from a local directory."""
        plugins = {}

        if not plugin_dir.exists() or not plugin_dir.is_dir():
            logger.warning(f"Plugin directory does not exist: {plugin_dir}")
            return plugins

        # Find all .py files in the directory
        python_files = list(plugin_dir.glob("*.py"))
        python_files = [f for f in python_files if f.name != "__init__.py"]

        for py_file in python_files:
            try:
                logger.info(f"Loading local plugin: {py_file.name}")
                plugin_name = py_file.stem
                plugin_obj = self._load_module_from_file(py_file)
                plugins[plugin_name] = plugin_obj
                logger.info(f"Successfully loaded local plugin: {plugin_name}")

            except Exception as e:
                logger.error(f"Failed to load local plugin {py_file}: {e}")

        return plugins

    def _load_module_from_file(self, file_path: Path) -> Any:
        """Load a Python module from file path."""
        module_name = f"raglib_plugin_{file_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)

        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module

    def get_loaded_plugins(self) -> dict[str, Any]:
        """Get already loaded plugins."""
        return self._loaded_plugins.copy()


# Global plugin loader instance
_plugin_loader = PluginLoader()


def discover(plugin_dirs: list[Union[str, Path]] = None) -> dict[str, Any]:
    """
    Convenience function to discover plugins.

    Args:
        plugin_dirs: Optional list of directories to scan for local plugins

    Returns:
        Dictionary mapping plugin names to their loaded objects
    """
    global _plugin_loader

    if plugin_dirs:
        _plugin_loader = PluginLoader(plugin_dirs)

    return _plugin_loader.discover()


def get_loaded_plugins() -> dict[str, Any]:
    """Get already loaded plugins."""
    return _plugin_loader.get_loaded_plugins()


def discover_and_register_techniques(
    plugin_dirs: list[Union[str, Path]] = None
) -> dict[str, Any]:
    """
    Discover plugins and automatically register any techniques they provide.

    This is a convenience function that discovers plugins and then scans them
    for RAGTechnique classes that should be registered.

    Args:
        plugin_dirs: Optional list of directories to scan for local plugins

    Returns:
        Dictionary mapping plugin names to their loaded objects
    """
    plugins = discover(plugin_dirs)

    # Scan plugins for techniques that might need registration
    for plugin_name, plugin_obj in plugins.items():
        try:
            _register_techniques_from_plugin(plugin_name, plugin_obj)
        except Exception as e:
            logger.error(f"Error registering techniques from plugin {plugin_name}: {e}")

    return plugins


def _register_techniques_from_plugin(plugin_name: str, plugin_obj: Any) -> None:
    """
    Scan a plugin module for RAGTechnique classes and register them.

    This is a best-effort function that looks for classes that might be
    techniques and attempts to register them if they're not already registered.
    """
    if not hasattr(plugin_obj, '__dict__'):
        return

    from .core import RAGTechnique

    for attr_name in dir(plugin_obj):
        if attr_name.startswith('_'):
            continue

        attr = getattr(plugin_obj, attr_name, None)

        # Check if it's a class that inherits from RAGTechnique
        if (isinstance(attr, type) and
            issubclass(attr, RAGTechnique) and
            attr != RAGTechnique):

            try:
                # Check if already registered
                technique_name = getattr(attr, '_registry_name', attr.__name__.lower())

                if technique_name not in TechniqueRegistry.list():
                    logger.info(
                        f"Auto-registering technique {technique_name} "
                        f"from plugin {plugin_name}"
                    )
                    TechniqueRegistry.register(technique_name)(attr)

            except Exception as e:
                logger.error(
                    f"Failed to register technique {attr.__name__} "
                    f"from plugin {plugin_name}: {e}"
                )
