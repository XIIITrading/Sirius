# backtest/plugins/plugin_loader.py
"""
Dynamic plugin loader for backtest calculations.
Discovers and loads all plugins following the standard structure.
"""

import os
import sys
import importlib
import logging
from pathlib import Path
from typing import Dict, List, Optional

from plugins.base_plugin import BacktestPlugin, PluginRegistry

logger = logging.getLogger(__name__)


class PluginLoader:
    """Dynamically loads calculation plugins"""
    
    def __init__(self, plugins_dir: Optional[Path] = None):
        """
        Initialize plugin loader.
        
        Args:
            plugins_dir: Directory containing plugins (default: ./plugins)
        """
        if plugins_dir is None:
            self.plugins_dir = Path(__file__).parent
        else:
            self.plugins_dir = Path(plugins_dir)
            
        self.registry = PluginRegistry()
        
        # Ensure plugins directory is in path
        if str(self.plugins_dir) not in sys.path:
            sys.path.insert(0, str(self.plugins_dir))
    
    def load_all_plugins(self) -> Dict[str, BacktestPlugin]:
        """
        Discover and load all plugins in the plugins directory.
        
        Returns:
            Dict of adapter_name -> plugin instance
        """
        logger.info(f"Loading plugins from {self.plugins_dir}")
        
        # Find all plugin directories
        plugin_dirs = [
            d for d in self.plugins_dir.iterdir()
            if d.is_dir() and not d.name.startswith('_') and d.name != 'base_plugin'
        ]
        
        for plugin_dir in plugin_dirs:
            try:
                self._load_plugin(plugin_dir)
            except Exception as e:
                logger.error(f"Failed to load plugin from {plugin_dir}: {e}")
        
        loaded_plugins = self.registry.get_all_plugins()
        logger.info(f"Loaded {len(loaded_plugins)} plugins: {list(loaded_plugins.keys())}")
        
        return loaded_plugins
    
    def _load_plugin(self, plugin_dir: Path) -> None:
        """
        Load a single plugin from directory.
        
        Args:
            plugin_dir: Path to plugin directory
        """
        plugin_name = plugin_dir.name
        
        # Check for required files
        init_file = plugin_dir / '__init__.py'
        plugin_file = plugin_dir / 'plugin.py'
        
        if not init_file.exists() or not plugin_file.exists():
            logger.debug(f"Skipping {plugin_name}: missing required files")
            return
        
        try:
            # Import the plugin module
            module_name = f"{plugin_name}"
            module = importlib.import_module(module_name)
            
            # Get the plugin instance
            if hasattr(module, 'plugin'):
                plugin = module.plugin
                
                # Validate it's a BacktestPlugin
                if isinstance(plugin, BacktestPlugin):
                    self.registry.register(plugin)
                    logger.info(f"Loaded plugin: {plugin.name} v{plugin.version}")
                else:
                    logger.error(f"Invalid plugin in {plugin_name}: not a BacktestPlugin instance")
            else:
                logger.error(f"No 'plugin' export found in {plugin_name}")
                
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}", exc_info=True)
    
    def get_registry(self) -> PluginRegistry:
        """Get the plugin registry"""
        return self.registry
    
    def get_plugin(self, adapter_name: str) -> Optional[BacktestPlugin]:
        """Get a specific plugin by adapter name"""
        return self.registry.get_plugin(adapter_name)
    
    def get_adapter_configs(self) -> Dict[str, Dict]:
        """
        Get adapter configurations for all plugins.
        Used by BacktestEngine to initialize adapters.
        """
        configs = {}
        for adapter_name, plugin in self.registry.get_all_plugins().items():
            configs[adapter_name] = {
                'adapter_class': plugin.adapter_class,
                'adapter_config': plugin.get_adapter_config()
            }
        return configs