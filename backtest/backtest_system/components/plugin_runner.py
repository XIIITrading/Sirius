"""
Plugin Runner - Discovers and executes plugins
"""

import importlib
import asyncio
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PluginRunner:
    """Discovers and runs backtest plugins"""
    
    def __init__(self):
        self.plugins = self._discover_plugins()
        logger.info(f"Discovered {len(self.plugins)} plugins")
        
    def _discover_plugins(self) -> Dict[str, Any]:
        """Discover available plugins"""
        plugins = {}
        
        # Get plugins directory
        plugins_dir = Path(__file__).parent.parent.parent / "plugins"
        
        # Find all plugin directories (excluding __pycache__ and base files)
        for plugin_dir in plugins_dir.iterdir():
            if (plugin_dir.is_dir() and 
                not plugin_dir.name.startswith('_') and
                plugin_dir.name != '__pycache__'):
                
                # Check if it has __init__.py with run_analysis
                init_file = plugin_dir / "__init__.py"
                if init_file.exists():
                    try:
                        # Import the plugin module
                        module_name = f"backtest.plugins.{plugin_dir.name}"
                        module = importlib.import_module(module_name)
                        
                        # Check if it has run_analysis function
                        if hasattr(module, 'run_analysis'):
                            plugin_name = getattr(module, 'PLUGIN_NAME', plugin_dir.name)
                            plugins[plugin_name] = module
                            logger.info(f"Loaded plugin: {plugin_name}")
                            
                    except Exception as e:
                        logger.error(f"Failed to load plugin {plugin_dir.name}: {e}")
                        
        return plugins
    
    def get_available_plugins(self) -> List[str]:
        """Get list of available plugin names"""
        return list(self.plugins.keys())
    
    async def run_single_plugin(self, plugin_name: str, symbol: str, 
                               entry_time: datetime, direction: str) -> Dict[str, Any]:
        """Run a single plugin"""
        if plugin_name not in self.plugins:
            return {
                'plugin_name': plugin_name,
                'timestamp': entry_time,
                'error': f"Plugin '{plugin_name}' not found"
            }
            
        plugin_module = self.plugins[plugin_name]
        
        try:
            # Run the plugin's analysis
            result = await plugin_module.run_analysis(symbol, entry_time, direction)
            return result
            
        except Exception as e:
            logger.error(f"Error running plugin {plugin_name}: {e}")
            return {
                'plugin_name': plugin_name,
                'timestamp': entry_time,
                'error': str(e),
                'signal': {
                    'direction': 'NEUTRAL',
                    'strength': 0,
                    'confidence': 0
                }
            }
    
    async def run_multiple_plugins(self, plugin_names: List[str], symbol: str,
                                  entry_time: datetime, direction: str) -> List[Dict[str, Any]]:
        """Run multiple plugins in parallel"""
        tasks = []
        
        for plugin_name in plugin_names:
            task = self.run_single_plugin(plugin_name, symbol, entry_time, direction)
            tasks.append(task)
            
        # Run all plugins concurrently
        results = await asyncio.gather(*tasks)
        
        return results