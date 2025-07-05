# backtest/backtest_system/components/plugin_runner.py
"""
Plugin Runner - Discovers and executes plugins with automatic progress tracking
"""

import importlib
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import logging
import inspect

logger = logging.getLogger(__name__)


class ProgressWrapper:
    """Wrapper that provides automatic progress tracking for plugins"""
    
    def __init__(self, plugin_name: str, progress_callback: Optional[Callable[[int, str], None]] = None):
        self.plugin_name = plugin_name
        self.progress_callback = progress_callback
        self.start_time = None
        self.phase_times = {
            'initialization': 2,  # seconds
            'data_fetching': 5,
            'processing': 15,
            'formatting': 2
        }
        self.current_phase = None
        self.phase_start = None
        
    async def run_with_estimated_progress(self, plugin_func, *args, **kwargs):
        """Run plugin with time-based progress estimation"""
        self.start_time = time.time()
        
        # Start progress monitoring task
        progress_task = asyncio.create_task(self._monitor_progress())
        
        try:
            # Run the actual plugin
            result = await plugin_func(*args, **kwargs)
            return result
        finally:
            # Stop progress monitoring
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass
            
            # Ensure we show 100% completion
            if self.progress_callback:
                self.progress_callback(100, "Complete")
    
    async def _monitor_progress(self):
        """Monitor progress based on estimated times"""
        try:
            phases = [
                ('initialization', 5, "Initializing..."),
                ('data_fetching', 25, "Fetching data..."),
                ('processing', 85, "Processing..."),
                ('formatting', 95, "Formatting results...")
            ]
            
            total_time = sum(self.phase_times.values())
            elapsed = 0
            
            for phase, target_pct, message in phases:
                phase_duration = self.phase_times[phase]
                phase_start = time.time()
                
                # Update progress during this phase
                while elapsed < sum(list(self.phase_times.values())[:phases.index((phase, target_pct, message)) + 1]):
                    await asyncio.sleep(0.5)
                    elapsed = time.time() - self.start_time
                    
                    # Calculate progress within this phase
                    phase_elapsed = time.time() - phase_start
                    phase_progress = min(phase_elapsed / phase_duration, 1.0)
                    
                    # Calculate overall progress
                    base_pct = target_pct - (phase_duration / total_time * 100)
                    current_pct = base_pct + (phase_progress * (phase_duration / total_time * 100))
                    
                    if self.progress_callback:
                        self.progress_callback(int(current_pct), message)
                        
        except asyncio.CancelledError:
            pass


class PluginRunner:
    """Discovers and runs backtest plugins with automatic progress tracking"""
    
    def __init__(self, data_manager=None):
        """
        Initialize plugin runner.
        
        Args:
            data_manager: Optional PolygonDataManager instance for tracking requests
        """
        self.plugins = self._discover_plugins()
        self.plugin_metadata = self._analyze_plugins()
        self.data_manager = data_manager
        self._plugin_data_managers = {}  # Track data managers by plugin
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
                            
                            # Try to find data manager in the plugin
                            self._find_plugin_data_manager(plugin_name, module)
                            
                    except Exception as e:
                        logger.error(f"Failed to load plugin {plugin_dir.name}: {e}")
                        
        return plugins
    
    def _find_plugin_data_manager(self, plugin_name: str, module):
        """Find and store reference to plugin's data manager"""
        try:
            # Check if plugin has a _plugin instance with data_manager
            if hasattr(module, '_plugin'):
                plugin_instance = getattr(module, '_plugin')
                if hasattr(plugin_instance, 'data_manager'):
                    self._plugin_data_managers[plugin_name] = plugin_instance.data_manager
                    logger.debug(f"Found data_manager in {plugin_name}")
        except Exception as e:
            logger.debug(f"No data_manager found in {plugin_name}: {e}")
    
    def _analyze_plugins(self) -> Dict[str, Dict]:
        """Analyze plugins to determine their complexity and features"""
        metadata = {}
        
        for plugin_name, module in self.plugins.items():
            meta = {
                'has_progress': hasattr(module, 'run_analysis_with_progress'),
                'is_complex': False,
                'estimated_time': 10  # Default 10 seconds
            }
            
            # Check if it's a complex plugin based on name or features
            if any(keyword in plugin_name.lower() for keyword in ['imbalance', 'volume', 'flow', 'tick']):
                meta['is_complex'] = True
                meta['estimated_time'] = 30
            elif 'structure' in plugin_name.lower():
                meta['estimated_time'] = 15
            elif 'ema' in plugin_name.lower():
                meta['estimated_time'] = 5
            
            metadata[plugin_name] = meta
            
        return metadata
    
    def get_available_plugins(self) -> List[str]:
        """Get list of available plugin names"""
        return list(self.plugins.keys())
    
    def set_data_manager(self, data_manager):
        """Set the data manager for tracking requests"""
        self.data_manager = data_manager
    
    async def run_single_plugin(self, plugin_name: str, symbol: str, 
                               entry_time: datetime, direction: str,
                               progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Run a single plugin and return its results without modification.
        """
        if plugin_name not in self.plugins:
            return {
                'plugin_name': plugin_name,
                'timestamp': entry_time,
                'error': f"Plugin '{plugin_name}' not found"
            }
            
        plugin_module = self.plugins[plugin_name]
        plugin_meta = self.plugin_metadata.get(plugin_name, {})
        
        # Set current plugin in data manager(s)
        self._set_current_plugin(plugin_name)
        
        try:
            # IMPORTANT: Set data manager in plugin if it has set_data_manager function
            if hasattr(plugin_module, 'set_data_manager') and self.data_manager:
                logger.info(f"Setting data manager for {plugin_name}")
                plugin_module.set_data_manager(self.data_manager)
            
            # Check if plugin supports progress reporting
            if plugin_meta.get('has_progress'):
                logger.info(f"Running {plugin_name} with native progress support")
                result = await plugin_module.run_analysis_with_progress(
                    symbol, entry_time, direction, progress_callback
                )
                
            # Complex plugin without progress support
            elif plugin_meta.get('is_complex'):
                logger.info(f"Running complex plugin {plugin_name} with estimated progress")
                
                wrapper = ProgressWrapper(plugin_name, progress_callback)
                wrapper.phase_times = {
                    'initialization': 3,
                    'data_fetching': 10,
                    'processing': 25,
                    'formatting': 2
                }
                
                if progress_callback:
                    progress_callback(0, "Note: This plugin doesn't report detailed progress")
                
                result = await wrapper.run_with_estimated_progress(
                    plugin_module.run_analysis, symbol, entry_time, direction
                )
                
            # Simple plugin with estimated progress
            else:
                logger.info(f"Running simple plugin {plugin_name} with estimated progress")
                
                wrapper = ProgressWrapper(plugin_name, progress_callback)
                result = await wrapper.run_with_estimated_progress(
                    plugin_module.run_analysis, symbol, entry_time, direction
                )
            
            # Return the result exactly as provided by the plugin
            return result
            
        except Exception as e:
            logger.error(f"Error running plugin {plugin_name}: {e}")
            import traceback
            traceback.print_exc()
            
            if progress_callback:
                progress_callback(100, f"Error: {str(e)}")
            
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
    
    def _set_current_plugin(self, plugin_name: str):
        """Set current plugin name in all available data managers"""
        # Set in main data manager if available
        if self.data_manager:
            self.data_manager.set_current_plugin(plugin_name)
            logger.debug(f"Set current plugin to {plugin_name} in main data manager")
        
        # Also set in plugin's own data manager if it has one
        if plugin_name in self._plugin_data_managers:
            plugin_dm = self._plugin_data_managers[plugin_name]
            if hasattr(plugin_dm, 'set_current_plugin'):
                plugin_dm.set_current_plugin(plugin_name)
                logger.debug(f"Set current plugin to {plugin_name} in plugin's data manager")
    
    async def run_multiple_plugins(self, plugin_names: List[str], symbol: str,
                                  entry_time: datetime, direction: str,
                                  progress_callback: Optional[Callable[[str, int, str], None]] = None) -> List[Dict[str, Any]]:
        """Run multiple plugins with progress tracking"""
        results = []
        
        for plugin_name in plugin_names:
            # Create plugin-specific callback
            plugin_callback = None
            if progress_callback:
                plugin_callback = lambda pct, msg: progress_callback(plugin_name, pct, msg)
            
            # Run plugin
            result = await self.run_single_plugin(
                plugin_name, symbol, entry_time, direction, plugin_callback
            )
            results.append(result)
        
        # After all plugins run, generate data report if data manager is available
        if self.data_manager and hasattr(self.data_manager, 'generate_data_report'):
            try:
                logger.info("Generating data manager report...")
                json_file, summary_file = self.data_manager.generate_data_report()
                logger.info(f"Data report generated: {summary_file}")
            except Exception as e:
                logger.error(f"Failed to generate data report: {e}")
        
        return results
    
    def get_plugin_info(self, plugin_name: str) -> Dict[str, Any]:
        """Get information about a plugin"""
        if plugin_name not in self.plugins:
            return {'error': 'Plugin not found'}
        
        meta = self.plugin_metadata.get(plugin_name, {})
        module = self.plugins[plugin_name]
        
        return {
            'name': plugin_name,
            'version': getattr(module, 'PLUGIN_VERSION', 'Unknown'),
            'has_progress': meta.get('has_progress', False),
            'is_complex': meta.get('is_complex', False),
            'estimated_time': meta.get('estimated_time', 10),
            'has_data_manager': plugin_name in self._plugin_data_managers,
            'recommendation': 'Consider implementing run_analysis_with_progress()' 
                            if meta.get('is_complex') and not meta.get('has_progress') else None
        }
    
    def get_all_data_managers(self) -> Dict[str, Any]:
        """Get all data managers for reporting"""
        managers = {}
        
        if self.data_manager:
            managers['main'] = self.data_manager
        
        for plugin_name, dm in self._plugin_data_managers.items():
            managers[f'plugin_{plugin_name}'] = dm
        
        return managers
    
    def generate_consolidated_report(self):
        """Generate a consolidated report from all data managers"""
        all_managers = self.get_all_data_managers()
        
        if not all_managers:
            logger.warning("No data managers found for reporting")
            return
        
        # Generate report from main data manager if available
        if 'main' in all_managers and hasattr(all_managers['main'], 'generate_data_report'):
            try:
                logger.info("Generating consolidated data report...")
                json_file, summary_file = all_managers['main'].generate_data_report()
                logger.info(f"Consolidated report generated: {summary_file}")
                return json_file, summary_file
            except Exception as e:
                logger.error(f"Failed to generate consolidated report: {e}")
                return None, None
        else:
            logger.warning("Main data manager not available for reporting")
            return None, None