"""
Impact Success Plugin
Tracks success rates of large order price impacts
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

# Plugin metadata
PLUGIN_NAME = "Impact Success"
PLUGIN_VERSION = "1.0.0"
PLUGIN_DESCRIPTION = "Tracks success rates of large order price impacts"


async def run_analysis(symbol: str, entry_time: datetime, direction: str) -> Dict[str, Any]:
    """
    Run Impact Success analysis.
    
    Args:
        symbol: Stock symbol
        entry_time: Entry time for analysis
        direction: Trade direction (LONG/SHORT)
        
    Returns:
        Analysis results dictionary
    """
    try:
        # Import here to avoid circular imports
        from .plugin import Plugin
        
        # Create plugin instance
        plugin = Plugin()
        
        # Get data manager from somewhere (this is a bit tricky)
        # For now, we'll import and create one
        from backtest.data.polygon_data_manager import PolygonDataManager
        data_manager = PolygonDataManager(disable_polygon_cache=True)
        
        # Run the analysis
        result = await plugin.run(symbol, entry_time, direction, data_manager)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in Impact Success analysis: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'plugin_name': PLUGIN_NAME,
            'timestamp': entry_time,
            'symbol': symbol,
            'error': str(e),
            'signal': {'direction': 'ERROR', 'strength': 0, 'confidence': 0}
        }


async def run_analysis_with_progress(symbol: str, entry_time: datetime, direction: str,
                                   progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
    """
    Run Impact Success analysis with progress tracking.
    
    Args:
        symbol: Stock symbol
        entry_time: Entry time for analysis
        direction: Trade direction (LONG/SHORT)
        progress_callback: Progress callback function
        
    Returns:
        Analysis results dictionary
    """
    try:
        # Import here to avoid circular imports
        from .plugin import Plugin
        from backtest.data.polygon_data_manager import PolygonDataManager
        
        # Create plugin instance
        plugin = Plugin()
        
        # Create data manager
        data_manager = PolygonDataManager(disable_polygon_cache=True)
        data_manager.set_current_plugin(PLUGIN_NAME)
        
        # Run with progress
        result = await plugin.run(symbol, entry_time, direction, data_manager, progress_callback)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in Impact Success analysis: {e}")
        
        if progress_callback:
            progress_callback(100, f"Error: {str(e)}")
            
        return {
            'plugin_name': PLUGIN_NAME,
            'timestamp': entry_time,
            'symbol': symbol,
            'error': str(e),
            'signal': {'direction': 'ERROR', 'strength': 0, 'confidence': 0}
        }


# Export everything needed
__all__ = ['run_analysis', 'run_analysis_with_progress', 'PLUGIN_NAME', 'PLUGIN_VERSION']