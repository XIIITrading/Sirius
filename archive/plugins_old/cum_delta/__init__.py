# backtest/plugins/cum_delta/__init__.py
"""
Cumulative Delta Analysis Plugin
Real-time order flow analysis tracking buying vs selling pressure through bid/ask classification.
Provides tick-by-tick delta calculation with multi-timeframe analysis.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, Callable

from .plugin import CumDeltaPlugin

# Configure logging
logger = logging.getLogger(__name__)

# Create plugin instance
_plugin = CumDeltaPlugin()

# Export the main interface
async def run_analysis(symbol: str, entry_time: datetime, direction: str) -> Dict[str, Any]:
    """
    Run cumulative delta analysis.
    
    This is the main entry point for the plugin.
    
    Args:
        symbol: Stock symbol
        entry_time: Entry time (UTC)
        direction: 'LONG' or 'SHORT'
        
    Returns:
        Complete analysis results formatted for display
    """
    try:
        # Validate inputs
        if not _plugin.validate_inputs(symbol, entry_time, direction):
            raise ValueError("Invalid input parameters")
        
        # Run analysis
        result = await _plugin.run_analysis(symbol, entry_time, direction)
        
        logger.info(f"Cumulative delta analysis complete for {symbol} at {entry_time}")
        return result
        
    except Exception as e:
        logger.error(f"Error in cumulative delta analysis: {e}")
        # Return error result
        return {
            'plugin_name': _plugin.name,
            'timestamp': entry_time,
            'error': str(e),
            'signal': {
                'direction': 'NEUTRAL',
                'strength': 0,
                'confidence': 0
            }
        }

# Export plugin metadata
PLUGIN_NAME = _plugin.name
PLUGIN_VERSION = _plugin.version

# Export progress-enabled version
async def run_analysis_with_progress(symbol: str, entry_time: datetime, direction: str,
                                   progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
    """
    Run cumulative delta analysis with progress reporting.
    
    Args:
        symbol: Stock symbol
        entry_time: Entry time (UTC)
        direction: 'LONG' or 'SHORT'
        progress_callback: Function to call with (percentage, message)
        
    Returns:
        Complete analysis results
    """
    try:
        if not _plugin.validate_inputs(symbol, entry_time, direction):
            raise ValueError("Invalid input parameters")
        
        # Run with progress
        result = await _plugin.run_analysis_with_progress(
            symbol, entry_time, direction, progress_callback
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in cumulative delta analysis: {e}")
        if progress_callback:
            progress_callback(100, f"Error: {str(e)}")
            
        return {
            'plugin_name': _plugin.name,
            'timestamp': entry_time,
            'error': str(e),
            'signal': {
                'direction': 'NEUTRAL',
                'strength': 0,
                'confidence': 0
            }
        }

# Optional: Export configuration function for UI
def get_config():
    """Get plugin configuration for UI settings"""
    return _plugin.get_config()