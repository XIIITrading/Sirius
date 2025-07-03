# backtest/plugins/bid_ask_ratio/__init__.py

"""
Bid/Ask Ratio Plugin
Real-time buy/sell pressure visualization with 30-minute rolling window.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, Callable

from .plugin import BidAskRatioPlugin

# Configure logging
logger = logging.getLogger(__name__)

# Create plugin instance
_plugin = BidAskRatioPlugin()

# Export the main interface
async def run_analysis(symbol: str, entry_time: datetime, direction: str) -> Dict[str, Any]:
    """
    Run Bid/Ask Ratio analysis without progress tracking.
    
    This is for backward compatibility.
    
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
        
        logger.info(f"Bid/Ask Ratio analysis complete for {symbol} at {entry_time}")
        return result
        
    except Exception as e:
        logger.error(f"Error in Bid/Ask Ratio analysis: {e}")
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

# Export the progress-enabled version
async def run_analysis_with_progress(symbol: str, entry_time: datetime, direction: str,
                                   progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
    """
    Run Bid/Ask Ratio analysis with progress tracking.
    
    This is the preferred interface for the plugin runner.
    
    Args:
        symbol: Stock symbol
        entry_time: Entry time (UTC)
        direction: 'LONG' or 'SHORT'
        progress_callback: Optional callback for progress updates
        
    Returns:
        Complete analysis results formatted for display
    """
    try:
        # Validate inputs
        if not _plugin.validate_inputs(symbol, entry_time, direction):
            raise ValueError("Invalid input parameters")
        
        # Run analysis with progress
        result = await _plugin.run_analysis_with_progress(
            symbol, entry_time, direction, progress_callback
        )
        
        logger.info(f"Bid/Ask Ratio analysis complete for {symbol} at {entry_time}")
        return result
        
    except Exception as e:
        logger.error(f"Error in Bid/Ask Ratio analysis: {e}")
        # Call progress callback with error if provided
        if progress_callback:
            progress_callback(100, f"Error: {str(e)}")
        
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

# Optional: Export configuration function for UI
def get_config():
    """Get plugin configuration for UI settings"""
    return _plugin.get_config()