# backtest/plugins/m15_statistical_trend/__init__.py
"""
15-Minute Statistical Trend Plugin
Self-contained plugin for market regime analysis using statistical methods.
Provides daily trading bias and major trend identification.
"""

import logging
from datetime import datetime
from typing import Dict, Any

from .plugin import M15StatisticalTrendPlugin

# Configure logging
logger = logging.getLogger(__name__)

# Create plugin instance
_plugin = M15StatisticalTrendPlugin()

# Export the main interface
async def run_analysis(symbol: str, entry_time: datetime, direction: str) -> Dict[str, Any]:
    """
    Run 15-Minute Statistical Trend analysis for market regime.
    
    This is the single entry point for the plugin.
    
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
        
        logger.info(f"15-Min Statistical Trend analysis complete for {symbol} at {entry_time}")
        return result
        
    except Exception as e:
        logger.error(f"Error in 15-Min Statistical Trend analysis: {e}")
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