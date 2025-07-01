# backtest/plugins/m1_ema/__init__.py
"""
M1 EMA Crossover Plugin
Complete self-contained plugin for 1-minute EMA crossover analysis.
"""

import logging
from datetime import datetime
from typing import Dict, Any

from .plugin import M1EMAPlugin

logger = logging.getLogger(__name__)

# Create plugin instance
_plugin = M1EMAPlugin()

# Export the main interface
async def run_analysis(symbol: str, entry_time: datetime, direction: str) -> Dict[str, Any]:
    """
    Run M1 EMA Crossover analysis.
    
    This is the single entry point for the plugin.
    External systems only need to call this function.
    """
    try:
        # Validate inputs
        if not _plugin.validate_inputs(symbol, entry_time, direction):
            raise ValueError("Invalid input parameters")
        
        # Run analysis
        result = await _plugin.run_analysis(symbol, entry_time, direction)
        
        logger.info(f"M1 EMA analysis complete for {symbol} at {entry_time}")
        return result
        
    except Exception as e:
        logger.error(f"Error in M1 EMA analysis: {e}")
        # Return error result in standard format
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

# Export metadata for plugin discovery
PLUGIN_NAME = _plugin.name
PLUGIN_VERSION = _plugin.version