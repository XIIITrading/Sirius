# backtest/plugins/tick_flow_analysis/__init__.py
"""
Tick Flow Analysis Plugin
Real-time order flow analysis using tick-by-tick trade classification.
"""

import logging
from datetime import datetime
from typing import Dict, Any

from .plugin import TickFlowPlugin

# Configure logging
logger = logging.getLogger(__name__)

# Create plugin instance
_plugin = TickFlowPlugin()

# Export the main interface
async def run_analysis(symbol: str, entry_time: datetime, direction: str) -> Dict[str, Any]:
    """
    Run Tick Flow analysis.
    
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
        
        logger.info(f"Tick Flow analysis complete for {symbol} at {entry_time}")
        return result
        
    except Exception as e:
        logger.error(f"Error in Tick Flow analysis: {e}")
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