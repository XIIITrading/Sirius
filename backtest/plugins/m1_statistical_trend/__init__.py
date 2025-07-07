# backtest/plugins/m1_statistical_trend/__init__.py
"""
M1 Statistical Trend Plugin
Analyzes statistical trends using 1-minute bars
Pure passthrough implementation - no calculations here
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable

from backtest.data.polygon_data_manager import PolygonDataManager
from backtest.calculations.trend.statistical_trend_1min import (
    StatisticalTrend1MinSimplified,
    StatisticalSignal
)

logger = logging.getLogger(__name__)

# Plugin metadata
PLUGIN_NAME = "1-Min Statistical Trend"
PLUGIN_VERSION = "2.0.0"

# Global data manager instance (will be set by dashboard)
_data_manager: Optional[PolygonDataManager] = None


def set_data_manager(data_manager: PolygonDataManager):
    """Set the data manager instance"""
    global _data_manager
    _data_manager = data_manager


async def run_analysis(symbol: str, entry_time: datetime, direction: str) -> Dict[str, Any]:
    """
    Run analysis for the dashboard
    
    Args:
        symbol: Stock symbol
        entry_time: Entry time for analysis
        direction: Trade direction (LONG/SHORT)
        
    Returns:
        Standardized signal dictionary
    """
    try:
        if not _data_manager:
            raise ValueError("Data manager not set. Call set_data_manager first.")
        
        # Set current plugin for tracking
        _data_manager.set_current_plugin(PLUGIN_NAME)
        
        # Create analyzer with default parameters
        analyzer = StatisticalTrend1MinSimplified(lookback_periods=10)
        
        # Calculate data range needed
        # Need at least lookback_periods of data before entry_time
        start_time = entry_time - timedelta(minutes=20)  # Get extra for safety
        
        # Fetch data through data manager (with circuit breaker protection)
        bars_df = await _data_manager.load_bars(
            symbol=symbol,
            start_time=start_time,
            end_time=entry_time,
            timeframe='1min'
        )
        
        if bars_df.empty:
            return _create_error_signal(entry_time, f"No data available for {symbol}")
        
        logger.info(f"Fetched {len(bars_df)} bars from {bars_df.index.min()} to {bars_df.index.max()}")
        
        # Trim to exactly what we need if we got more
        if len(bars_df) > analyzer.lookback_periods:
            bars_df = bars_df.iloc[-analyzer.lookback_periods:]
        
        # Send to calculator - pure passthrough
        signal = analyzer.analyze(
            symbol=symbol,
            bars_df=bars_df,
            entry_time=entry_time
        )
        
        logger.info(f"Received signal: {signal.signal} with {signal.confidence:.1f}% confidence")
        
        # Format for dashboard
        return _interpret_signal(signal, entry_time, direction)
        
    except Exception as e:
        logger.error(f"Error in {PLUGIN_NAME}: {e}")
        return _create_error_signal(entry_time, str(e))


async def run_analysis_with_progress(symbol: str, entry_time: datetime, 
                                   direction: str, 
                                   progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
    """
    Run analysis with progress reporting
    
    Args:
        symbol: Stock symbol
        entry_time: Entry time
        direction: Trade direction
        progress_callback: Optional callback for progress updates
        
    Returns:
        Standardized signal dictionary
    """
    if progress_callback:
        progress_callback(0, "Starting statistical trend analysis...")
    
    if progress_callback:
        progress_callback(20, "Fetching price data...")
    
    # Run the analysis
    result = await run_analysis(symbol, entry_time, direction)
    
    if progress_callback:
        progress_callback(100, "Analysis complete")
    
    return result


def _interpret_signal(signal: StatisticalSignal, entry_time: datetime, 
                     direction: str) -> Dict[str, Any]:
    """
    Format calculator output for dashboard display.
    No calculations - just data restructuring.
    """
    # Check alignment
    alignment_map = {
        'LONG': ['STRONG BUY', 'BUY', 'WEAK BUY'],
        'SHORT': ['STRONG SELL', 'SELL', 'WEAK SELL']
    }
    aligned = signal.signal in alignment_map.get(direction, [])
    
    # Map signal to dashboard format
    signal_descriptions = {
        'STRONG BUY': 'Strong Bullish (Trend > 2x volatility)',
        'BUY': 'Bullish (Trend > 1x volatility)', 
        'WEAK BUY': 'Weak Bullish (Trend > 0.5x volatility)',
        'STRONG SELL': 'Strong Bearish (Trend > 2x volatility)',
        'SELL': 'Bearish (Trend > 1x volatility)',
        'WEAK SELL': 'Weak Bearish (Trend > 0.5x volatility)',
        'NEUTRAL': 'Neutral (No clear trend)'
    }
    
    # For dashboard compatibility
    signal_map = {
        'STRONG BUY': 'BULLISH',
        'BUY': 'BULLISH',
        'WEAK BUY': 'BULLISH',
        'STRONG SELL': 'BEARISH',
        'SELL': 'BEARISH',
        'WEAK SELL': 'BEARISH',
        'NEUTRAL': 'NEUTRAL'
    }
    
    # Build display table
    table_data = [
        ['Signal', f"{signal.signal}"],
        ['Confidence', f"{signal.confidence:.1f}%"],
        ['Trend Strength', f"{signal.trend_strength:.2f}%"],
        ['Volatility Adjusted', f"{signal.volatility_adjusted_strength:.2f}"],
        ['Volume Confirmation', "Yes" if signal.volume_confirmation else "No"],
        ['Direction Alignment', "Aligned ✓" if aligned else "Opposed ✗"]
    ]
    
    # Build description
    desc = signal_descriptions.get(signal.signal, signal.signal)
    
    if aligned:
        desc += f" ✅ Aligned with {direction}"
    else:
        desc += f" ⚠️ Not aligned with {direction}"
    
    return {
        'plugin_name': PLUGIN_NAME,
        'timestamp': entry_time,
        'signal': {
            'direction': signal_map[signal.signal],
            'strength': float(signal.trend_strength),
            'confidence': float(signal.confidence)
        },
        'details': {
            'raw_signal': signal.signal,
            'signal_description': signal_descriptions.get(signal.signal, signal.signal),
            'volatility_adjusted_strength': float(signal.volatility_adjusted_strength),
            'volume_confirmation': signal.volume_confirmation,
            'aligned': aligned
        },
        'display_data': {
            'summary': f"{signal.signal} - {signal.confidence:.0f}% confidence",
            'description': desc,
            'table_data': table_data
        }
    }


def _create_error_signal(entry_time: datetime, error_msg: str) -> Dict[str, Any]:
    """Create error signal response"""
    return {
        'plugin_name': PLUGIN_NAME,
        'timestamp': entry_time,
        'error': error_msg,
        'signal': {
            'direction': 'NEUTRAL',
            'strength': 0,
            'confidence': 0
        }
    }


# Export configuration function for UI
def get_config():
    """Get plugin configuration for UI settings"""
    return {
        'lookback_periods': 10,
        'data_minutes': 15
    }