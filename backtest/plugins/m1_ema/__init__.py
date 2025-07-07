# backtest/plugins/m1_ema/__init__.py
"""
M1 EMA Crossover Plugin
Interprets signals from dashboard and coordinates with data system
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable

from backtest.data.polygon_data_manager import PolygonDataManager
from backtest.calculations.indicators.m1_ema import M1EMACalculator, M1EMAResult

logger = logging.getLogger(__name__)

# Plugin metadata
PLUGIN_NAME = "1-Min EMA Crossover"
PLUGIN_VERSION = "1.0.0"

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
        
        # Define data requirements
        lookback_minutes = 120  # 2 hours for EMA calculation
        start_time = entry_time - timedelta(minutes=lookback_minutes)
        
        # Fetch data through data manager (with circuit breaker protection)
        bars = await _data_manager.load_bars(
            symbol=symbol,
            start_time=start_time,
            end_time=entry_time,
            timeframe='1min'
        )
        
        if bars.empty:
            return _create_error_signal(entry_time, "No data available")
        
        # Run calculation - use correct parameter names
        calculator = M1EMACalculator(ema_short=9, ema_long=21)
        result = calculator.calculate(bars)
        
        if not result:
            return _create_error_signal(entry_time, "Insufficient data for calculation")
        
        # Interpret result into signal
        return _interpret_signal(result, entry_time, direction)
        
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
        progress_callback(0, "Starting analysis...")
    
    if progress_callback:
        progress_callback(20, "Fetching data...")
    
    # Run the analysis
    result = await run_analysis(symbol, entry_time, direction)
    
    if progress_callback:
        progress_callback(100, "Complete")
    
    return result


def _interpret_signal(result: M1EMAResult, entry_time: datetime, 
                     direction: str) -> Dict[str, Any]:
    """Interpret calculation result into standardized signal"""
    
    # Map internal signal to dashboard format
    signal_map = {
        'BULL': 'BULLISH',
        'BEAR': 'BEARISH',
        'NEUTRAL': 'NEUTRAL'
    }
    
    signal_direction = signal_map.get(result.signal, 'NEUTRAL')
    
    # Build display data
    table_data = [
        ["Timeframe", "1-Minute"],
        ["EMA 9", f"${result.ema_9:.2f}"],
        ["EMA 21", f"${result.ema_21:.2f}"],
        ["Spread", f"${result.spread:.2f} ({result.spread_pct:.2f}%)"],
        ["Trend Strength", f"{result.trend_strength:.0f}%"],
        ["Signal Strength", f"{result.signal_strength:.0f}%"],
        ["Price Position", result.price_position.replace('_', ' ').title()]
    ]
    
    if result.is_crossover:
        table_data.append(["Crossover", result.crossover_type.title()])
    
    # Direction alignment
    alignment = "Neutral"
    if signal_direction != "NEUTRAL":
        if (direction == "LONG" and signal_direction == "BULLISH") or \
           (direction == "SHORT" and signal_direction == "BEARISH"):
            alignment = "Aligned ✓"
        else:
            alignment = "Opposed ✗"
    table_data.append(["Direction Alignment", alignment])
    
    # Add volume information
    table_data.append(["Last Volume", f"{result.last_1min_volume:,.0f}"])
    
    return {
        'plugin_name': PLUGIN_NAME,
        'timestamp': entry_time,
        'signal': {
            'direction': signal_direction,
            'strength': float(result.signal_strength),
            'confidence': float(result.signal_strength)
        },
        'details': {
            'timeframe': '1-minute',
            'ema_9': result.ema_9,
            'ema_21': result.ema_21,
            'ema_spread': result.spread,
            'ema_spread_pct': result.spread_pct,
            'trend_strength': result.trend_strength,
            'is_crossover': result.is_crossover,
            'crossover_type': result.crossover_type,
            'price_position': result.price_position,
            'last_price': result.last_1min_close,
            'bars_processed': result.bars_processed
        },
        'display_data': {
            'summary': f"1-Min EMA 9/21 - {signal_direction}",
            'description': result.reason,
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