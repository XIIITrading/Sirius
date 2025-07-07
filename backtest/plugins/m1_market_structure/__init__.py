# backtest/plugins/m1_market_structure/__init__.py
"""
M1 Market Structure Plugin
Analyzes market structure using fractal-based swing point detection
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable

from backtest.data.polygon_data_manager import PolygonDataManager
from backtest.calculations.market_structure.m1_market_structure import (
    MarketStructureAnalyzer,
    MarketStructureSignal
)

logger = logging.getLogger(__name__)

# Plugin metadata
PLUGIN_NAME = "1-Min Market Structure"
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
        
        # Create analyzer
        analyzer = MarketStructureAnalyzer(
            fractal_length=5,
            buffer_size=200,
            min_candles_required=21,
            bars_needed=200
        )
        
        # Calculate data requirements
        bars_needed = analyzer.get_required_bars()
        # Add buffer for weekends/holidays (roughly 3x to be safe)
        estimated_minutes = bars_needed * 3
        start_time = entry_time - timedelta(minutes=estimated_minutes)
        
        # Fetch data through data manager (with circuit breaker protection)
        bars = await _data_manager.load_bars(
            symbol=symbol,
            start_time=start_time,
            end_time=entry_time,
            timeframe='1min'
        )
        
        if bars.empty:
            return _create_error_signal(entry_time, "No data available")
        
        # Trim to exact number of bars needed if we got more
        if len(bars) > bars_needed:
            bars = bars.iloc[-bars_needed:]
        
        # Run analysis up to entry time
        signal = analyzer.process_bars_dataframe(symbol, bars, entry_time)
        
        if not signal:
            # No signal generated, return current state
            current_state = analyzer.get_current_analysis(symbol)
            if current_state:
                return _interpret_signal(current_state, entry_time, direction, is_current_state=True)
            else:
                return _create_error_signal(entry_time, "Insufficient data for analysis")
        
        # Interpret result into signal
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
        progress_callback(0, "Starting market structure analysis...")
    
    if progress_callback:
        progress_callback(20, "Fetching price data...")
    
    # Run the analysis
    result = await run_analysis(symbol, entry_time, direction)
    
    if progress_callback:
        progress_callback(100, "Analysis complete")
    
    return result


def _interpret_signal(signal: MarketStructureSignal, entry_time: datetime, 
                     direction: str, is_current_state: bool = False) -> Dict[str, Any]:
    """Interpret market structure signal into standardized format"""
    
    # Map internal signal to dashboard format
    signal_map = {
        'BULL': 'BULLISH',
        'BEAR': 'BEARISH',
        'NEUTRAL': 'NEUTRAL'
    }
    
    signal_direction = signal_map.get(signal.signal, 'NEUTRAL')
    
    # Extract metrics
    metrics = signal.metrics
    
    # Build display data table
    table_data = [
        ["Timeframe", "1-Minute"],
        ["Current Trend", metrics.get('current_trend', 'N/A')],
        ["Structure Type", signal.structure_type],
        ["Signal Strength", f"{signal.strength:.0f}%"]
    ]
    
    # Add fractal levels if available
    if metrics.get('last_high_fractal'):
        table_data.append(["Last High Fractal", f"${metrics['last_high_fractal']:.2f}"])
    if metrics.get('last_low_fractal'):
        table_data.append(["Last Low Fractal", f"${metrics['last_low_fractal']:.2f}"])
    
    # Add break information
    if metrics.get('last_break_type'):
        table_data.append(["Last Break", metrics['last_break_type']])
        if metrics.get('last_break_price'):
            table_data.append(["Break Price", f"${metrics['last_break_price']:.2f}"])
    
    # Add statistics
    table_data.extend([
        ["Total Fractals", str(metrics.get('fractal_count', 0))],
        ["Structure Breaks", str(metrics.get('structure_breaks', 0))],
        ["Trend Changes", str(metrics.get('trend_changes', 0))],
        ["Candles Analyzed", str(metrics.get('candles_processed', 0))]
    ])
    
    # Direction alignment
    alignment = "Neutral"
    if signal_direction != "NEUTRAL" and not is_current_state:
        if (direction == "LONG" and signal_direction == "BULLISH") or \
           (direction == "SHORT" and signal_direction == "BEARISH"):
            alignment = "Aligned ✓"
        else:
            alignment = "Opposed ✗"
    table_data.append(["Direction Alignment", alignment])
    
    # Adjust strength for current state (no new signal)
    strength = signal.strength if not is_current_state else signal.strength * 0.8
    
    return {
        'plugin_name': PLUGIN_NAME,
        'timestamp': entry_time,
        'signal': {
            'direction': signal_direction,
            'strength': float(strength),
            'confidence': float(strength)
        },
        'details': {
            'timeframe': '1-minute',
            'current_trend': metrics.get('current_trend', 'NEUTRAL'),
            'structure_type': signal.structure_type,
            'last_high_fractal': metrics.get('last_high_fractal'),
            'last_low_fractal': metrics.get('last_low_fractal'),
            'last_break_type': metrics.get('last_break_type'),
            'last_break_price': metrics.get('last_break_price'),
            'fractal_count': metrics.get('fractal_count', 0),
            'structure_breaks': metrics.get('structure_breaks', 0),
            'trend_changes': metrics.get('trend_changes', 0),
            'candles_processed': metrics.get('candles_processed', 0),
            'is_current_state': is_current_state
        },
        'display_data': {
            'summary': f"1-Min Market Structure - {signal_direction} ({signal.structure_type})",
            'description': signal.reason,
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