# backtest/plugins/m5_market_structure/__init__.py
"""
M5 Market Structure Plugin
Analyzes market structure using fractal-based swing point detection on 5-minute timeframe
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
import pandas as pd

from backtest.data.polygon_data_manager import PolygonDataManager
from backtest.calculations.market_structure.m5_market_structure import (
    M5MarketStructureAnalyzer,
    MarketStructureSignal
)

logger = logging.getLogger(__name__)

# Plugin metadata
PLUGIN_NAME = "5-Min Market Structure"
PLUGIN_VERSION = "1.0.0"

# Global data manager instance (will be set by dashboard)
_data_manager: Optional[PolygonDataManager] = None


def set_data_manager(data_manager: PolygonDataManager):
    """Set the data manager instance"""
    global _data_manager
    _data_manager = data_manager


def _aggregate_to_5min(bars_1min: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-minute bars to 5-minute bars"""
    if bars_1min.empty:
        return pd.DataFrame()
    
    # Resample to 5-minute bars
    bars_5min = bars_1min.resample('5T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return bars_5min


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
        analyzer = M5MarketStructureAnalyzer(
            fractal_length=3,         # Smaller for 5-min timeframe
            buffer_size=100,         # 100 5-min bars = 500 minutes
            min_candles_required=15,
            bars_needed=100         # Request 100 5-min bars
        )
        
        # Calculate data requirements
        # We need 100 5-min bars = 500 1-min bars
        bars_needed_5min = analyzer.get_required_bars()
        bars_needed_1min = bars_needed_5min * 5
        
        # Add buffer for weekends/holidays (roughly 3x to be safe)
        estimated_minutes = bars_needed_1min * 3
        start_time = entry_time - timedelta(minutes=estimated_minutes)
        
        # Fetch 1-minute data through data manager (with circuit breaker protection)
        bars_1min = await _data_manager.load_bars(
            symbol=symbol,
            start_time=start_time,
            end_time=entry_time,
            timeframe='1min'
        )
        
        if bars_1min.empty:
            return _create_error_signal(entry_time, "No data available")
        
        # Aggregate to 5-minute bars
        bars_5min = _aggregate_to_5min(bars_1min)
        
        if bars_5min.empty:
            return _create_error_signal(entry_time, "No 5-minute bars after aggregation")
        
        logger.info(f"Aggregated {len(bars_1min)} 1-min bars to {len(bars_5min)} 5-min bars")
        
        # Trim to exact number of bars needed if we got more
        if len(bars_5min) > bars_needed_5min:
            bars_5min = bars_5min.iloc[-bars_needed_5min:]
        
        # Run analysis up to entry time
        signal = analyzer.process_bars_dataframe(symbol, bars_5min, entry_time)
        
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
        progress_callback(0, "Starting 5-minute market structure analysis...")
    
    if progress_callback:
        progress_callback(20, "Fetching 1-minute data...")
    
    if progress_callback:
        progress_callback(50, "Aggregating to 5-minute bars...")
    
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
        ["Timeframe", "5-Minute"],
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
        ["5-Min Candles", str(metrics.get('candles_processed', 0))]
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
    
    # Add significance note for 5-minute timeframe
    description = signal.reason
    if signal.structure_type:
        description += ". Mid-timeframe signal - moderate significance."
    
    return {
        'plugin_name': PLUGIN_NAME,
        'timestamp': entry_time,
        'signal': {
            'direction': signal_direction,
            'strength': float(strength),
            'confidence': float(strength)
        },
        'details': {
            'timeframe': '5-minute',
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
            'summary': f"5-Min Market Structure - {signal_direction} ({signal.structure_type})",
            'description': description,
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