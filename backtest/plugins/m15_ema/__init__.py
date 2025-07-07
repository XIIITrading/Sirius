# backtest/plugins/m15_ema/__init__.py
"""
M15 EMA Crossover Plugin
Interprets signals from dashboard and coordinates with data system
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable

from backtest.data.polygon_data_manager import PolygonDataManager
from backtest.calculations.indicators.m15_ema import M15EMACalculator, M15EMAResult

logger = logging.getLogger(__name__)

# Plugin metadata
PLUGIN_NAME = "15-Min EMA Crossover"
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
        
        # Define bar requirements
        required_1min_bars = 390  # 26 fifteen-minute bars * 15 = 390
        
        # Start with a reasonable lookback and extend if needed
        lookback_days = 3
        max_lookback_days = 10
        bars_1min = None
        
        logger.info(f"Attempting to fetch {required_1min_bars} 1-minute bars for {symbol}")
        
        while lookback_days <= max_lookback_days:
            start_time = entry_time - timedelta(days=lookback_days)
            
            logger.info(f"Trying {lookback_days}-day lookback: {start_time} to {entry_time}")
            
            # Fetch 1-minute data
            bars_1min = await _data_manager.load_bars(
                symbol=symbol,
                start_time=start_time,
                end_time=entry_time,
                timeframe='1min'
            )
            
            if bars_1min.empty:
                logger.warning(f"No data returned for {lookback_days}-day lookback")
            else:
                logger.info(f"Fetched {len(bars_1min)} bars with {lookback_days}-day lookback")
                
                if len(bars_1min) >= required_1min_bars:
                    # We have enough bars - trim to exactly what we need
                    bars_1min = bars_1min.iloc[-required_1min_bars:]
                    logger.info(f"Successfully obtained required {required_1min_bars} bars")
                    break
            
            # Try more days
            lookback_days += 2
        
        # Check if we got enough data
        if bars_1min is None or bars_1min.empty:
            return _create_error_signal(entry_time, "No data available")
        
        if len(bars_1min) < required_1min_bars:
            return _create_error_signal(
                entry_time, 
                f"Insufficient data: only {len(bars_1min)} bars available, need {required_1min_bars}"
            )
        
        # Run calculation with the exact number of bars needed
        calculator = M15EMACalculator(ema_short=9, ema_long=21)
        result = calculator.calculate(bars_1min, timeframe='1min')
        
        if not result:
            return _create_error_signal(entry_time, "Calculation failed")
        
        # Interpret result into signal
        return _interpret_signal(result, entry_time, direction)
        
    except Exception as e:
        logger.error(f"Error in {PLUGIN_NAME}: {e}", exc_info=True)
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
        progress_callback(0, "Starting 15-minute EMA analysis...")
    
    if progress_callback:
        progress_callback(20, "Fetching required 1-minute bars...")
    
    # Run the analysis
    result = await run_analysis(symbol, entry_time, direction)
    
    if progress_callback:
        progress_callback(100, "Complete")
    
    return result


def _interpret_signal(result: M15EMAResult, entry_time: datetime, 
                     direction: str) -> Dict[str, Any]:
    """Interpret calculation result into standardized signal"""
    
    # Map internal signal to dashboard format
    signal_map = {
        'BULL': 'BULLISH',
        'BEAR': 'BEARISH',
        'NEUTRAL': 'NEUTRAL'
    }
    
    signal_direction = signal_map.get(result.signal, 'NEUTRAL')
    strength = result.signal_strength
    
    # Build display data
    table_data = [
        ["Timeframe", "15-Minute"],
        ["EMA 9", f"${result.ema_9:.2f}"],
        ["EMA 21", f"${result.ema_21:.2f}"],
        ["Spread", f"${result.spread:.2f} ({result.spread_pct:.2f}%)"],
        ["Trend Strength", f"{result.trend_strength:.0f}%"],
        ["Signal Strength", f"{strength:.0f}%"],
        ["15m Bars Processed", f"{result.bars_processed}"],
        ["Price Position", result.price_position.title()],
        ["Last Close", f"${result.last_15min_close:.2f}"],
        ["Last Volume", f"{result.last_15min_volume:,.0f}"]
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
    
    return {
        'plugin_name': PLUGIN_NAME,
        'timestamp': entry_time,
        'signal': {
            'direction': signal_direction,
            'strength': float(strength),
            'confidence': float(strength)
        },
        'details': {
            'timeframe': '15-minute',
            'ema_9': result.ema_9,
            'ema_21': result.ema_21,
            'ema_spread': result.spread,
            'ema_spread_pct': result.spread_pct,
            'trend_strength': result.trend_strength,
            'is_crossover': result.is_crossover,
            'crossover_type': result.crossover_type,
            'price_position': result.price_position,
            'bars_processed': result.bars_processed,
            'last_close': result.last_15min_close,
            'last_volume': result.last_15min_volume,
            'bars_used': 390  # Always using exactly 390 bars now
        },
        'display_data': {
            'summary': f"15-Min EMA 9/21 - {signal_direction}",
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