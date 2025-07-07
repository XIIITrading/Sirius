# backtest/plugins/m5_statistical_trend/__init__.py
"""
5-Minute Statistical Trend Plugin
Modernized single-file plugin following M1 pattern
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from backtest.calculations.trend.statistical_trend_5min import StatisticalTrend5Min, PositionSignal5Min

# Configure logging
logger = logging.getLogger(__name__)

# Plugin metadata
PLUGIN_NAME = "5-Min Statistical Trend"
PLUGIN_VERSION = "2.0.0"

# Global data manager instance (set by dashboard)
_data_manager = None


def set_data_manager(data_manager):
    """
    Set the data manager instance.
    Called by the dashboard during initialization.
    
    Args:
        data_manager: PolygonDataManager instance
    """
    global _data_manager
    _data_manager = data_manager
    logger.info(f"{PLUGIN_NAME} data manager set")


async def run_analysis(symbol: str, entry_time: datetime, direction: str) -> Dict[str, Any]:
    """
    Run 5-Minute Statistical Trend analysis.
    Pure passthrough to calculation module - no calculations here.
    
    Args:
        symbol: Stock symbol
        entry_time: Entry time (UTC)
        direction: 'LONG' or 'SHORT'
        
    Returns:
        Complete analysis results formatted for dashboard
    """
    try:
        # Validate data manager
        if not _data_manager:
            raise ValueError("Data manager not set. Call set_data_manager() first.")
        
        # Set current plugin for data manager
        _data_manager.set_current_plugin(PLUGIN_NAME)
        
        # Create analyzer
        analyzer = StatisticalTrend5Min(lookback_periods=10)
        
        # Calculate time range
        lookback_minutes = analyzer.lookback_periods * 5 + 10  # 60 minutes total
        start_time = entry_time - timedelta(minutes=lookback_minutes)
        
        logger.info(f"Fetching 5-min bars for {symbol} from {start_time} to {entry_time}")
        
        # Use load_bars ONLY - with correct parameters
        bars_df = await _data_manager.load_bars(
            symbol=symbol,
            start_time=start_time,
            end_time=entry_time,
            timeframe='5min'
        )
        
        if bars_df is None or bars_df.empty:
            raise ValueError(f"No bar data available for {symbol}")
        
        logger.info(f"Fetched {len(bars_df)} bars for analysis")
        
        # Use analyze method for simplified version
        signal = analyzer.analyze(symbol, bars_df, entry_time)
        
        # Format results for dashboard
        return _interpret_signal(signal, entry_time, direction)
        
    except Exception as e:
        logger.error(f"Error in {PLUGIN_NAME} analysis: {e}")
        return _create_error_signal(entry_time, str(e))


async def run_analysis_with_progress(
    symbol: str, 
    entry_time: datetime, 
    direction: str,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Run analysis with optional progress callback.
    
    Args:
        symbol: Stock symbol
        entry_time: Entry time (UTC)
        direction: 'LONG' or 'SHORT'
        progress_callback: Optional callback for progress updates
        
    Returns:
        Complete analysis results
    """
    if progress_callback:
        progress_callback(0.1, "Starting 5-min statistical trend analysis...")
    
    try:
        if progress_callback:
            progress_callback(0.3, "Fetching 5-minute bar data...")
        
        result = await run_analysis(symbol, entry_time, direction)
        
        if progress_callback:
            progress_callback(1.0, "Analysis complete")
        
        return result
        
    except Exception as e:
        if progress_callback:
            progress_callback(1.0, f"Error: {str(e)}")
        raise


def _interpret_signal(signal: PositionSignal5Min, entry_time: datetime, direction: str) -> Dict[str, Any]:
    """
    Format calculation results for dashboard display.
    
    Args:
        signal: PositionSignal5Min from analyzer
        entry_time: Entry time
        direction: Trade direction
        
    Returns:
        Formatted results dictionary
    """
    # Determine alignment
    aligned = False
    if (direction == 'LONG' and signal.bias == 'BULLISH') or \
       (direction == 'SHORT' and signal.bias == 'BEARISH'):
        aligned = True
    
    # Build summary description
    description = f"{signal.signal}"
    if signal.volume_confirmation:
        description += " with volume confirmation"
    
    if aligned:
        description += f" | ✅ Bias aligned with {direction} trade"
    else:
        description += f" | ⚠️ Bias not aligned with {direction} trade"
    
    # Build summary rows for display
    summary_rows = [
        ['Signal', signal.signal],
        ['Market Bias', signal.bias],
        ['Confidence', f"{signal.confidence:.0f}%"],
        ['Trend Strength', f"{signal.trend_strength:.2f}%"],
        ['Volatility-Adjusted', f"{signal.volatility_adjusted_strength:.2f}x"],
        ['Volume Confirmation', '✅ Yes' if signal.volume_confirmation else '❌ No'],
    ]
    
    # Add interpretation
    if signal.volatility_adjusted_strength >= 1.5:
        summary_rows.append(['Interpretation', 'Strong trend relative to noise'])
    elif signal.volatility_adjusted_strength >= 0.75:
        summary_rows.append(['Interpretation', 'Moderate trend with good clarity'])
    else:
        summary_rows.append(['Interpretation', 'Weak or noisy market conditions'])
    
    return {
        'plugin_name': PLUGIN_NAME,
        'timestamp': entry_time,
        'signal': {
            'direction': signal.bias,  # Dashboard expects bias format
            'strength': signal.volatility_adjusted_strength * 33.33,  # Scale to 0-100
            'confidence': signal.confidence
        },
        'details': {
            'signal': signal.signal,
            'bias': signal.bias,
            'trend_strength': signal.trend_strength,
            'volatility_adjusted_strength': signal.volatility_adjusted_strength,
            'volume_confirmation': signal.volume_confirmation,
            'aligned': aligned,
            'price': signal.price
        },
        'display_data': {
            'summary': f"{signal.signal} - {signal.bias} bias",
            'description': description,
            'table_data': summary_rows
        }
    }


def _create_error_signal(entry_time: datetime, error_message: str) -> Dict[str, Any]:
    """
    Create standardized error response.
    
    Args:
        entry_time: Entry time
        error_message: Error description
        
    Returns:
        Error response dictionary
    """
    return {
        'plugin_name': PLUGIN_NAME,
        'timestamp': entry_time,
        'error': error_message,
        'signal': {
            'direction': 'NEUTRAL',
            'strength': 0,
            'confidence': 0
        },
        'details': {
            'error': error_message
        },
        'display_data': {
            'summary': 'Analysis Error',
            'description': f"Error: {error_message}",
            'table_data': [['Error', error_message]]
        }
    }


# Optional: Export configuration function for UI
def get_config():
    """Get plugin configuration for UI settings"""
    return {
        'lookback_periods': 10,
        'timeframe': '5min',
        'analysis_minutes': 50,  # 10 periods * 5 minutes
        'version': PLUGIN_VERSION
    }