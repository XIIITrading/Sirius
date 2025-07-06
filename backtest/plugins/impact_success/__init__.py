"""
Impact Success Plugin
Tracks success rates of large order price impacts
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
import pandas as pd
import numpy as np

from modules.calculations.order_flow.impact_success import ImpactSuccessTracker
from modules.calculations.order_flow.buy_sell_ratio import Trade

logger = logging.getLogger(__name__)

# Plugin metadata
PLUGIN_NAME = "Impact Success"
PLUGIN_VERSION = "2.0.0"
PLUGIN_DESCRIPTION = "Tracks success rates of large order price impacts"

# Plugin configuration
CONFIG = {
    'stats_window_minutes': 15,
    'min_trades_for_std': 100,
    'ratio_threshold': 1.5,
    'stdev_threshold': 1.25,
    'pressure_window_seconds': 60,  # 1-minute bars
    'history_points': 1800  # 30 minutes of data
}

# Initialize tracker at module level
tracker = ImpactSuccessTracker(
    stats_window_minutes=CONFIG['stats_window_minutes'],
    min_trades_for_std=CONFIG['min_trades_for_std'],
    ratio_threshold=CONFIG['ratio_threshold'],
    stdev_threshold=CONFIG['stdev_threshold'],
    pressure_window_seconds=CONFIG['pressure_window_seconds'],
    history_points=CONFIG['history_points']
)

# Module-level data manager (set by backtesting system)
_data_manager = None


def set_data_manager(data_manager):
    """
    Called by the plugin runner to provide the data manager.
    This is how the backtesting system provides data access to plugins.
    """
    global _data_manager
    _data_manager = data_manager
    logger.info(f"Data manager set for {PLUGIN_NAME}")


async def run_analysis(symbol: str, entry_time: datetime, direction: str) -> Dict[str, Any]:
    """
    Run Impact Success analysis.
    
    Args:
        symbol: Stock symbol
        entry_time: Entry time for analysis
        direction: Trade direction (LONG/SHORT)
        
    Returns:
        Analysis results dictionary
    """
    # Use the data manager that was set by the plugin runner
    if _data_manager is None:
        logger.error("Data manager not set. Plugin runner should call set_data_manager() first.")
        return {
            'plugin_name': PLUGIN_NAME,
            'timestamp': entry_time,
            'symbol': symbol,
            'error': 'Data manager not initialized',
            'signal': {'direction': 'ERROR', 'strength': 0, 'confidence': 0}
        }
    
    return await _run_analysis_internal(symbol, entry_time, direction, None)


async def run_analysis_with_progress(symbol: str, entry_time: datetime, direction: str,
                                   progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
    """
    Run Impact Success analysis with progress tracking.
    
    Args:
        symbol: Stock symbol
        entry_time: Entry time for analysis
        direction: Trade direction (LONG/SHORT)
        progress_callback: Progress callback function
        
    Returns:
        Analysis results dictionary
    """
    # Use the data manager that was set by the plugin runner
    if _data_manager is None:
        logger.error("Data manager not set. Plugin runner should call set_data_manager() first.")
        if progress_callback:
            progress_callback(100, "Error: Data manager not initialized")
        return {
            'plugin_name': PLUGIN_NAME,
            'timestamp': entry_time,
            'symbol': symbol,
            'error': 'Data manager not initialized',
            'signal': {'direction': 'ERROR', 'strength': 0, 'confidence': 0}
        }
    
    return await _run_analysis_internal(symbol, entry_time, direction, progress_callback)


async def _run_analysis_internal(symbol: str, entry_time: datetime, direction: str,
                                progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
    """
    Internal analysis implementation that uses the module-level data manager.
    """
    try:
        if progress_callback:
            progress_callback(0, "Fetching trade and quote data...")
        
        # Fetch data window (30 minutes before entry)
        end_time = entry_time
        start_time = entry_time - timedelta(minutes=30)
        
        # Use the module-level data manager
        logger.info(f"Fetching trades for {symbol} from {start_time} to {end_time}")
        trades_df = await _data_manager.load_trades(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time
        )
        
        if progress_callback:
            progress_callback(25, f"Loaded {len(trades_df)} trades")
        
        # Fetch quotes
        logger.info(f"Fetching quotes for {symbol}")
        quotes_df = await _data_manager.load_quotes(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time
        )
        
        if progress_callback:
            progress_callback(50, f"Loaded {len(quotes_df)} quotes")
        
        # Process data
        if progress_callback:
            progress_callback(60, "Processing large orders...")
            
        # Process all the data
        _process_data(trades_df, quotes_df, symbol)
        
        if progress_callback:
            progress_callback(80, "Calculating pressure metrics...")
        
        # Get current stats from the tracker
        stats = tracker.get_current_stats(symbol)
        
        # Generate signal based on pressure
        signal = _generate_signal(stats, direction)
        
        # Get the complete display package
        display_data = _prepare_complete_display_package(stats, symbol, entry_time)
        
        if progress_callback:
            progress_callback(100, "Analysis complete")
        
        return {
            'plugin_name': PLUGIN_NAME,
            'timestamp': entry_time,
            'symbol': symbol,
            'direction': direction,
            'signal': signal,
            'display_data': display_data,
            'stats': stats
        }
        
    except Exception as e:
        logger.error(f"Error in Impact Success analysis: {e}")
        import traceback
        traceback.print_exc()
        
        if progress_callback:
            progress_callback(100, f"Error: {str(e)}")
            
        return {
            'plugin_name': PLUGIN_NAME,
            'timestamp': entry_time,
            'symbol': symbol,
            'error': str(e),
            'signal': {'direction': 'ERROR', 'strength': 0, 'confidence': 0}
        }


def _process_data(trades_df: pd.DataFrame, quotes_df: pd.DataFrame, symbol: str) -> None:
    """
    Process historical data through the tracker.
    """
    # Add symbol column if not present
    if 'symbol' not in trades_df.columns:
        trades_df['symbol'] = symbol
    if 'symbol' not in quotes_df.columns:
        quotes_df['symbol'] = symbol
    
    # First update all quotes
    for timestamp, quote_row in quotes_df.iterrows():
        tracker.update_quote(
            symbol=symbol,
            bid=quote_row['bid'],
            ask=quote_row['ask'],
            timestamp=timestamp
        )
    
    # Then process trades
    large_order_count = 0
    for timestamp, trade_row in trades_df.iterrows():
        # Get the most recent quote at this time
        quotes_before = quotes_df[quotes_df.index <= timestamp]
        if not quotes_before.empty:
            latest_quote = quotes_before.iloc[-1]
            bid = latest_quote['bid']
            ask = latest_quote['ask']
        else:
            bid = None
            ask = None
        
        trade = Trade(
            symbol=symbol,
            price=trade_row['price'],
            size=int(trade_row['size']),
            timestamp=timestamp,
            bid=bid,
            ask=ask
        )
        
        # Process trade - tracker handles everything
        large_order = tracker.process_trade(trade)
        if large_order:
            large_order_count += 1
    
    logger.info(f"Processed {len(trades_df)} trades, detected {large_order_count} large orders")


def _generate_signal(stats: Dict[str, Any], direction: str) -> Dict[str, Any]:
    """Generate trading signal from pressure stats"""
    net_pressure = stats.get('net_pressure', 0)
    total_volume = stats.get('total_buy_volume', 0) + stats.get('total_sell_volume', 0)
    
    # No signal if no large orders
    if total_volume == 0:
        return {
            'direction': 'NEUTRAL',
            'strength': 0,
            'confidence': 0,
            'reason': 'No large orders detected'
        }
    
    # Calculate signal strength based on pressure imbalance
    pressure_ratio = abs(net_pressure) / total_volume if total_volume > 0 else 0
    strength = min(100, pressure_ratio * 100)
    
    # Confidence based on volume
    confidence = min(100, (total_volume / 10000) * 100)  # Scale by expected volume
    
    # Determine signal direction
    pressure_direction = stats.get('pressure_direction', 'NEUTRAL')
    
    # Check alignment with intended direction
    aligned = False
    if direction == 'LONG' and pressure_direction == 'BULLISH':
        aligned = True
    elif direction == 'SHORT' and pressure_direction == 'BEARISH':
        aligned = True
    
    # Build reason
    if pressure_direction == 'BULLISH':
        reason = f"Buy pressure dominates ({stats.get('total_buy_volume', 0):,} vs {stats.get('total_sell_volume', 0):,})"
    elif pressure_direction == 'BEARISH':
        reason = f"Sell pressure dominates ({stats.get('total_sell_volume', 0):,} vs {stats.get('total_buy_volume', 0):,})"
    else:
        reason = "Balanced large order flow"
    
    if aligned:
        reason += f" - Supports {direction}"
    elif pressure_direction != 'NEUTRAL':
        reason += f" - Contradicts {direction}"
    
    return {
        'direction': pressure_direction,
        'strength': strength,
        'confidence': confidence,
        'reason': reason,
        'aligned': aligned
    }


def _prepare_complete_display_package(stats: Dict[str, Any], symbol: str, entry_time: datetime) -> Dict[str, Any]:
    """
    Prepare complete display data package with pressure chart data.
    """
    # Get chart data from tracker
    chart_data = tracker.get_chart_data(symbol)
    
    # Format description text
    description_lines = [
        f"Buy Volume: {stats.get('total_buy_volume', 0):,} ({stats.get('buy_order_count', 0)} orders)",
        f"Sell Volume: {stats.get('total_sell_volume', 0):,} ({stats.get('sell_order_count', 0)} orders)",
        f"Net Pressure: {stats.get('net_pressure', 0):+,}",
        f"Status: {stats.get('interpretation', 'No data')}"
    ]
    description = '\n'.join(description_lines)
    
    # Prepare table data for display
    table_data = [
        ("Total Buy Volume", f"{stats.get('total_buy_volume', 0):,}"),
        ("Total Sell Volume", f"{stats.get('total_sell_volume', 0):,}"),
        ("Net Pressure", f"{stats.get('net_pressure', 0):+,}"),
        ("Buy Order Count", str(stats.get('buy_order_count', 0))),
        ("Sell Order Count", str(stats.get('sell_order_count', 0))),
        ("Cumulative Pressure", f"{stats.get('current_cumulative', 0):+,}"),
        ("Pressure Direction", stats.get('pressure_direction', 'NEUTRAL'))
    ]
    
    # Return complete display package
    return {
        'summary': f"Large Order Pressure: {stats.get('net_pressure', 0):+,}",
        'description': description,
        'table_data': table_data,
        'chart_widget': {
            'module': 'backtest.plugins.impact_success.chart',
            'type': 'ImpactSuccessChart',
            'data': chart_data,
            'entry_time': entry_time.isoformat() if entry_time else None
        }
    }


# Export everything needed
__all__ = [
    'run_analysis', 
    'run_analysis_with_progress', 
    'set_data_manager',
    'PLUGIN_NAME', 
    'PLUGIN_VERSION', 
    'PLUGIN_DESCRIPTION'
]