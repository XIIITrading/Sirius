"""
Impact Success Plugin
Tracks success rates of large order price impacts
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import pandas as pd
import numpy as np

from backtest.calculations.order_flow.impact_success import ImpactSuccessTracker
from backtest.data.trade_quote_aligner import TradeQuoteAligner

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
    'history_points': 1800,  # 30 minutes of data
    'max_quote_age_ms': 500,  # Maximum quote staleness for alignment
    'min_confidence': 0.7,  # Minimum confidence for aligned trades
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


# Simple Trade dataclass for compatibility with ImpactSuccessTracker
@dataclass
class Trade:
    """Simple trade representation for the tracker"""
    symbol: str
    price: float
    size: int
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None


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
            progress_callback(0, "Initializing components...")
        
        # Set the current plugin for tracking
        _data_manager.set_current_plugin(PLUGIN_NAME)
        
        # Initialize TradeQuoteAligner
        aligner = TradeQuoteAligner(
            max_quote_age_ms=CONFIG['max_quote_age_ms'],
            min_confidence_threshold=CONFIG['min_confidence']
        )
        
        # Fetch data window (30 minutes before entry)
        end_time = entry_time
        start_time = entry_time - timedelta(minutes=30)
        
        if progress_callback:
            progress_callback(10, "Fetching trade data...")
        
        # Fetch trades and quotes separately (following buy_sell_ratio pattern)
        logger.info(f"Fetching trades for {symbol} from {start_time} to {end_time}")
        trades_df = await _data_manager.load_trades(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time
        )
        
        if progress_callback:
            progress_callback(30, f"Loaded {len(trades_df)} trades")
        
        # Fetch quotes
        logger.info(f"Fetching quotes for {symbol}")
        quotes_df = await _data_manager.load_quotes(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time
        )
        
        if progress_callback:
            progress_callback(50, f"Loaded {len(quotes_df)} quotes")
        
        if trades_df.empty or quotes_df.empty:
            raise ValueError(f"No trade or quote data available for {symbol}")
        
        if progress_callback:
            progress_callback(60, f"Aligning {len(trades_df):,} trades with {len(quotes_df):,} quotes...")
        
        # Align trades with quotes using TradeQuoteAligner
        aligned_df, alignment_report = aligner.align_trades_quotes(trades_df, quotes_df)
        
        if aligned_df.empty:
            raise ValueError("No trades could be aligned with quotes")
        
        logger.info(f"Successfully aligned {alignment_report.aligned_trades}/{alignment_report.total_trades} trades")
        
        if progress_callback:
            progress_callback(70, "Processing large orders...")
            
        # Process the aligned data
        _process_aligned_data(aligned_df, symbol)
        
        if progress_callback:
            progress_callback(80, "Calculating pressure metrics...")
        
        # Get current stats from the tracker
        stats = tracker.get_current_stats(symbol)
        
        # Generate signal based on pressure
        signal = _generate_signal(stats, direction)
        
        # Get the complete display package
        display_data = _prepare_complete_display_package(stats, symbol, entry_time, alignment_report)
        
        if progress_callback:
            progress_callback(100, "Analysis complete")
        
        return {
            'plugin_name': PLUGIN_NAME,
            'timestamp': entry_time,
            'symbol': symbol,
            'direction': direction,
            'signal': signal,
            'display_data': display_data,
            'stats': stats,
            'alignment_report': {
                'aligned_trades': alignment_report.aligned_trades,
                'total_trades': alignment_report.total_trades,
                'avg_quote_age_ms': alignment_report.avg_quote_age_ms
            }
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


def _process_aligned_data(aligned_df: pd.DataFrame, symbol: str) -> None:
    """
    Process aligned trade data through the tracker.
    
    The aligned_df from TradeQuoteAligner contains:
    - All original trade columns (price, size, etc.)
    - All original quote columns with 'quote_' prefix (quote_bid, quote_ask, etc.)
    - trade_side (buy/sell classification)
    - confidence, alignment_method, quote_age_ms
    """
    if aligned_df.empty:
        logger.warning(f"No aligned trade data for {symbol}")
        return
    
    # Ensure the index is datetime
    if not isinstance(aligned_df.index, pd.DatetimeIndex):
        aligned_df.index = pd.to_datetime(aligned_df.index)
    
    # First, update the tracker with all quotes (using bid/ask from aligned data)
    # This ensures the tracker has quote context
    processed_quotes = set()
    for idx in range(len(aligned_df)):
        row = aligned_df.iloc[idx]
        timestamp = aligned_df.index[idx]
        
        # Ensure timestamp is a datetime object
        if not isinstance(timestamp, datetime):
            timestamp = pd.Timestamp(timestamp).to_pydatetime()
        
        # Handle different column naming conventions from TradeQuoteAligner
        bid = row.get('quote_bid', row.get('bid'))
        ask = row.get('quote_ask', row.get('ask'))
        
        # Create a unique key for quote updates to avoid duplicates
        if pd.notna(bid) and pd.notna(ask):
            quote_key = (timestamp, bid, ask)
            if quote_key not in processed_quotes:
                tracker.update_quote(
                    symbol=symbol,
                    bid=bid,
                    ask=ask,
                    timestamp=timestamp
                )
                processed_quotes.add(quote_key)
    
    # Process trades through the tracker
    large_order_count = 0
    for idx in range(len(aligned_df)):
        row = aligned_df.iloc[idx]
        timestamp = aligned_df.index[idx]
        
        # Ensure timestamp is a datetime object
        if not isinstance(timestamp, datetime):
            timestamp = pd.Timestamp(timestamp).to_pydatetime()
        
        # Get trade data - handle different column names
        price = row.get('trade_price', row.get('price'))
        size = row.get('trade_size', row.get('size'))
        bid = row.get('quote_bid', row.get('bid'))
        ask = row.get('quote_ask', row.get('ask'))
        
        # Create a Trade object for the tracker
        trade = Trade(
            symbol=symbol,
            price=price,
            size=int(size),
            timestamp=timestamp,
            bid=bid,  # May be None if no quote available
            ask=ask   # May be None if no quote available
        )
        
        # Process trade - tracker handles large order detection
        large_order = tracker.process_trade(trade)
        if large_order:
            large_order_count += 1
            
            # Log if we have a mismatch between aligner classification and tracker's
            if 'trade_side' in row and row['trade_side'] != 'unknown':
                tracker_side = large_order.side.value.lower()
                aligned_side = row['trade_side'].lower()
                if tracker_side != aligned_side and tracker_side != 'unknown':
                    logger.debug(f"Classification mismatch at {timestamp}: "
                               f"aligned={aligned_side}, tracker={tracker_side}")
    
    logger.info(f"Processed {len(aligned_df)} aligned trades, "
                f"detected {large_order_count} large orders")


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


def _prepare_complete_display_package(stats: Dict[str, Any], symbol: str, entry_time: datetime, 
                                     alignment_report=None) -> Dict[str, Any]:
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
    
    # Add alignment stats if available
    if alignment_report:
        description_lines.extend([
            f"Aligned Trades: {alignment_report.aligned_trades:,} / {alignment_report.total_trades:,}",
            f"Avg Quote Age: {alignment_report.avg_quote_age_ms:.1f} ms"
        ])
    
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
    
    # Add alignment stats to table if available
    if alignment_report:
        table_data.extend([
            ("Aligned Trades", f"{alignment_report.aligned_trades:,} / {alignment_report.total_trades:,}"),
            ("Avg Quote Age", f"{alignment_report.avg_quote_age_ms:.1f} ms")
        ])
    
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