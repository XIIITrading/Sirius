# backtest/plugins/buy_sell_ratio/__init__.py

"""
Bid/Ask Ratio Plugin
Real-time buy/sell pressure visualization with 30-minute rolling window.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Callable, List
import pandas as pd
import numpy as np

from modules.calculations.order_flow.buy_sell_ratio import (
    SimpleDeltaTracker, Trade, MinuteBar
)

# Configure logging
logger = logging.getLogger(__name__)

# Plugin metadata
PLUGIN_NAME = "Bid/Ask Ratio Tracker"
PLUGIN_VERSION = "1.0.0"

# Configuration
CONFIG = {
    'window_minutes': 30,  # Rolling window for display
    'lookback_minutes': 35,  # Fetch extra data for processing
    'chart_y_min': -1.25,
    'chart_y_max': 1.25,
    'reference_lines': [0, 0.25, 0.5, 0.75],  # Dotted lines on chart
    'sampling_interval_seconds': 60,  # 1-minute bars
}


def get_config() -> Dict[str, Any]:
    """Get plugin configuration for UI settings"""
    return CONFIG.copy()


def validate_inputs(symbol: str, entry_time: datetime, direction: str) -> bool:
    """Validate input parameters"""
    if not symbol or not isinstance(symbol, str):
        logger.error(f"Invalid symbol: {symbol}")
        return False
    
    if not isinstance(entry_time, datetime):
        logger.error(f"Invalid entry_time type: {type(entry_time)}")
        return False
        
    if direction not in ['LONG', 'SHORT']:
        logger.error(f"Invalid direction: {direction}")
        return False
        
    return True


async def run_analysis(symbol: str, entry_time: datetime, direction: str) -> Dict[str, Any]:
    """
    BACKWARD COMPATIBILITY: Run analysis without data_manager parameter.
    This function creates its own data manager for legacy compatibility.
    
    Args:
        symbol: Stock symbol
        entry_time: Entry time (UTC)
        direction: 'LONG' or 'SHORT'
        
    Returns:
        Complete analysis results formatted for display
    """
    logger.warning("run_analysis called without data_manager - using legacy mode")
    
    # Import here to avoid circular imports
    from backtest.data.polygon_data_manager import PolygonDataManager
    
    # Create our own data manager for backward compatibility
    data_manager = PolygonDataManager()
    
    # Call the new interface
    return await run_analysis_with_data_manager(data_manager, symbol, entry_time, direction)


async def run_analysis_with_progress(symbol: str, entry_time: datetime, direction: str,
                                   progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
    """
    BACKWARD COMPATIBILITY: Run analysis without data_manager parameter.
    This function creates its own data manager for legacy compatibility.
    
    Args:
        symbol: Stock symbol
        entry_time: Entry time (UTC)
        direction: 'LONG' or 'SHORT'
        progress_callback: Optional callback for progress updates
        
    Returns:
        Complete analysis results formatted for display
    """
    logger.warning("run_analysis_with_progress called without data_manager - using legacy mode")
    
    # Import here to avoid circular imports
    from backtest.data.polygon_data_manager import PolygonDataManager
    
    # Create our own data manager for backward compatibility
    data_manager = PolygonDataManager()
    
    # Call the new interface
    return await run_analysis_with_data_manager_and_progress(
        data_manager, symbol, entry_time, direction, progress_callback
    )


async def run_analysis_with_data_manager(data_manager, symbol: str, entry_time: datetime, 
                                        direction: str) -> Dict[str, Any]:
    """
    NEW INTERFACE: Run analysis with provided data_manager.
    
    Args:
        data_manager: Centralized PolygonDataManager instance
        symbol: Stock symbol
        entry_time: Entry time (UTC)
        direction: 'LONG' or 'SHORT'
        
    Returns:
        Complete analysis results formatted for display
    """
    return await run_analysis_with_data_manager_and_progress(
        data_manager, symbol, entry_time, direction, None
    )


async def run_analysis_with_data_manager_and_progress(data_manager, symbol: str, 
                                                     entry_time: datetime, direction: str,
                                                     progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
    """
    NEW INTERFACE: Run analysis with provided data_manager and progress tracking.
    
    Args:
        data_manager: Centralized PolygonDataManager instance
        symbol: Stock symbol
        entry_time: Entry time (UTC)
        direction: 'LONG' or 'SHORT'
        progress_callback: Optional callback for progress updates
        
    Returns:
        Complete analysis results formatted for display
    """
    try:
        # Validate inputs
        if not validate_inputs(symbol, entry_time, direction):
            raise ValueError("Invalid input parameters")
        
        # Set current plugin on data manager
        data_manager.set_current_plugin(PLUGIN_NAME)
        
        # Progress helper
        def report_progress(percentage: int, message: str):
            if progress_callback:
                progress_callback(percentage, message)
            logger.info(f"Progress: {percentage}% - {message}")
        
        # 1. Initialize tracker
        report_progress(5, "Initializing delta tracker...")
        tracker = SimpleDeltaTracker(window_minutes=CONFIG['window_minutes'])
        
        # 2. Calculate data fetch window
        start_time = entry_time - timedelta(minutes=CONFIG['lookback_minutes'])
        end_time = entry_time
        
        # 3. Fetch trade and quote data
        report_progress(10, "Fetching trade data...")
        trades_df = await data_manager.load_trades(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time
        )
        
        report_progress(20, "Fetching quote data...")
        quotes_df = await data_manager.load_quotes(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time
        )
        
        if trades_df.empty:
            raise ValueError(f"No trade data available for {symbol}")
        
        report_progress(30, f"Processing {len(trades_df):,} trades and {len(quotes_df):,} quotes...")
        
        # 4. Update tracker with quotes first
        quote_count = 0
        for timestamp, quote_data in quotes_df.iterrows():
            tracker.update_quote(
                symbol=symbol,
                bid=float(quote_data['bid']),
                ask=float(quote_data['ask']),
                timestamp=timestamp.to_pydatetime() if hasattr(timestamp, 'to_pydatetime') else timestamp
            )
            quote_count += 1
            
            if quote_count % 1000 == 0:
                progress = 30 + int((quote_count / len(quotes_df)) * 20)
                report_progress(progress, f"Processed {quote_count:,} quotes...")
        
        # 5. Process trades
        report_progress(50, "Processing trades...")
        completed_bars = []
        trades_processed = 0
        
        for i, (timestamp, trade_data) in enumerate(trades_df.iterrows()):
            # Create trade object
            trade = Trade(
                symbol=symbol,
                price=float(trade_data['price']),
                size=int(trade_data['size']),
                timestamp=timestamp.to_pydatetime() if hasattr(timestamp, 'to_pydatetime') else timestamp,
                bid=float(trade_data['bid']) if 'bid' in trade_data else None,
                ask=float(trade_data['ask']) if 'ask' in trade_data else None
            )
            
            # Process trade and check if minute completed
            completed_bar = tracker.process_trade(trade)
            if completed_bar:
                completed_bars.append(completed_bar)
            
            trades_processed += 1
            
            # Update progress
            if trades_processed % 1000 == 0:
                progress = 50 + int((trades_processed / len(trades_df)) * 40)
                report_progress(progress, f"Processed {trades_processed:,} trades, {len(completed_bars)} bars...")
        
        # 6. Get chart data
        report_progress(90, "Generating chart data...")
        chart_data = tracker.get_chart_data(symbol)
        
        # Filter to last 30 minutes before entry
        if chart_data:
            filtered_chart_data = []
            for point in chart_data:
                point_time = datetime.fromisoformat(point['timestamp'])
                if point_time <= entry_time and point_time >= (entry_time - timedelta(minutes=30)):
                    filtered_chart_data.append(point)
            chart_data = filtered_chart_data
        
        # 7. Get analysis metrics
        latest_ratio = tracker.get_latest_ratio(symbol)
        summary_stats = tracker.get_summary_stats(symbol)
        
        # 8. Determine signal based on ratio and direction
        signal_assessment = _assess_signal(latest_ratio, summary_stats, direction)
        
        # 9. Format results
        report_progress(95, "Formatting results...")
        result = _format_results(
            chart_data, latest_ratio, summary_stats, signal_assessment,
            entry_time, direction, completed_bars
        )
        
        report_progress(100, "Analysis complete")
        return result
        
    except Exception as e:
        logger.error(f"Error in Bid/Ask Ratio analysis: {e}")
        if progress_callback:
            progress_callback(100, f"Error: {str(e)}")
        
        # Return error result
        return {
            'plugin_name': PLUGIN_NAME,
            'timestamp': entry_time,
            'error': str(e),
            'signal': {
                'direction': 'NEUTRAL',
                'strength': 0,
                'confidence': 0
            }
        }


def _assess_signal(latest_ratio: Optional[float], 
                  summary_stats: Dict[str, Any], 
                  direction: str) -> Dict[str, Any]:
    """Assess signal strength based on ratio and direction"""
    
    if latest_ratio is None:
        return {
            'aligned': False,
            'strength': 0,
            'confidence': 0,
            'signal_direction': 'NEUTRAL'
        }
    
    # Determine signal direction
    if latest_ratio > 0.5:
        signal_direction = 'BULLISH'
    elif latest_ratio < -0.5:
        signal_direction = 'BEARISH'
    elif latest_ratio > 0.25:
        signal_direction = 'LEAN_BULLISH'
    elif latest_ratio < -0.25:
        signal_direction = 'LEAN_BEARISH'
    else:
        signal_direction = 'NEUTRAL'
    
    # Calculate strength (0-100)
    strength = min(100, abs(latest_ratio) * 100)
    
    # Calculate confidence based on consistency
    avg_ratio = summary_stats.get('avg_ratio', 0)
    ratio_consistency = 1 - abs(latest_ratio - avg_ratio) / max(abs(latest_ratio), abs(avg_ratio), 0.1)
    confidence = max(0, min(100, ratio_consistency * 100))
    
    # Check alignment with intended direction
    aligned = False
    if direction == 'LONG' and latest_ratio > 0:
        aligned = True
    elif direction == 'SHORT' and latest_ratio < 0:
        aligned = True
    
    return {
        'aligned': aligned,
        'strength': strength,
        'confidence': confidence,
        'signal_direction': signal_direction
    }


def _format_results(chart_data: List[Dict], latest_ratio: Optional[float],
                   summary_stats: Dict[str, Any], signal_assessment: Dict[str, Any],
                   entry_time: datetime, direction: str, 
                   completed_bars: List[MinuteBar]) -> Dict[str, Any]:
    """Format results for display"""
    
    # Build summary display rows
    summary_rows = []
    
    if latest_ratio is not None:
        summary_rows.extend([
            ['Current Ratio', f"{latest_ratio:+.3f}"],
            ['Average Ratio', f"{summary_stats.get('avg_ratio', 0):+.3f}"],
            ['Max Ratio', f"{summary_stats.get('max_ratio', 0):+.3f}"],
            ['Min Ratio', f"{summary_stats.get('min_ratio', 0):+.3f}"],
            ['Total Volume', f"{summary_stats.get('total_volume', 0):,.0f}"],
            ['Minutes Tracked', f"{summary_stats.get('minutes_tracked', 0)}"]
        ])
    
    # Recent bars for detailed view
    recent_bars_data = []
    for bar in completed_bars[-10:]:  # Last 10 bars
        recent_bars_data.append([
            bar.timestamp.strftime('%H:%M:%S'),
            f"{bar.weighted_pressure:+.3f}",
            f"{bar.positive_volume:,.0f}",
            f"{bar.negative_volume:,.0f}",
            f"{bar.total_volume:,.0f}"
        ])
    
    # Create description
    description = f"Buy/Sell Ratio: {latest_ratio:+.3f} | "
    if signal_assessment['aligned']:
        description += f"✅ Supports {direction}"
    else:
        description += f"❌ Contradicts {direction}"
    
    if latest_ratio is not None:
        if latest_ratio > 0.75:
            description += " | Strong buying pressure"
        elif latest_ratio < -0.75:
            description += " | Strong selling pressure"
        elif abs(latest_ratio) < 0.25:
            description += " | Balanced flow"
    
    # Format chart data for visualization
    chart_config = {
        'module': 'backtest.plugins.buy_sell_ratio.chart',
        'type': 'BidAskRatioChart',
        'data': chart_data,
        'entry_time': entry_time.isoformat() if entry_time else None
    }
    
    return {
        'plugin_name': PLUGIN_NAME,
        'timestamp': entry_time,
        'signal': {
            'direction': signal_assessment['signal_direction'],
            'strength': float(signal_assessment['strength']),
            'confidence': float(signal_assessment['confidence'])
        },
        'details': {
            'current_ratio': latest_ratio,
            'average_ratio': summary_stats.get('avg_ratio', 0),
            'max_ratio': summary_stats.get('max_ratio', 0),
            'min_ratio': summary_stats.get('min_ratio', 0),
            'total_volume': summary_stats.get('total_volume', 0),
            'minutes_tracked': summary_stats.get('minutes_tracked', 0),
            'aligned': signal_assessment['aligned']
        },
        'display_data': {
            'summary': f"Bid/Ask Ratio: {latest_ratio:+.3f}" if latest_ratio is not None else "No Data",
            'description': description,
            'table_data': summary_rows,
            'recent_bars': {
                'headers': ['Time', 'Ratio', 'Buy Vol', 'Sell Vol', 'Total Vol'],
                'rows': recent_bars_data
            },
            'chart_widget': chart_config
        }
    }