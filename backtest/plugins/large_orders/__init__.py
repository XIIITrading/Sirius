"""
Large Orders Grid Plugin
Displays successful large order trades in a grid format
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, List

from modules.calculations.order_flow.large_orders import LargeOrderDetector, ImpactStatus
from modules.calculations.order_flow.buy_sell_ratio import Trade
from backtest.data.polygon_data_manager import PolygonDataManager

logger = logging.getLogger(__name__)

# Plugin metadata
PLUGIN_NAME = "Large Orders Grid"
PLUGIN_VERSION = "1.0.1"
PLUGIN_DESCRIPTION = "Displays successful large order trades with impact analysis"

# Configuration
CONFIG = {
    'stats_window_minutes': 15,
    'impact_window_seconds': 1,
    'min_trades_for_std': 100,
    'ratio_threshold': 1.5,
    'stdev_threshold': 1.25,
    'max_orders_display': 50,  # Increased from 30
    'show_all_orders': False,  # Set to True to show all large orders regardless of impact
    'min_impact_for_success': 0.5  # Minimum impact in spread units for success
}

# Create module-level data manager
_data_manager = None

def _get_data_manager():
    """Get or create the data manager instance"""
    global _data_manager
    if _data_manager is None:
        _data_manager = PolygonDataManager()
        _data_manager.set_current_plugin(PLUGIN_NAME)
    return _data_manager


async def run_analysis(symbol: str, entry_time: datetime, direction: str) -> Dict[str, Any]:
    """
    Run Large Orders Grid analysis.
    
    Args:
        symbol: Stock symbol
        entry_time: Entry time for analysis
        direction: Trade direction (LONG/SHORT)
        
    Returns:
        Analysis results dictionary
    """
    data_manager = _get_data_manager()
    return await run_analysis_with_data_manager(symbol, entry_time, direction, data_manager, None)


async def run_analysis_with_progress(symbol: str, entry_time: datetime, direction: str,
                                   progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
    """
    Run Large Orders Grid analysis with progress tracking.
    
    Args:
        symbol: Stock symbol
        entry_time: Entry time for analysis
        direction: Trade direction (LONG/SHORT)
        progress_callback: Progress callback function
        
    Returns:
        Analysis results dictionary
    """
    data_manager = _get_data_manager()
    return await run_analysis_with_data_manager(symbol, entry_time, direction, data_manager, progress_callback)


async def run_analysis_with_data_manager(symbol: str, entry_time: datetime, direction: str, 
                                       data_manager: Any, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Run Large Orders Grid analysis with provided data manager.
    
    This is the main entry point that contains all analysis logic.
    
    Args:
        symbol: Stock symbol
        entry_time: Entry time for analysis
        direction: Trade direction (LONG/SHORT)
        data_manager: PolygonDataManager instance
        progress_callback: Optional progress callback
        
    Returns:
        Analysis results dictionary with grid data
    """
    try:
        # Ensure data manager has plugin name set
        data_manager.set_current_plugin(PLUGIN_NAME)
        
        # Initialize detector
        detector = LargeOrderDetector(
            stats_window_minutes=CONFIG['stats_window_minutes'],
            impact_window_seconds=CONFIG['impact_window_seconds'],
            min_trades_for_std=CONFIG['min_trades_for_std'],
            ratio_threshold=CONFIG['ratio_threshold'],
            stdev_threshold=CONFIG['stdev_threshold']
        )
        
        if progress_callback:
            progress_callback(0, "Fetching trade and quote data...")
        
        # Fetch data window (30 minutes before entry)
        end_time = entry_time
        start_time = entry_time - timedelta(minutes=30)
        
        # Fetch trades
        trades_df = await data_manager.load_trades(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time
        )
        
        if progress_callback:
            progress_callback(25, f"Loaded {len(trades_df)} trades")
        
        # Fetch quotes
        quotes_df = await data_manager.load_quotes(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time
        )
        
        if progress_callback:
            progress_callback(50, f"Loaded {len(quotes_df)} quotes")
        
        # Process data
        if progress_callback:
            progress_callback(60, "Detecting large orders...")
            
        # Process all the data
        _process_data(detector, trades_df, quotes_df, symbol)
        
        if progress_callback:
            progress_callback(80, "Analyzing order impact...")
        
        # Get diagnostic information about all large orders
        diagnostics = _get_order_diagnostics(detector, symbol)
        
        # Get orders to display (successful or all based on config)
        if CONFIG['show_all_orders']:
            display_orders = _get_all_large_orders(detector, symbol)
        else:
            display_orders = _get_successful_orders(detector, symbol)
        
        # Generate signal based on order analysis
        signal = _generate_signal(display_orders, direction)
        
        # Get the complete display package with diagnostics
        display_data = _prepare_complete_display_package(detector, display_orders, symbol, diagnostics)
        
        if progress_callback:
            progress_callback(100, "Analysis complete")
        
        return {
            'plugin_name': PLUGIN_NAME,
            'timestamp': entry_time,
            'symbol': symbol,
            'direction': direction,
            'signal': signal,
            'display_data': display_data,
            'large_orders_count': len(display_orders),
            'diagnostics': diagnostics  # Add diagnostic info
        }
        
    except Exception as e:
        logger.error(f"Error in Large Orders Grid plugin: {e}")
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


def _process_data(detector: LargeOrderDetector, trades_df: pd.DataFrame, 
                  quotes_df: pd.DataFrame, symbol: str) -> None:
    """Process historical data through the detector."""
    # Add symbol column if not present
    if 'symbol' not in trades_df.columns:
        trades_df['symbol'] = symbol
    if 'symbol' not in quotes_df.columns:
        quotes_df['symbol'] = symbol
    
    # First update all quotes
    for timestamp, quote_row in quotes_df.iterrows():
        detector.update_quote(
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
        
        # Process trade - detector handles everything
        large_order = detector.process_trade(trade)
        if large_order:
            large_order_count += 1
    
    logger.info(f"Processed {len(trades_df)} trades, detected {large_order_count} large orders")


def _get_order_diagnostics(detector: LargeOrderDetector, symbol: str) -> Dict[str, Any]:
    """Get diagnostic information about all large orders"""
    if symbol not in detector.completed_orders:
        return {}
    
    all_orders = list(detector.completed_orders[symbol])
    
    # Count by impact status
    status_counts = {}
    for status in ImpactStatus:
        status_counts[status.value] = sum(1 for o in all_orders if o.impact_status == status)
    
    # Analyze impact magnitudes
    impacts = [abs(o.impact_magnitude) if o.impact_magnitude else 0 for o in all_orders]
    
    # Analyze spreads at execution
    spreads = [o.spread_at_execution for o in all_orders if o.spread_at_execution > 0]
    
    diagnostics = {
        'total_large_orders': len(all_orders),
        'status_breakdown': status_counts,
        'impact_stats': {
            'min': min(impacts) if impacts else 0,
            'max': max(impacts) if impacts else 0,
            'mean': np.mean(impacts) if impacts else 0,
            'median': np.median(impacts) if impacts else 0,
            'std': np.std(impacts) if impacts else 0
        },
        'spread_stats': {
            'min': min(spreads) if spreads else 0,
            'max': max(spreads) if spreads else 0,
            'mean': np.mean(spreads) if spreads else 0,
            'median': np.median(spreads) if spreads else 0
        }
    }
    
    # Add impact distribution
    if impacts:
        bins = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, float('inf')]
        hist, _ = np.histogram(impacts, bins=bins)
        diagnostics['impact_distribution'] = {
            '0-0.1': int(hist[0]),
            '0.1-0.25': int(hist[1]),
            '0.25-0.5': int(hist[2]),
            '0.5-0.75': int(hist[3]),
            '0.75-1.0': int(hist[4]),
            '1.0-2.0': int(hist[5]),
            '>2.0': int(hist[6])
        }
    
    return diagnostics


def _get_all_large_orders(detector: LargeOrderDetector, symbol: str) -> List[Dict[str, Any]]:
    """Get all large orders for display (not just successful ones)"""
    if symbol not in detector.completed_orders:
        return []
    
    # Get all completed orders
    all_orders = list(detector.completed_orders[symbol])
    
    # Sort by timestamp descending (most recent first)
    all_orders.sort(key=lambda o: o.timestamp, reverse=True)
    
    # Limit to configured maximum
    all_orders = all_orders[:CONFIG['max_orders_display']]
    
    # Convert to grid data format
    grid_data = []
    for order in all_orders:
        grid_data.append({
            'timestamp': order.timestamp,
            'price': order.price,
            'size': order.size,
            'side': order.side.value,
            'impact_magnitude': order.impact_magnitude if order.impact_magnitude else 0.0,
            'impact_status': order.impact_status.value,
            'size_vs_avg': order.size_vs_avg,
            'size_vs_stdev': order.size_vs_stdev,
            'vwap_1s': order.vwap_1s,
            'volume_1s': order.volume_1s,
            'trade_count_1s': order.trade_count_1s,
            'spread_at_execution': order.spread_at_execution
        })
    
    return grid_data


def _get_successful_orders(detector: LargeOrderDetector, symbol: str) -> List[Dict[str, Any]]:
    """Get successful large orders for display"""
    if symbol not in detector.completed_orders:
        return []
    
    # Get all completed orders
    all_orders = list(detector.completed_orders[symbol])
    
    # Filter for successful orders only
    successful_orders = [
        order for order in all_orders 
        if order.impact_status == ImpactStatus.SUCCESS
    ]
    
    # Sort by timestamp descending (most recent first)
    successful_orders.sort(key=lambda o: o.timestamp, reverse=True)
    
    # Limit to configured maximum
    successful_orders = successful_orders[:CONFIG['max_orders_display']]
    
    # Convert to grid data format
    grid_data = []
    for order in successful_orders:
        grid_data.append({
            'timestamp': order.timestamp,
            'price': order.price,
            'size': order.size,
            'side': order.side.value,
            'impact_magnitude': order.impact_magnitude if order.impact_magnitude else 0.0,
            'impact_status': order.impact_status.value,
            'size_vs_avg': order.size_vs_avg,
            'size_vs_stdev': order.size_vs_stdev,
            'vwap_1s': order.vwap_1s,
            'volume_1s': order.volume_1s,
            'trade_count_1s': order.trade_count_1s,
            'spread_at_execution': order.spread_at_execution
        })
    
    return grid_data


def _generate_signal(display_orders: List[Dict[str, Any]], direction: str) -> Dict[str, Any]:
    """Generate trading signal from large orders"""
    if not display_orders:
        return {
            'direction': 'NEUTRAL',
            'strength': 0,
            'confidence': 0,
            'reason': 'No large orders detected'
        }
    
    # Analyze all large orders (not just successful ones)
    buy_count = sum(1 for o in display_orders if o['side'] == 'BUY')
    sell_count = sum(1 for o in display_orders if o['side'] == 'SELL')
    
    # Count high-impact orders (>0.25 spreads)
    high_impact_buys = sum(1 for o in display_orders 
                          if o['side'] == 'BUY' and abs(o['impact_magnitude']) > 0.25)
    high_impact_sells = sum(1 for o in display_orders 
                           if o['side'] == 'SELL' and abs(o['impact_magnitude']) > 0.25)
    
    # Calculate average impact magnitude
    avg_impact = np.mean([abs(o['impact_magnitude']) for o in display_orders])
    
    # Determine signal direction based on order balance and impact
    total_buy_score = buy_count + (high_impact_buys * 2)
    total_sell_score = sell_count + (high_impact_sells * 2)
    
    if total_buy_score > total_sell_score * 1.3:
        signal_direction = 'BULLISH'
        strength = min(100, (total_buy_score / max(1, total_sell_score)) * 20)
    elif total_sell_score > total_buy_score * 1.3:
        signal_direction = 'BEARISH'
        strength = min(100, (total_sell_score / max(1, total_buy_score)) * 20)
    else:
        signal_direction = 'NEUTRAL'
        strength = 30
    
    # Confidence based on order count and average impact
    confidence = min(100, len(display_orders) * 2 + avg_impact * 30)
    
    # Check alignment
    aligned = (signal_direction == 'BULLISH' and direction == 'LONG') or \
             (signal_direction == 'BEARISH' and direction == 'SHORT')
    
    # Build reason
    reason = f"{len(display_orders)} large orders: {buy_count} buys, {sell_count} sells. "
    reason += f"High-impact: {high_impact_buys} buys, {high_impact_sells} sells. "
    reason += f"Avg impact: {avg_impact:.3f} spreads"
    
    return {
        'direction': signal_direction,
        'strength': strength,
        'confidence': confidence,
        'reason': reason,
        'aligned': aligned
    }


def _prepare_complete_display_package(detector: LargeOrderDetector, 
                                    display_orders: List[Dict[str, Any]], 
                                    symbol: str,
                                    diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare complete display data package with grid configuration."""
    # Calculate summary statistics
    total_orders = len(display_orders)
    
    # Get diagnostic stats
    total_detected = diagnostics.get('total_large_orders', 0)
    status_breakdown = diagnostics.get('status_breakdown', {})
    impact_stats = diagnostics.get('impact_stats', {})
    
    if total_orders > 0:
        buy_orders = [o for o in display_orders if o['side'] == 'BUY']
        sell_orders = [o for o in display_orders if o['side'] == 'SELL']
        
        avg_impact = np.mean([abs(o['impact_magnitude']) for o in display_orders])
        max_impact = max([abs(o['impact_magnitude']) for o in display_orders])
        
        total_volume = sum(o['size'] for o in display_orders)
        avg_size = np.mean([o['size'] for o in display_orders])
        
        # Build summary based on what we're showing
        if CONFIG['show_all_orders']:
            summary = f"{total_orders} large orders shown (of {total_detected} total)"
        else:
            summary = f"{total_orders} successful large orders (of {total_detected} total)"
        
        description_lines = [
            f"Total Volume: {total_volume:,} shares",
            f"Average Size: {avg_size:,.0f} shares",
            f"Average Impact: {avg_impact:.3f} spreads",
            f"Maximum Impact: {max_impact:.3f} spreads",
            "",
            "Impact Status Breakdown:",
            f"  SUCCESS: {status_breakdown.get('SUCCESS', 0)}",
            f"  FAILURE: {status_breakdown.get('FAILURE', 0)}",
            f"  CONTESTED: {status_breakdown.get('CONTESTED', 0)}",
            f"  INSUFFICIENT: {status_breakdown.get('INSUFFICIENT', 0)}",
            "",
            f"Median Impact: {impact_stats.get('median', 0):.3f} spreads"
        ]
    else:
        summary = f"No large orders to display (of {total_detected} detected)"
        description_lines = [
            f"Total large orders detected: {total_detected}",
            "",
            "Impact Status Breakdown:",
            f"  SUCCESS: {status_breakdown.get('SUCCESS', 0)}",
            f"  FAILURE: {status_breakdown.get('FAILURE', 0)}",
            f"  CONTESTED: {status_breakdown.get('CONTESTED', 0)}",
            f"  INSUFFICIENT: {status_breakdown.get('INSUFFICIENT', 0)}",
            "",
            f"Average impact magnitude: {impact_stats.get('mean', 0):.3f} spreads",
            f"Median impact magnitude: {impact_stats.get('median', 0):.3f} spreads",
            "",
            "Note: Success requires price movement > 0.5 spreads within 1 second"
        ]
        buy_orders = []
        sell_orders = []
        avg_impact = 0
        max_impact = 0
    
    description = '\n'.join(description_lines)
    
    # Prepare table data for summary display
    summary_stats = detector.get_summary_stats(symbol)
    detection_stats = summary_stats.get('large_order_stats', {})
    
    table_data = [
        ("Total Large Orders", str(total_detected)),
        ("Successful Orders", str(status_breakdown.get('SUCCESS', 0))),
        ("Failed Orders", str(status_breakdown.get('FAILURE', 0))),
        ("Success Rate", f"{detection_stats.get('impact_success_rate', 0):.1%}"),
        ("Avg Impact (all)", f"{impact_stats.get('mean', 0):.3f}"),
        ("Median Impact (all)", f"{impact_stats.get('median', 0):.3f}"),
        ("Orders Shown", str(total_orders))
    ]
    
    # Configure grid columns based on what we're showing
    columns = [
        {'name': 'Timestamp', 'field': 'timestamp', 'width': 150},
        {'name': 'Price', 'field': 'price', 'width': 100},
        {'name': 'Size', 'field': 'size', 'width': 100},
        {'name': 'Side', 'field': 'side', 'width': 80},
        {'name': 'Impact', 'field': 'impact_magnitude', 'width': 100}
    ]
    
    # Add status column if showing all orders
    if CONFIG['show_all_orders']:
        columns.append({'name': 'Status', 'field': 'impact_status', 'width': 100})
    
    # Return complete display package
    return {
        'summary': summary,
        'description': description,
        'table_data': table_data,
        'grid_widget': {
            'module': 'backtest.plugins.large_orders.grid',
            'type': 'LargeOrdersGrid',
            'data': display_orders,
            'config': {
                'title': 'Large Orders Analysis',
                'columns': columns,
                'highlight_threshold': 0.5  # Highlight orders with >0.5 spread impact
            }
        }
    }


# Export everything needed
__all__ = ['run_analysis', 'run_analysis_with_progress', 'run_analysis_with_data_manager', 
           'PLUGIN_NAME', 'PLUGIN_VERSION', 'PLUGIN_DESCRIPTION', 'CONFIG']