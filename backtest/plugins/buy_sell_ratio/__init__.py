# backtest/plugins/buy_sell_ratio/__init__.py

"""
Bid/Ask Ratio Plugin
Real-time buy/sell pressure visualization using pre-aligned trade data.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, List
import pandas as pd

from backtest.calculations.order_flow.buy_sell_ratio import BuySellRatioCalculator, MinuteBar
from backtest.data.trade_quote_aligner import TradeQuoteAligner

# Configure logging
logger = logging.getLogger(__name__)

# Plugin metadata
PLUGIN_NAME = "Bid/Ask Ratio Tracker"
PLUGIN_VERSION = "2.0.0"

# Configuration
CONFIG = {
    'window_minutes': 30,  # Rolling window for display
    'lookback_minutes': 35,  # Fetch extra data for processing
    'chart_y_min': -1.25,
    'chart_y_max': 1.25,
    'reference_lines': [0, 0.25, 0.5, 0.75],  # Dotted lines on chart
    'min_confidence': 0.5,  # Minimum confidence for aligned trades
    'max_quote_age_ms': 500,  # Maximum quote staleness
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
    """
    logger.warning("run_analysis called without data_manager - using legacy mode")
    from backtest.data.polygon_data_manager import PolygonDataManager
    data_manager = PolygonDataManager()
    return await run_analysis_with_data_manager(data_manager, symbol, entry_time, direction)


async def run_analysis_with_progress(symbol: str, entry_time: datetime, direction: str,
                                   progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
    """
    BACKWARD COMPATIBILITY: Run analysis without data_manager parameter.
    """
    logger.warning("run_analysis_with_progress called without data_manager - using legacy mode")
    from backtest.data.polygon_data_manager import PolygonDataManager
    data_manager = PolygonDataManager()
    return await run_analysis_with_data_manager_and_progress(
        data_manager, symbol, entry_time, direction, progress_callback
    )


async def run_analysis_with_data_manager(data_manager, symbol: str, entry_time: datetime, 
                                        direction: str) -> Dict[str, Any]:
    """
    NEW INTERFACE: Run analysis with provided data_manager.
    """
    return await run_analysis_with_data_manager_and_progress(
        data_manager, symbol, entry_time, direction, None
    )


async def run_analysis_with_data_manager_and_progress(data_manager, symbol: str, 
                                                     entry_time: datetime, direction: str,
                                                     progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
    """
    Run analysis using TradeQuoteAligner for pre-aligned data.
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
        
        # 1. Initialize components
        report_progress(5, "Initializing components...")
        calculator = BuySellRatioCalculator(
            window_minutes=CONFIG['window_minutes'],
            min_confidence=CONFIG['min_confidence']
        )
        
        aligner = TradeQuoteAligner(
            max_quote_age_ms=CONFIG['max_quote_age_ms'],
            min_confidence_threshold=CONFIG['min_confidence']
        )
        
        # 2. Calculate data fetch window
        start_time = entry_time - timedelta(minutes=CONFIG['lookback_minutes'])
        end_time = entry_time
        
        # 3. Fetch raw data
        report_progress(10, "Fetching trade data...")
        trades_df = await data_manager.load_trades(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time
        )
        
        report_progress(30, "Fetching quote data...")
        quotes_df = await data_manager.load_quotes(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time
        )
        
        if trades_df.empty or quotes_df.empty:
            raise ValueError(f"No trade or quote data available for {symbol}")
        
        report_progress(50, f"Aligning {len(trades_df):,} trades with {len(quotes_df):,} quotes...")
        
        # 4. Align trades with quotes - THIS IS THE KEY STEP
        aligned_df, alignment_report = aligner.align_trades_quotes(trades_df, quotes_df)
        
        if aligned_df.empty:
            raise ValueError("No trades could be aligned with quotes")
        
        report_progress(70, f"Processing {len(aligned_df):,} aligned trades into minute bars...")
        
        # 5. Process aligned trades into minute bars
        minute_bars = calculator.process_aligned_trades(aligned_df, symbol)
        
        if not minute_bars:
            raise ValueError("No minute bars could be calculated")
        
        # 6. Filter to last 30 minutes before entry
        filtered_bars = [
            bar for bar in minute_bars 
            if bar.timestamp <= entry_time and bar.timestamp >= (entry_time - timedelta(minutes=30))
        ]
        
        # 7. Get chart data and stats
        report_progress(85, "Generating visualization data...")
        chart_data = calculator.get_chart_data(filtered_bars)
        summary_stats = calculator.get_summary_stats(filtered_bars)
        
        # 8. Determine signal
        latest_ratio = filtered_bars[-1].weighted_pressure if filtered_bars else None
        signal_assessment = _assess_signal(latest_ratio, summary_stats, direction)
        
        # 9. Format results
        report_progress(95, "Formatting results...")
        result = _format_results(
            chart_data, latest_ratio, summary_stats, signal_assessment,
            entry_time, direction, filtered_bars, alignment_report
        )
        
        report_progress(100, "Analysis complete")
        return result
        
    except Exception as e:
        logger.error(f"Error in Bid/Ask Ratio analysis: {e}", exc_info=True)
        if progress_callback:
            progress_callback(100, f"Error: {str(e)}")
        
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
    
    # Calculate confidence based on consistency and data quality
    avg_ratio = summary_stats.get('avg_ratio', 0)
    ratio_consistency = 1 - abs(latest_ratio - avg_ratio) / max(abs(latest_ratio), abs(avg_ratio), 0.1)
    
    # Factor in classification rate and average confidence
    classification_rate = summary_stats.get('classification_rate', 0)
    avg_confidence = summary_stats.get('avg_confidence', 0)
    
    # Combined confidence
    confidence = max(0, min(100, ratio_consistency * classification_rate * avg_confidence * 100))
    
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
                   minute_bars: List[MinuteBar],
                   alignment_report) -> Dict[str, Any]:
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
            ['Minutes Tracked', f"{summary_stats.get('minutes_tracked', 0)}"],
            ['Classification Rate', f"{summary_stats.get('classification_rate', 0)*100:.1f}%"],
            ['Avg Confidence', f"{summary_stats.get('avg_confidence', 0)*100:.1f}%"]
        ])
    
    # Add alignment stats
    if alignment_report:
        summary_rows.extend([
            ['Aligned Trades', f"{alignment_report.aligned_trades:,} / {alignment_report.total_trades:,}"],
            ['Avg Quote Age', f"{alignment_report.avg_quote_age_ms:.1f} ms"]
        ])
    
    # Recent bars for detailed view
    recent_bars_data = []
    for bar in minute_bars[-10:]:  # Last 10 bars
        recent_bars_data.append([
            bar.timestamp.strftime('%H:%M:%S'),
            f"{bar.weighted_pressure:+.3f}",
            f"{bar.positive_volume:,.0f}",
            f"{bar.negative_volume:,.0f}",
            f"{bar.total_volume:,.0f}",
            f"{bar.avg_confidence*100:.0f}%"
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
    
    # Add data quality indicator
    classification_rate = summary_stats.get('classification_rate', 0)
    if classification_rate < 0.8:
        description += f" | ⚠️ Low classification rate ({classification_rate*100:.0f}%)"
    
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
            'classification_rate': summary_stats.get('classification_rate', 0),
            'avg_confidence': summary_stats.get('avg_confidence', 0),
            'aligned': signal_assessment['aligned']
        },
        'display_data': {
            'summary': f"Bid/Ask Ratio: {latest_ratio:+.3f}" if latest_ratio is not None else "No Data",
            'description': description,
            'table_data': summary_rows,
            'recent_bars': {
                'headers': ['Time', 'Ratio', 'Buy Vol', 'Sell Vol', 'Total Vol', 'Confidence'],
                'rows': recent_bars_data
            },
            'chart_widget': chart_config
        }
    }