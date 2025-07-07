# backtest/plugins/bid_ask_imbalance/__init__.py
"""
Bid/Ask Imbalance Analysis Plugin
Real-time order flow analysis through bid/ask execution imbalance detection.
Consolidated implementation using modular PolygonDataManager.
Version 3.7.0 - Optimized version with pre-aligned data support
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Callable, List, Tuple
import pandas as pd
import numpy as np

# Import the modular data manager
from backtest.data.polygon_data_manager import PolygonDataManager
from backtest.calculations.order_flow.bid_ask_imbal import (
    BidAskImbalance, Trade, Quote, BidAskSignal
)

# Configure logging
logger = logging.getLogger(__name__)

# Plugin metadata
PLUGIN_NAME = "Bid/Ask Imbalance Analysis"
PLUGIN_VERSION = "3.7.0"

# Create single data manager instance at module level
_data_manager = PolygonDataManager()

# Plugin configuration - optimized for production use
_config = {
    'imbalance_lookback': 8000,  # Total trades to track
    'trades_per_bar': 100,  # Number of trades per bar index
    'spread_history_seconds': 600,  # 10 minutes of spread history
    'quote_sync_tolerance_ms': 500,  # Base sync tolerance
    'aggression_threshold': 0.7,  # Threshold for aggressive trades
    'smoothing_alpha': 0.1,  # EWMA smoothing factor
    'lookback_minutes': 20,  # Minutes of data to fetch before entry
    'quote_window_minutes': 15,  # Extended quote window
    'quote_forward_minutes': 5,   # Forward window
    'max_trades_to_process': 15000,  # Maximum trades to process
    'count_sample_interval': 10,  # Increased from 2 to 10 for better performance
    'adaptive_sync': True,  # Enable adaptive sync tolerance
    'use_prealigned_data': True  # Enable pre-aligned data usage
}


def get_config() -> Dict[str, Any]:
    """Get plugin configuration for UI settings"""
    return _config.copy()


def validate_inputs(symbol: str, entry_time: datetime, direction: str) -> bool:
    """Validate input parameters"""
    if not symbol or not isinstance(symbol, str):
        return False
    
    if not isinstance(entry_time, datetime):
        return False
        
    if direction not in ['LONG', 'SHORT']:
        return False
        
    return True


def _adjust_sync_tolerance_adaptive(trades_df: pd.DataFrame, quotes_df: pd.DataFrame) -> int:
    """
    Adaptively adjust quote sync tolerance based on data characteristics.
    Enhanced for better handling of quote gaps.
    """
    if trades_df.empty or quotes_df.empty:
        return _config['quote_sync_tolerance_ms']
    
    # Calculate quote intervals
    quote_intervals = quotes_df.index.to_series().diff().dt.total_seconds() * 1000
    median_quote_interval = quote_intervals.median()
    p95_quote_interval = quote_intervals.quantile(0.95)
    
    # Base tolerance on quote sparsity - use 95th percentile for robustness
    if p95_quote_interval > 5000:  # Very sparse quotes
        suggested_tolerance = min(int(p95_quote_interval * 0.5), 10000)  # Cap at 10 seconds
        logger.info(f"Very sparse quotes detected (95th percentile: {p95_quote_interval:.0f}ms)")
    elif median_quote_interval > 500:  # Sparse quotes
        suggested_tolerance = min(int(median_quote_interval * 2), 5000)  # Cap at 5 seconds
        logger.info(f"Sparse quotes detected (median interval: {median_quote_interval:.0f}ms)")
    elif median_quote_interval > 200:  # Moderate quote frequency
        suggested_tolerance = min(int(median_quote_interval * 1.5), 1000)
        logger.info(f"Moderate quote frequency (median interval: {median_quote_interval:.0f}ms)")
    else:  # Dense quotes
        suggested_tolerance = _config['quote_sync_tolerance_ms']
    
    # Ensure minimum tolerance
    suggested_tolerance = max(suggested_tolerance, 200)
    
    logger.info(f"Adaptive sync tolerance: {suggested_tolerance}ms (default: {_config['quote_sync_tolerance_ms']}ms)")
    return int(suggested_tolerance)


async def run_analysis(symbol: str, entry_time: datetime, direction: str) -> Dict[str, Any]:
    """
    Run Bid/Ask Imbalance analysis without progress tracking.
    
    Args:
        symbol: Stock symbol
        entry_time: Entry time (UTC)
        direction: 'LONG' or 'SHORT'
        
    Returns:
        Complete analysis results formatted for display
    """
    return await run_analysis_with_progress(symbol, entry_time, direction, None)


async def run_analysis_with_progress(symbol: str, entry_time: datetime, direction: str,
                                   progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
    """
    Run Bid/Ask Imbalance analysis with progress tracking.
    Optimized version using pre-aligned data when available.
    
    Args:
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
        _data_manager.set_current_plugin(PLUGIN_NAME)
        
        # Progress helper
        def report_progress(percentage: int, message: str):
            if progress_callback:
                progress_callback(percentage, message)
            logger.info(f"Progress: {percentage}% - {message}")
        
        # 1. Check if data manager supports pre-aligned data
        use_prealigned = _config['use_prealigned_data'] and hasattr(_data_manager, 'load_aligned_trades')
        
        if use_prealigned:
            logger.info("Using optimized pre-aligned data processing")
            return await _run_analysis_optimized(symbol, entry_time, direction, report_progress)
        else:
            logger.info("Using standard processing (pre-aligned data not available)")
            return await _run_analysis_standard(symbol, entry_time, direction, report_progress)
        
    except Exception as e:
        logger.error(f"Error in Bid/Ask Imbalance analysis: {e}")
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


async def _run_analysis_optimized(symbol: str, entry_time: datetime, direction: str,
                                report_progress: Callable[[int, str], None]) -> Dict[str, Any]:
    """
    Optimized analysis using pre-aligned trade data.
    This is significantly faster as quote synchronization is already done.
    """
    # 1. Fetch pre-aligned data
    report_progress(5, "Fetching pre-aligned trade data...")
    
    start_time = entry_time - timedelta(minutes=_config['lookback_minutes'])
    
    try:
        # Try to get pre-aligned trades
        aligned_trades = await _data_manager.load_aligned_trades(
            symbol=symbol,
            start_time=start_time,
            end_time=entry_time,
            alignment_tolerance_ms=_config['quote_sync_tolerance_ms']
        )
        
        if aligned_trades.empty:
            raise ValueError(f"No aligned trade data available for {symbol}")
        
        original_trade_count = len(aligned_trades)
        report_progress(10, f"Fetched {original_trade_count:,} pre-aligned trades")
        
    except Exception as e:
        logger.warning(f"Failed to load pre-aligned data: {e}. Falling back to standard processing.")
        return await _run_analysis_standard(symbol, entry_time, direction, report_progress)
    
    # 2. Initialize analyzer
    report_progress(15, "Initializing analyzer...")
    analyzer = BidAskImbalance(
        imbalance_lookback=_config['imbalance_lookback'],
        trades_per_bar=_config['trades_per_bar'],
        spread_history_seconds=_config['spread_history_seconds'],
        quote_sync_tolerance_ms=_config['quote_sync_tolerance_ms'],
        aggression_threshold=_config['aggression_threshold'],
        smoothing_alpha=_config['smoothing_alpha']
    )
    
    # 3. Sample trades if needed
    if len(aligned_trades) > _config['max_trades_to_process']:
        report_progress(20, f"Optimizing data for analysis...")
        aligned_trades = aligned_trades.iloc[::_config['count_sample_interval']].copy()
        report_progress(25, f"Optimized to {len(aligned_trades):,} trades")
    else:
        report_progress(25, "Using all available data")
    
    # 4. Process trades with pre-aligned quotes
    report_progress(30, f"Processing {len(aligned_trades):,} trades...")
    
    signals = []
    last_signal = None
    trades_processed = 0
    update_interval = max(1, len(aligned_trades) // 20)
    
    # Convert to numpy arrays for faster access
    trade_times = aligned_trades.index.to_numpy()
    trade_prices = aligned_trades['price'].to_numpy() 
    trade_sizes = aligned_trades['size'].to_numpy()
    trade_bids = aligned_trades['bid'].to_numpy() if 'bid' in aligned_trades else None
    trade_asks = aligned_trades['ask'].to_numpy() if 'ask' in aligned_trades else None
    
    # Build quote history from aligned data
    if trade_bids is not None and trade_asks is not None:
        report_progress(35, "Building quote history from aligned data...")
        
        # Get unique quote updates
        unique_quotes = aligned_trades[['bid', 'ask', 'bid_size', 'ask_size']].drop_duplicates()
        
        for idx, quote_data in unique_quotes.iterrows():
            quote = Quote(
                symbol=symbol,
                bid=float(quote_data['bid']),
                ask=float(quote_data['ask']),
                bid_size=int(quote_data.get('bid_size', 100)),
                ask_size=int(quote_data.get('ask_size', 100)),
                timestamp=idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else idx
            )
            analyzer.process_quote(quote)
    
    # Process trades
    report_progress(45, "Analyzing trades...")
    
    for i in range(len(trade_times)):
        # Stop at entry time
        if trade_times[i] >= entry_time:
            break
        
        # Create trade object with pre-aligned quote data
        trade = Trade(
            symbol=symbol,
            price=float(trade_prices[i]),
            size=int(trade_sizes[i]),
            timestamp=trade_times[i].to_pydatetime() if hasattr(trade_times[i], 'to_pydatetime') else trade_times[i],
            conditions=None
        )
        
        # Add pre-aligned quote data if available
        if trade_bids is not None and trade_asks is not None:
            trade.bid = float(trade_bids[i])
            trade.ask = float(trade_asks[i])
        
        # Process trade and get signal
        signal = analyzer.process_trade(trade)
        
        if signal:
            signals.append(signal)
            last_signal = signal
        
        trades_processed += 1
        
        # Update progress
        if i % update_interval == 0 and i > 0:
            progress = 45 + int((i / len(trade_times)) * 45)
            report_progress(
                min(progress, 90),
                f"Processed {trades_processed:,} trades, {len(signals):,} signals"
            )
    
    # 5. Complete analysis
    report_progress(92, "Generating final analysis...")
    
    if not last_signal:
        if symbol in analyzer.latest_signals:
            last_signal = analyzer.latest_signals[symbol]
        else:
            raise ValueError("No signals generated from analysis")
    
    # Get summaries
    session_summary = analyzer.get_session_summary(symbol)
    bar_summary = analyzer.get_bar_index_summary(symbol)
    
    # 6. Format results
    report_progress(95, "Formatting results...")
    
    result = _format_results(
        last_signal, signals, session_summary,
        bar_summary, analyzer, entry_time, direction
    )
    
    # Add performance stats
    result['performance_stats'] = {
        'original_trades': original_trade_count,
        'processed_trades': trades_processed,
        'sampling_ratio': trades_processed / original_trade_count if original_trade_count > 0 else 1,
        'signals_generated': len(signals),
        'sync_rate': 100.0,  # Pre-aligned data has 100% sync
        'processing_mode': 'optimized_prealigned',
        'count_interval': _config['count_sample_interval']
    }
    
    report_progress(100, f"Complete - {last_signal.signal_type}")
    
    return result


async def _run_analysis_standard(symbol: str, entry_time: datetime, direction: str,
                               report_progress: Callable[[int, str], None]) -> Dict[str, Any]:
    """
    Standard analysis implementation (original logic).
    Used when pre-aligned data is not available.
    """
    # 1. Fetch data using modular data manager
    report_progress(5, "Fetching trade and quote data...")
    trades_df, quotes_df = await _fetch_data(symbol, entry_time)
    
    if trades_df.empty:
        raise ValueError(f"No trade data available for {symbol}")
    
    original_trade_count = len(trades_df)
    original_quote_count = len(quotes_df)
    report_progress(10, f"Fetched {original_trade_count:,} trades and {original_quote_count:,} quotes")
    
    # 2. Analyze data characteristics and adjust sync tolerance if needed
    adaptive_tolerance = _config['quote_sync_tolerance_ms']
    if _config['adaptive_sync']:
        adaptive_tolerance = _adjust_sync_tolerance_adaptive(trades_df, quotes_df)
    
    # 3. Initialize analyzer with adaptive settings
    report_progress(15, f"Initializing analyzer (sync tolerance: {adaptive_tolerance}ms)...")
    analyzer = BidAskImbalance(
        imbalance_lookback=_config['imbalance_lookback'],
        trades_per_bar=_config['trades_per_bar'],
        spread_history_seconds=_config['spread_history_seconds'],
        quote_sync_tolerance_ms=adaptive_tolerance,
        aggression_threshold=_config['aggression_threshold'],
        smoothing_alpha=_config['smoothing_alpha']
    )
    
    # 4. Filter quotes to prevent cleanup issues
    quote_window_start = entry_time - timedelta(minutes=_config['quote_window_minutes'])
    quote_window_end = entry_time + timedelta(minutes=_config['quote_forward_minutes'])
    
    quotes_filtered = quotes_df[
        (quotes_df.index >= quote_window_start) & 
        (quotes_df.index <= quote_window_end)
    ]
    
    logger.info(f"Quote filtering: {len(quotes_df)} -> {len(quotes_filtered)} quotes")
    logger.info(f"Quote window: {quote_window_start} to {quote_window_end}")
    
    # Use filtered quotes
    quotes_df = quotes_filtered
    
    # 5. Sample trades if needed - using count-based sampling
    if len(trades_df) > _config['max_trades_to_process']:
        report_progress(20, f"Optimizing data for analysis...")
        trades_df = _sample_trades(trades_df)
        quotes_df = _sample_quotes(quotes_df)
        report_progress(25, f"Optimized to {len(trades_df):,} trades and {len(quotes_df):,} quotes")
    else:
        report_progress(25, "Using all available data")
    
    # 6. Process quotes - convert to numpy arrays for faster iteration
    report_progress(30, "Building quote history...")
    
    quote_times = quotes_df.index.to_numpy()
    quote_bids = quotes_df['bid'].to_numpy()
    quote_asks = quotes_df['ask'].to_numpy()
    quote_bid_sizes = quotes_df['bid_size'].to_numpy() if 'bid_size' in quotes_df else np.full(len(quotes_df), 100)
    quote_ask_sizes = quotes_df['ask_size'].to_numpy() if 'ask_size' in quotes_df else np.full(len(quotes_df), 100)
    
    quote_count = 0
    quote_update_interval = max(1, len(quotes_df) // 20)
    
    for i in range(len(quote_times)):
        quote_timestamp = quote_times[i]
        if hasattr(quote_timestamp, 'to_pydatetime'):
            quote_timestamp = quote_timestamp.to_pydatetime()
            
        quote = Quote(
            symbol=symbol,
            bid=float(quote_bids[i]),
            ask=float(quote_asks[i]),
            bid_size=int(quote_bid_sizes[i]),
            ask_size=int(quote_ask_sizes[i]),
            timestamp=quote_timestamp
        )
        analyzer.process_quote(quote)
        quote_count += 1
        
        if i % quote_update_interval == 0 and i > 0:
            progress = 30 + int((i / len(quotes_df)) * 15)
            report_progress(progress, f"Processed {quote_count:,} quotes")
    
    # 7. Process trades - convert to numpy arrays
    report_progress(45, f"Analyzing {len(trades_df):,} trades...")
    
    trade_times = trades_df.index.to_numpy()
    trade_prices = trades_df['price'].to_numpy()
    trade_sizes = trades_df['size'].to_numpy()
    trade_conditions = trades_df['conditions'].to_numpy() if 'conditions' in trades_df else None
    
    signals = []
    last_signal = None
    trades_processed = 0
    update_interval = max(1, len(trades_df) // 20)
    sync_success = 0
    sync_fail = 0
    
    for i in range(len(trade_times)):
        # Stop at entry time
        if trade_times[i] >= entry_time:
            break
        
        trade_timestamp = trade_times[i]
        if hasattr(trade_timestamp, 'to_pydatetime'):
            trade_timestamp = trade_timestamp.to_pydatetime()
        
        # Create trade object
        trade = Trade(
            symbol=symbol,
            price=float(trade_prices[i]),
            size=int(trade_sizes[i]),
            timestamp=trade_timestamp,
            conditions=trade_conditions[i] if trade_conditions is not None else None
        )
        
        # Check sync before processing (for logging)
        synced_quote = analyzer._get_synchronized_quote(trade)
        if synced_quote:
            sync_success += 1
        else:
            sync_fail += 1
        
        # Process trade and get signal
        signal = analyzer.process_trade(trade)
        
        if signal:
            signals.append(signal)
            last_signal = signal
        
        trades_processed += 1
        
        # Update progress
        if i % update_interval == 0 and i > 0:
            progress = 45 + int((i / len(trades_df)) * 45)
            sync_rate = sync_success / (sync_success + sync_fail) * 100 if (sync_success + sync_fail) > 0 else 0
            report_progress(
                min(progress, 90),
                f"Processed {trades_processed:,} trades, {len(signals):,} signals, sync: {sync_rate:.0f}%"
            )
    
    # Log sync statistics
    total_sync_attempts = sync_success + sync_fail
    sync_rate = 0
    if total_sync_attempts > 0:
        sync_rate = sync_success / total_sync_attempts * 100
        logger.info(f"Trade/Quote sync rate: {sync_rate:.1f}% ({sync_success}/{total_sync_attempts})")
    
    # 8. Complete analysis
    report_progress(92, "Generating final analysis...")
    
    if not last_signal:
        if symbol in analyzer.latest_signals:
            last_signal = analyzer.latest_signals[symbol]
        else:
            raise ValueError("No signals generated from analysis")
    
    # Get summaries
    session_summary = analyzer.get_session_summary(symbol)
    bar_summary = analyzer.get_bar_index_summary(symbol)
    
    # 9. Format results
    report_progress(95, "Formatting results...")
    
    result = _format_results(
        last_signal, signals, session_summary,
        bar_summary, analyzer, entry_time, direction
    )
    
    # Add performance stats
    result['performance_stats'] = {
        'original_trades': original_trade_count,
        'processed_trades': trades_processed,
        'original_quotes': original_quote_count,
        'filtered_quotes': len(quotes_filtered),
        'processed_quotes': quote_count,
        'sampling_ratio': trades_processed / original_trade_count if original_trade_count > 0 else 1,
        'signals_generated': len(signals),
        'sync_rate': sync_rate,
        'sync_tolerance_ms': adaptive_tolerance,
        'quote_window_minutes': _config['quote_window_minutes'],
        'count_interval': _config['count_sample_interval'],
        'processing_mode': 'standard'
    }
    
    report_progress(100, f"Complete - {last_signal.signal_type}")
    
    return result


async def _fetch_data(symbol: str, entry_time: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch required trade and quote data using modular data manager"""
    # Calculate time range
    start_time = entry_time - timedelta(minutes=_config['lookback_minutes'])
    
    logger.info(f"Fetching {symbol} data from {start_time} to {entry_time}")
    
    # Fetch trade data using modular data manager
    trades = await _data_manager.load_trades(
        symbol=symbol,
        start_time=start_time,
        end_time=entry_time
    )
    
    # Fetch quote data using modular data manager
    quotes = await _data_manager.load_quotes(
        symbol=symbol,
        start_time=start_time,
        end_time=entry_time
    )
    
    return trades, quotes


def _sample_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sample trades using count-based method.
    Takes every Nth trade based on count_sample_interval.
    """
    interval = _config['count_sample_interval']
    
    # Sort by time to ensure proper ordering
    trades_df = trades_df.sort_index()
    
    # Take every Nth trade
    sampled = trades_df.iloc[::interval].copy()
    
    logger.info(f"Trade sampling (every {interval} trades): {len(trades_df)} → {len(sampled)} trades")
    return sampled


def _sample_quotes(quotes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sample quotes to ensure good coverage for trade sync.
    Uses half the interval of trades for better coverage.
    """
    # Use half the trade interval for quotes to ensure better coverage
    interval = max(1, _config['count_sample_interval'] // 2)
    
    # Sort by time
    quotes_df = quotes_df.sort_index()
    
    # Take every Nth quote
    sampled = quotes_df.iloc[::interval].copy()
    
    logger.info(f"Quote sampling (every {interval} quotes): {len(quotes_df)} → {len(sampled)} quotes")
    return sampled


def _format_results(final_signal: BidAskSignal, all_signals: List[BidAskSignal],
                   session_summary: Dict, bar_summary: Optional[List[Dict]],
                   analyzer: BidAskImbalance, entry_time: datetime, 
                   direction: str) -> Dict[str, Any]:
    """Format results for display including bar indices"""
    
    # Extract key metrics from final signal
    components = final_signal.components
    
    # Calculate signal statistics
    strong_bull_signals = sum(1 for s in all_signals if s.bull_score >= 2)
    strong_bear_signals = sum(1 for s in all_signals if s.bear_score >= 2)
    avg_confidence = np.mean([s.confidence for s in all_signals]) if all_signals else 0
    
    # Build summary display rows
    summary_rows = [
        ['Signal', f"{final_signal.signal_type}"],
        ['Strength', f"{final_signal.signal_strength}"],
        ['Bull/Bear Score', f"{final_signal.bull_score}/{final_signal.bear_score}"],
        ['Confidence', f"{final_signal.confidence:.0%}"],
        ['Imbalance', f"{components.smoothed_imbalance:+.2%}"],
        ['Aggression', f"{components.aggression_ratio:.0%}"],
        ['Spread', f"{components.spread_ratio_1min:.2f}x normal"],
        ['Liquidity', components.liquidity_state.upper()],
        ['Buy Volume', f"{components.buy_volume:,.0f}"],
        ['Sell Volume', f"{components.sell_volume:,.0f}"]
    ]
    
    # Add session summary
    if session_summary:
        summary_rows.extend([
            ['Session Imbalance', f"{session_summary['session_imbalance']:+.2%}"],
            ['Extreme Periods', f"{session_summary['extreme_periods']}"]
        ])
    
    # Format bar indices for display
    bar_index_rows = []
    if components.bar_indices:
        for bar in components.bar_indices:
            bar_index_rows.append([
                f"Bar {bar.bar_index}",
                f"{bar.weighted_imbalance:+.1%}",
                f"{bar.aggression_ratio:.0%}",
                f"{bar.buy_volume:,.0f}",
                f"{bar.sell_volume:,.0f}",
                f"{bar.total_volume:,.0f}",
                bar.time_range[0].strftime('%H:%M:%S')
            ])
    
    # Signal history for detailed display
    signal_history = []
    for i, sig in enumerate(all_signals[-10:]):  # Last 10 signals
        signal_history.append([
            sig.timestamp.strftime('%H:%M:%S'),
            f"{sig.components.smoothed_imbalance:+.1%}",
            f"{sig.bull_score}/{sig.bear_score}",
            sig.signal_strength,
            f"{sig.current_price:.2f}"
        ])
    
    # Determine alignment and confidence
    aligned = False
    confidence = final_signal.confidence * 100
    
    if direction == 'LONG':
        if final_signal.bull_score > final_signal.bear_score:
            aligned = True
        elif final_signal.bear_score > final_signal.bull_score:
            confidence = max(0, 100 - confidence)
    else:  # SHORT
        if final_signal.bear_score > final_signal.bull_score:
            aligned = True
        elif final_signal.bull_score > final_signal.bear_score:
            confidence = max(0, 100 - confidence)
    
    # Create description with bar trend info
    description = final_signal.reason
    
    # Add bar trend description
    if components.bar_indices and len(components.bar_indices) >= 3:
        recent_imb = np.mean([b.weighted_imbalance for b in components.bar_indices[:3]])
        older_imb = np.mean([b.weighted_imbalance for b in components.bar_indices[3:6]]) if len(components.bar_indices) >= 6 else recent_imb
        
        if abs(recent_imb - older_imb) > 0.1:
            if recent_imb > older_imb:
                description += " | 📈 Momentum accelerating"
            else:
                description += " | 📉 Momentum decelerating"
    
    if final_signal.warnings:
        description += f" | ⚠️ {', '.join(final_signal.warnings)}"
    
    if aligned:
        description += f" | ✅ Supports {direction}"
    else:
        description += f" | ❌ Contradicts {direction}"
    
    # Map signal to direction
    if final_signal.bull_score > final_signal.bear_score:
        signal_direction = 'BULLISH'
    elif final_signal.bear_score > final_signal.bull_score:
        signal_direction = 'BEARISH'
    else:
        signal_direction = 'NEUTRAL'
    
    return {
        'plugin_name': PLUGIN_NAME,
        'timestamp': entry_time,
        'signal': {
            'direction': signal_direction,
            'strength': float(final_signal.signal_strength == 'EXCEPTIONAL' and 100 or 
                             final_signal.signal_strength == 'STRONG' and 75 or
                             final_signal.signal_strength == 'MODERATE' and 50 or 25),
            'confidence': float(confidence)
        },
        'details': {
            'signal_type': final_signal.signal_type,
            'bull_score': final_signal.bull_score,
            'bear_score': final_signal.bear_score,
            'imbalance': components.smoothed_imbalance,
            'aggression_ratio': components.aggression_ratio,
            'spread_ratio': components.spread_ratio_1min,
            'liquidity_state': components.liquidity_state,
            'buy_volume': components.buy_volume,
            'sell_volume': components.sell_volume,
            'total_signals': len(all_signals),
            'strong_bull_signals': strong_bull_signals,
            'strong_bear_signals': strong_bear_signals,
            'aligned': aligned,
            'num_bar_indices': len(components.bar_indices) if components.bar_indices else 0
        },
        'display_data': {
            'summary': f"{final_signal.signal_type} - {final_signal.signal_strength}",
            'description': description,
            'table_data': summary_rows,
            'bar_indices': {
                'headers': ['Bar', 'Imbalance', 'Aggression', 'Buy Vol', 'Sell Vol', 'Total Vol', 'Start Time'],
                'rows': bar_index_rows
            },
            'signal_history': {
                'headers': ['Time', 'Imbalance', 'Bull/Bear', 'Strength', 'Price'],
                'rows': signal_history
            },
            'chart_markers': _get_chart_markers(final_signal, all_signals)
        }
    }


def _get_chart_markers(final_signal: BidAskSignal, 
                      all_signals: List[BidAskSignal]) -> List[Dict]:
    """Get chart markers for visualization"""
    markers = []
    
    # Add markers for exceptional signals
    for signal in all_signals:
        if signal.signal_strength == 'EXCEPTIONAL':
            markers.append({
                'time': signal.timestamp,
                'type': 'imbalance_extreme',
                'label': signal.signal_type.replace('_', ' '),
                'color': 'green' if signal.bull_score > signal.bear_score else 'red'
            })
    
    # Final signal marker
    if final_signal.signal_strength in ['EXCEPTIONAL', 'STRONG']:
        markers.append({
            'time': final_signal.timestamp,
            'type': 'final_signal',
            'label': f"FINAL: {final_signal.signal_type}",
            'color': 'darkgreen' if final_signal.bull_score > final_signal.bear_score else 'darkred'
        })
        
    return markers