# backtest/plugins/bid_ask_imbalance/plugin.py
"""
Bid/Ask Imbalance Plugin Implementation with Performance Optimizations
FIXED: Quote filtering to prevent cleanup issues during backtesting
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple, Callable
import pandas as pd
import numpy as np
import logging

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backtest.plugins.base_plugin import BacktestPlugin
from modules.calculations.order_flow.bid_ask_imbal import (
    BidAskImbalance, Trade, Quote, BidAskSignal
)
from backtest.data.polygon_data_manager import PolygonDataManager

logger = logging.getLogger(__name__)


class BidAskImbalancePlugin(BacktestPlugin):
    """Plugin for real-time bid/ask imbalance analysis with performance optimization"""
    
    def __init__(self):
        self.data_manager = PolygonDataManager()
        self.config = {
            'imbalance_lookback': 1000,  # Total trades to track
            'trades_per_bar': 100,  # Number of trades per bar index
            'spread_history_seconds': 600,  # 10 minutes of spread history
            'quote_sync_tolerance_ms': 100,  # Max ms between trade and quote
            'aggression_threshold': 0.7,  # Threshold for aggressive trades
            'smoothing_alpha': 0.1,  # EWMA smoothing factor
            'lookback_minutes': 20,  # Minutes of data to fetch before entry
            # QUOTE FILTERING (NEW)
            'quote_window_minutes': 8,  # Process quotes from X minutes before entry
            'quote_forward_minutes': 2,  # Process quotes until X minutes after entry
            # PERFORMANCE SETTINGS
            'max_trades_to_process': 5000,  # Cap total trades processed
            'sampling_method': 'time',  # 'time', 'volume', or 'count'
            'time_sample_seconds': 1,  # For time-based sampling
            'count_sample_interval': 10,  # Process every Nth trade
            'enable_parallel_processing': True,
            'performance_mode': True  # Enable performance optimizations
        }
        
    @property
    def name(self) -> str:
        return "Bid/Ask Imbalance Analysis"
    
    @property
    def version(self) -> str:
        return "3.1.0"  # Updated version for quote filtering fix
    
    def get_config(self) -> Dict[str, Any]:
        """Get plugin configuration"""
        return self.config.copy()
    
    def validate_inputs(self, symbol: str, entry_time: datetime, direction: str) -> bool:
        """Validate input parameters"""
        if not symbol or not isinstance(symbol, str):
            return False
        
        if not isinstance(entry_time, datetime):
            return False
            
        if direction not in ['LONG', 'SHORT']:
            return False
            
        return True
    
    def set_performance_mode(self, enabled: bool):
        """Enable or disable performance mode"""
        self.config['performance_mode'] = enabled
        if enabled:
            self.config['max_trades_to_process'] = 3000
            self.config['time_sample_seconds'] = 2
        else:
            self.config['max_trades_to_process'] = 10000
            self.config['time_sample_seconds'] = 1
    
    async def run_analysis(self, symbol: str, entry_time: datetime, direction: str) -> Dict[str, Any]:
        """Run analysis without progress reporting (backward compatibility)"""
        return await self.run_analysis_with_progress(symbol, entry_time, direction, None)
    
    async def run_analysis_with_progress(self, symbol: str, entry_time: datetime, 
                                       direction: str,
                                       progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Run analysis with optimizations and progress reporting"""
        try:
            # Progress helper
            def report_progress(percentage: int, message: str):
                if progress_callback:
                    progress_callback(percentage, message)
                logger.info(f"Progress: {percentage}% - {message}")
            
            # 1. Initialize analyzer
            report_progress(5, "Initializing analyzer...")
            analyzer = BidAskImbalance(
                imbalance_lookback=self.config['imbalance_lookback'],
                trades_per_bar=self.config['trades_per_bar'],
                spread_history_seconds=self.config['spread_history_seconds'],
                quote_sync_tolerance_ms=self.config['quote_sync_tolerance_ms'],
                aggression_threshold=self.config['aggression_threshold'],
                smoothing_alpha=self.config['smoothing_alpha']
            )
            
            # 2. Fetch data
            report_progress(10, "Fetching trade and quote data...")
            trades_df, quotes_df = await self._fetch_data(symbol, entry_time)
            
            if trades_df.empty:
                raise ValueError(f"No trade data available for {symbol}")
            
            original_trade_count = len(trades_df)
            original_quote_count = len(quotes_df)
            report_progress(20, f"Fetched {original_trade_count:,} trades and {original_quote_count:,} quotes")
            
            # 3. CRITICAL FIX: Filter quotes to prevent cleanup issues
            # When processing historical data, we need to limit quotes to a window
            # that won't be cleaned up by the time we process trades
            quote_window_start = entry_time - timedelta(minutes=self.config['quote_window_minutes'])
            quote_window_end = entry_time + timedelta(minutes=self.config['quote_forward_minutes'])
            
            quotes_filtered = quotes_df[
                (quotes_df.index >= quote_window_start) & 
                (quotes_df.index <= quote_window_end)
            ]
            
            logger.info(f"Quote filtering: {len(quotes_df)} -> {len(quotes_filtered)} quotes")
            logger.info(f"Quote window: {quote_window_start} to {quote_window_end}")
            
            # Use filtered quotes
            quotes_df = quotes_filtered
            
            # 4. OPTIMIZATION: Sample data if too much (after filtering)
            if self.config['performance_mode'] and len(trades_df) > self.config['max_trades_to_process']:
                report_progress(25, f"Optimizing data for performance...")
                trades_df = self._sample_trades_efficiently(trades_df, quotes_df)
                quotes_df = self._sample_quotes_efficiently(quotes_df)
                report_progress(30, f"Optimized to {len(trades_df):,} trades and {len(quotes_df):,} quotes")
            
            # 5. Process quotes
            report_progress(35, "Building quote history...")
            
            quote_count = 0
            quote_update_interval = max(1, len(quotes_df) // 10)
            
            for i, (timestamp, quote_data) in enumerate(quotes_df.iterrows()):
                # Convert timestamp if needed
                if hasattr(timestamp, 'to_pydatetime'):
                    quote_timestamp = timestamp.to_pydatetime()
                else:
                    quote_timestamp = timestamp
                    
                quote = Quote(
                    symbol=symbol,
                    bid=float(quote_data['bid']),
                    ask=float(quote_data['ask']),
                    bid_size=int(quote_data.get('bid_size', 100)),
                    ask_size=int(quote_data.get('ask_size', 100)),
                    timestamp=quote_timestamp
                )
                analyzer.process_quote(quote)
                quote_count += 1
                
                if i % quote_update_interval == 0 and i > 0:
                    progress = 35 + int((i / len(quotes_df)) * 10)  # 35-45%
                    report_progress(progress, f"Processed {quote_count:,} quotes")
            
            # Log quote history health
            if symbol in analyzer.quote_history and analyzer.quote_history[symbol]:
                history = list(analyzer.quote_history[symbol])
                logger.info(f"Quote history: {len(history)} quotes, "
                           f"spanning {(history[-1].timestamp - history[0].timestamp).total_seconds():.1f}s")
            
            # 6. Process trades
            report_progress(45, f"Analyzing {len(trades_df):,} trades...")
            
            signals = []
            last_signal = None
            trades_processed = 0
            update_interval = max(1, len(trades_df) // 20)  # Update progress 20 times
            sync_success = 0
            sync_fail = 0
            
            for i, (timestamp, trade_data) in enumerate(trades_df.iterrows()):
                # Stop at entry time
                if timestamp >= entry_time:
                    break
                
                # Convert timestamp if needed
                if hasattr(timestamp, 'to_pydatetime'):
                    trade_timestamp = timestamp.to_pydatetime()
                else:
                    trade_timestamp = timestamp
                
                # Create trade object
                trade = Trade(
                    symbol=symbol,
                    price=float(trade_data['price']),
                    size=int(trade_data['size']),
                    timestamp=trade_timestamp,
                    conditions=trade_data.get('conditions')
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
                    progress = 45 + int((i / len(trades_df)) * 45)  # 45-90%
                    sync_rate = sync_success / (sync_success + sync_fail) * 100 if (sync_success + sync_fail) > 0 else 0
                    report_progress(
                        min(progress, 90),
                        f"Processed {trades_processed:,} trades, {len(signals):,} signals, sync: {sync_rate:.0f}%"
                    )
            
            # Log sync statistics
            total_sync_attempts = sync_success + sync_fail
            if total_sync_attempts > 0:
                sync_rate = sync_success / total_sync_attempts * 100
                logger.info(f"Trade/Quote sync rate: {sync_rate:.1f}% ({sync_success}/{total_sync_attempts})")
                if sync_rate < 50:
                    logger.warning(f"Low sync rate detected! Consider adjusting quote_window_minutes")
            
            # 7. Complete analysis
            report_progress(92, "Generating final analysis...")
            
            if not last_signal:
                # Try to get the latest analysis if no signals
                if symbol in analyzer.latest_signals:
                    last_signal = analyzer.latest_signals[symbol]
                else:
                    raise ValueError("No signals generated from analysis")
            
            # Get summaries
            session_summary = analyzer.get_session_summary(symbol)
            bar_summary = analyzer.get_bar_index_summary(symbol)
            
            # 8. Format results
            report_progress(95, "Formatting results...")
            
            result = self._format_results(
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
                'sync_rate': sync_rate if total_sync_attempts > 0 else 0,
                'performance_mode': self.config['performance_mode'],
                'quote_window_minutes': self.config['quote_window_minutes']
            }
            
            report_progress(100, f"Complete - {last_signal.signal_type}")
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            if progress_callback:
                progress_callback(100, f"Error: {str(e)}")
            raise
    
    def _sample_trades_efficiently(self, trades_df: pd.DataFrame, quotes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Sample trades efficiently based on configured method.
        Reduces large datasets while preserving signal quality.
        """
        method = self.config['sampling_method']
        
        if method == 'time':
            # Time-based sampling - one trade per time interval
            interval = self.config['time_sample_seconds']
            
            # Ensure we have a DatetimeIndex
            if not isinstance(trades_df.index, pd.DatetimeIndex):
                # If 'timestamp' is a column, set it as index
                if 'timestamp' in trades_df.columns:
                    trades_df = trades_df.set_index('timestamp')
                else:
                    logger.error("No timestamp index or column found in trades DataFrame")
                    return trades_df
            
            # Group by time intervals and aggregate (use lowercase 's' for seconds)
            sampled = trades_df.groupby(pd.Grouper(freq=f'{interval}s')).agg({
                'price': ['first', 'last', 'min', 'max'],  # OHLC
                'size': 'sum',  # Total volume
                'conditions': 'first'
            })
            
            # Flatten multi-level columns
            sampled.columns = ['open', 'close', 'low', 'high', 'size', 'conditions']
            sampled['price'] = sampled['close']  # Use close as the price
            sampled = sampled.dropna()
            
            logger.info(f"Time sampling ({interval}s): {len(trades_df)} → {len(sampled)} trades")
            return sampled
            
        elif method == 'volume':
            # Volume-based sampling - sample at volume intervals
            trades_df['cum_volume'] = trades_df['size'].cumsum()
            total_volume = trades_df['size'].sum()
            volume_per_sample = total_volume / self.config['max_trades_to_process']
            
            sampled_indices = []
            last_volume_mark = 0
            
            for idx, row in trades_df.iterrows():
                if row['cum_volume'] >= last_volume_mark + volume_per_sample:
                    sampled_indices.append(idx)
                    last_volume_mark = row['cum_volume']
            
            # Always include first and last trades
            if len(trades_df) > 0:
                sampled_indices = [trades_df.index[0]] + sampled_indices + [trades_df.index[-1]]
                sampled_indices = sorted(list(set(sampled_indices)))
            
            sampled = trades_df.loc[sampled_indices].drop('cum_volume', axis=1)
            logger.info(f"Volume sampling: {len(trades_df)} → {len(sampled)} trades")
            return sampled
            
        else:  # count-based sampling
            # Simple interval sampling
            interval = max(1, len(trades_df) // self.config['max_trades_to_process'])
            sampled = trades_df.iloc[::interval]
            logger.info(f"Count sampling (every {interval}): {len(trades_df)} → {len(sampled)} trades")
            return sampled
    
    def _sample_quotes_efficiently(self, quotes_df: pd.DataFrame) -> pd.DataFrame:
        """Sample quotes to match trade sampling rate"""
        if self.config['sampling_method'] == 'time':
            # Match time-based sampling
            interval = self.config['time_sample_seconds']
            
            # Ensure we have a DatetimeIndex
            if not isinstance(quotes_df.index, pd.DatetimeIndex):
                # If 'timestamp' is a column, set it as index
                if 'timestamp' in quotes_df.columns:
                    quotes_df = quotes_df.set_index('timestamp')
                else:
                    logger.error("No timestamp index or column found in quotes DataFrame")
                    return quotes_df
            
            # Use lowercase 's' for seconds (pandas deprecation fix)
            sampled = quotes_df.groupby(pd.Grouper(freq=f'{interval}s')).last()
            sampled = sampled.dropna()
            logger.info(f"Quote time sampling: {len(quotes_df)} → {len(sampled)} quotes")
            return sampled
        else:
            # Simple interval sampling for quotes
            max_quotes = min(len(quotes_df), self.config['max_trades_to_process'] // 2)
            interval = max(1, len(quotes_df) // max_quotes)
            sampled = quotes_df.iloc[::interval]
            logger.info(f"Quote sampling (every {interval}): {len(quotes_df)} → {len(sampled)} quotes")
            return sampled
    
    async def _fetch_data(self, symbol: str, entry_time: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch required trade and quote data"""
        # Calculate time range
        start_time = entry_time - timedelta(minutes=self.config['lookback_minutes'])
        
        logger.info(f"Fetching {symbol} data from {start_time} to {entry_time}")
        
        # Fetch trade data
        trades = await self.data_manager.load_trades(
            symbol=symbol,
            start_time=start_time,
            end_time=entry_time,
            use_cache=True
        )
        
        # Fetch quote data
        quotes = await self.data_manager.load_quotes(
            symbol=symbol,
            start_time=start_time,
            end_time=entry_time,
            use_cache=True
        )
        
        return trades, quotes
    
    def _format_results(self, final_signal: BidAskSignal, all_signals: List[BidAskSignal],
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
            'plugin_name': self.name,
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
                'chart_markers': self._get_chart_markers(final_signal, all_signals)
            }
        }
    
    def _get_chart_markers(self, final_signal: BidAskSignal, 
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