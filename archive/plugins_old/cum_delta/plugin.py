# backtest/plugins/cum_delta/plugin.py
"""
Cumulative Delta Analysis Plugin Implementation
Tracks order flow through bid/ask trade classification
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple, Callable
import pandas as pd
import numpy as np
import logging
import asyncio

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backtest.plugins.base_plugin import BacktestPlugin
from modules.calculations.order_flow.cum_delta import (
    DeltaFlowAnalyzer, Trade, Quote, DeltaSignal, DeltaTimeSeries
)
from backtest.data.polygon_data_manager import PolygonDataManager

logger = logging.getLogger(__name__)


class CumDeltaPlugin(BacktestPlugin):
    """Plugin for cumulative delta order flow analysis"""
    
    def __init__(self):
        self.data_manager = PolygonDataManager()
        # Let the analyzer determine all the parameters
        self.analyzer = DeltaFlowAnalyzer()
        
    @property
    def name(self) -> str:
        return "Cumulative Delta Analysis"
    
    @property
    def version(self) -> str:
        return "2.0.0"
    
    def get_config(self) -> Dict[str, Any]:
        """Get plugin configuration from analyzer"""
        return {
            'warmup_config': self.analyzer.warmup_config,
            'timeframes': self.analyzer.timeframes,
            'buffer_size': self.analyzer.buffer_size,
            'time_series_lookback': self.analyzer.time_series_lookback
        }
    
    def validate_inputs(self, symbol: str, entry_time: datetime, direction: str) -> bool:
        """Validate input parameters"""
        if not symbol or not isinstance(symbol, str):
            return False
        
        if not isinstance(entry_time, datetime):
            return False
            
        if direction not in ['LONG', 'SHORT']:
            return False
            
        return True
    
    async def run_analysis(self, symbol: str, entry_time: datetime, direction: str) -> Dict[str, Any]:
        """Run the cumulative delta analysis without progress reporting"""
        return await self.run_analysis_with_progress(symbol, entry_time, direction, None)
    
    async def run_analysis_with_progress(self, symbol: str, entry_time: datetime, direction: str,
                                       progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Run the complete cumulative delta analysis with progress reporting.
        Let the analyzer determine all parameters.
        """
        try:
            # Progress update helper
            def update_progress(pct: int, msg: str):
                if progress_callback:
                    progress_callback(pct, msg)
                logger.info(f"Progress: {pct}% - {msg}")
            
            # 1. Initialize analyzer - it already has all the configuration
            update_progress(0, "Initializing analyzer...")
            
            # Enable warmup mode for backtesting
            self.analyzer.warmup_config['enabled'] = True
            
            # 2. Let analyzer determine lookback based on its configuration
            # The analyzer uses 45 minutes by default now
            lookback_minutes = 45  # This matches the analyzer's internal default
            lookback_start = entry_time - timedelta(minutes=lookback_minutes)
            
            # 3. Fetch required data
            update_progress(10, f"Fetching {lookback_minutes} minutes of data...")
            trades_df, quotes_df = await self._fetch_data(symbol, lookback_start, entry_time)
            
            if trades_df.empty:
                raise ValueError(f"No trade data available for {symbol}")
            
            update_progress(25, f"Retrieved {len(trades_df)} trades...")
            
            # 4. Use analyzer's warmup method
            update_progress(30, "Processing trades with analyzer...")
            
            # Initialize symbol
            self.analyzer.initialize_symbol(symbol)
            
            # Check for market open reset
            market_open = entry_time.replace(hour=14, minute=30, second=0, microsecond=0)  # 9:30 AM ET in UTC
            if lookback_start < market_open < entry_time:
                self.analyzer.reset_session(symbol, market_open)
            
            # Use the analyzer's warmup method with progress
            def warmup_progress(pct, msg):
                # Map warmup progress (0-100) to overall progress (30-90)
                overall_pct = 30 + int(pct * 0.6)
                update_progress(overall_pct, msg)
            
            last_signal = self.analyzer.warmup_with_trades(
                symbol=symbol,
                trades_df=trades_df,
                quotes_df=quotes_df,
                entry_time=entry_time,
                progress_callback=warmup_progress
            )
            
            # 5. Get final analysis
            update_progress(90, "Generating analysis...")
            
            if not last_signal:
                # Try to force complete the current bar
                last_signal = self.analyzer.force_complete_bar(symbol)
                if not last_signal:
                    raise ValueError("No signals generated - insufficient data")
            
            # Get time series data
            time_series = self.analyzer.get_delta_time_series(symbol)
            
            # 6. Format results
            update_progress(95, "Formatting results...")
            result = self._format_results(
                last_signal, 
                time_series, 
                entry_time, 
                direction, 
                self.analyzer.delta_aggregators[symbol].warmup_trades_processed
            )
            
            update_progress(100, "Analysis complete!")
            return result
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def _fetch_data(self, symbol: str, start_time: datetime, 
                         end_time: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch required trade and quote data"""
        logger.info(f"Fetching {symbol} data from {start_time} to {end_time}")
        
        # Fetch trade data
        trades = await self.data_manager.load_trades(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            use_cache=True
        )
        
        # Fetch quote data
        quotes = await self.data_manager.load_quotes(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            use_cache=True
        )
        
        logger.info(f"Fetched {len(trades)} trades and {len(quotes)} quotes")
        return trades, quotes
    
    def _format_results(self, signal: DeltaSignal, time_series: List[DeltaTimeSeries],
                       entry_time: datetime, direction: str, trades_processed: int) -> Dict[str, Any]:
        """Format results for display"""
        
        # Map signal to standard format
        signal_direction = signal.signal_type
        if 'ACCUMULATION' in signal_direction or 'BULLISH' in signal_direction:
            signal_direction = 'BULLISH'
        elif 'DISTRIBUTION' in signal_direction or 'BEARISH' in signal_direction:
            signal_direction = 'BEARISH'
        else:
            signal_direction = 'NEUTRAL'
        
        # Calculate strength (0-100)
        if signal.bull_score == 2 or signal.bear_score == 2:
            strength = 90.0
        elif signal.bull_score == 1 or signal.bear_score == 1:
            strength = 60.0
        else:
            strength = 30.0
        
        # Confidence based on signal confidence and alignment
        confidence = signal.confidence * 100
        aligned = False
        
        if (direction == 'LONG' and signal_direction == 'BULLISH') or \
           (direction == 'SHORT' and signal_direction == 'BEARISH'):
            aligned = True
        elif (direction == 'LONG' and signal_direction == 'BEARISH') or \
             (direction == 'SHORT' and signal_direction == 'BULLISH'):
            confidence = max(0, 100 - confidence)  # Invert if opposite
        
        # Build summary statistics
        cum_delta = signal.components.cumulative_delta
        
        # Time series analysis
        if time_series:
            recent_trend = self._analyze_trend(time_series)
            delta_chart = self._prepare_chart_data(time_series)
        else:
            recent_trend = "No data"
            delta_chart = None
        
        # Summary table
        summary_rows = [
            ['Signal', f"{signal.signal_type} ({signal.signal_strength})"],
            ['Cumulative Δ', f"{cum_delta:+,}"],
            ['Efficiency', f"{signal.components.efficiency:.2f}"],
            ['Absorption', f"{signal.components.absorption_score:.0%}"],
            ['Divergences', f"{len(signal.components.divergences)}"],
            ['Delta Rate', f"{signal.components.delta_rate:+.0f}/min"],
            ['Recent Trend', recent_trend],
            ['Trades Processed', f"{trades_processed:,}"]
        ]
        
        # Timeframe deltas
        tf_rows = []
        for tf, delta_val in signal.components.timeframe_deltas.items():
            tf_rows.append([f"{tf} Δ", f"{delta_val:+,}"])
        
        # Build description
        description = signal.reason
        if aligned:
            description += f" | ✅ Supports {direction}"
        else:
            description += f" | ⚠️ Contradicts {direction}"
        
        # Add warnings if any
        if signal.warnings:
            description += f" | ⚠️ {', '.join(signal.warnings)}"
        
        return {
            'plugin_name': self.name,
            'timestamp': entry_time,
            'signal': {
                'direction': signal_direction,
                'strength': float(strength),
                'confidence': float(confidence)
            },
            'details': {
                'cumulative_delta': cum_delta,
                'session_delta': self.analyzer.get_session_delta(symbol),
                'efficiency': signal.components.efficiency,
                'absorption_score': signal.components.absorption_score,
                'delta_rate': signal.components.delta_rate,
                'delta_volatility': signal.components.delta_volatility,
                'directional_efficiency': signal.components.directional_efficiency,
                'divergences': [d['description'] for d in signal.components.divergences],
                'trades_processed': trades_processed,
                'signal_type': signal.signal_type,
                'signal_strength': signal.signal_strength,
                'aligned': aligned,
                'calculation_time_ms': signal.calculation_time_ms
            },
            'display_data': {
                'summary': f"{signal.signal_type} - Δ: {cum_delta:+,}",
                'description': description,
                'table_data': summary_rows,
                'timeframe_deltas': {
                    'headers': ['Timeframe', 'Delta'],
                    'rows': tf_rows
                },
                'time_series_chart': delta_chart,
                'chart_markers': self._get_chart_markers(signal)
            }
        }
    
    def _analyze_trend(self, time_series: List[DeltaTimeSeries]) -> str:
        """Analyze recent trend from time series"""
        if len(time_series) < 5:
            return "Insufficient data"
        
        # Look at last 5 minutes
        recent = time_series[-5:]
        deltas = [ts.period_delta for ts in recent]
        
        # Calculate trend
        positive = sum(1 for d in deltas if d > 0)
        total_delta = sum(deltas)
        
        if positive >= 4:
            return f"Strong buying ({positive}/5 positive)"
        elif positive >= 3:
            return f"Buying pressure ({positive}/5 positive)"
        elif positive <= 1:
            return f"Strong selling ({5-positive}/5 negative)"
        elif positive <= 2:
            return f"Selling pressure ({5-positive}/5 negative)"
        else:
            return "Mixed"
    
    def _prepare_chart_data(self, time_series: List[DeltaTimeSeries]) -> Dict[str, Any]:
        """Prepare time series data for charting"""
        if not time_series:
            return None
        
        return {
            'timestamps': [ts.timestamp.isoformat() for ts in time_series],
            'cumulative_delta': [ts.cumulative_delta for ts in time_series],
            'period_delta': [ts.period_delta for ts in time_series],
            'efficiency': [ts.efficiency for ts in time_series],
            'close_price': [ts.close_price for ts in time_series],
            'volume': [ts.period_volume for ts in time_series]
        }
    
    def _get_chart_markers(self, signal: DeltaSignal) -> List[Dict]:
        """Get chart markers for visualization"""
        markers = []
        
        # Strong accumulation/distribution
        if signal.components.cumulative_delta > 5000:
            markers.append({
                'type': 'strong_accumulation',
                'label': 'STRONG ACC',
                'color': 'green'
            })
        elif signal.components.cumulative_delta < -5000:
            markers.append({
                'type': 'strong_distribution',
                'label': 'STRONG DIST',
                'color': 'red'
            })
        
        # Divergences
        for div in signal.components.divergences:
            if div['type'] == 'bullish':
                markers.append({
                    'type': 'bullish_divergence',
                    'label': 'BULL DIV',
                    'color': 'lime'
                })
            else:
                markers.append({
                    'type': 'bearish_divergence',
                    'label': 'BEAR DIV',
                    'color': 'orange'
                })
        
        # High absorption
        if signal.components.absorption_score > 0.7:
            markers.append({
                'type': 'absorption',
                'label': 'ABSORPTION',
                'color': 'yellow'
            })
            
        return markers[:3]  # Limit to 3 most important markers