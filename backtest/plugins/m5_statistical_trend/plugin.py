# backtest/plugins/m5_statistical_trend/plugin.py
"""
5-Minute Statistical Trend Plugin Implementation
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backtest.plugins.base_plugin import BacktestPlugin
from modules.calculations.trend.statistical_trend_5min import StatisticalTrend5Min, PositionSignal5Min
from backtest.data.polygon_data_manager import PolygonDataManager

logger = logging.getLogger(__name__)


class M5StatisticalTrendPlugin(BacktestPlugin):
    """Plugin for 5-minute statistical trend analysis"""
    
    def __init__(self):
        # Initialize data manager with smaller extension window
        self.data_manager = PolygonDataManager(
            extend_window_bars=2  # Only extend by 2 bars as specified
        )
        self.config = {
            'lookback_minutes': 75,   # Fetch 75 minutes of data (15 x 5-min bars)
            'analysis_bars': 10,      # Analyze last 10 bars (50 minutes)
            'timeframe': '5min',
            'short_lookback': 3,      # 15-min trend (3 x 5-min)
            'medium_lookback': 5,     # 25-min trend (5 x 5-min)
            'long_lookback': 10       # 50-min trend (10 x 5-min)
        }
        
    @property
    def name(self) -> str:
        return "5-Min Statistical Trend"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
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
    
    async def run_analysis(self, symbol: str, entry_time: datetime, direction: str) -> Dict[str, Any]:
        """
        Run the complete 5-minute statistical trend analysis.
        """
        try:
            # 1. Initialize trend calculator
            trend_calc = StatisticalTrend5Min(
                short_lookback=self.config['short_lookback'],
                medium_lookback=self.config['medium_lookback'],
                long_lookback=self.config['long_lookback']
            )
            
            # 2. Fetch required data (75 minutes)
            bars_df = await self._fetch_data(symbol, entry_time)
            
            if bars_df.empty:
                raise ValueError(f"No bar data available for {symbol}")
            
            # 3. Feed historical data to calculator
            signal_history = []
            final_signal = None
            
            logger.info(f"Processing {len(bars_df)} 5-min bars for trend analysis")
            
            for timestamp, bar in bars_df.iterrows():
                # Stop processing after entry time
                if timestamp > entry_time:
                    break
                
                # Update trend calculator with bar data
                signal = trend_calc.update_bar(
                    symbol=symbol,
                    open_price=float(bar['open']),
                    high=float(bar['high']),
                    low=float(bar['low']),
                    close=float(bar['close']),
                    volume=float(bar['volume']),
                    timestamp=timestamp
                )
                
                # Store signal if we have one
                if signal:
                    signal_data = {
                        'timestamp': timestamp,
                        'price': signal.price,
                        'signal': signal.signal,
                        'bias': signal.bias,
                        'confidence': signal.confidence,
                        'strength': signal.strength,
                        'market_state': signal.market_state,
                        'short_direction': signal.short_trend.get('direction', 'N/A') if signal.short_trend else 'N/A',
                        'medium_direction': signal.medium_trend.get('direction', 'N/A') if signal.medium_trend else 'N/A',
                        'long_direction': signal.long_trend.get('direction', 'N/A') if signal.long_trend else 'N/A'
                    }
                    signal_history.append(signal_data)
                    
                    # Update final signal if at or before entry time
                    if timestamp <= entry_time:
                        final_signal = signal
            
            if not final_signal:
                raise ValueError("No signal generated at entry time")
            
            # 4. Format results
            return self._format_results(final_signal, signal_history, entry_time, direction)
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise
    
    async def _fetch_data(self, symbol: str, entry_time: datetime) -> pd.DataFrame:
        """Fetch required bar data using get_bars for better control"""
        # Use get_bars which gives us more control over the fetch
        logger.info(f"Fetching {symbol} 5-min bars for 75 minutes before {entry_time}")
        
        # Get exactly what we need - 75 minutes of data
        bars = self.data_manager.get_bars(
            symbol=symbol,
            entry_time=entry_time,
            lookback_hours=1.25,  # 75 minutes = 1.25 hours
            forward_bars=0,       # No forward bars needed
            timeframe='5min'
        )
        
        if bars is None:
            # Fallback to load_bars if get_bars fails
            start_time = entry_time - timedelta(minutes=self.config['lookback_minutes'])
            logger.warning(f"get_bars failed, trying load_bars from {start_time} to {entry_time}")
            
            bars = await self.data_manager.load_bars(
                symbol=symbol,
                start_time=start_time,
                end_time=entry_time,
                timeframe='5min',
                use_cache=True
            )
        
        logger.info(f"Fetched {len(bars) if bars is not None else 0} bars")
        
        return bars if bars is not None else pd.DataFrame()
    
    def _format_results(self, final_signal: PositionSignal5Min, signal_history: List[Dict],
                       entry_time: datetime, direction: str) -> Dict[str, Any]:
        """Format results for display"""
        
        # Get last 10 signals for display
        display_signals = signal_history[-10:] if len(signal_history) >= 10 else signal_history
        
        # Build detailed signal history table
        signal_rows = []
        for sig in display_signals:
            # Format timeframe indicators with arrows
            short_indicator = "↑" if sig['short_direction'] == 'bullish' else "↓" if sig['short_direction'] == 'bearish' else "→"
            medium_indicator = "↑" if sig['medium_direction'] == 'bullish' else "↓" if sig['medium_direction'] == 'bearish' else "→"
            long_indicator = "↑" if sig['long_direction'] == 'bullish' else "↓" if sig['long_direction'] == 'bearish' else "→"
            
            signal_rows.append([
                sig['timestamp'].strftime('%H:%M:%S'),
                f"${sig['price']:.2f}",
                sig['signal'],
                sig['bias'],
                f"{sig['confidence']:.0f}%",
                f"{sig['strength']:.0f}%",
                f"{short_indicator} {medium_indicator} {long_indicator}",
                sig['market_state']
            ])
        
        # Determine alignment
        aligned = False
        if (direction == 'LONG' and final_signal.bias == 'BULLISH') or \
           (direction == 'SHORT' and final_signal.bias == 'BEARISH'):
            aligned = True
        
        # Map bias directly to dashboard direction (bias already matches dashboard format)
        dashboard_direction = final_signal.bias
        
        # Build summary rows
        summary_rows = [
            ['Position Signal', final_signal.signal],
            ['Market Bias', final_signal.bias],
            ['Confidence', f"{final_signal.confidence:.0f}%"],
            ['Strength', f"{final_signal.strength:.0f}%"],
            ['Market State', final_signal.market_state],
            ['Recommendation', final_signal.recommendation]
        ]
        
        # Add detailed timeframe analysis
        if final_signal.short_trend:
            short = final_signal.short_trend
            summary_rows.append([
                '15-min Trend', 
                f"{short.get('direction', 'N/A').upper()} (Strength: {short.get('strength', 0):.0f}%, VWAP: {short.get('vwap_position', 0):+.2f}%)"
            ])
        if final_signal.medium_trend:
            medium = final_signal.medium_trend
            summary_rows.append([
                '25-min Trend', 
                f"{medium.get('direction', 'N/A').upper()} (Strength: {medium.get('strength', 0):.0f}%, Score: {medium.get('score', 0):.3f})"
            ])
        if final_signal.long_trend:
            long = final_signal.long_trend
            summary_rows.append([
                '50-min Trend', 
                f"{long.get('direction', 'N/A').upper()} (Strength: {long.get('strength', 0):.0f}%, Score: {long.get('score', 0):.3f})"
            ])
        
        # Create description
        description = f"{final_signal.signal}: {final_signal.recommendation}"
        if aligned:
            description += f" | ✅ Bias aligned with {direction} trade"
        else:
            description += f" | ⚠️ Bias not aligned with {direction} trade"
        
        # Add signal progression summary
        signal_counts = {}
        for sig in display_signals:
            signal_counts[sig['signal']] = signal_counts.get(sig['signal'], 0) + 1
        
        progression_summary = "Signal progression: " + ", ".join([f"{signal}={count}" for signal, count in signal_counts.items()])
        
        return {
            'plugin_name': self.name,
            'timestamp': entry_time,
            'signal': {
                'direction': dashboard_direction,
                'strength': float(final_signal.strength),
                'confidence': float(final_signal.confidence)
            },
            'details': {
                'position_signal': final_signal.signal,
                'market_bias': final_signal.bias,
                'market_state': final_signal.market_state,
                'recommendation': final_signal.recommendation,
                'short_trend': final_signal.short_trend,
                'medium_trend': final_signal.medium_trend,
                'long_trend': final_signal.long_trend,
                'aligned': aligned,
                'signals_analyzed': len(display_signals),
                'signal_progression': progression_summary
            },
            'display_data': {
                'summary': f"{final_signal.signal} - {final_signal.bias} bias",
                'description': description,
                'table_data': summary_rows,
                'signal_history': {
                    'headers': ['Time', 'Price', 'Signal', 'Bias', 'Conf%', 'Str%', 'Trends (15/25/50)', 'State'],
                    'rows': signal_rows
                }
            }
        }