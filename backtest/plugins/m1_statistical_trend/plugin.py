# backtest/plugins/m1_statistical_trend/plugin.py
"""
1-Minute Statistical Trend Plugin Implementation
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
from modules.calculations.trend.statistical_trend_1min import StatisticalTrend1Min, ScalperSignal
from backtest.data.polygon_data_manager import PolygonDataManager

logger = logging.getLogger(__name__)


class M1StatisticalTrendPlugin(BacktestPlugin):
    """Plugin for 1-minute statistical trend analysis"""
    
    def __init__(self):
        # Initialize data manager with smaller extension window
        self.data_manager = PolygonDataManager(
            extend_window_bars=5  # Only extend by 5 bars instead of 500
        )
        self.config = {
            'lookback_minutes': 15,  # Fetch 15 minutes of data
            'analysis_minutes': 10,  # Analyze last 10 minutes
            'micro_lookback': 3,     # 3-min micro trend
            'short_lookback': 5,     # 5-min short trend
            'medium_lookback': 10    # 10-min medium trend
        }
        
    @property
    def name(self) -> str:
        return "1-Min Statistical Trend"
    
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
        Run the complete 1-minute statistical trend analysis.
        """
        try:
            # 1. Initialize trend calculator
            trend_calc = StatisticalTrend1Min(
                micro_lookback=self.config['micro_lookback'],
                short_lookback=self.config['short_lookback'],
                medium_lookback=self.config['medium_lookback']
            )
            
            # 2. Fetch required data (15 minutes)
            bars_df = await self._fetch_data(symbol, entry_time)
            
            if bars_df.empty:
                raise ValueError(f"No bar data available for {symbol}")
            
            # 3. Feed historical data to calculator
            signal_history = []
            final_signal = None
            
            logger.info(f"Processing {len(bars_df)} bars for trend analysis")
            
            for timestamp, bar in bars_df.iterrows():
                # Stop processing after entry time
                if timestamp > entry_time:
                    break
                
                # Update trend calculator
                signal = trend_calc.update_price(
                    symbol=symbol,
                    price=float(bar['close']),
                    volume=float(bar['volume']),
                    timestamp=timestamp
                )
                
                # Store signal if we have one
                if signal:
                    signal_data = {
                        'timestamp': timestamp,
                        'price': float(bar['close']),
                        'signal': signal.signal,
                        'confidence': signal.confidence,
                        'strength': signal.strength,
                        'micro_direction': signal.micro_trend.get('direction', 'N/A'),
                        'short_direction': signal.short_trend.get('direction', 'N/A'),
                        'medium_direction': signal.medium_trend.get('direction', 'N/A'),
                        'momentum': signal.micro_trend.get('momentum', 0.0)
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
        logger.info(f"Fetching {symbol} bars for 15 minutes before {entry_time}")
        
        # Get exactly what we need - 15 minutes of data
        bars = self.data_manager.get_bars(
            symbol=symbol,
            entry_time=entry_time,
            lookback_hours=0.25,  # 15 minutes = 0.25 hours
            forward_bars=0,       # No forward bars needed
            timeframe='1min'
        )
        
        if bars is None:
            # Fallback to load_bars if get_bars fails
            start_time = entry_time - timedelta(minutes=self.config['lookback_minutes'])
            logger.warning(f"get_bars failed, trying load_bars from {start_time} to {entry_time}")
            
            bars = await self.data_manager.load_bars(
                symbol=symbol,
                start_time=start_time,
                end_time=entry_time,
                timeframe='1min',
                use_cache=True
            )
        
        logger.info(f"Fetched {len(bars) if bars is not None else 0} bars")
        
        return bars if bars is not None else pd.DataFrame()
    
    def _format_results(self, final_signal: ScalperSignal, signal_history: List[Dict],
                   entry_time: datetime, direction: str) -> Dict[str, Any]:
        """Format results for display"""
        
        # Get last 10 signals for display
        display_signals = signal_history[-10:] if len(signal_history) >= 10 else signal_history
        
        # Build detailed signal history table
        signal_rows = []
        for sig in display_signals:
            # Format timeframe indicators with arrows
            micro_indicator = "↑" if sig['micro_direction'] == 'bullish' else "↓" if sig['micro_direction'] == 'bearish' else "→"
            short_indicator = "↑" if sig['short_direction'] == 'bullish' else "↓" if sig['short_direction'] == 'bearish' else "→"
            medium_indicator = "↑" if sig['medium_direction'] == 'bullish' else "↓" if sig['medium_direction'] == 'bearish' else "→"
            
            signal_rows.append([
                sig['timestamp'].strftime('%H:%M:%S'),
                f"${sig['price']:.2f}",
                sig['signal'],
                f"{sig['confidence']:.0f}%",
                f"{sig['strength']:.0f}%",
                f"{micro_indicator} {short_indicator} {medium_indicator}",
                f"{sig['momentum']:.2f}%"
            ])
        
        # Determine alignment
        aligned = False
        if (direction == 'LONG' and final_signal.signal in ['STRONG BUY', 'BUY', 'SCALP BUY']) or \
        (direction == 'SHORT' and final_signal.signal in ['STRONG SELL', 'SELL', 'SCALP SELL']):
            aligned = True
        
        # Map signal to generic direction for dashboard
        signal_direction_map = {
            'STRONG BUY': 'BULLISH',
            'BUY': 'BULLISH',
            'SCALP BUY': 'BULLISH',
            'STRONG SELL': 'BEARISH',
            'SELL': 'BEARISH',
            'SCALP SELL': 'BEARISH',
            'NEUTRAL': 'NEUTRAL'
        }
        
        dashboard_direction = signal_direction_map.get(final_signal.signal, 'NEUTRAL')
        
        # Build summary rows
        summary_rows = [
            ['Final Signal', final_signal.signal],
            ['Confidence', f"{final_signal.confidence:.0f}%"],
            ['Strength', f"{final_signal.strength:.0f}%"],
            ['Target Hold', final_signal.target_hold],
            ['Reason', final_signal.reason[:50] + '...' if len(final_signal.reason) > 50 else final_signal.reason]
        ]
        
        # Add detailed timeframe analysis
        if final_signal.micro_trend:
            micro = final_signal.micro_trend
            summary_rows.append([
                '3-min Trend', 
                f"{micro.get('direction', 'N/A').upper()} (Momentum: {micro.get('momentum', 0):.2f}%, Strength: {micro.get('strength', 0):.0f}%)"
            ])
        if final_signal.short_trend:
            short = final_signal.short_trend
            summary_rows.append([
                '5-min Trend', 
                f"{short.get('direction', 'N/A').upper()} (Strength: {short.get('strength', 0):.0f}%, Score: {short.get('score', 0):.3f})"
            ])
        if final_signal.medium_trend:
            medium = final_signal.medium_trend
            summary_rows.append([
                '10-min Trend', 
                f"{medium.get('direction', 'N/A').upper()} (Strength: {medium.get('strength', 0):.0f}%, Score: {medium.get('score', 0):.3f})"
            ])
        
        # Create description
        description = f"{final_signal.signal}: {final_signal.reason}"
        if aligned:
            description += f" | ✅ Aligned with {direction} trade"
        else:
            description += f" | ⚠️ Not aligned with {direction} trade"
        
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
                'trading_signal': final_signal.signal,
                'target_hold': final_signal.target_hold,
                'reason': final_signal.reason,
                'micro_trend': final_signal.micro_trend,
                'short_trend': final_signal.short_trend,
                'medium_trend': final_signal.medium_trend,
                'aligned': aligned,
                'signals_analyzed': len(display_signals),
                'signal_progression': progression_summary
            },
            'display_data': {
                'summary': f"{final_signal.signal} - {final_signal.confidence:.0f}% confidence",
                'description': description,
                'table_data': summary_rows,
                'signal_history': {
                    'headers': ['Time', 'Price', 'Signal', 'Conf%', 'Str%', 'Trends (3/5/10)', 'Mom%'],
                    'rows': signal_rows
                }
            }
        }