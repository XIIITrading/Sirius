# backtest/plugins/m15_statistical_trend/plugin.py
"""
15-Minute Statistical Trend Plugin Implementation
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
from modules.calculations.trend.statistical_trend_15min import StatisticalTrend15Min, MarketRegimeSignal
from backtest.data.polygon_data_manager import PolygonDataManager

logger = logging.getLogger(__name__)


class M15StatisticalTrendPlugin(BacktestPlugin):
    """Plugin for 15-minute statistical trend analysis and market regime detection"""
    
    def __init__(self):
        # Initialize data manager
        self.data_manager = PolygonDataManager(
            extend_window_bars=5  # Small extension for 15-min bars
        )
        self.config = {
            'lookback_minutes': 375,  # 25 bars * 15 minutes = 6.25 hours
            'analysis_bars': 10,      # Show last 10 bars in history
            'short_lookback': 3,      # 45-min trend
            'medium_lookback': 5,     # 75-min trend
            'long_lookback': 10       # 150-min trend
        }
        
    @property
    def name(self) -> str:
        return "15-Min Statistical Trend"
    
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
        Run the complete 15-minute statistical trend analysis.
        """
        try:
            # 1. Initialize trend calculator (REACTIVE version)
            trend_calc = StatisticalTrend15Min(
                short_lookback=self.config['short_lookback'],
                medium_lookback=self.config['medium_lookback'],
                long_lookback=self.config['long_lookback']
            )
            
            # 2. Fetch required data (375 minutes / 25 bars)
            bars_df = await self._fetch_data(symbol, entry_time)
            
            if bars_df.empty:
                raise ValueError(f"No bar data available for {symbol}")
            
            # 3. Feed historical data to calculator
            signal_history = []
            final_signal = None
            
            logger.info(f"Processing {len(bars_df)} 15-min bars for regime analysis")
            
            for timestamp, bar in bars_df.iterrows():
                # Stop processing after entry time
                if timestamp > entry_time:
                    break
                
                # Update trend calculator with OHLCV data
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
                    signal_data = self._extract_signal_data(signal, timestamp)
                    signal_history.append(signal_data)
                    
                    # Update final signal if at or before entry time
                    if timestamp <= entry_time:
                        final_signal = signal
            
            if not final_signal:
                raise ValueError("No regime signal generated at entry time")
            
            # 4. Format results
            return self._format_results(final_signal, signal_history, entry_time, direction)
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise
    
    def _extract_signal_data(self, signal: MarketRegimeSignal, timestamp: datetime) -> Dict:
        """Extract display data from MarketRegimeSignal"""
        # Map regime to position signal
        position_signal_map = {
            'BULL MARKET': 'TREND UP',
            'BEAR MARKET': 'TREND DOWN',
            'RANGE BOUND': 'RANGING',
            'TRANSITIONING': 'RANGING'
        }
        
        # Map daily bias to market bias
        market_bias_map = {
            'LONG ONLY': 'BULLISH',
            'LONG BIAS': 'BULLISH',
            'SHORT ONLY': 'BEARISH',
            'SHORT BIAS': 'BEARISH',
            'BOTH WAYS': 'NEUTRAL',
            'STAY OUT': 'NEUTRAL'
        }
        
        # Map volatility state to market state
        market_state_map = {
            'EXTREME': 'VOLATILE',
            'HIGH': 'VOLATILE',
            'NORMAL': 'CONSOLIDATING',
            'LOW': 'CONSOLIDATING'
        }
        
        # Get trend directions for display
        trend_arrows = []
        for trend_data in [signal.short_trend, signal.medium_trend, signal.long_trend]:
            if not trend_data:
                trend_arrows.append('→')
            elif trend_data.get('direction') == 'bullish':
                trend_arrows.append('↑')
            elif trend_data.get('direction') == 'bearish':
                trend_arrows.append('↓')
            else:
                trend_arrows.append('→')
        
        return {
            'timestamp': timestamp,
            'price': signal.price,
            'position_signal': position_signal_map.get(signal.regime, 'RANGING'),
            'market_bias': market_bias_map.get(signal.daily_bias, 'NEUTRAL'),
            'confidence': signal.confidence,
            'strength': signal.strength,
            'trend_arrows': ' '.join(trend_arrows),
            'market_state': market_state_map.get(signal.volatility_state, 'UNKNOWN'),
            'regime': signal.regime,
            'daily_bias': signal.daily_bias,
            'trading_notes': signal.trading_notes,
            'short_trend': signal.short_trend,
            'medium_trend': signal.medium_trend,
            'long_trend': signal.long_trend
        }
    
    async def _fetch_data(self, symbol: str, entry_time: datetime) -> pd.DataFrame:
        """Fetch required bar data"""
        # Calculate start time for 375 minutes (25 bars)
        start_time = entry_time - timedelta(minutes=self.config['lookback_minutes'])
        
        logger.info(f"Fetching {symbol} 15-min bars from {start_time} to {entry_time}")
        
        bars = await self.data_manager.load_bars(
            symbol=symbol,
            start_time=start_time,
            end_time=entry_time,
            timeframe='15min',
            use_cache=True
        )
        
        logger.info(f"Fetched {len(bars) if bars is not None else 0} 15-min bars")
        
        return bars if bars is not None else pd.DataFrame()
    
    def _format_results(self, final_signal: MarketRegimeSignal, signal_history: List[Dict],
                       entry_time: datetime, direction: str) -> Dict[str, Any]:
        """Format results for display"""
        
        # Get last 10 signals for display
        display_signals = signal_history[-self.config['analysis_bars']:] if len(signal_history) >= self.config['analysis_bars'] else signal_history
        
        # Build signal history table
        signal_rows = []
        for sig in display_signals:
            signal_rows.append([
                sig['timestamp'].strftime('%H:%M:%S'),
                f"${sig['price']:.2f}",
                sig['position_signal'],
                sig['market_bias'],
                f"{sig['confidence']:.0f}%",
                f"{sig['strength']:.0f}%",
                sig['trend_arrows'],
                sig['market_state']
            ])
        
        # Extract final signal data
        final_data = self._extract_signal_data(final_signal, entry_time)
        
        # Determine alignment
        aligned = False
        if (direction == 'LONG' and final_data['market_bias'] == 'BULLISH') or \
           (direction == 'SHORT' and final_data['market_bias'] == 'BEARISH'):
            aligned = True
        
        # Map to dashboard direction
        dashboard_direction = final_data['market_bias']
        
        # Extract recommendation from trading notes
        recommendation = final_signal.trading_notes
        # Try to extract just the main recommendation if it's too long
        if '|' in recommendation:
            parts = recommendation.split('|')
            recommendation = parts[0].strip()
        
        # Build summary rows
        summary_rows = [
            ['Position Signal', final_data['position_signal']],
            ['Market Bias', final_data['market_bias']],
            ['Confidence', f"{final_data['confidence']:.0f}%"],
            ['Strength', f"{final_data['strength']:.0f}%"],
            ['Market State', final_data['market_state']],
            ['Recommendation', recommendation[:60] + '...' if len(recommendation) > 60 else recommendation]
        ]
        
        # Add timeframe details
        if final_signal.short_trend:
            short = final_signal.short_trend
            summary_rows.append([
                '45-min Trend',
                f"{short.get('direction', 'N/A').upper()} (Strength: {short.get('strength', 0):.0f}%, VWAP: {short.get('vwap_position', 0):+.2f}%)"
            ])
        
        if final_signal.medium_trend:
            medium = final_signal.medium_trend
            summary_rows.append([
                '75-min Trend',
                f"{medium.get('direction', 'N/A').upper()} (Strength: {medium.get('strength', 0):.0f}%, Score: {medium.get('score', 0):.3f})"
            ])
        
        if final_signal.long_trend:
            long = final_signal.long_trend
            summary_rows.append([
                '150-min Trend',
                f"{long.get('direction', 'N/A').upper()} (Strength: {long.get('strength', 0):.0f}%, Score: {long.get('score', 0):.3f})"
            ])
        
        # Create description
        description = f"{final_data['position_signal']}: {final_data['regime']} - {final_signal.trading_notes[:100]}"
        if aligned:
            description += f" | ✅ Aligned with {direction} trade"
        else:
            description += f" | ⚠️ Not aligned with {direction} trade"
        
        # Signal progression summary
        regime_counts = {}
        bias_counts = {}
        for sig in display_signals:
            regime_counts[sig['position_signal']] = regime_counts.get(sig['position_signal'], 0) + 1
            bias_counts[sig['market_bias']] = bias_counts.get(sig['market_bias'], 0) + 1
        
        progression_summary = f"Regime progression: {', '.join([f'{k}={v}' for k, v in regime_counts.items()])}"
        
        return {
            'plugin_name': self.name,
            'timestamp': entry_time,
            'signal': {
                'direction': dashboard_direction,
                'strength': float(final_signal.strength),
                'confidence': float(final_signal.confidence)
            },
            'details': {
                'regime': final_signal.regime,
                'daily_bias': final_signal.daily_bias,
                'position_signal': final_data['position_signal'],
                'market_bias': final_data['market_bias'],
                'market_state': final_data['market_state'],
                'volatility_state': final_signal.volatility_state,
                'trading_notes': final_signal.trading_notes,
                'key_levels': final_signal.key_levels,
                'short_trend': final_signal.short_trend,
                'medium_trend': final_signal.medium_trend,
                'long_trend': final_signal.long_trend,
                'aligned': aligned,
                'signals_analyzed': len(display_signals),
                'regime_progression': progression_summary
            },
            'display_data': {
                'summary': f"{final_data['position_signal']} - {final_data['market_bias']} - {final_signal.confidence:.0f}% confidence",
                'description': description,
                'table_data': summary_rows,
                'signal_history': {
                    'headers': ['Time', 'Price', 'Signal', 'Bias', 'Conf%', 'Str%', 'Trends (45/75/150)', 'State'],
                    'rows': signal_rows
                }
            }
        }