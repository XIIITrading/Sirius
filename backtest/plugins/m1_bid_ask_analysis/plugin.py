# backtest/plugins/m1_bid_ask_analysis/plugin.py
"""
M1 Bid/Ask Analysis Plugin Implementation
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
from modules.calculations.volume.m1_bid_ask_analysis import M1VolumeAnalyzer, BidAskVolumeBar
from backtest.data.polygon_data_manager import PolygonDataManager

logger = logging.getLogger(__name__)


class M1BidAskPlugin(BacktestPlugin):
    """Plugin for 1-minute bid/ask volume analysis"""
    
    def __init__(self):
        self.data_manager = PolygonDataManager()
        self.config = {
            'lookback_bars': 14,
            'aggressive_threshold': 0.65,  # 65% threshold for signal
            'lookback_minutes': 20  # Fetch extra to ensure 14 complete bars
        }
        
    @property
    def name(self) -> str:
        return "1-Min Bid/Ask Analysis"
    
    @property
    def version(self) -> str:
        return "2.0.0"  # Updated version
    
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
        Run the complete M1 bid/ask analysis.
        """
        try:
            # 1. Initialize analyzer
            analyzer = M1VolumeAnalyzer(lookback_bars=self.config['lookback_bars'])
            
            # 2. Fetch required data
            trades_df, quotes_df = await self._fetch_data(symbol, entry_time)
            
            if trades_df.empty:
                raise ValueError(f"No trade data available for {symbol}")
            
            # 3. Process trades with bid/ask context
            logger.info(f"Processing {len(trades_df)} trades with {len(quotes_df)} quotes")
            
            for timestamp, trade in trades_df.iterrows():
                # Stop at entry time
                if timestamp >= entry_time:
                    break
                
                # Find closest quote
                bid, ask = self._get_bid_ask_at_time(timestamp, quotes_df)
                
                # Process trade with context
                analyzer.process_trade_with_context(
                    symbol=symbol,
                    timestamp=timestamp,
                    price=float(trade['price']),
                    size=float(trade['size']),
                    bid=bid,
                    ask=ask
                )
            
            # 4. Get final analysis
            analysis_result = analyzer.force_complete_bar(symbol)
            
            if not analysis_result:
                raise ValueError("No analysis results generated")
            
            # 5. Format results
            return self._format_results(analysis_result, analyzer, entry_time, direction)
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise
    
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
    
    def _get_bid_ask_at_time(self, trade_time: datetime, quotes_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        """Get bid/ask prices at trade time"""
        if quotes_df.empty:
            return None, None
            
        # Find quotes at or before trade time
        quotes_before = quotes_df[quotes_df.index <= trade_time]
        
        if quotes_before.empty:
            return None, None
            
        # Get the most recent quote
        latest_quote = quotes_before.iloc[-1]
        
        return latest_quote.get('bid'), latest_quote.get('ask')
    
    def _format_results(self, analysis: Dict[str, Any], analyzer: M1VolumeAnalyzer,
                       entry_time: datetime, direction: str) -> Dict[str, Any]:
        """Format results for display"""
        
        # Get the bars for detailed display
        bars = analyzer.get_current_bars(analysis['symbol'])
        
        # Calculate bar-level statistics
        bull_wins = 0
        bear_wins = 0
        display_rows = []
        
        for bar in bars:
            # Determine winner
            if bar.above_ask_volume > bar.below_bid_volume:
                winner = 'BULLISH'
                bull_wins += 1
            elif bar.below_bid_volume > bar.above_ask_volume:
                winner = 'BEARISH'
                bear_wins += 1
            else:
                winner = 'NEUTRAL'
            
            # Add to display
            display_rows.append([
                bar.timestamp.strftime('%H:%M'),
                f"{bar.above_ask_volume:,.0f}",
                f"{bar.below_bid_volume:,.0f}",
                winner,
                f"{max(bar.aggressive_buy_ratio, bar.aggressive_sell_ratio):.0f}%"
            ])
        
        # Get metrics
        metrics = analysis['metrics']
        
        # Build summary display
        summary_rows = [
            ['Signal', f"{analysis['signal']} ({analysis['strength']:.0f}%)"],
            ['Bull Wins', f"{bull_wins} ({bull_wins/len(bars)*100:.0f}%)"],
            ['Bear Wins', f"{bear_wins} ({bear_wins/len(bars)*100:.0f}%)"],
            ['Above Ask Vol', f"{metrics['total_above_ask_volume']:,.0f}"],
            ['Below Bid Vol', f"{metrics['total_below_bid_volume']:,.0f}"],
            ['Aggressive Buy%', f"{metrics['aggressive_buy_ratio']:.1f}%"],
            ['Avg Spread', f"{metrics['avg_spread_bps']:.1f} bps"],
            ['Price Change', f"{metrics['price_change_pct']:+.2f}%"]
        ]
        
        # Determine confidence based on alignment
        confidence = analysis['strength']
        aligned = False
        
        if (direction == 'LONG' and analysis['signal'] == 'BULLISH') or \
           (direction == 'SHORT' and analysis['signal'] == 'BEARISH'):
            aligned = True
        elif (direction == 'LONG' and analysis['signal'] == 'BEARISH') or \
             (direction == 'SHORT' and analysis['signal'] == 'BULLISH'):
            confidence = max(0, 100 - confidence)  # Invert if opposite
        
        # Create description
        description = analysis['reason']
        if aligned:
            description += f" | ✅ Aligned with {direction}"
        else:
            description += f" | ⚠️ Contradicts {direction}"
        
        return {
            'plugin_name': self.name,
            'timestamp': entry_time,
            'signal': {
                'direction': analysis['signal'],
                'strength': float(analysis['strength']),
                'confidence': float(confidence)
            },
            'details': {
                'bars_analyzed': analysis['bars_analyzed'],
                'total_above_ask': metrics['total_above_ask_volume'],
                'total_below_bid': metrics['total_below_bid_volume'],
                'aggressive_buy_ratio': metrics['aggressive_buy_ratio'],
                'recent_buy_ratio': metrics['recent_buy_ratio'],
                'older_buy_ratio': metrics['older_buy_ratio'],
                'volume_acceleration': metrics['volume_acceleration'],
                'avg_spread_bps': metrics['avg_spread_bps'],
                'price_change_pct': metrics['price_change_pct'],
                'bull_wins': bull_wins,
                'bear_wins': bear_wins,
                'aligned': aligned
            },
            'display_data': {
                'summary': f"{analysis['signal']} - {analysis['strength']:.0f}%",
                'description': description,
                'table_data': summary_rows,
                'bar_details': {
                    'headers': ['Time', 'Above Ask', 'Below Bid', 'Winner', 'Strength'],
                    'rows': display_rows
                },
                'chart_markers': self._get_chart_markers(metrics)
            }
        }
    
    def _get_chart_markers(self, metrics: Dict) -> List[Dict]:
        """Get chart markers for visualization"""
        markers = []
        
        # Aggressive imbalance markers
        if metrics['aggressive_buy_ratio'] > 70:
            markers.append({
                'type': 'aggressive_buy',
                'label': 'AGG BUY',
                'color': 'green'
            })
        elif metrics['aggressive_buy_ratio'] < 30:
            markers.append({
                'type': 'aggressive_sell',
                'label': 'AGG SELL',
                'color': 'red'
            })
            
        return markers