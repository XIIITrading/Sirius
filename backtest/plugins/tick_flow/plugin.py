# backtest/plugins/tick_flow_analysis/plugin.py
"""
Tick Flow Analysis Plugin Implementation
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
from modules.calculations.volume.tick_flow import TickFlowAnalyzer, VolumeSignal
from backtest.data.polygon_data_manager import PolygonDataManager

logger = logging.getLogger(__name__)


class TickFlowPlugin(BacktestPlugin):
    """Plugin for tick-by-tick order flow analysis"""
    
    def __init__(self):
        self.data_manager = PolygonDataManager()
        self.config = {
            'buffer_size': 200,  # Number of trades to analyze
            'large_trade_multiplier': 3.0,  # Multiple of avg size for large trades
            'momentum_threshold': 60.0,  # % threshold for bull/bear signal
            'min_trades_required': 50,  # Minimum trades for valid signal
            'lookback_minutes': 5  # Minutes of data to fetch before entry
        }
        
    @property
    def name(self) -> str:
        return "Tick Flow Analysis"
    
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
        Run the complete tick flow analysis.
        """
        try:
            # 1. Initialize analyzer
            analyzer = TickFlowAnalyzer(
                buffer_size=self.config['buffer_size'],
                large_trade_multiplier=self.config['large_trade_multiplier'],
                momentum_threshold=self.config['momentum_threshold'],
                min_trades_required=self.config['min_trades_required']
            )
            
            # 2. Fetch required data
            trades_df = await self._fetch_trades(symbol, entry_time)
            
            if trades_df.empty:
                raise ValueError(f"No trade data available for {symbol}")
            
            # 3. Process trades up to entry time
            logger.info(f"Processing {len(trades_df)} trades for {symbol}")
            
            last_signal = None
            signals_generated = []
            
            for timestamp, trade in trades_df.iterrows():
                # Stop at entry time
                if timestamp >= entry_time:
                    break
                
                # Convert trade data to expected format
                trade_data = {
                    'timestamp': int(timestamp.timestamp() * 1000),  # Convert to milliseconds
                    'price': float(trade['price']),
                    'size': float(trade['size']),
                    'conditions': trade.get('conditions', [])
                }
                
                # Process trade
                signal = analyzer.process_trade(symbol, trade_data)
                
                if signal:
                    last_signal = signal
                    signals_generated.append(signal)
            
            # 4. Get final analysis
            if not last_signal:
                # Force analysis even if below min trades
                last_signal = analyzer.get_current_analysis(symbol)
                
            if not last_signal:
                raise ValueError("Insufficient trades for analysis")
            
            # 5. Format results
            return self._format_results(last_signal, analyzer, entry_time, direction, signals_generated)
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise
    
    async def _fetch_trades(self, symbol: str, entry_time: datetime) -> pd.DataFrame:
        """Fetch required trade data"""
        # Calculate time range
        start_time = entry_time - timedelta(minutes=self.config['lookback_minutes'])
        
        logger.info(f"Fetching {symbol} trades from {start_time} to {entry_time}")
        
        # Fetch trade data
        trades = await self.data_manager.load_trades(
            symbol=symbol,
            start_time=start_time,
            end_time=entry_time,
            use_cache=True
        )
        
        return trades
    
    def _format_results(self, signal: VolumeSignal, analyzer: TickFlowAnalyzer,
                       entry_time: datetime, direction: str, 
                       signals_generated: List[VolumeSignal]) -> Dict[str, Any]:
        """Format results for display"""
        
        # Get statistics
        stats = analyzer.get_statistics()
        metrics = signal.metrics
        
        # Build summary display
        summary_rows = [
            ['Signal', f"{signal.signal} ({signal.strength:.0f}%)"],
            ['Buy Volume %', f"{metrics['buy_volume_pct']:.1f}%"],
            ['Buy/Sell Trades', f"{metrics['buy_trades']}/{metrics['sell_trades']}"],
            ['Buy/Sell Volume', f"{metrics['buy_volume']:,.0f}/{metrics['sell_volume']:,.0f}"],
            ['Large Buys/Sells', f"{metrics['large_buy_trades']}/{metrics['large_sell_trades']}"],
            ['Momentum Score', f"{metrics['momentum_score']:+.1f}"],
            ['Trade Rate', f"{metrics['trade_rate']:.1f} trades/sec"],
            ['Avg Trade Size', f"{metrics['avg_trade_size']:,.0f}"]
        ]
        
        # Calculate signal history if available
        signal_history = []
        if signals_generated:
            # Get last 10 signals
            for sig in signals_generated[-10:]:
                signal_history.append([
                    sig.timestamp.strftime('%H:%M:%S'),
                    sig.signal,
                    f"{sig.strength:.0f}%",
                    f"{sig.metrics['buy_volume_pct']:.0f}%"
                ])
        
        # Determine confidence based on alignment
        confidence = signal.strength
        aligned = False
        
        if (direction == 'LONG' and signal.signal == 'BULLISH') or \
           (direction == 'SHORT' and signal.signal == 'BEARISH'):
            aligned = True
        elif (direction == 'LONG' and signal.signal == 'BEARISH') or \
             (direction == 'SHORT' and signal.signal == 'BULLISH'):
            confidence = max(0, 100 - confidence)  # Invert if opposite
        
        # Create description
        description = signal.reason
        if aligned:
            description += f" | ✅ Aligned with {direction}"
        else:
            description += f" | ⚠️ Contradicts {direction}"
        
        # Determine emoji for signal
        if signal.signal == 'BULLISH':
            emoji = '🟢'
        elif signal.signal == 'BEARISH':
            emoji = '🔴'
        else:
            emoji = '⚪'
        
        return {
            'plugin_name': self.name,
            'timestamp': entry_time,
            'signal': {
                'direction': signal.signal,
                'strength': float(signal.strength),
                'confidence': float(confidence)
            },
            'details': {
                'trades_analyzed': metrics['total_trades'],
                'buy_volume_pct': metrics['buy_volume_pct'],
                'momentum_score': metrics['momentum_score'],
                'large_buy_trades': metrics['large_buy_trades'],
                'large_sell_trades': metrics['large_sell_trades'],
                'trade_rate': metrics['trade_rate'],
                'price_trend': metrics['price_trend'],
                'trades_processed': stats['trades_processed'],
                'signals_generated': stats['signals_generated'],
                'aligned': aligned
            },
            'display_data': {
                'summary': f"{emoji} {signal.signal} - {signal.strength:.0f}%",
                'description': description,
                'table_data': summary_rows,
                'signal_history': {
                    'headers': ['Time', 'Signal', 'Strength', 'Buy %'],
                    'rows': signal_history
                } if signal_history else None,
                'chart_markers': self._get_chart_markers(metrics)
            }
        }
    
    def _get_chart_markers(self, metrics: Dict) -> List[Dict]:
        """Get chart markers for visualization"""
        markers = []
        
        # Buy/sell pressure markers
        if metrics['buy_volume_pct'] > 70:
            markers.append({
                'type': 'heavy_buying',
                'label': 'HEAVY BUY',
                'color': 'green'
            })
        elif metrics['buy_volume_pct'] < 30:
            markers.append({
                'type': 'heavy_selling',
                'label': 'HEAVY SELL',
                'color': 'red'
            })
        
        # Large trade markers
        if metrics['large_buy_trades'] > metrics['large_sell_trades'] * 2:
            markers.append({
                'type': 'large_buys',
                'label': 'LARGE BUYS',
                'color': 'darkgreen'
            })
        elif metrics['large_sell_trades'] > metrics['large_buy_trades'] * 2:
            markers.append({
                'type': 'large_sells',
                'label': 'LARGE SELLS',
                'color': 'darkred'
            })
            
        return markers