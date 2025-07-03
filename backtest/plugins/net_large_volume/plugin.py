"""
Net Large Volume Backtest Plugin
Pass-through adapter for historical data processing
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime

from modules.calculations.order_flow.net_large_volume import NetLargeVolumeTracker
from modules.calculations.order_flow.large_orders import LargeOrderDetector


class NetLargeVolumePlugin:
    """Backtest plugin for net large order volume tracking"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize plugin with configuration.
        
        Config options:
        - large_order_config: Dict passed to LargeOrderDetector
        - volume_tracker_config: Dict passed to NetLargeVolumeTracker
        """
        # Initialize large order detector
        large_order_config = config.get('large_order_config', {})
        self.large_order_detector = LargeOrderDetector(**large_order_config)
        
        # Initialize volume tracker
        volume_tracker_config = config.get('volume_tracker_config', {})
        self.volume_tracker = NetLargeVolumeTracker(**volume_tracker_config)
        
        # Track symbols
        self.symbols = set()
        
    def process_historical_data(self, 
                              trades_df: pd.DataFrame,
                              quotes_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Process historical trade and quote data from polygon_data_manager.
        
        Expected DataFrame columns:
        - trades_df: price, size, conditions, exchange (index: timestamp)
        - quotes_df: bid, ask, bid_size, ask_size, bid_exchange, ask_exchange (index: timestamp)
        
        Returns DataFrames ready for charting.
        """
        results = {}
        
        # Get unique symbols from trades
        if 'symbol' in trades_df.columns:
            symbols = trades_df['symbol'].unique()
        else:
            # If no symbol column, assume single symbol
            symbols = ['UNKNOWN']
            trades_df['symbol'] = 'UNKNOWN'
            quotes_df['symbol'] = 'UNKNOWN'
        
        # Process each symbol
        for symbol in symbols:
            self.symbols.add(symbol)
            
            # Filter data for this symbol
            symbol_trades = trades_df[trades_df['symbol'] == symbol] if 'symbol' in trades_df.columns else trades_df
            symbol_quotes = quotes_df[quotes_df['symbol'] == symbol] if 'symbol' in quotes_df.columns else quotes_df
            
            # Process chronologically
            chart_data = []
            
            # First update all quotes
            for timestamp, quote_row in symbol_quotes.iterrows():
                self.large_order_detector.update_quote(
                    symbol=symbol,
                    bid=quote_row['bid'],
                    ask=quote_row['ask'],
                    timestamp=timestamp
                )
            
            # Then process trades
            for timestamp, trade_row in symbol_trades.iterrows():
                # Create Trade object compatible with large_orders.py
                from modules.calculations.order_flow.buy_sell_ratio import Trade
                
                # Get the most recent quote for this trade
                quotes_before = symbol_quotes[symbol_quotes.index <= timestamp]
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
                
                # Check for large order
                large_order = self.large_order_detector.process_trade(trade)
                
                # If large order detected and completed, update volume
                if large_order and large_order.is_impact_complete:
                    volume_point = self.volume_tracker.process_large_order(large_order)
                    
                    if volume_point:
                        chart_data.append({
                            'timestamp': volume_point.timestamp,
                            'net_volume': volume_point.cumulative_net_volume,
                            'buy_volume': volume_point.cumulative_buy_volume,
                            'sell_volume': volume_point.cumulative_sell_volume,
                            'rate_of_change': volume_point.rate_of_change,
                            'session_high': volume_point.session_high,
                            'session_low': volume_point.session_low
                        })
            
            # Convert to DataFrame
            if chart_data:
                results[symbol] = pd.DataFrame(chart_data)
                results[symbol].set_index('timestamp', inplace=True)
            else:
                results[symbol] = pd.DataFrame()
        
        return results
    
    def get_signals(self, symbol: str) -> pd.DataFrame:
        """
        Get trading signals based on net volume.
        Returns DataFrame with signal timestamps and strengths.
        """
        # Get trend analysis
        trend = self.volume_tracker.get_trend_analysis(symbol)
        
        signals = []
        
        # Generate signals based on trend
        if trend['trend'] == 'ACCUMULATION' and trend['strength'] > 0.7:
            current_stats = self.volume_tracker.get_current_stats(symbol)
            signals.append({
                'timestamp': datetime.now(),
                'signal': 'BUY',
                'strength': trend['strength'],
                'reason': f"Strong accumulation: {trend['slope']:.0f} volume/min"
            })
        elif trend['trend'] == 'DISTRIBUTION' and trend['strength'] > 0.7:
            signals.append({
                'timestamp': datetime.now(),
                'signal': 'SELL', 
                'strength': trend['strength'],
                'reason': f"Strong distribution: {trend['slope']:.0f} volume/min"
            })
        
        return pd.DataFrame(signals)
    
    def get_summary_stats(self) -> Dict[str, Dict]:
        """Get summary statistics for all symbols"""
        stats = {}
        
        for symbol in self.symbols:
            stats[symbol] = {
                'current': self.volume_tracker.get_current_stats(symbol),
                'trend': self.volume_tracker.get_trend_analysis(symbol),
                'large_orders': self.large_order_detector.get_summary_stats(symbol)
            }
        
        return stats