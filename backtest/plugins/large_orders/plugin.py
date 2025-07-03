"""
Large Orders Grid Backtest Plugin
Displays successful large order trades with impact magnitude
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import logging

from modules.calculations.order_flow.large_orders import LargeOrderDetector, ImpactStatus

logger = logging.getLogger(__name__)


class Plugin:
    """Large Orders Grid plugin for backtest system"""
    
    # Plugin metadata
    name = "Large Orders Grid"
    version = "1.0.0"
    description = "Displays successful large order trades with impact analysis"
    author = "Trading System"
    
    def __init__(self):
        """Initialize plugin"""
        self.config = {
            'stats_window_minutes': 15,
            'impact_window_seconds': 1,
            'min_trades_for_std': 100,
            'ratio_threshold': 1.5,
            'stdev_threshold': 1.25,
            'max_orders_display': 30  # Maximum successful orders to display
        }
        
        # Initialize detector
        self.detector = LargeOrderDetector(
            stats_window_minutes=self.config['stats_window_minutes'],
            impact_window_seconds=self.config['impact_window_seconds'],
            min_trades_for_std=self.config['min_trades_for_std'],
            ratio_threshold=self.config['ratio_threshold'],
            stdev_threshold=self.config['stdev_threshold']
        )
        
        self.symbol = None
        self.entry_time = None
        
    async def run(self, symbol: str, entry_time: datetime, direction: str, 
                  data_manager: Any, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run the Large Orders Grid analysis.
        
        Args:
            symbol: Stock symbol
            entry_time: Entry time for analysis
            direction: Trade direction (LONG/SHORT)
            data_manager: Data manager for fetching data
            progress_callback: Optional callback for progress updates
            
        Returns:
            Analysis results dictionary with grid data
        """
        try:
            self.symbol = symbol
            self.entry_time = entry_time
            
            if progress_callback:
                progress_callback(0, "Fetching trade and quote data...")
            
            # Fetch data window (30 minutes before entry)
            end_time = entry_time
            start_time = entry_time - timedelta(minutes=30)
            
            # Fetch trades
            trades_df = await data_manager.load_trades(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time
            )
            
            if progress_callback:
                progress_callback(25, f"Loaded {len(trades_df)} trades")
            
            # Fetch quotes
            quotes_df = await data_manager.load_quotes(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time
            )
            
            if progress_callback:
                progress_callback(50, f"Loaded {len(quotes_df)} quotes")
            
            # Process data
            if progress_callback:
                progress_callback(60, "Detecting large orders...")
                
            # Process all the data
            self._process_data(trades_df, quotes_df, symbol)
            
            if progress_callback:
                progress_callback(80, "Filtering successful orders...")
            
            # Get successful large orders
            successful_orders = self._get_successful_orders(symbol)
            
            # Generate signal based on successful order analysis
            signal = self._generate_signal(successful_orders, direction)
            
            # Get the complete display package
            display_data = self._prepare_complete_display_package(successful_orders)
            
            if progress_callback:
                progress_callback(100, "Analysis complete")
            
            return {
                'plugin_name': self.name,
                'timestamp': entry_time,
                'symbol': symbol,
                'direction': direction,
                'signal': signal,
                'display_data': display_data,
                'large_orders_count': len(successful_orders)
            }
            
        except Exception as e:
            logger.error(f"Error in Large Orders Grid plugin: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'plugin_name': self.name,
                'timestamp': entry_time,
                'symbol': symbol,
                'error': str(e),
                'signal': {'direction': 'ERROR', 'strength': 0, 'confidence': 0}
            }
    
    def _process_data(self, trades_df: pd.DataFrame, quotes_df: pd.DataFrame, symbol: str) -> None:
        """
        Process historical data through the detector.
        """
        # Add symbol column if not present
        if 'symbol' not in trades_df.columns:
            trades_df['symbol'] = symbol
        if 'symbol' not in quotes_df.columns:
            quotes_df['symbol'] = symbol
        
        # First update all quotes
        for timestamp, quote_row in quotes_df.iterrows():
            self.detector.update_quote(
                symbol=symbol,
                bid=quote_row['bid'],
                ask=quote_row['ask'],
                timestamp=timestamp
            )
        
        # Then process trades
        large_order_count = 0
        for timestamp, trade_row in trades_df.iterrows():
            # Create Trade object
            from modules.calculations.order_flow.buy_sell_ratio import Trade
            
            # Get the most recent quote at this time
            quotes_before = quotes_df[quotes_df.index <= timestamp]
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
            
            # Process trade - detector handles everything
            large_order = self.detector.process_trade(trade)
            if large_order:
                large_order_count += 1
        
        logger.info(f"Processed {len(trades_df)} trades, detected {large_order_count} large orders")
    
    def _get_successful_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """Get successful large orders for display"""
        if symbol not in self.detector.completed_orders:
            return []
        
        # Get all completed orders
        all_orders = list(self.detector.completed_orders[symbol])
        
        # Filter for successful orders only
        successful_orders = [
            order for order in all_orders 
            if order.impact_status == ImpactStatus.SUCCESS
        ]
        
        # Sort by timestamp descending (most recent first)
        successful_orders.sort(key=lambda o: o.timestamp, reverse=True)
        
        # Limit to configured maximum
        successful_orders = successful_orders[:self.config['max_orders_display']]
        
        # Convert to grid data format
        grid_data = []
        for order in successful_orders:
            grid_data.append({
                'timestamp': order.timestamp,
                'price': order.price,
                'size': order.size,
                'side': order.side.value,
                'impact_magnitude': order.impact_magnitude if order.impact_magnitude else 0.0,
                'size_vs_avg': order.size_vs_avg,
                'size_vs_stdev': order.size_vs_stdev,
                'vwap_1s': order.vwap_1s,
                'volume_1s': order.volume_1s,
                'trade_count_1s': order.trade_count_1s
            })
        
        return grid_data
    
    def _generate_signal(self, successful_orders: List[Dict[str, Any]], direction: str) -> Dict[str, Any]:
        """Generate trading signal from successful large orders"""
        if not successful_orders:
            return {
                'direction': 'NEUTRAL',
                'strength': 0,
                'confidence': 0,
                'reason': 'No successful large orders detected'
            }
        
        # Analyze the successful orders
        buy_count = sum(1 for o in successful_orders if o['side'] == 'BUY')
        sell_count = sum(1 for o in successful_orders if o['side'] == 'SELL')
        
        # Calculate average impact magnitude
        avg_impact = np.mean([abs(o['impact_magnitude']) for o in successful_orders])
        
        # Determine signal direction based on order balance
        if buy_count > sell_count * 1.5:
            signal_direction = 'BULLISH'
            strength = min(100, (buy_count / max(1, sell_count)) * 20)
        elif sell_count > buy_count * 1.5:
            signal_direction = 'BEARISH'
            strength = min(100, (sell_count / max(1, buy_count)) * 20)
        else:
            signal_direction = 'NEUTRAL'
            strength = 30
        
        # Confidence based on average impact and count
        confidence = min(100, len(successful_orders) * 3 + avg_impact * 20)
        
        # Check alignment
        aligned = (signal_direction == 'BULLISH' and direction == 'LONG') or \
                 (signal_direction == 'BEARISH' and direction == 'SHORT')
        
        # Build reason
        reason = f"{len(successful_orders)} successful large orders: {buy_count} buys, {sell_count} sells. "
        reason += f"Avg impact: {avg_impact:.2f} spreads"
        
        return {
            'direction': signal_direction,
            'strength': strength,
            'confidence': confidence,
            'reason': reason,
            'aligned': aligned
        }
    
    def _prepare_complete_display_package(self, successful_orders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare complete display data package with grid configuration.
        """
        # Calculate summary statistics
        total_orders = len(successful_orders)
        
        if total_orders > 0:
            buy_orders = [o for o in successful_orders if o['side'] == 'BUY']
            sell_orders = [o for o in successful_orders if o['side'] == 'SELL']
            
            avg_impact = np.mean([abs(o['impact_magnitude']) for o in successful_orders])
            max_impact = max([abs(o['impact_magnitude']) for o in successful_orders])
            
            total_volume = sum(o['size'] for o in successful_orders)
            avg_size = np.mean([o['size'] for o in successful_orders])
            
            summary = f"{total_orders} successful large orders ({len(buy_orders)} buys, {len(sell_orders)} sells)"
            
            description_lines = [
                f"Total Volume: {total_volume:,} shares",
                f"Average Size: {avg_size:,.0f} shares",
                f"Average Impact: {avg_impact:.2f} spreads",
                f"Maximum Impact: {max_impact:.2f} spreads"
            ]
        else:
            summary = "No successful large orders detected"
            description_lines = ["No large orders with successful price impact were found in the analysis window"]
        
        description = '\n'.join(description_lines)
        
        # Prepare table data for summary display
        summary_stats = self.detector.get_summary_stats(self.symbol)
        detection_stats = summary_stats.get('large_order_stats', {})
        
        table_data = [
            ("Total Large Orders", str(detection_stats.get('total_detected', 0))),
            ("Successful Orders", str(total_orders)),
            ("Success Rate", f"{detection_stats.get('impact_success_rate', 0):.1%}"),
            ("Buy Orders", str(len(buy_orders) if total_orders > 0 else 0)),
            ("Sell Orders", str(len(sell_orders) if total_orders > 0 else 0)),
            ("Avg Impact Magnitude", f"{avg_impact:.2f}" if total_orders > 0 else "N/A"),
            ("Max Impact Magnitude", f"{max_impact:.2f}" if total_orders > 0 else "N/A")
        ]
        
        # Return complete display package
        return {
            'summary': summary,
            'description': description,
            'table_data': table_data,
            'grid_widget': {
                'module': 'backtest.plugins.large_orders.grid',
                'type': 'LargeOrdersGrid',
                'data': successful_orders,
                'config': {
                    'title': 'Successful Large Orders',
                    'columns': [
                        {'name': 'Timestamp', 'field': 'timestamp', 'width': 150},
                        {'name': 'Price', 'field': 'price', 'width': 100},
                        {'name': 'Size', 'field': 'size', 'width': 100},
                        {'name': 'Side', 'field': 'side', 'width': 80},
                        {'name': 'Impact', 'field': 'impact_magnitude', 'width': 100}
                    ]
                }
            }
        }