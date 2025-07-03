"""
Impact Success Backtest Plugin
Tracks large order volume pressure (buy vs sell imbalance)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta 
import logging

from modules.calculations.order_flow.impact_success import ImpactSuccessTracker

logger = logging.getLogger(__name__)


class Plugin:
    """Impact Success plugin for backtest system"""
    
    # Plugin metadata
    name = "Impact Success"
    version = "2.0.0"
    description = "Tracks large order volume pressure"
    author = "Trading System"
    
    def __init__(self):
        """Initialize plugin"""
        self.config = {
            'stats_window_minutes': 15,
            'min_trades_for_std': 100,
            'ratio_threshold': 1.5,
            'stdev_threshold': 1.25,
            'pressure_window_seconds': 60,  # 1-minute bars
            'history_points': 1800  # 30 minutes of data
        }
        
        # Initialize consolidated tracker
        self.tracker = ImpactSuccessTracker(
            stats_window_minutes=self.config['stats_window_minutes'],
            min_trades_for_std=self.config['min_trades_for_std'],
            ratio_threshold=self.config['ratio_threshold'],
            stdev_threshold=self.config['stdev_threshold'],
            pressure_window_seconds=self.config['pressure_window_seconds'],
            history_points=self.config['history_points']
        )
        
        self.symbol = None
        self.entry_time = None
        
    async def run(self, symbol: str, entry_time: datetime, direction: str, 
                  data_manager: Any, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run the Impact Success analysis.
        
        Args:
            symbol: Stock symbol
            entry_time: Entry time for analysis
            direction: Trade direction (LONG/SHORT)
            data_manager: Data manager for fetching data
            progress_callback: Optional callback for progress updates
            
        Returns:
            Analysis results dictionary with complete data for display
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
                progress_callback(60, "Processing large orders...")
                
            # Process all the data
            self._process_data(trades_df, quotes_df, symbol)
            
            if progress_callback:
                progress_callback(80, "Calculating pressure metrics...")
            
            # Get current stats from the tracker
            stats = self.tracker.get_current_stats(symbol)
            
            # Generate signal based on pressure
            signal = self._generate_signal(stats, direction)
            
            # Get the complete display package
            display_data = self._prepare_complete_display_package(stats)
            
            if progress_callback:
                progress_callback(100, "Analysis complete")
            
            return {
                'plugin_name': self.name,
                'timestamp': entry_time,
                'symbol': symbol,
                'direction': direction,
                'signal': signal,
                'display_data': display_data,
                'stats': stats
            }
            
        except Exception as e:
            logger.error(f"Error in Impact Success plugin: {e}")
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
        Process historical data through the tracker.
        """
        # Add symbol column if not present
        if 'symbol' not in trades_df.columns:
            trades_df['symbol'] = symbol
        if 'symbol' not in quotes_df.columns:
            quotes_df['symbol'] = symbol
        
        # First update all quotes
        for timestamp, quote_row in quotes_df.iterrows():
            self.tracker.update_quote(
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
            
            # Process trade - tracker handles everything
            large_order = self.tracker.process_trade(trade)
            if large_order:
                large_order_count += 1
        
        logger.info(f"Processed {len(trades_df)} trades, detected {large_order_count} large orders")
    
    def _generate_signal(self, stats: Dict[str, Any], direction: str) -> Dict[str, Any]:
        """Generate trading signal from pressure stats"""
        net_pressure = stats.get('net_pressure', 0)
        total_volume = stats.get('total_buy_volume', 0) + stats.get('total_sell_volume', 0)
        
        # No signal if no large orders
        if total_volume == 0:
            return {
                'direction': 'NEUTRAL',
                'strength': 0,
                'confidence': 0,
                'reason': 'No large orders detected'
            }
        
        # Calculate signal strength based on pressure imbalance
        pressure_ratio = abs(net_pressure) / total_volume if total_volume > 0 else 0
        strength = min(100, pressure_ratio * 100)
        
        # Confidence based on volume
        confidence = min(100, (total_volume / 10000) * 100)  # Scale by expected volume
        
        # Determine signal direction
        pressure_direction = stats.get('pressure_direction', 'NEUTRAL')
        
        # Check alignment with intended direction
        aligned = False
        if direction == 'LONG' and pressure_direction == 'BULLISH':
            aligned = True
        elif direction == 'SHORT' and pressure_direction == 'BEARISH':
            aligned = True
        
        # Build reason
        if pressure_direction == 'BULLISH':
            reason = f"Buy pressure dominates ({stats.get('total_buy_volume', 0):,} vs {stats.get('total_sell_volume', 0):,})"
        elif pressure_direction == 'BEARISH':
            reason = f"Sell pressure dominates ({stats.get('total_sell_volume', 0):,} vs {stats.get('total_buy_volume', 0):,})"
        else:
            reason = "Balanced large order flow"
        
        if aligned:
            reason += f" - Supports {direction}"
        elif pressure_direction != 'NEUTRAL':
            reason += f" - Contradicts {direction}"
        
        return {
            'direction': pressure_direction,
            'strength': strength,
            'confidence': confidence,
            'reason': reason,
            'aligned': aligned
        }
    
    def _prepare_complete_display_package(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare complete display data package with pressure chart data.
        """
        # Get chart data from tracker
        chart_data = self.tracker.get_chart_data(self.symbol)
        
        # Format description text
        description_lines = [
            f"Buy Volume: {stats.get('total_buy_volume', 0):,} ({stats.get('buy_order_count', 0)} orders)",
            f"Sell Volume: {stats.get('total_sell_volume', 0):,} ({stats.get('sell_order_count', 0)} orders)",
            f"Net Pressure: {stats.get('net_pressure', 0):+,}",
            f"Status: {stats.get('interpretation', 'No data')}"
        ]
        description = '\n'.join(description_lines)
        
        # Prepare table data for display
        table_data = [
            ("Total Buy Volume", f"{stats.get('total_buy_volume', 0):,}"),
            ("Total Sell Volume", f"{stats.get('total_sell_volume', 0):,}"),
            ("Net Pressure", f"{stats.get('net_pressure', 0):+,}"),
            ("Buy Order Count", str(stats.get('buy_order_count', 0))),
            ("Sell Order Count", str(stats.get('sell_order_count', 0))),
            ("Cumulative Pressure", f"{stats.get('current_cumulative', 0):+,}"),
            ("Pressure Direction", stats.get('pressure_direction', 'NEUTRAL'))
        ]
        
        # Return complete display package
        return {
            'summary': f"Large Order Pressure: {stats.get('net_pressure', 0):+,}",
            'description': description,
            'table_data': table_data,
            'chart_widget': {
                'module': 'backtest.plugins.impact_success.chart',
                'type': 'ImpactSuccessChart',
                'data': chart_data,
                'entry_time': self.entry_time.isoformat() if self.entry_time else None
            }
        }


# Create plugin instance for use by the runner
_plugin = Plugin()

# Expose the run method for the plugin runner
async def run_analysis(symbol: str, entry_time: datetime, direction: str) -> Dict[str, Any]:
    """
    Entry point for the plugin runner.
    Note: This expects the data_manager to be injected by the runner.
    """
    # The plugin runner should inject the data_manager
    from backtest.data.polygon_data_manager import PolygonDataManager
    data_manager = PolygonDataManager()
    
    return await _plugin.run(symbol, entry_time, direction, data_manager)

# Expose run method with progress support
async def run_analysis_with_progress(symbol: str, entry_time: datetime, direction: str,
                                   progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Entry point with progress callback support"""
    from backtest.data.polygon_data_manager import PolygonDataManager
    data_manager = PolygonDataManager()
    
    return await _plugin.run(symbol, entry_time, direction, data_manager, progress_callback)

# Add data_manager support for plugin runner
async def run_analysis_with_data_manager(symbol: str, entry_time: datetime, direction: str, 
                                       data_manager: Any, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Entry point with provided data_manager - preferred method"""
    return await _plugin.run(symbol, entry_time, direction, data_manager, progress_callback)