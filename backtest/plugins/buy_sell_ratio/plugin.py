# backtest/plugins/bid_ask_ratio/plugin.py
"""
Bid/Ask Ratio Plugin Implementation
Tracks buy/sell pressure ratio over a 30-minute window using SimpleDeltaTracker
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Callable
import pandas as pd
import numpy as np
import logging

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backtest.plugins.base_plugin import BacktestPlugin
from modules.calculations.order_flow.buy_sell_ratio import (
    SimpleDeltaTracker, Trade, MinuteBar
)
from backtest.data.polygon_data_manager import PolygonDataManager

logger = logging.getLogger(__name__)


class BidAskRatioPlugin(BacktestPlugin):
    """Plugin for bid/ask ratio visualization using SimpleDeltaTracker"""
    
    def __init__(self):
        self.data_manager = PolygonDataManager()
        self.config = {
            'window_minutes': 30,  # Rolling window for display
            'lookback_minutes': 35,  # Fetch extra data for processing
            'chart_y_min': -1.25,
            'chart_y_max': 1.25,
            'reference_lines': [0, 0.25, 0.5, 0.75],  # Dotted lines on chart
            'sampling_interval_seconds': 60,  # 1-minute bars
        }
        
    @property
    def name(self) -> str:
        return "Bid/Ask Ratio Tracker"
    
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
        """Run analysis without progress reporting (backward compatibility)"""
        return await self.run_analysis_with_progress(symbol, entry_time, direction, None)
    
    async def run_analysis_with_progress(self, symbol: str, entry_time: datetime, 
                                       direction: str,
                                       progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Run analysis with progress reporting"""
        try:
            # Progress helper
            def report_progress(percentage: int, message: str):
                if progress_callback:
                    progress_callback(percentage, message)
                logger.info(f"Progress: {percentage}% - {message}")
            
            # 1. Initialize tracker
            report_progress(5, "Initializing delta tracker...")
            tracker = SimpleDeltaTracker(window_minutes=self.config['window_minutes'])
            
            # 2. Calculate data fetch window
            start_time = entry_time - timedelta(minutes=self.config['lookback_minutes'])
            end_time = entry_time
            
            # 3. Fetch trade and quote data
            report_progress(10, "Fetching trade data...")
            trades_df = await self.data_manager.load_trades(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                use_cache=True
            )
            
            report_progress(20, "Fetching quote data...")
            quotes_df = await self.data_manager.load_quotes(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                use_cache=True
            )
            
            if trades_df.empty:
                raise ValueError(f"No trade data available for {symbol}")
            
            report_progress(30, f"Processing {len(trades_df):,} trades and {len(quotes_df):,} quotes...")
            
            # 4. Update tracker with quotes first
            quote_count = 0
            for timestamp, quote_data in quotes_df.iterrows():
                tracker.update_quote(
                    symbol=symbol,
                    bid=float(quote_data['bid']),
                    ask=float(quote_data['ask']),
                    timestamp=timestamp.to_pydatetime() if hasattr(timestamp, 'to_pydatetime') else timestamp
                )
                quote_count += 1
                
                if quote_count % 1000 == 0:
                    progress = 30 + int((quote_count / len(quotes_df)) * 20)
                    report_progress(progress, f"Processed {quote_count:,} quotes...")
            
            # 5. Process trades
            report_progress(50, "Processing trades...")
            completed_bars = []
            trades_processed = 0
            
            for i, (timestamp, trade_data) in enumerate(trades_df.iterrows()):
                # Create trade object
                trade = Trade(
                    symbol=symbol,
                    price=float(trade_data['price']),
                    size=int(trade_data['size']),
                    timestamp=timestamp.to_pydatetime() if hasattr(timestamp, 'to_pydatetime') else timestamp,
                    bid=float(trade_data['bid']) if 'bid' in trade_data else None,
                    ask=float(trade_data['ask']) if 'ask' in trade_data else None
                )
                
                # Process trade and check if minute completed
                completed_bar = tracker.process_trade(trade)
                if completed_bar:
                    completed_bars.append(completed_bar)
                
                trades_processed += 1
                
                # Update progress
                if trades_processed % 1000 == 0:
                    progress = 50 + int((trades_processed / len(trades_df)) * 40)
                    report_progress(progress, f"Processed {trades_processed:,} trades, {len(completed_bars)} bars...")
            
            # 6. Get chart data
            report_progress(90, "Generating chart data...")
            chart_data = tracker.get_chart_data(symbol)
            
            # Filter to last 30 minutes before entry
            if chart_data:
                filtered_chart_data = []
                for point in chart_data:
                    point_time = datetime.fromisoformat(point['timestamp'])
                    if point_time <= entry_time and point_time >= (entry_time - timedelta(minutes=30)):
                        filtered_chart_data.append(point)
                chart_data = filtered_chart_data
            
            # 7. Get analysis metrics
            latest_ratio = tracker.get_latest_ratio(symbol)
            summary_stats = tracker.get_summary_stats(symbol)
            
            # 8. Determine signal based on ratio and direction
            signal_assessment = self._assess_signal(latest_ratio, summary_stats, direction)
            
            # 9. Format results
            report_progress(95, "Formatting results...")
            result = self._format_results(
                chart_data, latest_ratio, summary_stats, signal_assessment,
                entry_time, direction, completed_bars
            )
            
            report_progress(100, "Analysis complete")
            return result
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            if progress_callback:
                progress_callback(100, f"Error: {str(e)}")
            raise
    
    def _assess_signal(self, latest_ratio: Optional[float], 
                      summary_stats: Dict[str, Any], 
                      direction: str) -> Dict[str, Any]:
        """Assess signal strength based on ratio and direction"""
        
        if latest_ratio is None:
            return {
                'aligned': False,
                'strength': 0,
                'confidence': 0,
                'signal_direction': 'NEUTRAL'
            }
        
        # Determine signal direction
        if latest_ratio > 0.5:
            signal_direction = 'BULLISH'
        elif latest_ratio < -0.5:
            signal_direction = 'BEARISH'
        elif latest_ratio > 0.25:
            signal_direction = 'LEAN_BULLISH'
        elif latest_ratio < -0.25:
            signal_direction = 'LEAN_BEARISH'
        else:
            signal_direction = 'NEUTRAL'
        
        # Calculate strength (0-100)
        strength = min(100, abs(latest_ratio) * 100)
        
        # Calculate confidence based on consistency
        avg_ratio = summary_stats.get('avg_ratio', 0)
        ratio_consistency = 1 - abs(latest_ratio - avg_ratio) / max(abs(latest_ratio), abs(avg_ratio), 0.1)
        confidence = max(0, min(100, ratio_consistency * 100))
        
        # Check alignment with intended direction
        aligned = False
        if direction == 'LONG' and latest_ratio > 0:
            aligned = True
        elif direction == 'SHORT' and latest_ratio < 0:
            aligned = True
        
        return {
            'aligned': aligned,
            'strength': strength,
            'confidence': confidence,
            'signal_direction': signal_direction
        }
    
    def _format_results(self, chart_data: List[Dict], latest_ratio: Optional[float],
                       summary_stats: Dict[str, Any], signal_assessment: Dict[str, Any],
                       entry_time: datetime, direction: str, 
                       completed_bars: List[MinuteBar]) -> Dict[str, Any]:
        """Format results for display"""
        
        # Build summary display rows
        summary_rows = []
        
        if latest_ratio is not None:
            summary_rows.extend([
                ['Current Ratio', f"{latest_ratio:+.3f}"],
                ['Average Ratio', f"{summary_stats.get('avg_ratio', 0):+.3f}"],
                ['Max Ratio', f"{summary_stats.get('max_ratio', 0):+.3f}"],
                ['Min Ratio', f"{summary_stats.get('min_ratio', 0):+.3f}"],
                ['Total Volume', f"{summary_stats.get('total_volume', 0):,.0f}"],
                ['Minutes Tracked', f"{summary_stats.get('minutes_tracked', 0)}"]
            ])
        
        # Recent bars for detailed view
        recent_bars_data = []
        for bar in completed_bars[-10:]:  # Last 10 bars
            recent_bars_data.append([
                bar.timestamp.strftime('%H:%M:%S'),
                f"{bar.buy_sell_ratio:+.3f}",
                f"{bar.buy_volume:,.0f}",
                f"{bar.sell_volume:,.0f}",
                f"{bar.total_volume:,.0f}"
            ])
        
        # Create description
        description = f"Buy/Sell Ratio: {latest_ratio:+.3f} | "
        if signal_assessment['aligned']:
            description += f"✅ Supports {direction}"
        else:
            description += f"❌ Contradicts {direction}"
        
        if latest_ratio > 0.75:
            description += " | Strong buying pressure"
        elif latest_ratio < -0.75:
            description += " | Strong selling pressure"
        elif abs(latest_ratio) < 0.25:
            description += " | Balanced flow"
        
        # Format chart data for visualization
        chart_config = {
            'type': 'line',
            'y_axis': {
                'min': self.config['chart_y_min'],
                'max': self.config['chart_y_max'],
                'reference_lines': [
                    {'value': line, 'style': 'dotted', 'color': '#666'} 
                    for line in self.config['reference_lines']
                ]
            },
            'x_axis': {
                'type': 'time',
                'range_minutes': 30
            },
            'data': chart_data
        }
        
        return {
            'plugin_name': self.name,
            'timestamp': entry_time,
            'signal': {
                'direction': signal_assessment['signal_direction'],
                'strength': float(signal_assessment['strength']),
                'confidence': float(signal_assessment['confidence'])
            },
            'details': {
                'current_ratio': latest_ratio,
                'average_ratio': summary_stats.get('avg_ratio', 0),
                'max_ratio': summary_stats.get('max_ratio', 0),
                'min_ratio': summary_stats.get('min_ratio', 0),
                'total_volume': summary_stats.get('total_volume', 0),
                'minutes_tracked': summary_stats.get('minutes_tracked', 0),
                'aligned': signal_assessment['aligned']
            },
            'display_data': {
                'summary': f"Bid/Ask Ratio: {latest_ratio:+.3f}",
                'description': description,
                'table_data': summary_rows,
                'recent_bars': {
                    'headers': ['Time', 'Ratio', 'Buy Vol', 'Sell Vol', 'Total Vol'],
                    'rows': recent_bars_data
                },
                'chart': chart_config
            }
        }