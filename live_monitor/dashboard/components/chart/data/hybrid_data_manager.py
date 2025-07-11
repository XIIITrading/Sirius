# live_monitor/dashboard/components/chart/data/hybrid_data_manager.py
"""
Hybrid data manager that combines historical REST data with real-time WebSocket
"""
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from collections import deque
import requests

from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from .models import Bar, TimeframeType
from .chart_data_aggregator import ChartDataAggregator

logger = logging.getLogger(__name__)


class HybridDataManager(QObject):
    """
    Manages chart data from multiple sources:
    1. Historical data from REST API
    2. Real-time AM events from WebSocket
    3. Trade aggregation as fallback
    """
    
    # Signal emitted when chart needs updating
    chart_update_ready = pyqtSignal(dict)
    
    def __init__(self, rest_url: str = "http://localhost:8200/api/v1/bars"):
        super().__init__()
        self.rest_url = rest_url
        self.current_symbol = None
        
        # Data aggregator for building higher timeframes
        self.aggregator = ChartDataAggregator(max_bars=500)
        
        # Track duplicate AM events
        self.last_am_timestamp = {}
        
        # Stats
        self.stats = {
            'historical_bars': 0,
            'am_bars': 0,
            'trade_bars': 0,
            'duplicates_filtered': 0
        }
        
        # Add minute tracking for complete bar detection
        self.current_minute = {}  # Track current minute being built per symbol
        self.last_complete_minute = {}  # Track last completed minute per symbol
        
        logger.info("HybridDataManager initialized with minute boundary detection")
        
    def change_symbol(self, symbol: str):
        """Change to new symbol - load historical data"""
        if symbol == self.current_symbol:
            return
            
        logger.info(f"[HYBRID] Changing symbol to {symbol}")
        self.current_symbol = symbol
        
        # Clear existing data
        self.aggregator.clear()
        self.last_am_timestamp.clear()
        
        # Clear minute tracking
        self.current_minute.clear()
        self.last_complete_minute.clear()
        
        # Load historical data
        self.load_historical_data(symbol)
        
    def load_historical_data(self, symbol: str, bars_to_load: int = 390):
        """Load historical minute bars from REST API"""
        try:
            logger.info(f"[HYBRID] Loading {bars_to_load} bars of historical data for {symbol}")
            
            response = requests.post(
                self.rest_url,
                json={
                    "symbol": symbol,
                    "timeframe": "1min",
                    "limit": bars_to_load  # Now configurable
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                bars_data = data.get('data', [])
                
                # Reset stats
                self.stats['historical_bars'] = 0
                
                # Process historical bars
                for bar_data in bars_data:
                    bar = Bar(
                        timestamp=datetime.fromisoformat(bar_data['timestamp'].replace('Z', '+00:00')),
                        open=bar_data['open'],
                        high=bar_data['high'],
                        low=bar_data['low'],
                        close=bar_data['close'],
                        volume=bar_data['volume'],
                        trades=bar_data.get('transactions', 0)
                    )
                    
                    # Add to aggregator
                    updates = self.aggregator.add_minute_bar(bar)
                    self.stats['historical_bars'] += 1
                
                logger.info(f"[HYBRID] Loaded {len(bars_data)} historical bars")
                
                # Emit full update for all timeframes
                self.emit_full_update()
                
        except Exception as e:
            logger.error(f"[HYBRID] Failed to load historical data: {e}")
            
    def process_am_event(self, data: dict):
        """Process AM event from WebSocket - only emit on complete bars"""
        if not self.current_symbol:
            return
            
        # Check for duplicate
        timestamp = data.get('s', data.get('timestamp', 0))
        symbol = data.get('symbol', data.get('sym'))
        
        if symbol != self.current_symbol:
            return
            
        # Filter duplicates
        last_ts = self.last_am_timestamp.get(symbol, 0)
        if timestamp == last_ts:
            self.stats['duplicates_filtered'] += 1
            return
            
        self.last_am_timestamp[symbol] = timestamp
        
        # Create bar from AM data
        try:
            # Convert timestamp to datetime
            bar_time = datetime.fromtimestamp(timestamp / 1000.0)
            bar_minute = bar_time.replace(second=0, microsecond=0)
            
            # Check if this is a new minute
            current_minute = self.current_minute.get(symbol)
            if current_minute is None:
                # First bar for this symbol
                self.current_minute[symbol] = bar_minute
                logger.info(f"First bar for {symbol} at minute {bar_minute}")
            elif bar_minute > current_minute:
                # New minute detected - the previous minute is complete
                logger.info(f"New minute detected for {symbol}: {bar_minute} > {current_minute}")
                
                # Create bar object
                bar = Bar(
                    timestamp=current_minute,  # Use the completed minute
                    open=data.get('o', data.get('open')),
                    high=data.get('h', data.get('high')),
                    low=data.get('l', data.get('low')),
                    close=data.get('c', data.get('close')),
                    volume=data.get('v', data.get('volume')),
                    trades=data.get('n', data.get('transactions')) or 0
                )
                
                # Add to aggregator
                updates = self.aggregator.add_minute_bar(bar)
                self.stats['am_bars'] += 1
                
                # Emit updates for completed bar
                for timeframe, update in updates.items():
                    update.symbol = symbol
                    self.emit_update(update, timeframe)
                    
                # Update tracking
                self.last_complete_minute[symbol] = current_minute
                self.current_minute[symbol] = bar_minute
                
                logger.info(f"Emitted update for completed minute: {current_minute}")
            else:
                # Same minute update - don't emit
                logger.debug(f"Same minute update for {symbol}: {bar_minute}")
                
        except Exception as e:
            logger.error(f"[HYBRID] Error processing AM event: {e}")
            
    def process_trade_bar(self, bar_data: dict):
        """Process a completed bar from trade aggregation"""
        # Similar to process_am_event but for trade-generated bars
        self.stats['trade_bars'] += 1
        # Implementation similar to above
        
    def emit_update(self, update, timeframe: str):
        """Emit chart update signal"""
        update_dict = {
            'symbol': self.current_symbol,
            'timeframe': timeframe,
            'bars': [bar.to_dict() for bar in update.bars],
            'is_update': update.is_update,
            'latest_bar_complete': update.latest_bar_complete
        }
        
        self.chart_update_ready.emit(update_dict)
        
    def emit_full_update(self):
        """Emit full update for all timeframes"""
        for timeframe in ['1m', '5m', '15m', '30m', '1h']:
            bars = self.aggregator.get_bars(timeframe)
            if bars:
                update_dict = {
                    'symbol': self.current_symbol,
                    'timeframe': timeframe,
                    'bars': [bar.to_dict() for bar in bars],
                    'is_update': False,
                    'latest_bar_complete': False
                }
                self.chart_update_ready.emit(update_dict)
                
    def get_stats(self) -> dict:
        """Get current statistics"""
        return self.stats.copy()