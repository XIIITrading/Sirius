# live_monitor/dashboard/components/chart/data/aggregate_data_handler.py
"""
Handles AM (Aggregates per Minute) WebSocket data using hybrid approach
"""
import logging
from typing import Dict, Optional
from datetime import datetime

from PyQt6.QtCore import QObject, pyqtSignal

from .models import Bar, ChartDataUpdate
from .hybrid_data_manager import HybridDataManager

logger = logging.getLogger(__name__)


class AggregateDataHandler(QObject):
    """
    Enhanced handler using hybrid data approach
    Routes AM events to hybrid manager which handles:
    - Historical data loading
    - Duplicate filtering
    - Higher timeframe aggregation
    """
    
    # Signals
    chart_data_updated = pyqtSignal(dict)  # This was missing!
    
    def __init__(self):
        super().__init__()
        
        # Use hybrid manager
        self.hybrid_manager = HybridDataManager()
        
        # Forward signals
        self.hybrid_manager.chart_update_ready.connect(self.chart_data_updated.emit)
        
        # Track current symbol
        self.current_symbol = None
        
        logger.info("AggregateDataHandler initialized with HybridDataManager")
        
    def process_aggregate(self, data: dict):
        """
        Process incoming AM (minute bar) data
        Routes to hybrid manager for processing
        """
        # Validate it's an aggregate event
        event_type = data.get('event_type', data.get('ev'))
        if event_type not in ['aggregate', 'AM', 'A']:
            return
            
        # Route to hybrid manager
        self.hybrid_manager.process_am_event(data)
        
    def set_symbol(self, symbol: str):
        """Set current symbol and load historical data"""
        if symbol != self.current_symbol:
            logger.info(f"AggregateDataHandler: Symbol changed to {symbol}")
            self.current_symbol = symbol
            self.hybrid_manager.change_symbol(symbol)
    
    def get_chart_data(self, timeframe: str, count: Optional[int] = None) -> dict:
        """
        Get historical chart data for a timeframe
        
        Args:
            timeframe: Timeframe to get data for
            count: Number of bars (None for all)
            
        Returns:
            ChartUpdate dict with full data
        """
        if not self.current_symbol:
            return {
                'symbol': '',
                'timeframe': timeframe,
                'bars': [],
                'is_update': False,
                'latest_bar_complete': False
            }
        
        # Get bars from hybrid manager's aggregator
        bars = self.hybrid_manager.aggregator.get_bars(timeframe, count)
        
        return {
            'symbol': self.current_symbol,
            'timeframe': timeframe,
            'bars': [bar.to_dict() for bar in bars],
            'is_update': False,  # Full refresh
            'latest_bar_complete': False
        }
    
    def clear_data(self):
        """Clear all chart data"""
        self.hybrid_manager.aggregator.clear()
        logger.info("Cleared all chart data")
        
    def get_stats(self) -> dict:
        """Get statistics from hybrid manager"""
        return self.hybrid_manager.get_stats()