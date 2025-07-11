# live_monitor/dashboard/components/chart/data/aggregate_data_handler.py
"""
Handles AM (Aggregates per Minute) WebSocket data using hybrid approach
"""
import logging
from typing import Dict, Optional
from datetime import datetime

from PyQt6.QtCore import QObject, pyqtSignal, QTimer

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
    
    def __init__(self, zone_calculator: Optional['ZoneCalculator'] = None):
        super().__init__()
        
        # Use hybrid manager
        self.hybrid_manager = HybridDataManager()
        
        # Forward signals
        self.hybrid_manager.chart_update_ready.connect(self._on_chart_update)
        
        # Track current symbol
        self.current_symbol = None
        
        # Zone calculator reference
        self.zone_calculator = zone_calculator
        
        # Add flag to track if initial data loaded
        self.initial_data_loaded = False
        
        logger.info("AggregateDataHandler initialized with HybridDataManager")
        
    def _on_chart_update(self, update_dict):
        """Forward chart updates from hybrid manager"""
        logger.debug(f"Forwarding chart update: {update_dict.get('timeframe')} with {len(update_dict.get('bars', []))} bars")
        self.chart_data_updated.emit(update_dict)
        
    def process_aggregate(self, data: dict):
        """Process incoming AM (minute bar) data"""
        # Validate it's an aggregate event
        event_type = data.get('event_type', data.get('ev'))
        if event_type not in ['aggregate', 'AM', 'A']:
            return
            
        # Only process if initial data has been loaded
        if not self.initial_data_loaded:
            logger.debug("Ignoring aggregate - waiting for initial data load")
            return
            
        # Route to hybrid manager
        self.hybrid_manager.process_am_event(data)
        
        # Note: Zone calculator updates are now handled by chart_data_updated signal
        
    def set_symbol(self, symbol: str):
        """Set current symbol and load historical data"""
        if symbol != self.current_symbol:
            logger.info(f"AggregateDataHandler: Symbol changed to {symbol}")
            self.current_symbol = symbol
            self.initial_data_loaded = False
            
            # This should trigger historical data loading
            logger.info(f"AggregateDataHandler: Calling hybrid_manager.change_symbol({symbol})")
            self.hybrid_manager.change_symbol(symbol)
            
            # After symbol change, request initial data update
            logger.info("AggregateDataHandler: Requesting initial chart data after symbol change")
            
            # Use QTimer to delay the initial emit slightly to ensure data is loaded
            QTimer.singleShot(500, self._emit_initial_data)
    
    def _emit_initial_data(self):
        """Emit initial data after REST load completes"""
        if not self.current_symbol:
            return
            
        initial_data = self.get_chart_data('1m')
        if initial_data['bars']:
            logger.info(f"AggregateDataHandler: Emitting initial data with {len(initial_data['bars'])} bars")
            self.initial_data_loaded = True
            self.chart_data_updated.emit(initial_data)
            
            # Update zone calculator with initial data
            if self.zone_calculator:
                self.zone_calculator.update_data(initial_data['bars'])
        else:
            logger.warning("AggregateDataHandler: No initial data available after symbol change")
            # Retry after another delay
            QTimer.singleShot(1000, self._emit_initial_data)
    
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
        
        logger.debug(f"get_chart_data: Retrieved {len(bars)} bars for {timeframe}")
        
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