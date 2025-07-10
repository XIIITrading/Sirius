# live_monitor/data/polygon_data_manager.py
"""
Main data manager for Polygon WebSocket integration
"""
import logging
from typing import Dict, Optional
from datetime import datetime

from PyQt6.QtCore import QObject, pyqtSignal

from .websocket_client import PolygonWebSocketClient
from .models import (
    TradeData, QuoteData, AggregateData,
    MarketDataUpdate, TickerCalculationData,
    ChartUpdate, EntrySignal, ExitSignal
)

logger = logging.getLogger(__name__)


class PolygonDataManager(QObject):
    """
    Central manager for Polygon data flow
    
    Handles WebSocket connection, data transformation, and signal emission
    for the Live Monitor Dashboard.
    """
    
    # UI Update Signals
    market_data_updated = pyqtSignal(dict)      # For TickerCalculations
    chart_data_updated = pyqtSignal(dict)       # For ChartWidget  
    calculation_updated = pyqtSignal(dict)      # For calculated metrics
    entry_signal_generated = pyqtSignal(dict)   # For PointCallEntry
    exit_signal_generated = pyqtSignal(dict)    # For PointCallExit
    
    # Connection Status
    connection_status_changed = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, server_url: Optional[str] = None):
        super().__init__()
        
        # Initialize WebSocket client
        self.ws_client = PolygonWebSocketClient(
            server_url or "ws://localhost:8200/ws/{client_id}",
            client_id="live_monitor_dashboard"
        )
        
        # Connect WebSocket signals
        self.ws_client.connected.connect(self._on_connected)
        self.ws_client.disconnected.connect(self._on_disconnected)
        self.ws_client.data_received.connect(self._on_data_received)
        self.ws_client.error_occurred.connect(self.error_occurred.emit)
        self.ws_client.connection_status.connect(self.connection_status_changed.emit)
        
        # Current state
        self.current_symbol: Optional[str] = None
        self.market_state: Dict[str, MarketDataUpdate] = {}
        
        # TODO: Initialize transformers here in next phase
        
    def connect(self):
        """Connect to Polygon WebSocket server"""
        logger.info("Connecting to Polygon server...")
        self.ws_client.connect_to_server()
    
    def disconnect(self):
        """Disconnect from server"""
        logger.info("Disconnecting from Polygon server...")
        self.ws_client.disconnect()
    
    def change_symbol(self, symbol: str):
        """Change the active symbol"""
        if symbol and symbol != self.current_symbol:
            logger.info(f"Changing symbol from {self.current_symbol} to {symbol}")
            self.current_symbol = symbol
            self.ws_client.change_symbol(symbol)
    
    def _on_connected(self):
        """Handle connection established"""
        logger.info("Connected to Polygon server")
        # Resubscribe to current symbol if set
        if self.current_symbol:
            self.ws_client.change_symbol(self.current_symbol)
    
    def _on_disconnected(self):
        """Handle disconnection"""
        logger.warning("Disconnected from Polygon server")
    
    def _on_data_received(self, data: dict):
        """
        Process incoming WebSocket data
        
        Routes data to appropriate transformers and emits signals
        """
        event_type = data.get('event_type')
        symbol = data.get('symbol')
        
        if not event_type or not symbol:
            return
        
        # Only process data for current symbol
        if symbol != self.current_symbol:
            return
        
        try:
            # Route based on event type
            if event_type == 'trade':
                self._process_trade(data)
            elif event_type == 'quote':
                self._process_quote(data)
            elif event_type == 'aggregate':
                self._process_aggregate(data)
                
        except Exception as e:
            logger.error(f"Error processing {event_type} data: {e}")
    
    def _process_trade(self, trade: TradeData):
        """Process trade data"""
        # Update market state
        symbol = trade['symbol']
        
        # Create/update market data
        if symbol not in self.market_state:
            self.market_state[symbol] = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'last_price': trade['price'],
                'bid': 0.0,
                'ask': 0.0,
                'spread': 0.0,
                'mid_price': trade['price'],
                'volume': 0,
                'last_size': trade['size']
            }
        else:
            self.market_state[symbol]['last_price'] = trade['price']
            self.market_state[symbol]['last_size'] = trade['size']
            self.market_state[symbol]['timestamp'] = datetime.now()
        
        # Emit update
        self._emit_market_update(symbol)
    
    def _process_quote(self, quote: QuoteData):
        """Process quote data"""
        symbol = quote['symbol']
        
        # Update market state
        if symbol not in self.market_state:
            self.market_state[symbol] = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'last_price': 0.0,
                'bid': quote['bid_price'],
                'ask': quote['ask_price'],
                'spread': quote['ask_price'] - quote['bid_price'],
                'mid_price': (quote['bid_price'] + quote['ask_price']) / 2,
                'volume': 0,
                'bid_size': quote['bid_size'],
                'ask_size': quote['ask_size']
            }
        else:
            state = self.market_state[symbol]
            state['bid'] = quote['bid_price']
            state['ask'] = quote['ask_price']
            state['spread'] = quote['ask_price'] - quote['bid_price']
            state['mid_price'] = (quote['bid_price'] + quote['ask_price']) / 2
            state['bid_size'] = quote['bid_size']
            state['ask_size'] = quote['ask_size']
            state['timestamp'] = datetime.now()
        
        # Emit update
        self._emit_market_update(symbol)
    
    def _process_aggregate(self, aggregate: AggregateData):
        """Process aggregate/bar data"""
        # TODO: Implement chart data aggregation
        # For now, just update volume
        symbol = aggregate['symbol']
        if symbol in self.market_state:
            self.market_state[symbol]['volume'] = aggregate['volume']
            self._emit_market_update(symbol)
    
    def _emit_market_update(self, symbol: str):
        """Emit market data update for UI"""
        if symbol in self.market_state:
            # Convert to TickerCalculationData format
            state = self.market_state[symbol]
            
            ticker_data: TickerCalculationData = {
                'last_price': state['last_price'],
                'bid': state['bid'],
                'ask': state['ask'],
                'spread': state['spread'],
                'mid_price': state['mid_price'],
                'volume': state['volume'],
                'change': None,  # TODO: Calculate from daily open
                'change_percent': None,
                'day_high': None,  # TODO: Track daily high
                'day_low': None,   # TODO: Track daily low
                'day_open': None,  # TODO: Get from morning data
                'last_update': datetime.now().strftime("%H:%M:%S"),
                'market_state': 'open'  # TODO: Determine from time
            }
            
            self.market_data_updated.emit(ticker_data)