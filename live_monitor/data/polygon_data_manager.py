# live_monitor/data/polygon_data_manager.py
"""
Main data manager for Polygon WebSocket integration
"""
import json
import logging
from typing import Dict, Optional
from datetime import datetime

from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from live_monitor.data.websocket_client import PolygonWebSocketClient
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
        
        # Aggregate handler will be set by dashboard
        self.aggregate_handler = None
        
        # Add heartbeat tracking
        self.last_data_time = {}  # Track last data time per symbol
        self.no_data_threshold = 30  # seconds
        
        # Create a timer to check data flow periodically
        self.heartbeat_timer = QTimer()
        self.heartbeat_timer.timeout.connect(self.check_data_flow)
        self.heartbeat_timer.start(5000)  # Check every 5 seconds
        
        logger.info("PolygonDataManager initialized with heartbeat monitoring")
        
    def set_aggregate_handler(self, handler):
        """Set the aggregate handler (to avoid circular imports)"""
        self.aggregate_handler = handler
        # Connect signals
        if self.aggregate_handler:
            self.aggregate_handler.chart_data_updated.connect(self.chart_data_updated.emit)
            logger.info("Aggregate handler connected")
    
    def connect(self):
        """Connect to Polygon WebSocket server"""
        logger.info("Connecting to Polygon server...")
        self.ws_client.connect_to_server()
    
    def disconnect(self):
        """Disconnect from server"""
        logger.info("Disconnecting from Polygon server...")
        self.heartbeat_timer.stop()  # Stop heartbeat timer
        self.ws_client.disconnect()
    
    def change_symbol(self, symbol: str):
        """Change the active symbol"""
        if symbol and symbol != self.current_symbol:
            # Store old symbol for cleanup
            old_symbol = self.current_symbol
            
            logger.info(f"Changing symbol from {old_symbol} to {symbol}")
            
            # Clear market state for old symbol
            if old_symbol and old_symbol in self.market_state:
                logger.info(f"Clearing market state for {old_symbol}")
                del self.market_state[old_symbol]
            
            # Clear last data time for old symbol
            if old_symbol and old_symbol in self.last_data_time:
                del self.last_data_time[old_symbol]
            
            # Update current symbol
            self.current_symbol = symbol
            
            # Notify aggregate handler to load historical data
            if self.aggregate_handler:
                self.aggregate_handler.set_symbol(symbol)
                
            # Change WebSocket subscription (this will handle unsubscribe)
            self.ws_client.change_symbol(symbol)
    
    def check_data_flow(self):
        """Check if we're receiving data for current symbol"""
        if not self.current_symbol:
            return
            
        last_time = self.last_data_time.get(self.current_symbol)
        if last_time:
            elapsed = (datetime.now() - last_time).total_seconds()
            if elapsed > self.no_data_threshold:
                logger.warning(f"No data received for {self.current_symbol} in {elapsed:.1f} seconds")
                # Emit a status update
                self.market_data_updated.emit({
                    'symbol': self.current_symbol,
                    'last_update': f"No data for {int(elapsed)}s",
                    'market_state': 'low_activity'
                })
        else:
            # No data received yet for this symbol
            logger.debug(f"No data received yet for {self.current_symbol}")
    
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
        
        # Also check for Polygon format
        if not event_type:
            event_type = data.get('ev')  # Polygon uses 'ev' for event type
        if not symbol:
            symbol = data.get('sym')  # Polygon uses 'sym' for symbol
        
        # Only process data for current symbol - add more explicit logging
        if symbol != self.current_symbol:
            if symbol:  # Only log if we actually have a symbol
                logger.debug(f"Ignoring data for {symbol}, current symbol is {self.current_symbol}")
            return
        
        # Update last data time for heartbeat
        if symbol == self.current_symbol:
            self.last_data_time[symbol] = datetime.now()
        
        logger.info(f"Processing: event_type={event_type}, symbol={symbol}")
        
        if not event_type or not symbol:
            logger.warning(f"Missing event_type or symbol in data: {data.keys()}")
            return
        
        try:
            # Route based on event type
            if event_type in ['trade', 'T']:  # Handle both formats
                self._process_trade(data)
            elif event_type in ['quote', 'Q']:
                self._process_quote(data)
            elif event_type in ['aggregate', 'A', 'AM']:  # Handle AM events!
                logger.info(f"Processing aggregate data: {event_type}")
                self._process_aggregate(data)
                
        except Exception as e:
            logger.error(f"Error processing {event_type} data: {e}", exc_info=True)
    
    def _process_trade(self, trade: TradeData):
        """Process trade data"""
        # Handle both field naming conventions
        symbol = trade.get('symbol', trade.get('sym'))
        price = trade.get('price', trade.get('p'))
        size = trade.get('size', trade.get('s'))
        
        # Update market state
        if symbol not in self.market_state:
            self.market_state[symbol] = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'last_price': price,
                'bid': 0.0,
                'ask': 0.0,
                'spread': 0.0,
                'mid_price': price,
                'volume': 0,
                'last_size': size
            }
        else:
            self.market_state[symbol]['last_price'] = price
            self.market_state[symbol]['last_size'] = size
            self.market_state[symbol]['timestamp'] = datetime.now()
        
        # Emit update
        self._emit_market_update(symbol)
    
    def _process_quote(self, quote: QuoteData):
        """Process quote data"""
        # Handle both field naming conventions
        symbol = quote.get('symbol', quote.get('sym'))
        bid_price = quote.get('bid_price', quote.get('bp'))
        ask_price = quote.get('ask_price', quote.get('ap'))
        bid_size = quote.get('bid_size', quote.get('bs'))
        ask_size = quote.get('ask_size', quote.get('as'))
        
        # Update market state
        if symbol not in self.market_state:
            self.market_state[symbol] = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'last_price': 0.0,
                'bid': bid_price,
                'ask': ask_price,
                'spread': ask_price - bid_price,
                'mid_price': (bid_price + ask_price) / 2,
                'volume': 0,
                'bid_size': bid_size,
                'ask_size': ask_size
            }
        else:
            state = self.market_state[symbol]
            state['bid'] = bid_price
            state['ask'] = ask_price
            state['spread'] = ask_price - bid_price
            state['mid_price'] = (bid_price + ask_price) / 2
            state['bid_size'] = bid_size
            state['ask_size'] = ask_size
            state['timestamp'] = datetime.now()
        
        # Emit update
        self._emit_market_update(symbol)
    
    def _process_aggregate(self, aggregate: AggregateData):
        """Process aggregate/bar data"""
        # Check if this is an AM (minute bar) event
        event_type = aggregate.get('event_type', aggregate.get('ev'))
        
        # Handle both possible formats from polygon WebSocket
        if event_type in ['AM', 'A', 'aggregate']:
            # Route to aggregate handler for chart data if available
            if self.aggregate_handler:
                self.aggregate_handler.process_aggregate(aggregate)
            else:
                logger.warning("No aggregate handler set - cannot process chart data")
        
        # Update volume for market data
        symbol = aggregate.get('symbol', aggregate.get('sym'))
        volume = aggregate.get('volume', aggregate.get('v', 0))
        
        if symbol in self.market_state:
            self.market_state[symbol]['volume'] = volume
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
    
    def get_chart_data(self, timeframe: str, count: Optional[int] = None) -> dict:
        """Get historical chart data"""
        if self.aggregate_handler:
            return self.aggregate_handler.get_chart_data(timeframe, count)
        else:
            return {
                'symbol': '',
                'timeframe': timeframe,
                'bars': [],
                'is_update': False,
                'latest_bar_complete': False
            }
    
    def test_chart_update(self):
        """Test the chart update pipeline with fake data"""
        if not self.current_symbol:
            logger.warning("No symbol set for test")
            return
            
        test_data = {
            'event_type': 'aggregate',
            'ev': 'AM',
            'symbol': self.current_symbol,
            'sym': self.current_symbol,
            's': int(datetime.now().timestamp() * 1000),  # start time in ms
            'o': 150.0,  # open
            'h': 151.0,  # high
            'l': 149.0,  # low
            'c': 150.5,  # close
            'v': 100000,  # volume
            'vw': 150.25, # vwap
            'n': 500     # number of trades
        }
        
        logger.info(f"Sending test aggregate data for {self.current_symbol}")
        self._on_data_received(test_data)  # Send through normal pipeline