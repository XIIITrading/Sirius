# live_monitor/data/polygon_data_manager.py
"""
Main data manager for Polygon WebSocket integration
All timestamps and operations in UTC
"""
import json
import logging
import time
from typing import Dict, Optional, List
from datetime import datetime, timezone

from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from live_monitor.data.websocket_client import PolygonWebSocketClient
from live_monitor.data.rest_client import PolygonRESTClient
from live_monitor.data.hist_request import HistoricalFetchCoordinator
from .models import (
    TradeData, QuoteData, AggregateData,
    MarketDataUpdate, TickerCalculationData,
    EntrySignal, ExitSignal
)

logger = logging.getLogger(__name__)


class PolygonDataManager(QObject):
    """
    Central manager for Polygon data flow
    
    Handles WebSocket connection, REST API calls, data transformation, 
    and signal emission for the Live Monitor Dashboard.
    All timestamps in UTC.
    """
    
    # UI Update Signals
    market_data_updated = pyqtSignal(dict)      # For TickerCalculations
    chart_data_updated = pyqtSignal(dict)       # For bar data accumulation
    calculation_updated = pyqtSignal(dict)      # For calculated metrics
    entry_signal_generated = pyqtSignal(dict)   # For PointCallEntry
    exit_signal_generated = pyqtSignal(dict)    # For PointCallExit
    
    # Historical data ready signals (forwarded from coordinator)
    ema_data_ready = pyqtSignal(dict)           # {'M1': df, 'M5': df, 'M15': df}
    structure_data_ready = pyqtSignal(dict)     # {'M1': df, 'M5': df, 'M15': df}
    trend_data_ready = pyqtSignal(dict)         # {'M1': df, 'M5': df, 'M15': df}
    zone_data_ready = pyqtSignal(dict)          # {'HVN': df, 'OrderBlocks': df}
    
    # Connection Status
    connection_status_changed = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)
    
    # Historical fetch progress
    historical_fetch_progress = pyqtSignal(dict)  # Progress updates
    
    def __init__(self, server_url: Optional[str] = None):
        super().__init__()
        
        # Initialize WebSocket client
        self.ws_client = PolygonWebSocketClient(
            server_url or "ws://localhost:8200/ws/{client_id}",
            client_id="live_monitor_dashboard"
        )
        
        # Initialize REST client
        self.rest_client = PolygonRESTClient(
            base_url="http://localhost:8200"
        )
        
        # Initialize historical fetch coordinator
        self.hist_coordinator = HistoricalFetchCoordinator(self.rest_client)
        
        # Connect WebSocket signals
        self.ws_client.connected.connect(self._on_connected)
        self.ws_client.disconnected.connect(self._on_disconnected)
        self.ws_client.data_received.connect(self._on_data_received)
        self.ws_client.error_occurred.connect(self.error_occurred.emit)
        self.ws_client.connection_status.connect(self.connection_status_changed.emit)
        
        # Connect REST client signals
        self.rest_client.error_occurred.connect(self.error_occurred.emit)
        
        # Connect historical coordinator signals
        self.hist_coordinator.all_fetches_completed.connect(self._on_historical_ready)
        self.hist_coordinator.fetch_progress.connect(self._on_fetch_progress)
        self.hist_coordinator.fetch_error.connect(self._on_fetch_error)
        
        # Forward data ready signals
        self.hist_coordinator.ema_data_ready.connect(self.ema_data_ready.emit)
        self.hist_coordinator.structure_data_ready.connect(self.structure_data_ready.emit)
        self.hist_coordinator.trend_data_ready.connect(self.trend_data_ready.emit)
        self.hist_coordinator.zone_data_ready.connect(self.zone_data_ready.emit)
        
        # Current state
        self.current_symbol: Optional[str] = None
        self.market_state: Dict[str, MarketDataUpdate] = {}
        
        # Bar data accumulation
        self.accumulated_bars: List[Dict] = []
        self.max_bars = 2000  # Keep last 2000 bars
        
        # Add heartbeat tracking
        self.last_data_time = {}  # Track last data time per symbol
        self.no_data_threshold = 30  # seconds
        
        # Track if we've fetched initial historical data
        self.initial_data_fetched = {}
        
        # Track historical data availability
        self.historical_data_available = {}
        
        # Create a timer to check data flow periodically
        self.heartbeat_timer = QTimer()
        self.heartbeat_timer.timeout.connect(self.check_data_flow)
        self.heartbeat_timer.start(5000)  # Check every 5 seconds
        
        logger.info("PolygonDataManager initialized with REST, WebSocket, and Historical coordinator")
    
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
            
            # Clear accumulated bars
            self.accumulated_bars.clear()
            
            # Reset initial data fetch flag
            self.initial_data_fetched[symbol] = False
            self.historical_data_available[symbol] = False
            
            # Update current symbol
            self.current_symbol = symbol
                
            # Change WebSocket subscription (this will handle unsubscribe)
            self.ws_client.change_symbol(symbol)
            
            # Trigger coordinated historical fetch instead of single fetch
            # This will fetch data for all calculation types in parallel
            QTimer.singleShot(1000, lambda: self._start_historical_fetch(symbol))
    
    def _start_historical_fetch(self, symbol: str):
        """Start the coordinated historical fetch"""
        if symbol != self.current_symbol:
            logger.debug(f"Skipping historical fetch for {symbol}, current is {self.current_symbol}")
            return
        
        logger.info(f"Starting coordinated historical fetch for {symbol}")
        
        # Start the fetch - priority mode will fetch high priority first
        self.hist_coordinator.fetch_all_for_symbol(symbol, priority_mode=True)
    
    def _on_fetch_progress(self, progress: dict):
        """Handle fetch progress updates"""
        logger.info(f"Historical fetch progress: {progress['completed']}/{progress['total']} "
                   f"({progress['percentage']:.0f}%)")
        
        # Forward progress to UI
        self.historical_fetch_progress.emit(progress)
    
    def _on_fetch_error(self, error_data: dict):
        """Handle fetch errors"""
        symbol = error_data.get('symbol')
        errors = error_data.get('errors', [])
        
        if symbol == self.current_symbol:
            error_msg = f"Historical fetch errors: {'; '.join(errors)}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
    
    def _on_historical_ready(self, symbol: str):
        """Handle when all historical data is ready"""
        if symbol != self.current_symbol:
            logger.debug(f"Ignoring historical ready for {symbol}, current is {self.current_symbol}")
            return
        
        logger.info(f"All historical data ready for {symbol}")
        
        # Mark that we have historical data
        self.initial_data_fetched[symbol] = True
        self.historical_data_available[symbol] = True
        
        # Update last data time to prevent immediate refetch
        self.last_data_time[symbol] = datetime.now(timezone.utc)
        
        # Get M1 data from coordinator for accumulated bars
        m1_data = self._get_m1_data_for_accumulation()
        if m1_data is not None:
            self._initialize_accumulated_bars(m1_data)
        
        # Emit a general chart update to notify UI
        if self.accumulated_bars:
            self.chart_data_updated.emit({
                'symbol': symbol,
                'bars': self.accumulated_bars,
                'is_update': False,  # Full refresh
                'source': 'historical'
            })
            
            # Also update market data with latest bar
            latest_bar = self.accumulated_bars[-1]
            self._update_market_data_from_bar(symbol, latest_bar)
    
    def _get_m1_data_for_accumulation(self):
        """Get 1-minute data from coordinator for bar accumulation"""
        # Try to get M1 data from any of the fetchers that use 1-minute data
        fetcher_priority = ['M1_MarketStructure', 'HVN', 'OrderBlocks', 'M1_EMA']
        
        for fetcher_name in fetcher_priority:
            fetcher = self.hist_coordinator.get_fetcher(fetcher_name)
            if fetcher and fetcher.cache is not None:
                logger.info(f"Using {fetcher_name} data for bar accumulation")
                return fetcher.cache
        
        logger.warning("No 1-minute data available for bar accumulation")
        return None
    
    def _initialize_accumulated_bars(self, df):
        """Initialize accumulated bars from historical data DataFrame"""
        # Convert DataFrame to list of bar dicts
        bars = []
        for timestamp, row in df.iterrows():
            bar = {
                'timestamp': timestamp,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
                'vwap': row.get('vwap', 0),
                'trades': row.get('trades', 0)
            }
            bars.append(bar)
        
        # Limit to max bars
        if len(bars) > self.max_bars:
            bars = bars[-self.max_bars:]
        
        self.accumulated_bars = bars
        logger.info(f"Initialized {len(bars)} accumulated bars from historical data")
    
    def fetch_historical_bars(self, symbol: str, bars_needed: int = 200):
        """
        Legacy method - now just triggers the coordinator
        Kept for backward compatibility
        """
        logger.info(f"Legacy fetch_historical_bars called for {symbol}")
        self._start_historical_fetch(symbol)
    
    def _update_market_data_from_bar(self, symbol: str, bar: Dict):
        """Update market data display from a bar"""
        ticker_data: TickerCalculationData = {
            'last_price': bar['close'],
            'bid': bar['close'] - 0.01,  # Estimate
            'ask': bar['close'] + 0.01,  # Estimate
            'spread': 0.02,
            'mid_price': bar['close'],
            'volume': bar['volume'],
            'change': None,
            'change_percent': None,
            'day_high': bar['high'],
            'day_low': bar['low'],
            'day_open': bar['open'],
            'last_update': datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
            'market_state': self.rest_client.get_market_session()
        }
        
        self.market_data_updated.emit(ticker_data)
    
    def check_data_flow(self):
        """Check if we're receiving data for current symbol"""
        if not self.current_symbol:
            return
            
        current_time = datetime.now(timezone.utc)
        last_time = self.last_data_time.get(self.current_symbol)
        
        if last_time:
            elapsed = (current_time - last_time).total_seconds()
            
            if elapsed > self.no_data_threshold:
                market_session = self.rest_client.get_market_session()
                
                if market_session == 'regular':
                    # Market is open but no data
                    logger.warning(f"No data received for {self.current_symbol} in {elapsed:.1f} seconds during market hours")
                    self.error_occurred.emit(f"No data for {elapsed:.0f}s")
                else:
                    # Market closed or pre/after hours
                    logger.info(f"Market session: {market_session}, no real-time data expected")
                    
                    # Fetch historical data if we haven't already
                    if not self.initial_data_fetched.get(self.current_symbol, False):
                        logger.info("Fetching initial historical data...")
                        self._start_historical_fetch(self.current_symbol)
                    elif elapsed > 300:  # Refresh every 5 minutes when market closed
                        logger.info("Refreshing historical data...")
                        self._start_historical_fetch(self.current_symbol)
                        self.last_data_time[self.current_symbol] = current_time
        else:
            # No data received yet for this symbol
            logger.debug(f"No data received yet for {self.current_symbol}")
            if not self.initial_data_fetched.get(self.current_symbol, False):
                self._start_historical_fetch(self.current_symbol)
    
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
        
        # Only process data for current symbol
        if symbol != self.current_symbol:
            if symbol:  # Only log if we actually have a symbol
                logger.debug(f"Ignoring data for {symbol}, current symbol is {self.current_symbol}")
            return
        
        # Update last data time for heartbeat
        if symbol == self.current_symbol:
            self.last_data_time[symbol] = datetime.now(timezone.utc)
        
        logger.debug(f"Processing: event_type={event_type}, symbol={symbol}")
        
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
                logger.debug(f"Processing aggregate data: {event_type}")
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
                'timestamp': datetime.now(timezone.utc),
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
            self.market_state[symbol]['timestamp'] = datetime.now(timezone.utc)
        
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
                'timestamp': datetime.now(timezone.utc),
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
            state['timestamp'] = datetime.now(timezone.utc)
        
        # Emit update
        self._emit_market_update(symbol)
    
    def _process_aggregate(self, aggregate: AggregateData):
        """Process aggregate/bar data"""
        # Check if this is an AM (minute bar) event
        event_type = aggregate.get('event_type', aggregate.get('ev'))
        
        # Handle both possible formats from polygon WebSocket
        if event_type in ['AM', 'A', 'aggregate']:
            # Convert to bar format for accumulation
            symbol = aggregate.get('symbol', aggregate.get('sym'))
            timestamp = aggregate.get('s', aggregate.get('timestamp', 0))
            
            # Create bar data with UTC timestamp
            bar_data = {
                'timestamp': datetime.fromtimestamp(timestamp / 1000.0, tz=timezone.utc),
                'open': aggregate.get('o', aggregate.get('open', 0)),
                'high': aggregate.get('h', aggregate.get('high', 0)),
                'low': aggregate.get('l', aggregate.get('low', 0)),
                'close': aggregate.get('c', aggregate.get('close', 0)),
                'volume': aggregate.get('v', aggregate.get('volume', 0)),
                'trades': aggregate.get('n', aggregate.get('transactions', 0))
            }
            
            # Accumulate bar
            self.accumulated_bars.append(bar_data)
            
            # Trim to max bars
            if len(self.accumulated_bars) > self.max_bars:
                self.accumulated_bars = self.accumulated_bars[-self.max_bars:]
            
            # Emit chart data update with accumulated bars
            chart_update = {
                'symbol': symbol,
                'timeframe': '1m',
                'bars': [bar_data],  # Send just the new bar
                'is_update': True,
                'latest_bar_complete': True,
                'source': 'websocket'
            }
            
            self.chart_data_updated.emit(chart_update)
            logger.debug(f"Emitted chart update for {symbol} with new bar")
        
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
                'last_update': datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                'market_state': self.rest_client.get_market_session()
            }
            
            self.market_data_updated.emit(ticker_data)
    
    def get_accumulated_bars(self) -> List[Dict]:
        """Get all accumulated bars"""
        return self.accumulated_bars.copy()
    
    def is_historical_data_ready(self, symbol: str) -> bool:
        """Check if historical data is available for a symbol"""
        return self.historical_data_available.get(symbol, False)
    
    def get_historical_fetch_status(self) -> Dict:
        """Get the current status of historical data fetching"""
        return self.hist_coordinator.get_fetch_status()
    
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
            's': int(datetime.now(timezone.utc).timestamp() * 1000),  # start time in ms
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