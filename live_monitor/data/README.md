# Live Monitor Data Module

## Overview

The `live_monitor/data/` module is a real-time market data management system designed for intraday trading applications. It provides WebSocket connectivity to Polygon.io data feeds, processes market events (trades, quotes, and aggregates), and emits Qt signals for UI updates in a trading dashboard.

## Architecture
data/
├── init.py                 # Module exports
├── polygon_data_manager.py     # Central data orchestrator
├── websocket_client.py         # Async WebSocket client with Qt integration
└── models/
├── init.py            # Model exports
├── market_data.py         # Market data structures
└── signals.py             # Trading signal structures

## Core Components

### 1. PolygonDataManager (`polygon_data_manager.py`)

The central hub for all data flow in the application. Manages WebSocket connections, transforms raw market data, and emits signals for UI consumption.

**Key Features:**
- Single symbol focus with dynamic symbol switching
- Heartbeat monitoring for data flow health
- Bar data accumulation (maintains last 2000 bars)
- Signal generation for entry/exit points
- Thread-safe Qt signal emission

**Signals Emitted:**
```python
market_data_updated = pyqtSignal(dict)      # TickerCalculationData format
chart_data_updated = pyqtSignal(dict)       # Bar data for charts
calculation_updated = pyqtSignal(dict)      # Calculated metrics
entry_signal_generated = pyqtSignal(dict)   # Entry signals
exit_signal_generated = pyqtSignal(dict)    # Exit signals
connection_status_changed = pyqtSignal(bool) # Connection state
error_occurred = pyqtSignal(str)            # Error messages
Key Methods:
pythondef __init__(self, server_url: Optional[str] = None)
    # Initialize with optional custom server URL
    # Default: "ws://localhost:8200/ws/{client_id}"

def connect(self)
    # Establish WebSocket connection

def disconnect(self)
    # Clean disconnection with heartbeat timer stop

def change_symbol(self, symbol: str)
    # Switch to new symbol, clearing old state
    # Handles unsubscription from previous symbol

def get_accumulated_bars(self) -> List[Dict]
    # Retrieve copy of accumulated bar data

def check_data_flow(self)
    # Heartbeat check - runs every 5 seconds
    # Warns if no data received for 30+ seconds
Data Processing Pipeline:

Raw WebSocket data received via _on_data_received()
Routed by event type (trade/quote/aggregate)
State updated in market_state dictionary
Appropriate signals emitted for UI updates

2. PolygonWebSocketClient (websocket_client.py)
Async WebSocket client with Qt thread integration. Handles the low-level WebSocket connection with automatic reconnection.
Key Features:

Runs asyncio event loop in separate QThread
Exponential backoff reconnection (1s to 30s max)
Subscription management for symbols and channels
Thread-safe method calls from Qt main thread

Connection Flow:
python# 1. Initialize client
client = PolygonWebSocketClient(
    server_url="ws://localhost:8200/ws/{client_id}",
    client_id="live_monitor"
)

# 2. Connect
client.connect_to_server()

# 3. Subscribe to symbols
client.subscribe(["AAPL", "MSFT"])
# Or single symbol
client.change_symbol("AAPL")

# 4. Receive data via signals
client.data_received.connect(process_data)
WebSocket Protocol:
json// Subscribe message
{
    "action": "subscribe",
    "symbols": ["AAPL"],
    "channels": ["T", "Q", "AM"]  // Trades, Quotes, Minute Aggregates
}

// Unsubscribe message
{
    "action": "unsubscribe",
    "symbols": ["AAPL"]
}

// Incoming market data
{
    "type": "market_data",
    "data": {
        "ev": "T",        // Event type
        "sym": "AAPL",    // Symbol
        "p": 150.25,      // Price
        "s": 100,         // Size
        "t": 1234567890   // Timestamp (ms)
    }
}
3. Data Models
Market Data Models (models/market_data.py)
TradeData
python{
    'event_type': 'trade',
    'symbol': 'AAPL',
    'timestamp': 1234567890000,  # milliseconds
    'price': 150.25,
    'size': 100,
    'conditions': [14, 37],       # Trade conditions
    'exchange': 'NASDAQ'          # Optional
}
QuoteData
python{
    'event_type': 'quote',
    'symbol': 'AAPL',
    'timestamp': 1234567890000,
    'bid_price': 150.20,
    'bid_size': 300,
    'ask_price': 150.25,
    'ask_size': 500,
    'exchange': 'NASDAQ'          # Optional
}
AggregateData (Minute bars)
python{
    'event_type': 'aggregate',
    'symbol': 'AAPL',
    'timestamp': 1234567890000,
    'open': 150.00,
    'high': 150.50,
    'low': 149.75,
    'close': 150.25,
    'volume': 100000,
    'vwap': 150.10,               # Optional
    'transactions': 500           # Optional
}
TickerCalculationData (For UI display)
python{
    'last_price': 150.25,
    'bid': 150.20,
    'ask': 150.25,
    'spread': 0.05,
    'mid_price': 150.225,
    'volume': 1000000,
    'change': 2.50,               # Optional
    'change_percent': 1.69,       # Optional
    'day_high': 151.00,           # Optional
    'day_low': 149.00,            # Optional
    'day_open': 147.75,           # Optional
    'last_update': '14:35:22',
    'market_state': 'open'        # 'open', 'closed', 'pre', 'post'
}
Trading Signal Models (models/signals.py)
EntrySignal
python{
    'time': '14:35:22',
    'signal_type': 'LONG',        # 'LONG' or 'SHORT'
    'price': '150.25',
    'signal': 'RSI Oversold + Support Test',
    'strength': 'Strong',         # 'Strong', 'Medium', 'Weak'
    'notes': 'Volume confirmation present',
    'timestamp': datetime,
    'symbol': 'AAPL'
}
ExitSignal
python{
    'time': '14:45:15',
    'exit_type': 'TARGET',        # 'TARGET', 'STOP', 'TRAIL'
    'price': '151.50',
    'pnl': '+0.83%',
    'signal': 'Price target reached',
    'urgency': 'Normal',          # 'Urgent', 'Warning', 'Normal'
    'timestamp': datetime,
    'symbol': 'AAPL',
    'pnl_value': 1.25
}
Usage Examples
Basic Setup and Connection
pythonfrom live_monitor.data import PolygonDataManager

# Initialize data manager
data_manager = PolygonDataManager()

# Connect status monitoring
data_manager.connection_status_changed.connect(
    lambda connected: print(f"Connected: {connected}")
)

# Market data updates
data_manager.market_data_updated.connect(
    lambda data: print(f"Price: ${data['last_price']}")
)

# Chart updates
data_manager.chart_data_updated.connect(
    lambda data: print(f"New bar: {data['bars']}")
)

# Connect and start receiving data
data_manager.connect()

# Change symbol
data_manager.change_symbol("AAPL")
Handling Data in UI Components
pythonclass TradingWidget(QWidget):
    def __init__(self, data_manager):
        super().__init__()
        self.data_manager = data_manager
        
        # Connect to data updates
        self.data_manager.market_data_updated.connect(
            self.update_ticker_display
        )
        self.data_manager.chart_data_updated.connect(
            self.update_chart
        )
        
    def update_ticker_display(self, data: dict):
        # Update UI with latest prices
        self.price_label.setText(f"${data['last_price']:.2f}")
        self.spread_label.setText(f"Spread: ${data['spread']:.2f}")
        
    def update_chart(self, data: dict):
        if data['is_update']:
            # Append new bar to existing chart
            self.chart.append_bar(data['bars'][0])
        else:
            # Full chart refresh
            self.chart.set_data(data['bars'])
Custom Signal Generation
python# Extend PolygonDataManager for custom signals
class CustomDataManager(PolygonDataManager):
    def _process_trade(self, trade):
        super()._process_trade(trade)
        
        # Add custom signal logic
        if self._check_entry_condition(trade):
            signal = {
                'time': datetime.now().strftime("%H:%M:%S"),
                'signal_type': 'LONG',
                'price': str(trade['price']),
                'signal': 'Custom indicator triggered',
                'strength': 'Medium',
                'symbol': trade['symbol']
            }
            self.entry_signal_generated.emit(signal)
Data Flow Diagram
Polygon WebSocket Server
         |
         v
PolygonWebSocketClient (websocket_client.py)
    - Async WebSocket connection
    - Running in QThread
    - Handles reconnection
         |
         | (data_received signal)
         v
PolygonDataManager (polygon_data_manager.py)
    - Routes by event type
    - Updates market state
    - Accumulates bars
    - Checks data heartbeat
         |
         | (Multiple signals)
         v
    +----+----+----+----+
    |    |    |    |    |
    v    v    v    v    v
  Market Chart Entry Exit Error
  Updates Data Signals Signals
Threading Model

Main Thread: Qt application and UI
WebSocket Thread: Dedicated QThread running asyncio event loop
Communication: Thread-safe via Qt signals/slots

Error Handling

Connection Errors: Automatic reconnection with exponential backoff
Data Errors: Logged and error_occurred signal emitted
Symbol Changes: Old symbol state cleared before switching
No Data Detection: Heartbeat monitoring alerts after 30 seconds

Configuration
Server Settings
python# Default server URL
server_url = "ws://localhost:8200/ws/{client_id}"

# Heartbeat settings
no_data_threshold = 30  # seconds
heartbeat_check_interval = 5000  # milliseconds

# Bar accumulation
max_bars = 2000  # Maximum bars to keep in memory

# Reconnection settings
reconnect_delay = 1.0
max_reconnect_delay = 30.0
max_reconnect_attempts = 10
Subscription Channels

T: Trade events
Q: Quote events
AM: Aggregate minute bars

Testing
Test Chart Updates
python# Built-in test method for chart updates
data_manager.test_chart_update()
This sends synthetic aggregate data through the processing pipeline to verify chart updates are working correctly.
Logging
The module uses Python's standard logging framework:
pythonimport logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Module loggers
logger = logging.getLogger('live_monitor.data.polygon_data_manager')
logger = logging.getLogger('live_monitor.data.websocket_client')
Notes for Developers

Single Symbol Focus: The system is designed for monitoring one symbol at a time. Symbol changes clear previous state.
Data Format Flexibility: The system handles both standard format (symbol, price) and Polygon format (sym, p) field names.
Memory Management: Bar accumulation is limited to 2000 bars to prevent memory issues during long sessions.
Qt Integration: All data updates are delivered via Qt signals, making UI integration straightforward.
Extensibility: The modular design allows easy addition of new data sources or signal generation logic.

Common Issues

No Data Received: Check WebSocket server is running and symbol is valid
Connection Drops: Monitor connection_status_changed signal
High Memory Usage: Reduce max_bars setting if needed
Delayed Updates: Check heartbeat logs for data flow issues


This comprehensive README provides everything an AI agent would need to understand and work with your data module, including architecture, usage patterns, data structures, and troubleshooting information. It's structured to be both a reference guide and a practical manual for implementation.