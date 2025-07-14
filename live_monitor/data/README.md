# Live Monitor Data Module

## Overview

The `live_monitor/data/` module is a comprehensive real-time and historical market data management system designed for intraday trading applications. It provides WebSocket connectivity to Polygon.io data feeds, processes market events (trades, quotes, and aggregates), manages historical data fetching, and emits Qt signals for UI updates in a trading dashboard.

## Architecture
data/
├── init.py                     # Module exports
├── polygon_data_manager.py         # Central data orchestrator
├── websocket_client.py             # Async WebSocket client with Qt integration
├── rest_client.py                  # REST API client for historical data
├── models/                         # Data models
│   ├── init.py                # Model exports
│   ├── market_data.py             # Market data structures
│   └── signals.py                 # Trading signal structures
└── hist_request/                   # Historical data fetching subsystem
├── init.py                # Fetcher exports
├── base_fetcher.py            # Base class for all fetchers
├── ema_fetchers.py            # EMA calculation fetchers
├── market_structure_fetchers.py # Market structure fetchers
├── trend_fetchers.py          # Statistical trend fetchers
├── zone_fetchers.py           # HVN and Order Block fetchers
└── fetch_coordinator.py       # Orchestrates parallel fetching

## Core Components

### 1. PolygonDataManager (`polygon_data_manager.py`)

The central hub for all data flow in the application. Manages WebSocket connections, REST API calls, historical data coordination, transforms raw market data, and emits signals for UI consumption.

**Key Features:**
- Single symbol focus with dynamic symbol switching
- Real-time WebSocket data processing
- Historical data coordination via HistoricalFetchCoordinator
- Heartbeat monitoring for data flow health
- Bar data accumulation (maintains last 2000 bars)
- Signal generation for entry/exit points
- Thread-safe Qt signal emission
- Automatic historical data fetching when market is closed

**Real-time Data Signals:**
```python
market_data_updated = pyqtSignal(dict)      # TickerCalculationData format
chart_data_updated = pyqtSignal(dict)       # Bar data for charts
calculation_updated = pyqtSignal(dict)      # Calculated metrics
entry_signal_generated = pyqtSignal(dict)   # Entry signals
exit_signal_generated = pyqtSignal(dict)    # Exit signals
connection_status_changed = pyqtSignal(bool) # Connection state
error_occurred = pyqtSignal(str)            # Error messages
Historical Data Signals:
pythonema_data_ready = pyqtSignal(dict)           # {'M1': df, 'M5': df, 'M15': df}
structure_data_ready = pyqtSignal(dict)     # {'M1': df, 'M5': df, 'M15': df}
trend_data_ready = pyqtSignal(dict)         # {'M1': df, 'M5': df, 'M15': df}
zone_data_ready = pyqtSignal(dict)          # {'HVN': df, 'OrderBlocks': df}
historical_fetch_progress = pyqtSignal(dict) # Fetch progress updates
2. PolygonWebSocketClient (websocket_client.py)
Async WebSocket client with Qt thread integration. Handles the low-level WebSocket connection with automatic reconnection.
Key Features:

Runs asyncio event loop in separate QThread
Exponential backoff reconnection (1s to 30s max)
Subscription management for symbols and channels
Thread-safe method calls from Qt main thread

3. PolygonRESTClient (rest_client.py)
REST API client for fetching historical data from the Polygon server.
Key Features:

Historical bar fetching with configurable timeframes
Market session detection (pre, regular, after, closed)
UTC timestamp handling throughout
Error handling with Qt signal emission

Available Methods:
pythonfetch_bars(symbol, timespan='1min', multiplier=1, limit=200)
is_market_open() -> bool
get_market_session() -> str  # 'pre', 'regular', 'after', 'closed'
4. Historical Data Request System (hist_request/)
A sophisticated subsystem for fetching and managing historical data for various technical analysis calculations. Uses a coordinator pattern to manage multiple parallel fetches efficiently.
Key Components:

HistoricalFetchCoordinator: Orchestrates all historical data fetching
BaseHistoricalFetcher: Abstract base class for all fetchers
Specialized Fetchers: 11 different fetchers optimized for specific calculations

Fetcher Types:

EMA Fetchers (35 bars each)

M1EMAFetcher: 1-minute bars for EMA
M5EMAFetcher: 5-minute bars for EMA
M15EMAFetcher: 15-minute bars for EMA


Market Structure Fetchers

M1MarketStructureFetcher: 200 1-minute bars
M5MarketStructureFetcher: 100 5-minute bars
M15MarketStructureFetcher: 60 15-minute bars


Statistical Trend Fetchers (15 bars each)

M1StatisticalTrendFetcher
M5StatisticalTrendFetcher
M15StatisticalTrendFetcher


Zone Analysis Fetchers (200 1-minute bars, shared cache)

HVNFetcher: High Volume Node analysis
OrderBlocksFetcher: Order block detection



Priority-Based Fetching:
pythonpriority_groups = {
    'high': ['M1_EMA', 'M5_EMA'],  # Fastest signals
    'medium': ['M15_EMA', 'M1_MarketStructure', 'M5_MarketStructure'],
    'low': ['M15_MarketStructure', 'M1_StatisticalTrend', 
            'M5_StatisticalTrend', 'M15_StatisticalTrend', 'HVN', 'OrderBlocks']
}
5. Data Models
Market Data Models (models/market_data.py)

TradeData: Individual trade events
QuoteData: Bid/ask updates
AggregateData: OHLCV bars
TickerCalculationData: UI display format
MarketDataUpdate: Combined market state

Trading Signal Models (models/signals.py)

EntrySignal: Trading entry signals
ExitSignal: Trading exit signals

Data Flow
Real-time Data Flow
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
Historical Data Flow
Symbol Change Request
         |
         v
PolygonDataManager
         |
         v
HistoricalFetchCoordinator
         |
    +----+----+----+
    |    |    |    |
    v    v    v    v
  High  Med  Low  Priority
  Pri   Pri  Pri  Groups
    |    |    |
    v    v    v
Individual Fetchers
    |
    v
PolygonRESTClient
    |
    v
Polygon REST API
Usage Examples
Basic Setup
pythonfrom live_monitor.data import PolygonDataManager

# Initialize data manager
data_manager = PolygonDataManager()

# Connect status monitoring
data_manager.connection_status_changed.connect(
    lambda connected: print(f"Connected: {connected}")
)

# Real-time data updates
data_manager.market_data_updated.connect(
    lambda data: print(f"Price: ${data['last_price']}")
)

# Historical data updates
data_manager.ema_data_ready.connect(
    lambda data: print(f"EMA data ready for timeframes: {data.keys()}")
)

# Connect and start
data_manager.connect()
data_manager.change_symbol("AAPL")
Handling Historical Data
pythonclass TradingWidget(QWidget):
    def __init__(self, data_manager):
        super().__init__()
        self.data_manager = data_manager
        
        # Connect to historical data signals
        self.data_manager.ema_data_ready.connect(self.process_ema_data)
        self.data_manager.structure_data_ready.connect(self.process_structure_data)
        self.data_manager.zone_data_ready.connect(self.process_zone_data)
        
    def process_ema_data(self, data: dict):
        # data contains {'M1': DataFrame, 'M5': DataFrame, 'M15': DataFrame}
        for timeframe, df in data.items():
            print(f"{timeframe} EMA data: {len(df)} bars")
            # Calculate EMAs...
    
    def process_structure_data(self, data: dict):
        # Process market structure data
        pass
Monitoring Fetch Progress
python# Monitor historical data fetching progress
data_manager.historical_fetch_progress.connect(
    lambda progress: print(f"Fetching: {progress['completed']}/{progress['total']} "
                          f"({progress['percentage']:.0f}%)")
)

# Check fetch status
status = data_manager.get_historical_fetch_status()
print(f"In progress: {status['in_progress']}")
print(f"Completed: {status['results']}")
Configuration
Server Settings
python# WebSocket settings
ws_server_url = "ws://localhost:8200/ws/{client_id}"

# REST API settings
rest_base_url = "http://localhost:8200"

# Heartbeat settings
no_data_threshold = 30  # seconds
heartbeat_check_interval = 5000  # milliseconds

# Bar accumulation
max_bars = 2000  # Maximum bars to keep in memory

# Reconnection settings
reconnect_delay = 1.0
max_reconnect_delay = 30.0
max_reconnect_attempts = 10
Historical Fetch Settings
python# Cache validity
cache_duration = 300  # 5 minutes

# Retry settings
max_retries = 3

# Priority fetching
use_priority_mode = True  # Fetch high priority first
Threading Model

Main Thread: Qt application and UI
WebSocket Thread: Dedicated QThread running asyncio event loop
Historical Fetches: Run in main thread using Qt's event loop
Communication: Thread-safe via Qt signals/slots

Error Handling
Connection Errors

Automatic reconnection with exponential backoff
Connection status signals for UI feedback

Data Errors

Logged with appropriate severity
error_occurred signal emitted for UI notification

Historical Fetch Errors

Individual fetcher retry logic (3 attempts)
Fetch continues even if some fetchers fail
Error summary provided via fetch_error signal

Market Hours Awareness
The system automatically adjusts behavior based on market hours:

Regular Hours (13:30-21:00 UTC): Real-time data expected
Pre-market (09:00-13:30 UTC): Limited real-time data
After-hours (21:00-01:00 UTC): Limited real-time data
Closed: Automatic historical data refresh every 5 minutes

Testing
Test Real-time Updates
python# Built-in test method for chart updates
data_manager.test_chart_update()
Test Historical Fetch
python# Force refresh of historical data
data_manager.fetch_historical_bars("AAPL", bars_needed=200)
Logging
pythonimport logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Module loggers
'live_monitor.data.polygon_data_manager'
'live_monitor.data.websocket_client'
'live_monitor.data.rest_client'
'live_monitor.data.hist_request.fetch_coordinator'
'live_monitor.data.hist_request.base_fetcher'
Common Issues

No Data Received

Check WebSocket server is running
Verify symbol is valid
Check market hours


Historical Fetch Timeout

Check REST API server status
Reduce number of concurrent fetches
Check network connectivity


High Memory Usage

Reduce max_bars setting
Clear old symbol data more aggressively


Delayed Updates

Check heartbeat logs
Monitor fetch progress signals
Verify server performance



Performance Considerations

Historical fetches use shared cache for zone analysis
Priority-based fetching reduces initial load time
Bar accumulation limited to prevent memory issues
Automatic cache management for historical data