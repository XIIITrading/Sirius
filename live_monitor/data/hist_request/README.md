# Historical Data Request Module

## Overview

The `hist_request/` module is a sophisticated historical data fetching subsystem designed to efficiently retrieve and manage historical market data for various technical analysis calculations. It implements a coordinator pattern with specialized fetchers that can run in parallel while respecting dependencies and priorities.

## Architecture
hist_request/
├── init.py                      # Module exports
├── base_fetcher.py                  # Abstract base class for all fetchers
├── fetch_coordinator.py             # Orchestrates parallel fetching
├── ema_fetchers.py                  # EMA-specific fetchers (M1, M5, M15)
├── market_structure_fetchers.py     # Market structure analysis fetchers
├── trend_fetchers.py                # Statistical trend fetchers
└── zone_fetchers.py                 # HVN and Order Block fetchers

## Core Concepts

### 1. Fetcher Hierarchy

All fetchers inherit from `BaseHistoricalFetcher` which provides:
- Common signal interface
- Cache management (5-minute validity)
- Retry logic (3 attempts with exponential backoff)
- Data validation framework
- Qt signal emission

### 2. Fetcher Categories

#### EMA Fetchers
Optimized for Exponential Moving Average calculations:
- **M1EMAFetcher**: 35 x 1-minute bars (26 minimum)
- **M5EMAFetcher**: 35 x 5-minute bars (26 minimum)
- **M15EMAFetcher**: 35 x 15-minute bars (26 minimum)

#### Market Structure Fetchers
For fractal and structure break detection:
- **M1MarketStructureFetcher**: 200 x 1-minute bars (21 minimum)
- **M5MarketStructureFetcher**: 100 x 5-minute bars (15 minimum)
- **M15MarketStructureFetcher**: 60 x 15-minute bars (10 minimum)

#### Statistical Trend Fetchers
Minimal data for trend analysis:
- **M1StatisticalTrendFetcher**: 15 x 1-minute bars (10 minimum)
- **M5StatisticalTrendFetcher**: 15 x 5-minute bars (10 minimum)
- **M15StatisticalTrendFetcher**: 15 x 15-minute bars (10 minimum)

#### Zone Analysis Fetchers
For support/resistance zones (share cache):
- **HVNFetcher**: 200 x 1-minute bars (100 minimum)
- **OrderBlocksFetcher**: 200 x 1-minute bars (14 minimum)

### 3. Fetch Coordinator

The `HistoricalFetchCoordinator` manages all fetchers:

**Features:**
- Priority-based fetching (high → medium → low)
- Progress tracking with percentage completion
- Parallel execution within priority groups
- Error aggregation and reporting
- Data ready signals for each calculation type

**Priority Groups:**
```python
{
    'high': ['M1_EMA', 'M5_EMA'],                    # ~1-2 seconds
    'medium': ['M15_EMA', 'M1_MarketStructure',      # ~2-3 seconds
               'M5_MarketStructure'],
    'low': ['M15_MarketStructure',                   # ~3-5 seconds
            'M1_StatisticalTrend', 
            'M5_StatisticalTrend', 
            'M15_StatisticalTrend', 
            'HVN', 'OrderBlocks']
}
Signal Flow
Individual Fetcher Signals
Each fetcher emits:
pythonfetch_started = pyqtSignal(str)      # symbol
fetch_completed = pyqtSignal(dict)   # {'symbol', 'dataframe', 'metadata', 'fetcher'}
fetch_failed = pyqtSignal(dict)      # {'symbol', 'error', 'fetcher'}
Coordinator Signals
The coordinator emits:
pythonfetch_started = pyqtSignal(str)               # Overall fetch started
fetch_progress = pyqtSignal(dict)             # Progress updates
all_fetches_completed = pyqtSignal(str)       # All complete
fetch_error = pyqtSignal(dict)                # Error summary

# Data ready signals (emitted as soon as available)
ema_data_ready = pyqtSignal(dict)             # {'M1': df, 'M5': df, 'M15': df}
structure_data_ready = pyqtSignal(dict)       # Market structure data
trend_data_ready = pyqtSignal(dict)           # Statistical trend data
zone_data_ready = pyqtSignal(dict)            # {'HVN': df, 'OrderBlocks': df}
Usage
Basic Coordinator Usage
pythonfrom live_monitor.data.rest_client import PolygonRESTClient
from live_monitor.data.hist_request import HistoricalFetchCoordinator

# Initialize
rest_client = PolygonRESTClient()
coordinator = HistoricalFetchCoordinator(rest_client)

# Connect to signals
coordinator.fetch_progress.connect(
    lambda p: print(f"Progress: {p['completed']}/{p['total']}")
)
coordinator.ema_data_ready.connect(
    lambda data: print(f"EMA data ready: {data.keys()}")
)

# Start fetch
coordinator.fetch_all_for_symbol("AAPL", priority_mode=True)
Individual Fetcher Usage
pythonfrom live_monitor.data.hist_request import M1EMAFetcher

# Initialize
fetcher = M1EMAFetcher(rest_client)

# Connect signals
fetcher.fetch_completed.connect(
    lambda data: print(f"Got {len(data['dataframe'])} bars")
)

# Fetch data
fetcher.fetch_for_symbol("AAPL")
Accessing Fetcher Results
python# After fetch completes, access data from coordinator
m1_ema_data = coordinator.fetchers['M1_EMA'].cache
if m1_ema_data is not None:
    print(f"M1 EMA data: {len(m1_ema_data)} bars")
    print(f"Latest close: {m1_ema_data['close'].iloc[-1]}")
Data Formats
Fetcher Output Format
All fetchers return data via fetch_completed signal:
python{
    'symbol': 'AAPL',
    'dataframe': pd.DataFrame,  # Index: UTC timestamps
    'metadata': {
        'calculation_type': 'M1_EMA',
        'latest_close': 150.25,
        'bar_count': 35,
        # ... fetcher-specific metadata
    },
    'fetcher': 'M1_EMA',
    'timespan': '1min',
    'bars': 35
}
DataFrame Structure
All DataFrames have UTC timestamp index with columns:

open: Opening price
high: High price
low: Low price
close: Closing price
volume: Volume
vwap: Volume-weighted average price (optional)
trades: Number of trades (optional)

Cache Management
Cache Behavior

Each fetcher maintains its own cache
Cache validity: 5 minutes
Cache cleared on symbol change
Zone fetchers (HVN, OrderBlocks) share cache

Shared Zone Cache
python# Zone fetchers share data to avoid duplicate API calls
zone_cache = SharedZoneCache()
zone_cache.register('HVN')
zone_cache.register('OrderBlocks')
Error Handling
Retry Logic

Maximum 3 retry attempts per fetcher
Exponential backoff between retries
Fetch continues even if some fetchers fail

Error Types

Network Errors: Connection timeouts, API unavailable
Data Validation Errors: Invalid OHLC data, insufficient bars
API Errors: Rate limits, invalid symbols

Error Recovery
python# Monitor errors
coordinator.fetch_error.connect(
    lambda e: print(f"Errors: {e['errors']}")
)

# Check individual fetcher status
status = coordinator.get_fetch_status()
if status['errors'] > 0:
    print(f"Failed fetchers: {status['in_progress']}")
Performance Optimization
Parallel Fetching

Fetchers within same priority run in parallel
Network I/O handled asynchronously
Qt signals ensure thread-safe communication

Memory Management

DataFrames limited to requested bars
Old cache automatically cleared
Shared cache for zone analysis

Priority Execution
High-priority fetchers complete first, enabling:

Faster initial UI updates
Progressive data loading
Better user experience

Extending the System
Creating a Custom Fetcher
pythonfrom live_monitor.data.hist_request import BaseHistoricalFetcher

class CustomFetcher(BaseHistoricalFetcher):
    def __init__(self, rest_client):
        super().__init__(
            rest_client=rest_client,
            bars_needed=50,
            timespan='1min',
            name='Custom'
        )
    
    def get_minimum_bars(self) -> int:
        return 30
    
    def _validate_specific_data(self, df: pd.DataFrame) -> bool:
        # Custom validation logic
        return True
    
    def _get_metadata(self, df: pd.DataFrame) -> Dict:
        return {
            'calculation_type': 'Custom',
            'custom_metric': self.calculate_metric(df)
        }
Adding to Coordinator
python# In HistoricalFetchCoordinator._init_fetchers()
self.custom_fetcher = CustomFetcher(self.rest_client)
self.fetchers['Custom'] = self.custom_fetcher

# Add to priority group
self.priority_groups['low'].append('Custom')
Best Practices

Always Check Cache: Fetchers check cache before API calls
Handle Missing Data: Some fetchers may fail in pre-market
Monitor Progress: Use progress signals for UI feedback
Respect Priorities: Let high-priority complete first
Clean Up: Clear old data when changing symbols

Troubleshooting
Common Issues

Slow Fetching

Check network latency
Reduce bars_needed if possible
Use priority mode


Cache Misses

Verify cache validity duration
Check symbol changes
Monitor cache clearing


Incomplete Data

Some timeframes may have gaps
Pre-market has limited data
Check fetcher minimums



Debug Logging
pythonimport logging

# Enable fetcher debug logs
logging.getLogger('live_monitor.data.hist_request').setLevel(logging.DEBUG)

# Monitor specific fetcher
logging.getLogger('live_monitor.data.hist_request.ema_fetchers').setLevel(logging.DEBUG)

