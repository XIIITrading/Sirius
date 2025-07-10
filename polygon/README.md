1. README.md (Main Documentation)
markdown# Polygon Data Module - Complete Documentation

## Overview

The Polygon module is a comprehensive Python package for accessing Polygon.io market data. It provides a unified interface for fetching historical data, streaming real-time data, managing local caching, and serving data through REST/WebSocket APIs.

## Architecture
polygon/
├── Core Components
│   ├── config.py          # Configuration management (PolygonConfig)
│   ├── core.py           # API client (PolygonClient, PolygonSession)
│   ├── fetcher.py        # High-level data fetching (DataFetcher, BatchDataFetcher)
│   ├── storage.py        # Local caching (StorageManager, CacheMetadata)
│   ├── rate_limiter.py   # API rate limiting (RateLimiter, AsyncRateLimiter)
│   └── exceptions.py     # Custom exceptions
│
├── Validators
│   ├── data_quality.py   # DataQualityReport, validation orchestration
│   ├── symbol.py         # Symbol validation (validate_symbol_detailed)
│   ├── ohlcv.py         # OHLCV data validation
│   ├── gaps.py          # Gap detection
│   ├── anomalies.py     # Anomaly detection
│   └── market_hours.py  # Market hours validation
│
├── Real-time
│   └── websocket.py     # WebSocket client (PolygonWebSocketClient)
│
├── Server
│   └── polygon_server/
│       ├── server.py    # FastAPI application
│       ├── models.py    # Pydantic models
│       └── endpoints/   # REST/WebSocket endpoints
│
└── Utilities
└── utils.py         # Helper functions

## Core Classes

### PolygonConfig (config.py)
Central configuration management for the entire module.

**Key Attributes:**
- `api_key`: Your Polygon.io API key (from POLYGON_API_KEY env var)
- `base_url`: "https://api.polygon.io"
- `subscription_tier`: 'basic', 'starter', 'developer', or 'advanced'
- `data_dir`: Local storage directory (./data)
- `cache_enabled`: Whether to use local caching (default: True)
- `market_timezone`: pytz.UTC (all times in UTC)

**Usage:**
```python
from polygon.config import get_config
config = get_config()
PolygonClient (core.py)
Low-level API client for direct Polygon.io interaction.
Key Methods:

get_aggregates(ticker, multiplier, timespan, from_date, to_date)
get_ticker_details(ticker, date=None)
get_market_status()
search_tickers(search, active=True, limit=100)
validate_ticker(ticker)

Usage:
pythonfrom polygon.core import PolygonClient
client = PolygonClient()
data = client.get_aggregates('AAPL', 1, 'day', '2023-01-01', '2023-01-31')
DataFetcher (fetcher.py)
High-level interface for fetching data with caching, validation, and rate limiting.
Key Methods:

fetch_data(symbol, timeframe, start_date, end_date, use_cache=True)
fetch_multiple_symbols(symbols, timeframe, start_date, end_date)
fetch_latest_bars(symbols, timeframe='1min', bars=100)
update_cache(symbol, timeframe)
get_data_summary(symbol, timeframe, start_date, end_date)

Usage:
pythonfrom polygon.fetcher import DataFetcher
fetcher = DataFetcher()
df = fetcher.fetch_data('AAPL', '5min', '2023-01-01', '2023-01-31')
StorageManager (storage.py)
Manages local data caching using SQLite for metadata and Parquet files for OHLCV data.
Key Methods:

save_data(df, symbol, timeframe)
load_data(symbol, timeframe, start_date=None, end_date=None)
has_cache(symbol, timeframe, start_date=None, end_date=None)
get_missing_ranges(symbol, timeframe, start_date, end_date)
clear_cache(symbol=None, timeframe=None, older_than_days=None)
get_cache_statistics()

Database Schema:

cache_metadata: Tracks cached data ranges, file paths, checksums
cache_access_log: Records cache usage for analytics
cleanup_log: Tracks cache maintenance operations

RateLimiter (rate_limiter.py)
Intelligent rate limiting with per-minute and daily limits based on subscription tier.
Key Methods:

check_limit() -> (is_allowed, wait_time)
wait_if_needed(priority=5) -> seconds_waited
record_request(response_time=None, success=True)
get_current_usage() -> usage statistics
queue_request(callback, *args, priority=5)

Subscription Tiers:

basic: 5 requests/minute, 1,000/day
starter: 100 requests/minute, 50,000/day
developer: 1,000 requests/minute, 500,000/day
advanced: 10,000 requests/minute, unlimited/day

Data Validation
DataQualityReport (validators/data_quality.py)
Comprehensive validation results container.
Validation Types:

Symbol Validation: Validates ticker format and type
OHLCV Integrity: Checks price relationships, NaN values, duplicates
Gap Detection: Identifies missing data points
Anomaly Detection: Statistical outliers, extreme changes
Volume Profile: Volume distribution analysis
Market Hours: Validates trading session data

Usage:
pythonfrom polygon.validators import generate_validation_summary
summary = generate_validation_summary(df, 'AAPL', '5min', start_date, end_date)
Configuration
Environment Variables
bash# Required
POLYGON_API_KEY=your_api_key_here

# Optional
POLYGON_SUBSCRIPTION_TIER=advanced  # default: advanced
POLYGON_LOG_LEVEL=INFO             # default: INFO
Timeframe Formats
The module accepts flexible timeframe strings:

Minutes: '1min', '5min', '15min', '30min', '60min'
Hours: '1h', '4h', '1hour', '4hour'
Days: '1d', 'D', '1day'
Weeks: '1w', 'W', '1week'
Months: '1m', 'M', '1month'

Important Notes

UTC Timezone: All timestamps are in UTC (POLYGON_TIMEZONE = pytz.UTC)
Market Hours: Configured in UTC (e.g., 9:30 AM ET = 2:30 PM UTC)
Data Storage: Parquet files in ./data/parquet/symbols/{SYMBOL}/
Cache Database: SQLite at ./data/cache/polygon_cache.db

Basic Usage Examples
Fetch Historical Data
pythonfrom polygon import PolygonDataManager

# Initialize manager
manager = PolygonDataManager()

# Fetch daily data
df = manager.fetch_data('AAPL', '1day', '2023-01-01', '2023-12-31')

# Fetch with validation
df = manager.fetch_data('AAPL', '5min', '2023-01-01', '2023-01-31', validate=True)
Batch Operations
python# Fetch multiple symbols
symbols = ['AAPL', 'GOOGL', 'MSFT']
data = manager.fetch_multiple_symbols(symbols, '1day', '2023-01-01', '2023-12-31')

# Create universe dataset
universe = manager.batch_fetcher.fetch_universe(
    symbols, '1day', '2023-01-01', '2023-12-31', aligned=True
)
Cache Management
python# Check cache statistics
stats = manager.get_storage_statistics()

# Clear old data
manager.clear_cache(older_than_days=30)

# Update cache for symbol
manager.update_cache('AAPL', '5min')
Error Handling
The module provides specific exception types:

PolygonError: Base exception
PolygonAPIError: API response errors (with status codes)
PolygonAuthenticationError: Invalid API key
PolygonRateLimitError: Rate limit exceeded
PolygonSymbolError: Invalid ticker symbol
PolygonDataError: Data validation failures
PolygonTimeRangeError: Invalid date ranges
PolygonStorageError: Cache/storage errors
PolygonNetworkError: Connection issues
PolygonWebSocketError: WebSocket streaming errors

Module Initialization
python# Option 1: Use convenience functions
from polygon import fetch_data, fetch_latest
df = fetch_data('AAPL', '5min')
latest = fetch_latest('AAPL', '1min', bars=20)

# Option 2: Use the manager
from polygon import PolygonDataManager
manager = PolygonDataManager()
df = manager.fetch_data('AAPL', '5min', start_date, end_date)

# Option 3: Direct component access
from polygon.fetcher import DataFetcher
from polygon.storage import get_storage_manager
fetcher = DataFetcher()
storage = get_storage_manager()
Performance Optimization

Caching: Automatic local caching reduces API calls
Rate Limiting: Intelligent throttling with priority queuing
Batch Operations: Parallel fetching for multiple symbols
Data Compression: Parquet files with Snappy compression
Smart Merging: Efficiently combines cached and new data