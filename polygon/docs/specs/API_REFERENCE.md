API_Reference

## 2. API_REFERENCE.md

```markdown
# Polygon Module - API Reference

## Core API Methods

### PolygonDataManager

The main interface for all polygon operations.

```python
class PolygonDataManager:
    """
    Primary interface for polygon data operations.
    Coordinates fetching, caching, validation, and serving.
    """
fetch_data()
pythondef fetch_data(
    symbol: str,
    timeframe: str = '1day',
    start_date: Union[str, datetime] = None,
    end_date: Union[str, datetime] = None,
    use_cache: bool = True,
    validate: bool = True,
    fill_gaps: bool = False,
    adjust_splits: bool = True
) -> pd.DataFrame
Parameters:

symbol: Stock ticker (e.g., 'AAPL')
timeframe: Data granularity ('1min', '5min', '1hour', '1day', etc.)
start_date: Start date (YYYY-MM-DD or datetime object)
end_date: End date (defaults to today)
use_cache: Use local cache if available
validate: Run data quality validation
fill_gaps: Attempt to fill small data gaps
adjust_splits: Apply split adjustments

Returns: DataFrame with columns: open, high, low, close, volume, vwap, transactions
Raises:

PolygonSymbolError: Invalid symbol
PolygonTimeRangeError: Invalid date range
PolygonAPIError: API request failed

fetch_multiple_symbols()
pythondef fetch_multiple_symbols(
    symbols: List[str],
    timeframe: str,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    max_workers: int = 5,
    **kwargs
) -> Dict[str, pd.DataFrame]
Returns: Dictionary mapping symbols to DataFrames
search_symbols()
pythondef search_symbols(
    query: str,
    active_only: bool = True
) -> List[Dict[str, Any]]
Returns: List of matching ticker information
DataFetcher Methods
fetch_latest_bars()
pythondef fetch_latest_bars(
    symbols: Union[str, List[str]],
    timeframe: str = '1min',
    bars: int = 100
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]
Parameters:

symbols: Single symbol or list
timeframe: Bar timeframe
bars: Number of recent bars to fetch

update_cache()
pythondef update_cache(
    symbol: str,
    timeframe: str
) -> Dict[str, Any]
Returns:
python{
    'symbol': 'AAPL',
    'timeframe': '5min',
    'rows_added': 150,
    'cache_updated': True,
    'error': None
}
get_data_summary()
pythondef get_data_summary(
    symbol: str,
    timeframe: str,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime]
) -> Dict[str, Any]
Returns:
python{
    'symbol': 'AAPL',
    'timeframe': '5min',
    'requested_range': {
        'start': '2023-01-01T00:00:00',
        'end': '2023-01-31T23:59:59',
        'days': 31
    },
    'cache_coverage': {
        'percentage': 85.5,
        'cached_ranges': [...],
        'missing_ranges': [...]
    },
    'estimated_bars': 4030,
    'estimated_size_mb': 2.5,
    'estimated_api_calls': 2,
    'data_available': False
}
StorageManager Methods
save_data()
pythondef save_data(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    update_metadata: bool = True
) -> CacheMetadata
Returns: CacheMetadata object with file info
load_data()
pythondef load_data(
    symbol: str,
    timeframe: str,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None
) -> Optional[pd.DataFrame]
get_cache_statistics()
pythondef get_cache_statistics() -> Dict[str, Any]
Returns:
python{
    'total_entries': 150,
    'total_size_mb': 125.5,
    'top_symbols': [
        {'symbol': 'AAPL', 'count': 5, 'size': 10485760},
        ...
    ],
    'by_timeframe': [...],
    'recent_access': [...],
    'cache_directory': '/path/to/cache',
    'database_path': '/path/to/cache.db'
}
RateLimiter Methods
get_current_usage()
pythondef get_current_usage() -> Dict[str, Any]
Returns:
python{
    'minute': {
        'used': 45,
        'limit': 100,
        'remaining': 55,
        'usage_pct': 45.0
    },
    'daily': {
        'used': 15000,
        'limit': 50000,
        'remaining': 35000,
        'usage_pct': 30.0
    },
    'queue_size': 0,
    'tier': 'advanced'
}
estimate_time_for_requests()
pythondef estimate_time_for_requests(num_requests: int) -> Dict[str, Any]
Returns:
python{
    'total_time_seconds': 120,
    'wait_time_seconds': 60,
    'strategy': 'batched',
    'batches': 5
}
Validation Functions
validate_symbol_detailed()
pythondef validate_symbol_detailed(symbol: str) -> Dict[str, Any]
Returns:
python{
    'valid': True,
    'normalized': 'AAPL',
    'type': 'equity',
    'characteristics': [],
    'warnings': []
}
Symbol Types:

equity: Standard stock (AAPL)
equity_class: With share class (BRK.A)
preferred: Preferred stock (BAC-PD)
warrant: Warrant (NKLA.WS)
when_issued: When issued (AAPL.WI)
rights: Rights (AAPL.RT)
units: Units (IPOF.U)

generate_validation_summary()
pythondef generate_validation_summary(
    df: pd.DataFrame,
    symbol: str,
    expected_timeframe: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Dict[str, Any]
Returns: Comprehensive validation report with:

Overall quality score (0-100)
Quality rating (EXCELLENT, GOOD, FAIR, POOR)
Detailed reports for each validation type
Recommendations

detect_gaps()
pythondef detect_gaps(
    df: pd.DataFrame,
    timeframe: str,
    market_hours_only: bool = True
) -> Dict[str, Any]
Returns:
python{
    'gaps': [
        {
            'start': datetime,
            'end': datetime,
            'duration': '1:30:00',
            'missing_bars': 18
        }
    ],
    'gap_count': 3,
    'total_missing_bars': 42,
    'data_completeness_pct': 98.5,
    'largest_gap': {...}
}
detect_price_anomalies()
pythondef detect_price_anomalies(
    df: pd.DataFrame,
    zscore_threshold: float = 3.0,
    pct_change_threshold: float = 0.2
) -> Dict[str, Any]
Returns: Dictionary with:

extreme_changes: Price changes > threshold
statistical_outliers: Z-score outliers
volume_spikes: Abnormal volume
spread_anomalies: Unusual high-low spreads

Utility Functions
parse_timeframe()
pythondef parse_timeframe(timeframe: str) -> Tuple[int, str]
Examples:

'5min' -> (5, 'minute')
'1h' -> (1, 'hour')
'1d' -> (1, 'day')

calculate_bars_in_range()
pythondef calculate_bars_in_range(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    timeframe: str
) -> int
split_large_date_range()
pythondef split_large_date_range(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    max_days: int
) -> List[Tuple[datetime, datetime]]
Exception Reference
PolygonError Attributes
All exceptions inherit from PolygonError and include:

message: Error description
details: Additional context dictionary
timestamp: When error occurred (UTC)

PolygonAPIError
Additional attributes:

status_code: HTTP status code
response_body: Raw API response

PolygonRateLimitError
Additional attributes:

retry_after: Seconds to wait
limit_type: 'minute' or 'day'

PolygonSymbolError
Additional attributes:

symbol: The invalid symbol
suggestions: List of similar valid symbols

DataFrame Formats
OHLCV DataFrame
Index: DatetimeIndex (UTC timezone)
Columns:

open: float64
high: float64
low: float64
close: float64
volume: int64
vwap: float64 (optional)
transactions: int64 (optional)

Multi-Symbol DataFrame (aligned)
Index: DatetimeIndex
Columns: MultiIndex[(symbol, metric)]
Example: df[('AAPL', 'close')]