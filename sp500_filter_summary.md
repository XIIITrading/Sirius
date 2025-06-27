# S&P 500 Filter System - Comprehensive Summary

## Overview

The `sp500_filter/` directory implements a sophisticated market screening and ranking system for S&P 500 stocks. This system combines real-time data fetching from Polygon.io with advanced filtering algorithms to identify high-potential trading opportunities during pre-market sessions. The architecture follows a clean separation of concerns with distinct layers for data management, filtering logic, and integration.

## Core Components

### 1. **sp500_tickers.py** - Ticker List Management
**Purpose**: Maintains and validates the current S&P 500 constituent list.

**Key Features**:
- **Manual Ticker List**: Contains 503 S&P 500 tickers as of June 26, 2025
- **Staleness Detection**: Warns when list is older than 90 days
- **Validation Functions**: `verify_ticker()`, `get_ticker_count()`
- **Update Tracking**: `LAST_UPDATED` constant with automatic warnings

**Important Code Sections**:
```python
# Lines 8-9: Critical update tracking
LAST_UPDATED = "2025-06-26"  # ISO format YYYY-MM-DD
UPDATE_FREQUENCY_DAYS = 90  # Remind to update quarterly

# Lines 15-115: Complete S&P 500 ticker list
SP500_TICKERS = ['A', 'AAL', 'AAP', ...]  # 503 tickers

# Lines 117-135: Staleness checking logic
def check_update_status():
    last_update = datetime.strptime(LAST_UPDATED, "%Y-%m-%d")
    days_since_update = (datetime.now() - last_update).days
    if days_since_update > UPDATE_FREQUENCY_DAYS:
        warnings.warn(...)
```

**Connections**: 
- Referenced by `sp500_bridge.py` for ticker list
- Used by `update_sp500_list.py` for comparison
- Called by `test_integration.py` for validation

### 2. **market_filter.py** - Pure Calculation Engine
**Purpose**: Core filtering and scoring logic without data dependencies.

**Key Features**:
- **Configurable Filter Criteria**: Price ranges, volume thresholds, ATR requirements
- **Interest Score Calculation**: Multi-factor weighted scoring system
- **Filter Result Tracking**: Detailed pass/fail analysis per criterion
- **Score Explanation**: Breakdown of individual score components

**Important Data Structures**:
```python
# Lines 12-25: Filter criteria configuration
@dataclass
class FilterCriteria:
    min_price: float = 5.0
    max_price: float = 500.0
    min_avg_volume: float = 500_000
    min_premarket_volume_ratio: float = 0.10
    min_dollar_volume: float = 5_000_000
    min_atr: float = 2.0
    min_atr_percent: float = 10.0

# Lines 28-40: Scoring weights configuration
@dataclass
class InterestScoreWeights:
    premarket_volume_ratio: float = 0.40  # 40% weight
    atr_percentage: float = 0.25          # 25% weight
    dollar_volume_score: float = 0.20     # 20% weight
    premarket_volume_absolute: float = 0.10  # 10% weight
    price_atr_bonus: float = 0.05         # 5% weight
```

**Core Algorithm** (Lines 150-200):
```python
def apply_filters(self, market_data: pd.DataFrame) -> pd.DataFrame:
    # Apply 7 distinct filters:
    filters = {
        'price_min': df['price'] >= self.criteria.min_price,
        'price_max': df['price'] <= self.criteria.max_price,
        'avg_volume': df['avg_daily_volume'] >= self.criteria.min_avg_volume,
        'pm_volume_ratio': (df['premarket_volume'] / df['avg_daily_volume']) >= self.criteria.min_premarket_volume_ratio,
        'dollar_volume': df['dollar_volume'] >= self.criteria.min_dollar_volume,
        'atr_min': df['atr'] >= self.criteria.min_atr,
        'atr_percent': df['atr_percent'] >= self.criteria.min_atr_percent
    }
```

**Interest Score Calculation** (Lines 200-250):
- **Pre-market Volume Ratio**: Normalized to 100-point scale
- **ATR Percentage**: Volatility measure as percentage of price
- **Dollar Volume Score**: Normalized against $5M baseline
- **Pre-market Volume Absolute**: Log-scale normalization
- **Price-ATR Sweet Spot Bonus**: 100 points for 2-5% ATR range

**Connections**:
- Used by `sp500_bridge.py` as the core filtering engine
- Tested by `test_integration.py` with sample data
- Referenced in example usage functions

### 3. **sp500_bridge.py** - Data Coordination Layer
**Purpose**: Orchestrates data fetching from Polygon and coordinates with the market filter.

**Key Features**:
- **Parallel Data Fetching**: ThreadPoolExecutor for efficient API calls
- **Market Date Calculation**: Handles pre-market sessions and historical data
- **ATR Calculation**: 14-period Average True Range computation
- **Volume Analysis**: Pre-market vs. average daily volume ratios
- **Error Handling**: Graceful failure handling for individual tickers

**Important Methods**:
```python
# Lines 80-120: Main scan orchestration
def run_morning_scan(self, scan_time=None, lookback_days=14, progress_callback=None):
    # 1. Calculate market dates
    market_dates = self._get_market_dates(scan_time, lookback_days)
    
    # 2. Fetch data for all tickers in parallel
    market_data = self._fetch_all_ticker_data(market_dates, scan_time, progress_callback)
    
    # 3. Apply filters and calculate scores
    filtered_data = self.filter_engine.apply_filters(market_data)
    
    # 4. Rank by interest score
    ranked_data = self.filter_engine.rank_by_interest(filtered_data)

# Lines 230-290: Individual ticker data fetching
def _fetch_single_ticker_data(self, ticker, market_dates, scan_time):
    # Fetch historical data for ATR calculation
    historical_df = self._fetch_historical_data(ticker, market_dates['history_start'], market_dates['history_end'])
    
    # Calculate metrics
    atr = self._calculate_atr(historical_df)
    avg_volume = self._calculate_avg_volume(historical_df)
    
    # Fetch pre-market volume
    premarket_volume = self._fetch_premarket_volume(ticker, market_dates['premarket_start'], market_dates['premarket_end'])
```

**Data Flow**:
1. **Input**: S&P 500 ticker list from `sp500_tickers.py`
2. **Fetch**: Historical and pre-market data from Polygon
3. **Calculate**: ATR, average volume, and derived metrics
4. **Filter**: Apply criteria using `MarketFilter`
5. **Rank**: Sort by interest score
6. **Output**: Ranked DataFrame with scores and metadata

**Connections**:
- Imports `MarketFilter` from `market_filter.py`
- Uses `get_sp500_tickers()` from `sp500_tickers.py`
- Integrates with Polygon's `DataFetcher` class
- Tested by `test_integration.py`

### 4. **update_sp500_list.py** - Maintenance Utility
**Purpose**: Automated tool for updating the S&P 500 ticker list from Wikipedia.

**Key Features**:
- **Dual Fetching Methods**: Pandas `read_html()` with BeautifulSoup fallback
- **Change Detection**: Compares new list with current list
- **Backup Creation**: Automatic backup of current list before updates
- **Formatted Output**: Generates Python code ready for copy-paste

**Important Functions**:
```python
# Lines 15-50: Primary fetching method
def fetch_current_sp500():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0]
    tickers = sorted(df['Symbol'].str.strip().tolist())
    tickers = [ticker.replace('.', '-') for ticker in tickers]  # Handle special cases

# Lines 80-110: Change detection
def compare_with_current(new_tickers):
    current_set = set(current_tickers)
    new_set = set(new_tickers)
    added = new_set - current_set
    removed = current_set - new_set
    # Report changes...
```

**Connections**:
- Reads current list from `sp500_tickers.py`
- Creates backup files in same directory
- Generates new `sp500_tickers.py` content

### 5. **test_integration.py** - Comprehensive Testing
**Purpose**: Validates the complete system integration and functionality.

**Test Coverage**:
- **Import Testing**: Verifies all modules can be imported
- **Ticker List Validation**: Checks S&P 500 list functionality
- **Market Filter Testing**: Tests filtering with sample data
- **Polygon Connection**: Validates API connectivity
- **Full Integration**: End-to-end system test

**Key Test Functions**:
```python
# Lines 25-60: Import validation
def test_imports():
    # Tests all local and external imports
    from sp500_tickers import get_sp500_tickers, check_update_status
    from market_filter import MarketFilter, FilterCriteria
    from sp500_bridge import SP500Bridge
    from polygon import DataFetcher

# Lines 100-150: Market filter testing
def test_market_filter():
    # Creates sample data and tests filtering logic
    sample_data = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'GOOGL', ...],
        'price': [150.0, 300.0, 120.0, ...],
        # ... other metrics
    })
    filter_engine = MarketFilter(criteria=criteria)
    filtered = filter_engine.apply_filters(sample_data)
```

**Connections**:
- Tests all components in the system
- Validates integration with Polygon API
- Provides example usage patterns

## System Architecture

### Data Flow Architecture
```
S&P 500 Tickers (sp500_tickers.py)
           ↓
Polygon API (via sp500_bridge.py)
           ↓
Market Filter Engine (market_filter.py)
           ↓
Ranked Results (DataFrame)
```

### Component Dependencies
```
sp500_bridge.py
├── market_filter.py (MarketFilter, FilterCriteria)
├── sp500_tickers.py (get_sp500_tickers)
└── polygon/ (DataFetcher, PolygonConfig)

test_integration.py
├── sp500_bridge.py
├── market_filter.py
├── sp500_tickers.py
└── polygon/

update_sp500_list.py
└── sp500_tickers.py (for comparison)
```

## Key Algorithms and Logic

### 1. **Interest Score Calculation**
The system uses a weighted composite score with five components:

1. **Pre-market Volume Ratio** (40% weight): `(premarket_volume / avg_daily_volume) * 100`
2. **ATR Percentage** (25% weight): Volatility as percentage of price
3. **Dollar Volume Score** (20% weight): `(dollar_volume / 5M_baseline) * 100`
4. **Pre-market Volume Absolute** (10% weight): Log-scale normalization
5. **Price-ATR Sweet Spot Bonus** (5% weight): 100 points for 2-5% ATR range

### 2. **Filter Criteria**
Seven distinct filters applied sequentially:
- Price range: $5.00 - $500.00
- Minimum average daily volume: 500,000 shares
- Minimum pre-market volume ratio: 10% of average daily volume
- Minimum dollar volume: $5,000,000
- Minimum ATR: $2.00
- Minimum ATR percentage: 10% of price

### 3. **Parallel Data Fetching**
Uses ThreadPoolExecutor with configurable worker count (default: 10) to fetch data for all 503 S&P 500 tickers simultaneously, significantly reducing total scan time.

## Performance Characteristics

### Real Data Test Results
Based on the test output in `temp/market_filter_real_data_20250626_135710.md`:

- **Test Coverage**: 15 major S&P 500 stocks
- **Pass Rate**: 73.3% (11/15 stocks passed filters)
- **Average Interest Score**: 62.12
- **Score Standard Deviation**: 7.66
- **Top Performer**: NVDA with 75.6 score

### Scalability Considerations
- **Parallel Processing**: 10 concurrent workers for API calls
- **Caching**: Polygon data caching enabled by default
- **Error Handling**: Individual ticker failures don't stop the scan
- **Memory Efficiency**: Streaming data processing, not batch loading

## Maintenance and Updates

### Quarterly Maintenance Tasks
1. **Update S&P 500 List**: Run `update_sp500_list.py` to fetch latest constituents
2. **Review Filter Criteria**: Adjust thresholds based on market conditions
3. **Validate API Access**: Ensure Polygon API credentials are current
4. **Performance Testing**: Run `test_integration.py` to validate system health

### Configuration Management
- Filter criteria are configurable via `FilterCriteria` dataclass
- Scoring weights can be adjusted via `InterestScoreWeights`
- Parallel processing can be tuned via `parallel_workers` parameter
- Caching behavior controlled via `cache_enabled` parameter

## Integration Points

### External Dependencies
- **Polygon.io API**: Primary data source for market data
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **BeautifulSoup**: Web scraping for ticker list updates

### Internal Sirius System Integration
- **Polygon Module**: Uses existing `polygon/` directory for data fetching
- **Logging**: Integrated with Python logging system
- **Error Handling**: Consistent with project error handling patterns
- **Configuration**: Follows project configuration patterns

## Usage Patterns

### Basic Usage
```python
from sp500_bridge import SP500Bridge

# Initialize bridge
bridge = SP500Bridge()

# Run morning scan
results = bridge.run_morning_scan()

# Get top 10 stocks
top_stocks = results.head(10)
```

### Custom Configuration
```python
from market_filter import FilterCriteria, InterestScoreWeights
from sp500_bridge import SP500Bridge

# Custom filter criteria
criteria = FilterCriteria(
    min_price=10.0,
    max_price=200.0,
    min_avg_volume=1_000_000
)

# Custom scoring weights
weights = InterestScoreWeights(
    premarket_volume_ratio=0.50,
    atr_percentage=0.30,
    dollar_volume_score=0.20
)

# Initialize with custom settings
bridge = SP500Bridge(
    filter_criteria=criteria,
    score_weights=weights,
    parallel_workers=15
)
```

This system provides a robust, scalable solution for S&P 500 market screening with real-time data integration, sophisticated filtering algorithms, and comprehensive testing coverage. The modular architecture allows for easy maintenance, configuration, and extension of functionality. 