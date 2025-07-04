# backtest/data/debug/test_utils.py
"""
Shared utilities for testing data modules
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Any
from backtest.data.polygon_data_manager import PolygonDataManager


def parse_datetime(dt_str: str) -> datetime:
    """
    Parse datetime string to timezone-aware datetime.
    
    Args:
        dt_str: Datetime string in format 'YYYY-MM-DD HH:MM:SS'
        
    Returns:
        Timezone-aware datetime (UTC)
    """
    # Try parsing with common formats
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%Y-%m-%d',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%SZ'
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(dt_str, fmt)
            # Make timezone aware (UTC)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    
    raise ValueError(f"Could not parse datetime: {dt_str}")


def create_sample_bars(num_bars: int, 
                      base_price: float = 100.0,
                      start_time: Optional[datetime] = None,
                      timeframe: str = '1min') -> pd.DataFrame:
    """
    Create sample OHLCV bar data for testing.
    
    Args:
        num_bars: Number of bars to create
        base_price: Starting price
        start_time: Starting timestamp (default: now)
        timeframe: Bar timeframe ('1min', '5min', etc.)
        
    Returns:
        DataFrame with OHLCV data indexed by timestamp
    """
    if start_time is None:
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
    
    # Determine frequency
    freq_map = {
        '1min': 'min',
        '5min': '5min',
        '15min': '15min',
        '30min': '30min',
        '1hour': 'H',
        '1day': 'D'
    }
    freq = freq_map.get(timeframe, 'min')
    
    # Generate timestamps
    timestamps = pd.date_range(start=start_time, periods=num_bars, freq=freq, tz='UTC')
    
    # Generate price data with some randomness
    prices = []
    current_price = base_price
    
    for _ in range(num_bars):
        # Random walk
        change = np.random.normal(0, 0.002) * current_price  # 0.2% std dev
        current_price += change
        
        # Generate OHLC
        open_price = current_price
        high_price = open_price + abs(np.random.normal(0, 0.001)) * open_price
        low_price = open_price - abs(np.random.normal(0, 0.001)) * open_price
        close_price = np.random.uniform(low_price, high_price)
        
        # Volume with some pattern
        base_volume = 10000
        volume = int(base_volume * (1 + abs(np.random.normal(0, 0.5))))
        
        prices.append({
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
        
        current_price = close_price
    
    # Create DataFrame
    df = pd.DataFrame(prices, index=timestamps)
    return df


def create_sample_trades(num_trades: int,
                        base_price: float = 100.0,
                        start_time: Optional[datetime] = None) -> pd.DataFrame:
    """
    Create sample trade tick data for testing.
    
    Args:
        num_trades: Number of trades to create
        base_price: Starting price
        start_time: Starting timestamp
        
    Returns:
        DataFrame with trade data indexed by timestamp
    """
    if start_time is None:
        start_time = datetime.now(timezone.utc) - timedelta(minutes=30)
    
    trades = []
    current_time = start_time
    current_price = base_price
    
    for _ in range(num_trades):
        # Random time increment (0.1 to 5 seconds)
        current_time += timedelta(seconds=np.random.uniform(0.1, 5))
        
        # Price movement
        change = np.random.normal(0, 0.001) * current_price
        current_price += change
        
        # Trade size (log-normal distribution)
        size = int(np.random.lognormal(np.log(100), 1))
        
        # Exchange (weighted random)
        exchanges = ['A', 'B', 'C', 'D', 'K', 'N', 'P', 'Q', 'V', 'X', 'Y', 'Z']
        exchange = np.random.choice(exchanges, p=[0.3, 0.2, 0.15, 0.1] + [0.025]*8)
        
        trades.append({
            'timestamp': current_time,
            'price': round(current_price, 2),
            'size': size,
            'exchange': exchange,
            'conditions': []
        })
    
    df = pd.DataFrame(trades)
    df.set_index('timestamp', inplace=True)
    return df

def print_dataframe_summary(df: pd.DataFrame, label: str = "DataFrame"):
    """Print a summary of a dataframe"""
    if df.empty:
        print(f"{label}: Empty DataFrame")
    else:
        print(f"{label}: {len(df)} rows")
        print(f"  Time range: {df.index.min()} to {df.index.max()}")
        if 'close' in df.columns:
            print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        if 'volume' in df.columns:
            print(f"  Total volume: {df['volume'].sum():,}")
def create_sample_quotes(num_quotes: int,
                        base_price: float = 100.0,
                        spread: float = 0.02,
                        start_time: Optional[datetime] = None) -> pd.DataFrame:
    """
    Create sample quote (NBBO) data for testing.
    
    Args:
        num_quotes: Number of quotes to create
        base_price: Starting mid price
        spread: Typical bid-ask spread
        start_time: Starting timestamp
        
    Returns:
        DataFrame with quote data indexed by timestamp
    """
    if start_time is None:
        start_time = datetime.now(timezone.utc) - timedelta(minutes=30)
    
    quotes = []
    current_time = start_time
    current_mid = base_price
    
    for _ in range(num_quotes):
        # Random time increment (0.01 to 1 second)
        current_time += timedelta(seconds=np.random.uniform(0.01, 1))
        
        # Mid price movement
        change = np.random.normal(0, 0.0005) * current_mid
        current_mid += change
        
        # Spread variation
        current_spread = spread * (1 + np.random.normal(0, 0.2))
        current_spread = max(0.01, current_spread)  # Minimum spread
        
        # Bid/Ask
        bid = round(current_mid - current_spread/2, 2)
        ask = round(current_mid + current_spread/2, 2)
        
        # Sizes (typically larger at round numbers)
        is_round = (bid * 100) % 100 == 0
        size_multiplier = 2 if is_round else 1
        
        bid_size = int(np.random.lognormal(np.log(100 * size_multiplier), 0.5))
        ask_size = int(np.random.lognormal(np.log(100 * size_multiplier), 0.5))
        
        quotes.append({
            'timestamp': current_time,
            'bid': bid,
            'ask': ask,
            'bid_size': bid_size,
            'ask_size': ask_size
        })
    
    df = pd.DataFrame(quotes)
    df.set_index('timestamp', inplace=True)
    return df


def create_sample_data_with_issues(data_type: str = 'bars') -> pd.DataFrame:
    """
    Create sample data with various quality issues for testing validation.
    
    Args:
        data_type: Type of data ('bars', 'trades', 'quotes')
        
    Returns:
        DataFrame with intentional data quality issues
    """
    if data_type == 'bars':
        # Start with good data
        df = create_sample_bars(100)
        
        # Add issues
        # 1. Gap in timestamps
        df = df.drop(df.index[20:25])
        
        # 2. Invalid OHLC (high < low)
        df.iloc[30, df.columns.get_loc('high')] = df.iloc[30]['low'] - 1
        
        # 3. Zero volume
        df.iloc[40:43, df.columns.get_loc('volume')] = 0
        
        # 4. Extreme price spike
        df.iloc[50, df.columns.get_loc('close')] = df.iloc[49]['close'] * 1.5
        
        # 5. Negative price
        df.iloc[60, df.columns.get_loc('low')] = -1
        
        return df
    
    elif data_type == 'trades':
        df = create_sample_trades(100)
        
        # Add issues
        # 1. Out of order timestamps
        idx = df.index.to_list()
        idx[20], idx[21] = idx[21], idx[20]
        df.index = idx
        
        # 2. Zero size trades
        df.iloc[30:33, df.columns.get_loc('size')] = 0
        
        # 3. Extreme price
        df.iloc[40, df.columns.get_loc('price')] = df.iloc[39]['price'] * 10
        
        return df
    
    elif data_type == 'quotes':
        df = create_sample_quotes(100)
        
        # Add issues
        # 1. Crossed quotes (bid > ask)
        df.iloc[20, df.columns.get_loc('bid')] = df.iloc[20]['ask'] + 0.05
        
        # 2. Zero spread (locked market)
        df.iloc[30:35, df.columns.get_loc('bid')] = df.iloc[30:35]['ask']
        
        # 3. Wide spread
        df.iloc[40, df.columns.get_loc('ask')] = df.iloc[40]['bid'] * 1.1
        
        # 4. Zero sizes
        df.iloc[50:52, df.columns.get_loc('bid_size')] = 0
        
        return df
    
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, 
                      name1: str = "DataFrame 1", 
                      name2: str = "DataFrame 2") -> Dict[str, Any]:
    """
    Compare two dataframes and return differences.
    
    Args:
        df1, df2: DataFrames to compare
        name1, name2: Names for display
        
    Returns:
        Dict with comparison results
    """
    results = {
        'identical': False,
        'shape_match': df1.shape == df2.shape,
        'columns_match': list(df1.columns) == list(df2.columns),
        'index_match': df1.index.equals(df2.index),
        'differences': []
    }
    
    results['shape'] = {
        name1: df1.shape,
        name2: df2.shape
    }
    
    if results['columns_match'] and results['index_match'] and results['shape_match']:
        # Compare values
        diff_mask = (df1 != df2).any(axis=1)
        if not diff_mask.any():
            results['identical'] = True
        else:
            results['differences'] = df1[diff_mask].index.tolist()
            results['num_differences'] = diff_mask.sum()
    
    return results


def generate_market_hours_mask(start_time: datetime, end_time: datetime, 
                              timezone_str: str = 'America/New_York') -> pd.Series:
    """
    Generate a boolean mask for market hours.
    
    Args:
        start_time: Start of period
        end_time: End of period
        timezone_str: Market timezone
        
    Returns:
        Boolean Series indexed by minute
    """
    # Generate minute-level index
    idx = pd.date_range(start=start_time, end=end_time, freq='min', tz='UTC')
    
    # Convert to market timezone
    market_tz = idx.tz_convert(timezone_str)
    
    # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
    mask = (
        (market_tz.hour >= 9) & 
        (market_tz.hour < 16) &
        (market_tz.weekday < 5)  # Monday = 0, Friday = 4
    )
    
    # Handle 9:30 AM start
    mask = mask & ~((market_tz.hour == 9) & (market_tz.minute < 30))
    
    return pd.Series(mask, index=idx)


def calculate_data_stats(df: pd.DataFrame, data_type: str = 'bars') -> Dict[str, Any]:
    """
    Calculate statistics for a dataset.
    
    Args:
        df: DataFrame to analyze
        data_type: Type of data
        
    Returns:
        Dict with statistics
    """
    stats = {
        'count': len(df),
        'time_range': {
            'start': df.index.min(),
            'end': df.index.max(),
            'duration_minutes': (df.index.max() - df.index.min()).total_seconds() / 60
        }
    }
    
    if data_type == 'bars' and 'close' in df.columns:
        stats['price'] = {
            'mean': df['close'].mean(),
            'std': df['close'].std(),
            'min': df['close'].min(),
            'max': df['close'].max(),
            'range': df['close'].max() - df['close'].min()
        }
        
        if 'volume' in df.columns:
            stats['volume'] = {
                'total': df['volume'].sum(),
                'mean': df['volume'].mean(),
                'max': df['volume'].max()
            }
    
    elif data_type == 'trades' and 'price' in df.columns:
        stats['price'] = {
            'mean': df['price'].mean(),
            'std': df['price'].std(),
            'min': df['price'].min(),
            'max': df['price'].max()
        }
        
        if 'size' in df.columns:
            stats['size'] = {
                'total': df['size'].sum(),
                'mean': df['size'].mean(),
                'median': df['size'].median()
            }
    
    elif data_type == 'quotes' and all(col in df.columns for col in ['bid', 'ask']):
        spreads = df['ask'] - df['bid']
        stats['spread'] = {
            'mean': spreads.mean(),
            'std': spreads.std(),
            'min': spreads.min(),
            'max': spreads.max(),
            'mean_pct': (spreads / df['bid'] * 100).mean()
        }
    
    return stats


# Mock data manager for testing without API
class MockPolygonDataManager:
    """Mock PolygonDataManager for testing"""
    
    def __init__(self):
        self.call_count = 0
        self.memory_cache = {}
        self.file_cache = {}
    
    async def load_bars(self, symbol: str, start_time: datetime, 
                       end_time: datetime, timeframe: str = '1min') -> pd.DataFrame:
        """Mock load_bars method"""
        self.call_count += 1
        num_bars = int((end_time - start_time).total_seconds() / 60)
        return create_sample_bars(num_bars, start_time=start_time, timeframe=timeframe)
    
    async def load_trades(self, symbol: str, start_time: datetime, 
                         end_time: datetime) -> pd.DataFrame:
        """Mock load_trades method"""
        self.call_count += 1
        return create_sample_trades(100, start_time=start_time)
    
    async def load_quotes(self, symbol: str, start_time: datetime, 
                         end_time: datetime) -> pd.DataFrame:
        """Mock load_quotes method"""
        self.call_count += 1
        return create_sample_quotes(100, start_time=start_time)
    
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """Mock cache key generation"""
        return f"mock_key_{self.call_count}"
    
    def set_current_plugin(self, plugin_name: str):
        """Mock plugin setting"""
        pass
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Mock cache stats"""
        return {
            'memory_cache': {'hit_rate': 0.0, 'size': 0},
            'file_cache': {'total_size_mb': 0.0},
            'api_stats': {'api_calls': self.call_count}
        }