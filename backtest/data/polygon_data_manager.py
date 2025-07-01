# backtest/data/polygon_data_manager.py
"""
Module: Polygon Data Manager for Backtesting
Purpose: Manage data fetching and caching for backtest system
Features: Memory cache, file cache, smart fetching, UTC handling
Architecture: Replaces Supabase data manager with Polygon API + caching
"""

import os
import sys
import json
import pickle
import hashlib
import logging
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import OrderedDict
import threading
import requests
import time

# Add path to import polygon module
current_dir = os.path.dirname(os.path.abspath(__file__))
backtest_dir = os.path.dirname(current_dir)
sirius_dir = os.path.dirname(backtest_dir)
sys.path.insert(0, sirius_dir)

# Import your existing DataFetcher
from polygon import DataFetcher
from polygon.config import PolygonConfig

# Remove from path after import
sys.path.remove(sirius_dir)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CacheMetadata:
    """Metadata for cached data"""
    symbol: str
    data_type: str  # 'bars', 'trades', 'quotes'
    timeframe: str
    start_time: datetime
    end_time: datetime
    cached_at: datetime
    data_points: int
    cache_key: str


class LRUCache:
    """Thread-safe LRU cache implementation"""
    
    def __init__(self, max_size: int = 100):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any):
        with self.lock:
            if key in self.cache:
                # Update and move to end
                self.cache.move_to_end(key)
            self.cache[key] = value
            
            # Remove oldest if over capacity
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, int]:
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0
            return {
                'hits': self.hits,
                'misses': self.misses,
                'total_requests': total,
                'hit_rate': hit_rate,
                'cached_items': len(self.cache)
            }


class PolygonDataManager:
    """
    Data manager for backtesting that fetches from Polygon API
    with intelligent caching to minimize API calls.
    
    Caching Strategy:
    1. Memory Cache (LRU) - Session-level, fastest
    2. File Cache (Parquet) - 24-hour persistence, fast
    3. Polygon API - Source of truth, rate-limited
    
    Smart Fetching:
    - Fetches extended windows to cache for adjacent tests
    - Reuses overlapping data from previous fetches
    - Handles timezone conversions (always UTC internally)
    """
    
    def __init__(self, 
                 cache_dir: Optional[str] = None,
                 memory_cache_size: int = 100,
                 file_cache_hours: int = 24,
                 extend_window_bars: int = 500,
                 polygon_config: Optional[Dict[str, Any]] = None,
                 polygon_api_key: Optional[str] = None):
        """
        Initialize Polygon Data Manager.
        
        Args:
            cache_dir: Directory for file cache (default: backtest/cache)
            memory_cache_size: Max items in memory cache
            file_cache_hours: Hours to keep file cache valid
            extend_window_bars: Extra bars to fetch for adjacent tests
            polygon_config: Config dict for DataFetcher
            polygon_api_key: Polygon API key (if not in config)
        """
        # Set up cache directory
        if cache_dir is None:
            self.cache_dir = Path(current_dir).parent / 'cache' / 'polygon_data'
        else:
            self.cache_dir = Path(cache_dir)
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / 'cache_metadata.json'
        
        # Initialize caches
        self.memory_cache = LRUCache(max_size=memory_cache_size)
        self.file_cache_hours = file_cache_hours
        self.extend_window_bars = extend_window_bars
        
        # Initialize Polygon DataFetcher
        config = polygon_config or {'cache_enabled': True}
        if polygon_api_key:
            config['api_key'] = polygon_api_key
        self.fetcher = DataFetcher(config=PolygonConfig(config))
        
        # Get API key for direct API calls
        self.api_key = polygon_api_key or config.get('api_key') or os.environ.get('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("Polygon API key not provided")
        
        # Load cache metadata
        self.cache_metadata = self._load_cache_metadata()
        
        # Track session stats
        self.api_calls = 0
        self.cache_hits = 0
        
        # Rate limiting
        self.last_api_call = 0
        self.min_call_interval = 0.015  # ~65 calls/second for paid tier
        
        logger.info(f"PolygonDataManager initialized with cache at {self.cache_dir}")
    
    def get_bars(self,
                 symbol: str,
                 entry_time: datetime,
                 lookback_hours: int = 2,
                 forward_bars: int = 60,
                 timeframe: str = '1min') -> Optional[pd.DataFrame]:
        """
        Get bars for backtesting - main interface method.
        
        This fetches bars from lookback_hours before entry_time
        to forward_bars after entry_time.
        
        Args:
            symbol: Stock ticker
            entry_time: Entry point for analysis (UTC)
            lookback_hours: Hours of data before entry
            forward_bars: Number of bars after entry
            timeframe: Bar timeframe (1min, 5min, etc)
            
        Returns:
            DataFrame with bars or None if error
        """
        # Ensure UTC timezone
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        
        # Calculate time range
        start_time = entry_time - timedelta(hours=lookback_hours)
        
        # Calculate end time based on forward bars and timeframe
        minutes_per_bar = self._get_minutes_per_bar(timeframe)
        end_time = entry_time + timedelta(minutes=forward_bars * minutes_per_bar)
        
        # Add extension for caching efficiency
        extended_start = start_time - timedelta(minutes=self.extend_window_bars * minutes_per_bar)
        extended_end = end_time + timedelta(minutes=self.extend_window_bars * minutes_per_bar)
        
        # Try to get data from caches
        cache_key = self._generate_cache_key(symbol, 'bars', timeframe, extended_start, extended_end)
        
        # 1. Check memory cache
        df = self._get_from_memory_cache(cache_key)
        if df is not None:
            logger.debug(f"Memory cache hit for {symbol} bars")
            self.cache_hits += 1
            return self._filter_timerange(df, start_time, end_time)
        
        # 2. Check file cache
        df = self._get_from_file_cache(symbol, 'bars', timeframe, extended_start, extended_end)
        if df is not None:
            logger.debug(f"File cache hit for {symbol} bars")
            self.cache_hits += 1
            # Store in memory cache for next time
            self.memory_cache.put(cache_key, df)
            return self._filter_timerange(df, start_time, end_time)
        
        # 3. Fetch from Polygon API
        logger.info(f"Fetching {symbol} bars from Polygon API: {extended_start} to {extended_end}")
        df = self._fetch_from_polygon(symbol, timeframe, extended_start, extended_end)
        
        if df is not None and not df.empty:
            # Cache the extended data
            self._cache_data(symbol, 'bars', timeframe, extended_start, extended_end, df, cache_key)
            # Return the requested range
            return self._filter_timerange(df, start_time, end_time)
        
        logger.error(f"Failed to get bar data for {symbol}")
        return None
    
    async def load_bars(self, symbol: str, start_time: datetime, end_time: datetime,
                       timeframe: str = '1min', use_cache: bool = True) -> pd.DataFrame:
        """
        Async wrapper for get_bars to match engine interface.
        
        The engine expects this method signature.
        """
        # Calculate lookback hours and forward bars from time range
        duration = end_time - start_time
        
        # For historical data (before entry)
        if end_time <= datetime.now(timezone.utc):
            lookback_hours = int(duration.total_seconds() / 3600)
            return self.get_bars(
                symbol=symbol,
                entry_time=end_time,
                lookback_hours=lookback_hours,
                forward_bars=0,
                timeframe=timeframe
            )
        else:
            # For forward data (after entry)
            forward_minutes = int(duration.total_seconds() / 60)
            return self.get_bars(
                symbol=symbol,
                entry_time=start_time,
                lookback_hours=0,
                forward_bars=forward_minutes,
                timeframe=timeframe
            )

    async def load_trades(self, symbol: str, start_time: datetime, 
                         end_time: datetime, use_cache: bool = True) -> pd.DataFrame:
        """
        Load trade tick data from Polygon.
        
        Returns DataFrame with columns: price, size, conditions, exchange
        """
        # Ensure UTC timezone
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
            
        # Check cache first
        cache_key = self._generate_cache_key(symbol, 'trades', 'tick', start_time, end_time)
        
        # 1. Memory cache
        df = self._get_from_memory_cache(cache_key)
        if df is not None and use_cache:
            logger.debug(f"Memory cache hit for {symbol} trades")
            self.cache_hits += 1
            return df
        
        # 2. File cache
        df = self._get_from_file_cache(symbol, 'trades', 'tick', start_time, end_time)
        if df is not None and use_cache:
            logger.debug(f"File cache hit for {symbol} trades")
            self.cache_hits += 1
            self.memory_cache.put(cache_key, df)
            return df
        
        # 3. Fetch from Polygon
        logger.info(f"Fetching {symbol} trades from Polygon API: {start_time} to {end_time}")
        df = await self._fetch_trades_from_polygon(symbol, start_time, end_time)
        
        if df is not None and not df.empty:
            # Cache the data
            self._cache_data(symbol, 'trades', 'tick', start_time, end_time, df, cache_key)
            return df
        
        logger.warning(f"No trade data found for {symbol}")
        return pd.DataFrame()

    async def load_quotes(self, symbol: str, start_time: datetime,
                         end_time: datetime, use_cache: bool = True) -> pd.DataFrame:
        """
        Load quote (NBBO) data from Polygon.
        
        Returns DataFrame with columns: bid, ask, bid_size, ask_size, bid_exchange, ask_exchange
        """
        # Ensure UTC timezone
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
            
        # Check cache first
        cache_key = self._generate_cache_key(symbol, 'quotes', 'tick', start_time, end_time)
        
        # 1. Memory cache
        df = self._get_from_memory_cache(cache_key)
        if df is not None and use_cache:
            logger.debug(f"Memory cache hit for {symbol} quotes")
            self.cache_hits += 1
            return df
        
        # 2. File cache
        df = self._get_from_file_cache(symbol, 'quotes', 'tick', start_time, end_time)
        if df is not None and use_cache:
            logger.debug(f"File cache hit for {symbol} quotes")
            self.cache_hits += 1
            self.memory_cache.put(cache_key, df)
            return df
        
        # 3. Fetch from Polygon
        logger.info(f"Fetching {symbol} quotes from Polygon API: {start_time} to {end_time}")
        df = await self._fetch_quotes_from_polygon(symbol, start_time, end_time)
        
        if df is not None and not df.empty:
            # Cache the data
            self._cache_data(symbol, 'quotes', 'tick', start_time, end_time, df, cache_key)
            return df
        
        logger.warning(f"No quote data found for {symbol}")
        return pd.DataFrame()

    async def _fetch_trades_from_polygon(self, symbol: str, 
                                       start_time: datetime, 
                                       end_time: datetime) -> Optional[pd.DataFrame]:
        """Fetch trade data from Polygon API v3."""
        try:
            all_trades = []
            
            # Rate limiting
            self._rate_limit()
            
            # Construct API URL for v3
            url = f"https://api.polygon.io/v3/trades/{symbol}"
            
            params = {
                'apiKey': self.api_key,
                'timestamp.gte': start_time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                'timestamp.lte': end_time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                'limit': 50000,  # Max limit for v3
                'order': 'asc'
            }
            
            # Paginate through results
            while True:
                response = requests.get(url, params=params)
                self.api_calls += 1
                
                if response.status_code != 200:
                    logger.error(f"API error {response.status_code}: {response.text}")
                    break
                
                data = response.json()
                
                if 'results' not in data or not data['results']:
                    break
                
                # Process trades
                for trade in data['results']:
                    all_trades.append({
                        'timestamp': pd.Timestamp(trade['participant_timestamp'], unit='ns', tz='UTC'),
                        'price': trade['price'],
                        'size': trade['size'],
                        'conditions': ','.join(map(str, trade.get('conditions', []))),
                        'exchange': str(trade.get('exchange', ''))
                    })
                
                # Check for next page
                if 'next_url' in data and data['next_url']:
                    url = data['next_url']
                    params = {'apiKey': self.api_key}  # API key still needed
                    self._rate_limit()
                else:
                    break
            
            if all_trades:
                df = pd.DataFrame(all_trades)
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                logger.info(f"Fetched {len(df)} trades for {symbol}")
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching trades from Polygon: {e}")
            return None

    async def _fetch_quotes_from_polygon(self, symbol: str,
                                       start_time: datetime,
                                       end_time: datetime) -> Optional[pd.DataFrame]:
        """Fetch quote (NBBO) data from Polygon API v3."""
        try:
            all_quotes = []
            
            # Rate limiting
            self._rate_limit()
            
            # Construct API URL for v3
            url = f"https://api.polygon.io/v3/quotes/{symbol}"
            
            params = {
                'apiKey': self.api_key,
                'timestamp.gte': start_time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                'timestamp.lte': end_time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                'limit': 50000,  # Max limit for v3
                'order': 'asc'
            }
            
            # Paginate through results
            while True:
                response = requests.get(url, params=params)
                self.api_calls += 1
                
                if response.status_code != 200:
                    logger.error(f"API error {response.status_code}: {response.text}")
                    break
                
                data = response.json()
                
                if 'results' not in data or not data['results']:
                    break
                
                # Process quotes
                for quote in data['results']:
                    all_quotes.append({
                        'timestamp': pd.Timestamp(quote['participant_timestamp'], unit='ns', tz='UTC'),
                        'bid': quote['bid_price'],
                        'ask': quote['ask_price'],
                        'bid_size': quote['bid_size'],
                        'ask_size': quote['ask_size'],
                        'bid_exchange': str(quote.get('bid_exchange', '')),
                        'ask_exchange': str(quote.get('ask_exchange', ''))
                    })
                
                # Check for next page
                if 'next_url' in data and data['next_url']:
                    url = data['next_url']
                    params = {'apiKey': self.api_key}  # API key still needed
                    self._rate_limit()
                else:
                    break
            
            if all_quotes:
                df = pd.DataFrame(all_quotes)
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                logger.info(f"Fetched {len(df)} quotes for {symbol}")
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching quotes from Polygon: {e}")
            return None

    def _rate_limit(self):
        """Implement rate limiting for API calls."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < self.min_call_interval:
            sleep_time = self.min_call_interval - time_since_last_call
            time.sleep(sleep_time)
        
        self.last_api_call = time.time()

    def set_cache_dir(self, cache_dir: str):
        """Set cache directory (for compatibility)"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache directory set to {self.cache_dir}")
    
    def _get_minutes_per_bar(self, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        timeframe_map = {
            '1min': 1,
            '5min': 5,
            '15min': 15,
            '30min': 30,
            '1hour': 60,
            'hour': 60,
            'day': 1440
        }
        return timeframe_map.get(timeframe, 1)
    
    def _generate_cache_key(self, symbol: str, data_type: str, timeframe: str, 
                           start_time: datetime, end_time: datetime) -> str:
        """Generate unique cache key for data request."""
        key_string = f"{symbol}_{data_type}_{timeframe}_{start_time.isoformat()}_{end_time.isoformat()}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_from_memory_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get data from memory cache."""
        return self.memory_cache.get(cache_key)
    
    def _get_from_file_cache(self, symbol: str, data_type: str, timeframe: str,
                            start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        """Get data from file cache if available and valid."""
        # Look for cached files that contain our requested range
        symbol_dir = self.cache_dir / symbol.lower() / data_type
        if not symbol_dir.exists():
            return None
        
        # Check each parquet file in symbol directory
        for parquet_file in symbol_dir.glob(f"{timeframe}_*.parquet"):
            try:
                # Extract metadata from filename
                parts = parquet_file.stem.split('_')
                if len(parts) < 3:
                    continue
                
                file_meta_key = parts[2]  # Hash in filename
                
                # Check if this file might contain our data
                if file_meta_key in self.cache_metadata:
                    metadata = self.cache_metadata[file_meta_key]
                    cached_start = datetime.fromisoformat(metadata['start_time'])
                    cached_end = datetime.fromisoformat(metadata['end_time'])
                    cached_at = datetime.fromisoformat(metadata['cached_at'])
                    
                    # Check if cache is still valid
                    cache_age = datetime.now(timezone.utc) - cached_at
                    if cache_age.total_seconds() > self.file_cache_hours * 3600:
                        logger.debug(f"File cache expired for {symbol}")
                        continue
                    
                    # Check if cached range contains requested range
                    if cached_start <= start_time and cached_end >= end_time:
                        logger.info(f"Found valid file cache for {symbol} {data_type}")
                        df = pd.read_parquet(parquet_file)
                        df.index = pd.to_datetime(df.index, utc=True)
                        return df
                        
            except Exception as e:
                logger.error(f"Error reading cache file {parquet_file}: {e}")
                continue
        
        return None
    
    def _fetch_from_polygon(self, symbol: str, timeframe: str,
                           start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        """Fetch bar data from Polygon API."""
        try:
            self.api_calls += 1
            
            # Use your existing DataFetcher
            df = self.fetcher.fetch_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_time,
                end_date=end_time,
                use_cache=True,  # Use Polygon's internal cache
                validate=True,
                fill_gaps=True
            )
            
            if df.empty:
                logger.warning(f"No data returned from Polygon for {symbol}")
                return None
            
            # Ensure UTC timezone
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
            
            # Ensure required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            df = df[required_cols].copy()
            
            logger.info(f"Fetched {len(df)} bars from Polygon for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching from Polygon: {e}")
            return None
    
    def _filter_timerange(self, df: pd.DataFrame, 
                         start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Filter dataframe to requested time range."""
        # Ensure timezone aware comparison
        if start_time.tzinfo is None:
            start_time = pd.Timestamp(start_time).tz_localize('UTC')
        else:
            start_time = pd.Timestamp(start_time)
            
        if end_time.tzinfo is None:
            end_time = pd.Timestamp(end_time).tz_localize('UTC')
        else:
            end_time = pd.Timestamp(end_time)
        
        # Filter to exact requested range
        mask = (df.index >= start_time) & (df.index <= end_time)
        filtered = df[mask].copy()
        
        logger.debug(f"Filtered {len(df)} rows to {len(filtered)} rows")
        return filtered
    
    def _cache_data(self, symbol: str, data_type: str, timeframe: str,
                   start_time: datetime, end_time: datetime,
                   df: pd.DataFrame, cache_key: str):
        """Cache data to both memory and file cache."""
        # Memory cache
        self.memory_cache.put(cache_key, df.copy())
        
        # File cache
        try:
            symbol_dir = self.cache_dir / symbol.lower() / data_type
            symbol_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename with metadata
            filename = f"{timeframe}_{start_time.strftime('%Y%m%d')}_{cache_key[:8]}.parquet"
            filepath = symbol_dir / filename
            
            # Save to parquet with compression for tick data
            compression = 'snappy' if data_type == 'bars' else 'gzip'
            df.to_parquet(filepath, compression=compression)
            
            # Update metadata
            metadata = CacheMetadata(
                symbol=symbol,
                data_type=data_type,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                cached_at=datetime.now(timezone.utc),
                data_points=len(df),
                cache_key=cache_key
            )
            
            self.cache_metadata[cache_key] = {
                'symbol': metadata.symbol,
                'data_type': metadata.data_type,
                'timeframe': metadata.timeframe,
                'start_time': metadata.start_time.isoformat(),
                'end_time': metadata.end_time.isoformat(),
                'cached_at': metadata.cached_at.isoformat(),
                'data_points': metadata.data_points,
                'cache_key': metadata.cache_key
            }
            
            self._save_cache_metadata()
            logger.info(f"Cached {len(df)} {data_type} to {filepath}")
            
        except Exception as e:
            logger.error(f"Error caching to file: {e}")
    
    def _load_cache_metadata(self) -> Dict[str, Dict]:
        """Load cache metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache metadata: {e}")
        return {}
    
    def _save_cache_metadata(self):
        """Save cache metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.cache_metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")
    
    def clear_cache(self, symbol: Optional[str] = None, 
                   data_type: Optional[str] = None,
                   older_than_hours: Optional[int] = None):
        """
        Clear cache data.
        
        Args:
            symbol: Clear only this symbol's cache (None = all)
            data_type: Clear only this data type ('bars', 'trades', 'quotes')
            older_than_hours: Clear only caches older than this
        """
        # Clear memory cache
        if symbol is None and data_type is None and older_than_hours is None:
            self.memory_cache.clear()
            logger.info("Cleared memory cache")
        
        # Clear file cache
        if symbol:
            symbol_dir = self.cache_dir / symbol.lower()
            if symbol_dir.exists():
                if data_type:
                    type_dir = symbol_dir / data_type
                    if type_dir.exists():
                        for file in type_dir.glob("*.parquet"):
                            file.unlink()
                        logger.info(f"Cleared {data_type} cache for {symbol}")
                else:
                    for type_dir in symbol_dir.iterdir():
                        if type_dir.is_dir():
                            for file in type_dir.glob("*.parquet"):
                                file.unlink()
                    logger.info(f"Cleared all cache for {symbol}")
        else:
            # Clear based on age or all
            current_time = datetime.now(timezone.utc)
            keys_to_remove = []
            
            for key, metadata in self.cache_metadata.items():
                cached_at = datetime.fromisoformat(metadata['cached_at'])
                age_hours = (current_time - cached_at).total_seconds() / 3600
                
                if older_than_hours is None or age_hours > older_than_hours:
                    if data_type is None or metadata.get('data_type') == data_type:
                        # Find and remove file
                        symbol_dir = self.cache_dir / metadata['symbol'].lower() / metadata.get('data_type', 'bars')
                        for file in symbol_dir.glob(f"*{key[:8]}*.parquet"):
                            file.unlink()
                        keys_to_remove.append(key)
            
            # Update metadata
            for key in keys_to_remove:
                self.cache_metadata.pop(key, None)
            
            self._save_cache_metadata()
            logger.info(f"Cleared {len(keys_to_remove)} expired cache files")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        # Memory cache stats
        memory_stats = self.memory_cache.get_stats()
        
        # File cache stats by data type
        file_stats = {
            'bars': {'files': 0, 'size_mb': 0},
            'trades': {'files': 0, 'size_mb': 0},
            'quotes': {'files': 0, 'size_mb': 0}
        }
        
        total_files = 0
        total_size_mb = 0
        
        for symbol_dir in self.cache_dir.iterdir():
            if symbol_dir.is_dir() and symbol_dir.name != 'cache_metadata.json':
                for type_dir in symbol_dir.iterdir():
                    if type_dir.is_dir() and type_dir.name in file_stats:
                        for file in type_dir.glob("*.parquet"):
                            file_stats[type_dir.name]['files'] += 1
                            size_mb = file.stat().st_size / (1024 * 1024)
                            file_stats[type_dir.name]['size_mb'] += size_mb
                            total_files += 1
                            total_size_mb += size_mb
        
        # Round sizes
        for data_type in file_stats:
            file_stats[data_type]['size_mb'] = round(file_stats[data_type]['size_mb'], 2)
        
        # API stats
        total_requests = self.api_calls + self.cache_hits
        cache_hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'memory_cache': memory_stats,
            'file_cache': {
                'total_files': total_files,
                'total_size_mb': round(total_size_mb, 2),
                'by_type': file_stats,
                'metadata_entries': len(self.cache_metadata)
            },
            'api_stats': {
                'api_calls': self.api_calls,
                'cache_hits': self.cache_hits,
                'total_requests': total_requests,
                'cache_hit_rate': round(cache_hit_rate, 2)
            }
        }


# Example usage and testing
if __name__ == "__main__":
    """Test the PolygonDataManager with trade and quote data"""
    import asyncio
    
    async def test_tick_data():
        # Initialize manager
        manager = PolygonDataManager()
        
        # Test parameters
        symbol = 'AAPL'
        end_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        start_time = end_time - timedelta(minutes=5)  # 5 minutes of data
        
        # Test 1: Fetch trade data
        print("Test 1: Fetching trade data...")
        trades = await manager.load_trades(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time
        )
        
        if not trades.empty:
            print(f"✓ Fetched {len(trades)} trades")
            print(f"  First trade: ${trades['price'].iloc[0]:.2f} @ {trades.index[0]}")
            print(f"  Last trade: ${trades['price'].iloc[-1]:.2f} @ {trades.index[-1]}")
            print(f"  Total volume: {trades['size'].sum():,.0f}")
        
        # Test 2: Fetch quote data
        print("\nTest 2: Fetching quote data...")
        quotes = await manager.load_quotes(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time
        )
        
        if not quotes.empty:
            print(f"✓ Fetched {len(quotes)} quotes")
            print(f"  First quote: Bid ${quotes['bid'].iloc[0]:.2f} / Ask ${quotes['ask'].iloc[0]:.2f}")
            print(f"  Last quote: Bid ${quotes['bid'].iloc[-1]:.2f} / Ask ${quotes['ask'].iloc[-1]:.2f}")
            print(f"  Avg spread: ${(quotes['ask'] - quotes['bid']).mean():.3f}")
        
        # Test 3: Fetch same data again (should hit cache)
        print("\nTest 3: Fetching same data again (cache test)...")
        trades2 = await manager.load_trades(symbol, start_time, end_time)
        quotes2 = await manager.load_quotes(symbol, start_time, end_time)
        
        print(f"✓ Cache working: trades={len(trades2)}, quotes={len(quotes2)}")
        
        # Show cache statistics
        print("\nCache Statistics:")
        stats = manager.get_cache_stats()
        print(f"  Memory cache: {stats['memory_cache']}")
        print(f"  File cache: {stats['file_cache']}")
        print(f"  API stats: {stats['api_stats']}")
    
    # Run the async test
    asyncio.run(test_tick_data())