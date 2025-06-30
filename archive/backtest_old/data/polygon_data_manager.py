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
                 polygon_config: Optional[Dict[str, Any]] = None):
        """
        Initialize Polygon Data Manager.
        
        Args:
            cache_dir: Directory for file cache (default: backtest/cache)
            memory_cache_size: Max items in memory cache
            file_cache_hours: Hours to keep file cache valid
            extend_window_bars: Extra bars to fetch for adjacent tests
            polygon_config: Config dict for DataFetcher
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
        self.fetcher = DataFetcher(config=PolygonConfig(config))
        
        # Load cache metadata
        self.cache_metadata = self._load_cache_metadata()
        
        # Track session stats
        self.api_calls = 0
        self.cache_hits = 0
        
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
        cache_key = self._generate_cache_key(symbol, timeframe, extended_start, extended_end)
        
        # 1. Check memory cache
        df = self._get_from_memory_cache(cache_key)
        if df is not None:
            logger.debug(f"Memory cache hit for {symbol}")
            self.cache_hits += 1
            return self._filter_timerange(df, start_time, end_time)
        
        # 2. Check file cache
        df = self._get_from_file_cache(symbol, timeframe, extended_start, extended_end)
        if df is not None:
            logger.debug(f"File cache hit for {symbol}")
            self.cache_hits += 1
            # Store in memory cache for next time
            self.memory_cache.put(cache_key, df)
            return self._filter_timerange(df, start_time, end_time)
        
        # 3. Fetch from Polygon API
        logger.info(f"Fetching {symbol} from Polygon API: {extended_start} to {extended_end}")
        df = self._fetch_from_polygon(symbol, timeframe, extended_start, extended_end)
        
        if df is not None and not df.empty:
            # Cache the extended data
            self._cache_data(symbol, timeframe, extended_start, extended_end, df, cache_key)
            # Return the requested range
            return self._filter_timerange(df, start_time, end_time)
        
        logger.error(f"Failed to get data for {symbol}")
        return None
    
    # Async wrapper methods for engine compatibility
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
        Placeholder for trade data loading.
        Currently not implemented in Polygon data manager.
        """
        logger.warning("Trade data loading not implemented in PolygonDataManager")
        return pd.DataFrame()

    async def load_quotes(self, symbol: str, start_time: datetime,
                         end_time: datetime, use_cache: bool = True) -> pd.DataFrame:
        """
        Placeholder for quote data loading.
        Currently not implemented in Polygon data manager.
        """
        logger.warning("Quote data loading not implemented in PolygonDataManager")
        return pd.DataFrame()

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
    
    def _generate_cache_key(self, symbol: str, timeframe: str, 
                           start_time: datetime, end_time: datetime) -> str:
        """Generate unique cache key for data request."""
        key_string = f"{symbol}_{timeframe}_{start_time.isoformat()}_{end_time.isoformat()}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_from_memory_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get data from memory cache."""
        return self.memory_cache.get(cache_key)
    
    def _get_from_file_cache(self, symbol: str, timeframe: str,
                            start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        """Get data from file cache if available and valid."""
        # Look for cached files that contain our requested range
        symbol_dir = self.cache_dir / symbol.lower()
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
                        logger.info(f"Found valid file cache for {symbol}")
                        df = pd.read_parquet(parquet_file)
                        df.index = pd.to_datetime(df.index, utc=True)
                        return df
                        
            except Exception as e:
                logger.error(f"Error reading cache file {parquet_file}: {e}")
                continue
        
        return None
    
    def _fetch_from_polygon(self, symbol: str, timeframe: str,
                           start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        """Fetch data from Polygon API."""
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
        
        logger.debug(f"Filtered {len(df)} bars to {len(filtered)} bars")
        return filtered
    
    def _cache_data(self, symbol: str, timeframe: str,
                   start_time: datetime, end_time: datetime,
                   df: pd.DataFrame, cache_key: str):
        """Cache data to both memory and file cache."""
        # Memory cache
        self.memory_cache.put(cache_key, df.copy())
        
        # File cache
        try:
            symbol_dir = self.cache_dir / symbol.lower()
            symbol_dir.mkdir(exist_ok=True)
            
            # Create filename with metadata
            filename = f"{timeframe}_{start_time.strftime('%Y%m%d')}_{cache_key[:8]}.parquet"
            filepath = symbol_dir / filename
            
            # Save to parquet
            df.to_parquet(filepath, compression='snappy')
            
            # Update metadata
            metadata = CacheMetadata(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                cached_at=datetime.now(timezone.utc),
                data_points=len(df),
                cache_key=cache_key
            )
            
            self.cache_metadata[cache_key] = {
                'symbol': metadata.symbol,
                'timeframe': metadata.timeframe,
                'start_time': metadata.start_time.isoformat(),
                'end_time': metadata.end_time.isoformat(),
                'cached_at': metadata.cached_at.isoformat(),
                'data_points': metadata.data_points,
                'cache_key': metadata.cache_key
            }
            
            self._save_cache_metadata()
            logger.info(f"Cached {len(df)} bars to {filepath}")
            
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
    
    def clear_cache(self, symbol: Optional[str] = None, older_than_hours: Optional[int] = None):
        """
        Clear cache data.
        
        Args:
            symbol: Clear only this symbol's cache (None = all)
            older_than_hours: Clear only caches older than this
        """
        # Clear memory cache
        if symbol is None and older_than_hours is None:
            self.memory_cache.clear()
            logger.info("Cleared memory cache")
        
        # Clear file cache
        if symbol:
            symbol_dir = self.cache_dir / symbol.lower()
            if symbol_dir.exists():
                for file in symbol_dir.glob("*.parquet"):
                    file.unlink()
                logger.info(f"Cleared file cache for {symbol}")
        else:
            # Clear based on age or all
            current_time = datetime.now(timezone.utc)
            keys_to_remove = []
            
            for key, metadata in self.cache_metadata.items():
                cached_at = datetime.fromisoformat(metadata['cached_at'])
                age_hours = (current_time - cached_at).total_seconds() / 3600
                
                if older_than_hours is None or age_hours > older_than_hours:
                    # Find and remove file
                    symbol_dir = self.cache_dir / metadata['symbol'].lower()
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
        
        # File cache stats
        total_files = 0
        total_size_mb = 0
        
        for symbol_dir in self.cache_dir.iterdir():
            if symbol_dir.is_dir():
                for file in symbol_dir.glob("*.parquet"):
                    total_files += 1
                    total_size_mb += file.stat().st_size / (1024 * 1024)
        
        # API stats
        total_requests = self.api_calls + self.cache_hits
        cache_hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'memory_cache': memory_stats,
            'file_cache': {
                'total_files': total_files,
                'total_size_mb': round(total_size_mb, 2),
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
    """Test the PolygonDataManager"""
    
    # Initialize manager
    manager = PolygonDataManager()
    
    # Test case 1: Fetch data for AAPL
    print("Test 1: Fetching AAPL data...")
    entry_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    
    bars = manager.get_bars(
        symbol='AAPL',
        entry_time=entry_time,
        lookback_hours=2,
        forward_bars=60,
        timeframe='1min'
    )
    
    if bars is not None:
        print(f"✓ Fetched {len(bars)} bars")
        print(f"  Date range: {bars.index[0]} to {bars.index[-1]}")
        print(f"  Last close: ${bars['close'].iloc[-1]:.2f}")
    
    # Test case 2: Fetch same data again (should hit cache)
    print("\nTest 2: Fetching same data again...")
    bars2 = manager.get_bars(
        symbol='AAPL',
        entry_time=entry_time,
        lookback_hours=2,
        forward_bars=60,
        timeframe='1min'
    )
    
    if bars2 is not None:
        print(f"✓ Fetched {len(bars2)} bars (from cache)")
    
    # Test case 3: Adjacent time period (should use extended cache)
    print("\nTest 3: Fetching adjacent time period...")
    adjacent_entry = entry_time + timedelta(hours=1)
    
    bars3 = manager.get_bars(
        symbol='AAPL',
        entry_time=adjacent_entry,
        lookback_hours=2,
        forward_bars=60,
        timeframe='1min'
    )
    
    if bars3 is not None:
        print(f"✓ Fetched {len(bars3)} bars")
    
    # Show cache statistics
    print("\nCache Statistics:")
    stats = manager.get_cache_stats()
    print(f"  Memory cache: {stats['memory_cache']}")
    print(f"  File cache: {stats['file_cache']}")
    print(f"  API stats: {stats['api_stats']}")