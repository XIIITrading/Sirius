"""Unified cache manager coordinating memory and file caches"""
import hashlib
import logging
from datetime import datetime
from typing import Optional
import pandas as pd

from .memory_cache import LRUCache
from .file_cache import FileCache

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages both memory and file caching with unified interface"""
    
    def __init__(self, cache_dir: str, memory_size: int = 100, file_cache_hours: int = 24):
        self.memory_cache = LRUCache(max_size=memory_size)
        self.file_cache = FileCache(cache_dir=cache_dir, cache_hours=file_cache_hours)
        
    def get(self, symbol: str, data_type: str, timeframe: str,
            start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        """Get data from cache (memory first, then file)"""
        cache_key = self._generate_cache_key(symbol, data_type, timeframe, start_time, end_time)
        
        # Check memory cache
        df = self.memory_cache.get(cache_key)
        if df is not None:
            logger.debug(f"Memory cache hit for {symbol}")
            return df
        
        # Check file cache
        df = self.file_cache.get(symbol, data_type, timeframe, start_time, end_time)
        if df is not None:
            logger.debug(f"File cache hit for {symbol}")
            # Store in memory for next time
            self.memory_cache.put(cache_key, df)
            return df
        
        return None
    
    def put(self, symbol: str, data_type: str, timeframe: str,
            start_time: datetime, end_time: datetime, df: pd.DataFrame):
        """Store data in both caches"""
        cache_key = self._generate_cache_key(symbol, data_type, timeframe, start_time, end_time)
        
        # Memory cache
        self.memory_cache.put(cache_key, df.copy())
        
        # File cache
        self.file_cache.put(symbol, data_type, timeframe, start_time, end_time, df, cache_key)
    
    def _generate_cache_key(self, symbol: str, data_type: str, timeframe: str,
                           start_time: datetime, end_time: datetime) -> str:
        """Generate unique cache key for data request"""
        key_string = f"{symbol}_{data_type}_{timeframe}_{start_time.isoformat()}_{end_time.isoformat()}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def clear(self, symbol: Optional[str] = None):
        """Clear caches"""
        self.memory_cache.clear()
        if symbol:
            self.file_cache.clear(symbol)
    
    def get_stats(self):
        """Get combined cache statistics"""
        return {
            'memory_cache': self.memory_cache.get_stats(),
            'file_cache': self.file_cache.get_stats()
        }