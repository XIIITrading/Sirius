# backtest/data/polygon_data_manager/data_manager.py
"""Main PolygonDataManager coordinating all components"""
import os
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple
import pandas as pd
from dotenv import load_dotenv

from .api_client import PolygonAPIClient
from .cache_manager import CacheManager
from .request_tracker import RequestTracker
from .models import DataRequest

# Ensure environment is loaded
load_dotenv()

logger = logging.getLogger(__name__)


class PolygonDataManager:
    """
    Simplified data manager that coordinates API calls and caching.
    
    This is now a clean coordinator that delegates to specialized components.
    """
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 memory_cache_size: int = 100,
                 file_cache_hours: int = 24,
                 extend_window_bars: int = 500,
                 report_dir: Optional[str] = None):
        """
        Initialize with modular components.
        
        Args:
            api_key: Polygon API key (uses env var if not provided)
            cache_dir: Directory for file cache
            memory_cache_size: Max items in memory cache
            file_cache_hours: Hours to keep file cache valid
            extend_window_bars: Extra bars to fetch for caching
            report_dir: Directory for reports
        """
        # Get API key with proper validation
        if api_key is None:
            api_key = os.environ.get('POLYGON_API_KEY')
            if not api_key:
                # Try one more time with explicit load
                load_dotenv()
                api_key = os.environ.get('POLYGON_API_KEY')
                
        if not api_key:
            raise ValueError(
                "Polygon API key not found. Please either:\n"
                "1. Pass api_key parameter to PolygonDataManager\n"
                "2. Set POLYGON_API_KEY environment variable\n"
                "3. Create a .env file with POLYGON_API_KEY=your_key"
            )
            
        # Initialize components with validated key
        self.api_client = PolygonAPIClient(api_key)
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / 'cache' / 'polygon_data'
        self.cache_manager = CacheManager(cache_dir, memory_cache_size, file_cache_hours)
        
        # Request tracking
        if report_dir is None:
            report_dir = Path(__file__).parent.parent.parent / 'temp'
        self.request_tracker = RequestTracker(report_dir)
        
        # Configuration
        self.extend_window_bars = extend_window_bars
        
        # Statistics
        self.api_calls = 0
        self.cache_hits = 0
        
        logger.info("PolygonDataManager initialized with API key")
        
    def set_current_plugin(self, plugin_name: str):
        """Set the current plugin making requests"""
        self.request_tracker.set_current_plugin(plugin_name)
        
    async def load_bars(self, symbol: str, start_time: datetime, 
                       end_time: datetime, timeframe: str = '1min') -> pd.DataFrame:
        """Load bar data with caching"""
        start_tracking = time.time()
        
        # Ensure UTC
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
            
        # Create request object
        request = DataRequest(
            request_id=f"{self.request_tracker.current_plugin}_{symbol}_{timeframe}_{int(time.time()*1000)}",
            plugin_name=self.request_tracker.current_plugin,
            symbol=symbol,
            data_type='bars',
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            requested_at=datetime.now(timezone.utc),
            source='pending',
            success=False,
            returned_count=0
        )
        
        # Add extension for efficiency
        minutes_per_bar = self._get_minutes_per_bar(timeframe)
        extended_start = start_time - timedelta(minutes=self.extend_window_bars * minutes_per_bar)
        extended_end = end_time + timedelta(minutes=self.extend_window_bars * minutes_per_bar)
        
        try:
            # Check cache
            df = self.cache_manager.get(symbol, 'bars', timeframe, extended_start, extended_end)
            
            if df is not None:
                self.cache_hits += 1
                request.source = 'cache'
                filtered = self._filter_timerange(df, start_time, end_time)
                request.success = True
                request.returned_count = len(filtered)
                if len(filtered) > 0:
                    request.actual_start = filtered.index.min()
                    request.actual_end = filtered.index.max()
            else:
                # Fetch from API
                logger.info(f"Fetching {symbol} bars from Polygon API")
                request.source = 'polygon_api'
                self.api_calls += 1
                
                df = await self.api_client.fetch_bars(symbol, extended_start, extended_end, timeframe)
                
                if df is not None and not df.empty:
                    # Cache the data
                    self.cache_manager.put(symbol, 'bars', timeframe, extended_start, extended_end, df)
                    
                    # Return filtered range
                    filtered = self._filter_timerange(df, start_time, end_time)
                    request.success = True
                    request.returned_count = len(filtered)
                    if len(filtered) > 0:
                        request.actual_start = filtered.index.min()
                        request.actual_end = filtered.index.max()
                else:
                    filtered = pd.DataFrame()
                    request.success = False
                    request.error = "No data returned from API"
                    
        except Exception as e:
            logger.error(f"Error loading bars: {e}")
            filtered = pd.DataFrame()
            request.success = False
            request.error = str(e)
            
        # Track request
        request.processing_time_ms = (time.time() - start_tracking) * 1000
        self.request_tracker.track_request(request)
        
        return filtered
        
    async def load_trades(self, symbol: str, start_time: datetime,
                         end_time: datetime) -> pd.DataFrame:
        """Load trade data with caching"""
        start_tracking = time.time()
        
        # Ensure UTC
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
            
        # Create request object
        request = DataRequest(
            request_id=f"{self.request_tracker.current_plugin}_{symbol}_trades_{int(time.time()*1000)}",
            plugin_name=self.request_tracker.current_plugin,
            symbol=symbol,
            data_type='trades',
            timeframe='tick',
            start_time=start_time,
            end_time=end_time,
            requested_at=datetime.now(timezone.utc),
            source='pending',
            success=False,
            returned_count=0
        )
        
        try:
            # Check cache
            df = self.cache_manager.get(symbol, 'trades', 'tick', start_time, end_time)
            
            if df is not None:
                self.cache_hits += 1
                request.source = 'cache'
                request.success = True
                request.returned_count = len(df)
                if len(df) > 0:
                    request.actual_start = df.index.min()
                    request.actual_end = df.index.max()
            else:
                # Fetch from API
                logger.info(f"Fetching {symbol} trades from Polygon API")
                request.source = 'polygon_api'
                self.api_calls += 1
                
                df = await self.api_client.fetch_trades(symbol, start_time, end_time)
                
                if df is not None and not df.empty:
                    # Cache the data
                    self.cache_manager.put(symbol, 'trades', 'tick', start_time, end_time, df)
                    request.success = True
                    request.returned_count = len(df)
                    request.actual_start = df.index.min()
                    request.actual_end = df.index.max()
                else:
                    df = pd.DataFrame()
                    request.success = False
                    request.error = "No data returned from API"
                    
        except Exception as e:
            logger.error(f"Error loading trades: {e}")
            df = pd.DataFrame()
            request.success = False
            request.error = str(e)
            
        # Track request
        request.processing_time_ms = (time.time() - start_tracking) * 1000
        self.request_tracker.track_request(request)
        
        return df
        
    async def load_quotes(self, symbol: str, start_time: datetime,
                         end_time: datetime) -> pd.DataFrame:
        """Load quote data with caching"""
        start_tracking = time.time()
        
        # Ensure UTC
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
            
        # Create request object
        request = DataRequest(
            request_id=f"{self.request_tracker.current_plugin}_{symbol}_quotes_{int(time.time()*1000)}",
            plugin_name=self.request_tracker.current_plugin,
            symbol=symbol,
            data_type='quotes',
            timeframe='tick',
            start_time=start_time,
            end_time=end_time,
            requested_at=datetime.now(timezone.utc),
            source='pending',
            success=False,
            returned_count=0
        )
        
        try:
            # Check cache
            df = self.cache_manager.get(symbol, 'quotes', 'tick', start_time, end_time)
            
            if df is not None:
                self.cache_hits += 1
                request.source = 'cache'
                request.success = True
                request.returned_count = len(df)
                if len(df) > 0:
                    request.actual_start = df.index.min()
                    request.actual_end = df.index.max()
            else:
                # Fetch from API
                logger.info(f"Fetching {symbol} quotes from Polygon API")
                request.source = 'polygon_api'
                self.api_calls += 1
                
                df = await self.api_client.fetch_quotes(symbol, start_time, end_time)
                
                if df is not None and not df.empty:
                    # Cache the data
                    self.cache_manager.put(symbol, 'quotes', 'tick', start_time, end_time, df)
                    request.success = True
                    request.returned_count = len(df)
                    request.actual_start = df.index.min()
                    request.actual_end = df.index.max()
                else:
                    df = pd.DataFrame()
                    request.success = False
                    request.error = "No data returned from API"
                    
        except Exception as e:
            logger.error(f"Error loading quotes: {e}")
            df = pd.DataFrame()
            request.success = False
            request.error = str(e)
            
        # Track request
        request.processing_time_ms = (time.time() - start_tracking) * 1000
        self.request_tracker.track_request(request)
        
        return df
        
    def _get_minutes_per_bar(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
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
        
    def _filter_timerange(self, df: pd.DataFrame, 
                         start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Filter dataframe to requested time range"""
        if df.empty:
            return df
            
        # Ensure timezone aware comparison
        if start_time.tzinfo is None:
            start_time = pd.Timestamp(start_time).tz_localize('UTC')
        else:
            start_time = pd.Timestamp(start_time)
            
        if end_time.tzinfo is None:
            end_time = pd.Timestamp(end_time).tz_localize('UTC')
        else:
            end_time = pd.Timestamp(end_time)
        
        mask = (df.index >= start_time) & (df.index <= end_time)
        return df[mask].copy()
        
    def generate_data_report(self) -> Tuple[str, str]:
        """Generate comprehensive data request report"""
        return self.request_tracker.generate_report()
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache and API statistics"""
        cache_stats = self.cache_manager.get_stats()
        tracker_stats = self.request_tracker.get_stats()
        
        return {
            **cache_stats,
            'api_stats': {
                'api_calls': self.api_calls,
                'cache_hits': self.cache_hits,
                'total_requests': tracker_stats['total_requests'],
                'cache_hit_rate': (self.cache_hits / max(1, tracker_stats['total_requests'])) * 100
            }
        }
        
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cache data"""
        self.cache_manager.clear(symbol)
        logger.info(f"Cache cleared for {symbol if symbol else 'all symbols'}")