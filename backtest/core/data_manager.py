# backtest/core/data_manager.py
"""
Data management for backtesting using Supabase.
Handles loading historical bars, trades, and quotes from Supabase tables.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import hashlib
from supabase import create_client, Client
import os

logger = logging.getLogger(__name__)


class BacktestDataManager:
    """
    Manages data loading from Supabase for backtesting.
    Interfaces with existing Supabase tables for market data.
    """
    
    def __init__(self, supabase_url: Optional[str] = None, 
                 supabase_key: Optional[str] = None,
                 cache_locally: bool = True):
        """
        Initialize data manager with Supabase connection.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase anon/service key
            cache_locally: Whether to cache data locally for performance
        """
        # Get credentials from environment if not provided
        self.supabase_url = supabase_url or os.getenv('SUPABASE_URL')
        self.supabase_key = supabase_key or os.getenv('SUPABASE_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase credentials required")
            
        # Initialize Supabase client
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Local caching for performance
        self.cache_locally = cache_locally
        if cache_locally:
            self.cache_dir = Path('backtest/data/cache')
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory cache for recent queries
        self.memory_cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
        self.cache_ttl_minutes = 60
        
        # Track statistics
        self.db_queries = 0
        self.cache_hits = 0
        self.total_rows_fetched = 0
        
        logger.info(f"Initialized Supabase data manager")
        
    async def load_bars(self, symbol: str, start_time: datetime, 
                       end_time: datetime, timeframe: str = '1min',
                       use_cache: bool = True) -> pd.DataFrame:
        """
        Load historical bar data from Supabase.
        
        Args:
            symbol: Stock symbol
            start_time: Start time (UTC)
            end_time: End time (UTC)
            timeframe: Bar timeframe ('1min', '5min', etc.)
            use_cache: Whether to use cache
            
        Returns:
            DataFrame with OHLCV data, index is UTC datetime
        """
        # Ensure UTC
        start_time = self._ensure_utc(start_time)
        end_time = self._ensure_utc(end_time)
        
        # Check memory cache first
        cache_key = self._get_cache_key(symbol, start_time, end_time, timeframe, 'bars')
        
        if use_cache and cache_key in self.memory_cache:
            cached_data, cached_time = self.memory_cache[cache_key]
            if (datetime.now(timezone.utc) - cached_time).total_seconds() < self.cache_ttl_minutes * 60:
                self.cache_hits += 1
                logger.debug(f"Memory cache hit for {symbol} bars")
                return cached_data
                
        # Query Supabase
        self.db_queries += 1
        logger.info(f"Fetching {symbol} {timeframe} bars from Supabase")
        
        try:
            # Determine table name based on timeframe
            table_name = self._get_table_name(timeframe, 'bars')
            
            # Build query
            query = self.supabase.table(table_name).select("*")
            query = query.eq('symbol', symbol)
            query = query.gte('timestamp', start_time.isoformat())
            query = query.lte('timestamp', end_time.isoformat())
            query = query.order('timestamp')
            
            # Execute query with pagination for large datasets
            all_data = []
            limit = 10000
            offset = 0
            
            while True:
                response = query.range(offset, offset + limit - 1).execute()
                
                if response.data:
                    all_data.extend(response.data)
                    if len(response.data) < limit:
                        break
                    offset += limit
                else:
                    break
                    
            self.total_rows_fetched += len(all_data)
            
            # Convert to DataFrame
            if all_data:
                df = pd.DataFrame(all_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                df.set_index('timestamp', inplace=True)
                
                # Ensure proper column names
                df = df[['open', 'high', 'low', 'close', 'volume']]
                df = df.astype({
                    'open': float,
                    'high': float,
                    'low': float,
                    'close': float,
                    'volume': float
                })
            else:
                df = pd.DataFrame()
                
            # Cache the result
            if use_cache and not df.empty:
                self.memory_cache[cache_key] = (df, datetime.now(timezone.utc))
                
            return df
            
        except Exception as e:
            logger.error(f"Failed to load bars from Supabase: {e}")
            return pd.DataFrame()
            
    async def load_trades(self, symbol: str, start_time: datetime,
                         end_time: datetime, use_cache: bool = True) -> List[Dict]:
        """
        Load historical trade data from Supabase.
        
        Returns:
            List of trade dictionaries
        """
        start_time = self._ensure_utc(start_time)
        end_time = self._ensure_utc(end_time)
        
        # Check cache
        cache_key = self._get_cache_key(symbol, start_time, end_time, 'tick', 'trades')
        
        if use_cache and cache_key in self.memory_cache:
            cached_data, cached_time = self.memory_cache[cache_key]
            if (datetime.now(timezone.utc) - cached_time).total_seconds() < self.cache_ttl_minutes * 60:
                self.cache_hits += 1
                return cached_data
                
        self.db_queries += 1
        logger.info(f"Fetching {symbol} trades from Supabase")
        
        try:
            # Query trades table
            query = self.supabase.table('trades').select("*")
            query = query.eq('symbol', symbol)
            query = query.gte('timestamp', start_time.isoformat())
            query = query.lte('timestamp', end_time.isoformat())
            query = query.order('timestamp')
            
            # Fetch with pagination
            all_trades = []
            limit = 50000  # Larger limit for trades
            offset = 0
            
            while True:
                response = query.range(offset, offset + limit - 1).execute()
                
                if response.data:
                    all_trades.extend(response.data)
                    if len(response.data) < limit:
                        break
                    offset += limit
                else:
                    break
                    
            self.total_rows_fetched += len(all_trades)
            
            # Convert timestamp to datetime objects
            for trade in all_trades:
                trade['timestamp'] = pd.to_datetime(trade['timestamp'], utc=True)
                
            # Cache result
            if use_cache and all_trades:
                self.memory_cache[cache_key] = (all_trades, datetime.now(timezone.utc))
                
            return all_trades
            
        except Exception as e:
            logger.error(f"Failed to load trades from Supabase: {e}")
            return []
            
    async def load_quotes(self, symbol: str, start_time: datetime,
                         end_time: datetime, use_cache: bool = True) -> List[Dict]:
        """
        Load historical quote data from Supabase.
        
        Returns:
            List of quote dictionaries
        """
        start_time = self._ensure_utc(start_time)
        end_time = self._ensure_utc(end_time)
        
        # Check cache
        cache_key = self._get_cache_key(symbol, start_time, end_time, 'tick', 'quotes')
        
        if use_cache and cache_key in self.memory_cache:
            cached_data, cached_time = self.memory_cache[cache_key]
            if (datetime.now(timezone.utc) - cached_time).total_seconds() < self.cache_ttl_minutes * 60:
                self.cache_hits += 1
                return cached_data
                
        self.db_queries += 1
        logger.info(f"Fetching {symbol} quotes from Supabase")
        
        try:
            # Query quotes table
            query = self.supabase.table('quotes').select("*")
            query = query.eq('symbol', symbol)
            query = query.gte('timestamp', start_time.isoformat())
            query = query.lte('timestamp', end_time.isoformat())
            query = query.order('timestamp')
            
            # Fetch with pagination
            all_quotes = []
            limit = 50000
            offset = 0
            
            while True:
                response = query.range(offset, offset + limit - 1).execute()
                
                if response.data:
                    all_quotes.extend(response.data)
                    if len(response.data) < limit:
                        break
                    offset += limit
                else:
                    break
                    
            self.total_rows_fetched += len(all_quotes)
            
            # Convert timestamp to datetime objects
            for quote in all_quotes:
                quote['timestamp'] = pd.to_datetime(quote['timestamp'], utc=True)
                
            # Cache result
            if use_cache and all_quotes:
                self.memory_cache[cache_key] = (all_quotes, datetime.now(timezone.utc))
                
            return all_quotes
            
        except Exception as e:
            logger.error(f"Failed to load quotes from Supabase: {e}")
            return []
            
    async def save_backtest_result(self, result: Dict[str, Any]) -> str:
        """
        Save backtest result to Supabase.
        
        Args:
            result: Backtest result dictionary
            
        Returns:
            Result ID
        """
        try:
            # Prepare data for insertion
            result_data = {
                'symbol': result['config']['symbol'],
                'entry_time': result['config']['entry_time'],
                'direction': result['config']['direction'],
                'consensus_direction': result['aggregated_signal']['consensus_direction'],
                'final_pnl': result['forward_analysis']['final_pnl'],
                'max_favorable_move': result['forward_analysis']['max_favorable_move'],
                'max_adverse_move': result['forward_analysis']['max_adverse_move'],
                'entry_signals': result['entry_signals'],  # JSONB column
                'forward_analysis': result['forward_analysis'],  # JSONB column
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Insert into backtest_results table
            response = self.supabase.table('backtest_results').insert(result_data).execute()
            
            if response.data:
                return response.data[0]['id']
            else:
                raise Exception("Failed to insert result")
                
        except Exception as e:
            logger.error(f"Failed to save backtest result: {e}")
            raise
            
    async def query_backtest_results(self, 
                                   symbol: Optional[str] = None,
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None,
                                   direction: Optional[str] = None,
                                   min_pnl: Optional[float] = None,
                                   limit: int = 1000) -> List[Dict]:
        """
        Query backtest results from Supabase.
        
        Returns:
            List of result dictionaries
        """
        try:
            query = self.supabase.table('backtest_results').select("*")
            
            # Apply filters
            if symbol:
                query = query.eq('symbol', symbol)
            if direction:
                query = query.eq('direction', direction)
            if start_date:
                query = query.gte('entry_time', start_date.isoformat())
            if end_date:
                query = query.lte('entry_time', end_date.isoformat())
            if min_pnl is not None:
                query = query.gte('final_pnl', min_pnl)
                
            # Order by entry time descending
            query = query.order('entry_time', desc=True)
            query = query.limit(limit)
            
            response = query.execute()
            
            return response.data if response.data else []
            
        except Exception as e:
            logger.error(f"Failed to query backtest results: {e}")
            return []
            
    def _get_table_name(self, timeframe: str, data_type: str) -> str:
        """Get Supabase table name based on timeframe and data type"""
        if data_type == 'bars':
            # Map timeframe to table name
            table_map = {
                '1min': 'bars_1min',
                '5min': 'bars_5min',
                '15min': 'bars_15min',
                '1hour': 'bars_1hour',
                '1day': 'bars_1day'
            }
            return table_map.get(timeframe, 'bars_1min')
        else:
            return data_type  # 'trades' or 'quotes'
            
    def _ensure_utc(self, dt: datetime) -> datetime:
        """Ensure datetime is UTC"""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        elif dt.tzinfo != timezone.utc:
            return dt.astimezone(timezone.utc)
        return dt
        
    def _get_cache_key(self, symbol: str, start_time: datetime, 
                      end_time: datetime, timeframe: str, data_type: str) -> str:
        """Generate cache key for data request"""
        key_str = f"{symbol}_{start_time.isoformat()}_{end_time.isoformat()}_{timeframe}_{data_type}"
        return hashlib.md5(key_str.encode()).hexdigest()
        
    def clear_memory_cache(self) -> None:
        """Clear memory cache"""
        self.memory_cache.clear()
        logger.info("Cleared memory cache")
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get data manager statistics"""
        cache_size = len(self.memory_cache)
        hit_rate = (
            self.cache_hits / (self.db_queries + self.cache_hits) * 100
            if (self.db_queries + self.cache_hits) > 0 else 0
        )
        
        return {
            'db_queries': self.db_queries,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': round(hit_rate, 2),
            'total_rows_fetched': self.total_rows_fetched,
            'memory_cache_size': cache_size,
            'supabase_url': self.supabase_url.split('.')[0]  # Just the project name
        }