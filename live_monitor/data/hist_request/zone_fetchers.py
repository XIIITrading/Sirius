# live_monitor/data/hist_request/zone_fetchers.py
"""
Zone analysis specific historical data fetchers
Fetches data for HVN and Order Block calculations
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional

from .base_fetcher import BaseHistoricalFetcher


class SharedZoneCache:
    """Shared cache for zone fetchers to avoid duplicate fetches"""
    def __init__(self):
        self.cache: Optional[pd.DataFrame] = None
        self.cache_symbol: Optional[str] = None
        self.subscribers = set()
    
    def register(self, fetcher_name: str):
        """Register a fetcher as using this cache"""
        self.subscribers.add(fetcher_name)
    
    def unregister(self, fetcher_name: str):
        """Unregister a fetcher"""
        self.subscribers.discard(fetcher_name)
    
    def is_valid(self, symbol: str) -> bool:
        """Check if cache is valid for symbol"""
        return self.cache is not None and self.cache_symbol == symbol
    
    def update(self, symbol: str, df: pd.DataFrame):
        """Update shared cache"""
        self.cache = df.copy()
        self.cache_symbol = symbol
    
    def get(self) -> Optional[pd.DataFrame]:
        """Get cached data"""
        return self.cache.copy() if self.cache is not None else None
    
    def clear(self):
        """Clear cache"""
        self.cache = None
        self.cache_symbol = None


# Create shared cache instance
zone_cache = SharedZoneCache()


class HVNFetcher(BaseHistoricalFetcher):
    """Fetcher for HVN (High Volume Node) analysis"""
    
    def __init__(self, rest_client):
        super().__init__(
            rest_client=rest_client,
            bars_needed=200,  # Needs substantial data for volume profile
            timespan='1min',
            name='HVN'
        )
        # Register with shared cache
        zone_cache.register(self.name)
    
    def get_minimum_bars(self) -> int:
        """HVN needs at least 100 bars"""
        return 100
    
    def fetch_for_symbol(self, symbol: str, force_refresh: bool = False) -> None:
        """Override to check shared cache first"""
        # Check shared cache
        if not force_refresh and zone_cache.is_valid(symbol):
            df = zone_cache.get()
            if df is not None and len(df) >= self.get_minimum_bars():
                self.cache = df
                self.cache_symbol = symbol
                self._process_and_emit(symbol, df)
                return
        
        # Otherwise, proceed with normal fetch
        super().fetch_for_symbol(symbol, force_refresh)
    
    def _fetch_data(self, symbol: str) -> None:
        """Override to update shared cache"""
        try:
            bars = self.rest_client.fetch_bars(
                symbol=symbol,
                timespan=self.timespan,
                limit=self.bars_needed
            )
            
            if bars:
                df = self._bars_to_dataframe(bars)
                
                if self._validate_data(df):
                    # Update both local and shared cache
                    self.cache = df
                    self.cache_symbol = symbol
                    zone_cache.update(symbol, df)
                    
                    self._process_and_emit(symbol, df)
                else:
                    self._handle_fetch_error(symbol, "Data validation failed")
            else:
                self._handle_fetch_error(symbol, "No data returned from API")
                
        except Exception as e:
            self._handle_fetch_error(symbol, str(e))
    
    def _validate_specific_data(self, df: pd.DataFrame) -> bool:
        """HVN specific validation"""
        # Need valid volume data
        if df['volume'].min() < 0:
            return False
        
        # Need some volume activity
        if df['volume'].sum() == 0:
            return False
        
        # Check price integrity
        if df['close'].min() <= 0:
            return False
        
        return True
    
    def _get_metadata(self, df: pd.DataFrame) -> Dict:
        """Get HVN relevant metadata"""
        # Calculate volume metrics
        total_volume = df['volume'].sum()
        avg_volume_per_bar = df['volume'].mean()
        
        # Price range for HVN calculation
        price_range = (df['low'].min(), df['high'].max())
        price_levels = 100  # Default HVN levels
        
        # Volume concentration
        volume_std = df['volume'].std()
        volume_concentration = volume_std / avg_volume_per_bar if avg_volume_per_bar > 0 else 0
        
        return {
            'calculation_type': 'HVN',
            'total_volume': int(total_volume),
            'avg_volume': float(avg_volume_per_bar),
            'price_range': price_range,
            'price_levels': price_levels,
            'volume_concentration': float(volume_concentration),
            'bar_count': len(df)
        }


class OrderBlocksFetcher(BaseHistoricalFetcher):
    """Fetcher for Order Blocks analysis"""
    
    def __init__(self, rest_client):
        super().__init__(
            rest_client=rest_client,
            bars_needed=200,  # Same as HVN, can share cache
            timespan='1min',
            name='OrderBlocks'
        )
        self.swing_length = 7  # From OrderBlockAnalyzer
        # Register with shared cache
        zone_cache.register(self.name)
    
    def get_minimum_bars(self) -> int:
        """Order blocks need enough bars for swing detection"""
        return self.swing_length * 2
    
    def fetch_for_symbol(self, symbol: str, force_refresh: bool = False) -> None:
        """Override to check shared cache first"""
        # Check shared cache
        if not force_refresh and zone_cache.is_valid(symbol):
            df = zone_cache.get()
            if df is not None and len(df) >= self.get_minimum_bars():
                self.cache = df
                self.cache_symbol = symbol
                self._process_and_emit(symbol, df)
                return
        
        # Otherwise, proceed with normal fetch
        super().fetch_for_symbol(symbol, force_refresh)
    
    def _fetch_data(self, symbol: str) -> None:
        """Override to update shared cache"""
        try:
            bars = self.rest_client.fetch_bars(
                symbol=symbol,
                timespan=self.timespan,
                limit=self.bars_needed
            )
            
            if bars:
                df = self._bars_to_dataframe(bars)
                
                if self._validate_data(df):
                    # Update both local and shared cache
                    self.cache = df
                    self.cache_symbol = symbol
                    zone_cache.update(symbol, df)
                    
                    self._process_and_emit(symbol, df)
                else:
                    self._handle_fetch_error(symbol, "Data validation failed")
            else:
                self._handle_fetch_error(symbol, "No data returned from API")
                
        except Exception as e:
            self._handle_fetch_error(symbol, str(e))
    
    def _validate_specific_data(self, df: pd.DataFrame) -> bool:
        """Order blocks specific validation"""
        # Need clean OHLC for swing detection
        if df[['open', 'high', 'low', 'close']].min().min() <= 0:
            return False
        
        # Verify high >= low
        invalid_bars = df[df['high'] < df['low']]
        if not invalid_bars.empty:
            return False
        
        # Need enough bars for swing detection
        if len(df) < self.swing_length * 2:
            return False
        
        return True
    
    def _get_metadata(self, df: pd.DataFrame) -> Dict:
        """Get Order Blocks relevant metadata"""
        # Find potential swing points (simplified)
        highs = df['high'].values
        lows = df['low'].values
        
        # Count potential swings
        potential_swing_highs = 0
        potential_swing_lows = 0
        
        for i in range(self.swing_length, len(df) - self.swing_length):
            # Check for swing high
            if highs[i] == max(highs[i-self.swing_length:i+self.swing_length+1]):
                potential_swing_highs += 1
            
            # Check for swing low
            if lows[i] == min(lows[i-self.swing_length:i+self.swing_length+1]):
                potential_swing_lows += 1
        
        # Recent price action
        recent_trend = 'up' if df['close'].iloc[-1] > df['close'].iloc[-20] else 'down'
        
        return {
            'calculation_type': 'OrderBlocks',
            'potential_swing_highs': potential_swing_highs,
            'potential_swing_lows': potential_swing_lows,
            'recent_trend': recent_trend,
            'latest_close': float(df['close'].iloc[-1]),
            'swing_length': self.swing_length,
            'bar_count': len(df)
        }