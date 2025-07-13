# live_monitor/data/hist_request/ema_fetchers.py
"""
EMA-specific historical data fetchers
Fetches data optimized for EMA calculations
"""
import pandas as pd
from typing import Dict

from .base_fetcher import BaseHistoricalFetcher


class M1EMAFetcher(BaseHistoricalFetcher):
    """Fetcher for M1 EMA calculations"""
    
    def __init__(self, rest_client):
        super().__init__(
            rest_client=rest_client,
            bars_needed=35,  # Fetch 35 to use prior 26 for smooth calculations
            timespan='1min',
            name='M1_EMA'
        )
    
    def get_minimum_bars(self) -> int:
        """M1 EMA needs at least 26 bars"""
        return 26
    
    def _validate_specific_data(self, df: pd.DataFrame) -> bool:
        """M1 EMA specific validation"""
        # Check for reasonable price ranges
        if df['close'].min() <= 0:
            return False
        
        # Check for data gaps (more than 5 minutes between bars during market hours)
        if len(df) > 1:
            time_diffs = df.index.to_series().diff()
            max_gap = time_diffs.max()
            if max_gap.total_seconds() > 300:  # 5 minutes
                # Only a problem during market hours
                market_session = self.rest_client.get_market_session()
                if market_session == 'regular':
                    return False
        
        return True
    
    def _get_metadata(self, df: pd.DataFrame) -> Dict:
        """Get M1 EMA relevant metadata"""
        return {
            'calculation_type': 'M1_EMA',
            'latest_close': float(df['close'].iloc[-1]),
            'latest_volume': int(df['volume'].iloc[-1]),
            'price_range': (float(df['low'].min()), float(df['high'].max())),
            'total_volume': int(df['volume'].sum()),
            'has_smooth_data': len(df) >= self.bars_needed
        }


class M5EMAFetcher(BaseHistoricalFetcher):
    """Fetcher for M5 EMA calculations"""
    
    def __init__(self, rest_client):
        super().__init__(
            rest_client=rest_client,
            bars_needed=35,  # Native 5-minute bars
            timespan='5min',
            name='M5_EMA'
        )
    
    def get_minimum_bars(self) -> int:
        """M5 EMA needs at least 26 bars"""
        return 26
    
    def _validate_specific_data(self, df: pd.DataFrame) -> bool:
        """M5 EMA specific validation"""
        # Check for reasonable price ranges
        if df['close'].min() <= 0:
            return False
        
        # For 5-minute bars, gaps up to 15 minutes are acceptable
        if len(df) > 1:
            time_diffs = df.index.to_series().diff()
            max_gap = time_diffs.max()
            if max_gap.total_seconds() > 900:  # 15 minutes
                market_session = self.rest_client.get_market_session()
                if market_session == 'regular':
                    return False
        
        return True
    
    def _get_metadata(self, df: pd.DataFrame) -> Dict:
        """Get M5 EMA relevant metadata"""
        # Calculate 5-minute specific metrics
        recent_volatility = df['close'].pct_change().std() * 100
        
        return {
            'calculation_type': 'M5_EMA',
            'latest_close': float(df['close'].iloc[-1]),
            'latest_volume': int(df['volume'].iloc[-1]),
            'price_range': (float(df['low'].min()), float(df['high'].max())),
            'total_volume': int(df['volume'].sum()),
            'recent_volatility': float(recent_volatility),
            'bar_count': len(df)
        }


class M15EMAFetcher(BaseHistoricalFetcher):
    """Fetcher for M15 EMA calculations"""
    
    def __init__(self, rest_client):
        super().__init__(
            rest_client=rest_client,
            bars_needed=35,  # Native 15-minute bars
            timespan='15min',
            name='M15_EMA'
        )
    
    def get_minimum_bars(self) -> int:
        """M15 EMA needs at least 26 bars"""
        return 26
    
    def _validate_specific_data(self, df: pd.DataFrame) -> bool:
        """M15 EMA specific validation"""
        # Check for reasonable price ranges
        if df['close'].min() <= 0:
            return False
        
        # For 15-minute bars, larger gaps are acceptable
        if len(df) > 1:
            time_diffs = df.index.to_series().diff()
            max_gap = time_diffs.max()
            # Allow up to 1 hour gaps (could be lunch break or after-hours)
            if max_gap.total_seconds() > 3600:  # 1 hour
                market_session = self.rest_client.get_market_session()
                if market_session == 'regular':
                    return False
        
        return True
    
    def _get_metadata(self, df: pd.DataFrame) -> Dict:
        """Get M15 EMA relevant metadata"""
        # Calculate trend strength over the period
        first_close = df['close'].iloc[0]
        last_close = df['close'].iloc[-1]
        trend_pct = ((last_close - first_close) / first_close) * 100
        
        return {
            'calculation_type': 'M15_EMA',
            'latest_close': float(df['close'].iloc[-1]),
            'latest_volume': int(df['volume'].iloc[-1]),
            'price_range': (float(df['low'].min()), float(df['high'].max())),
            'total_volume': int(df['volume'].sum()),
            'trend_percentage': float(trend_pct),
            'session_high': float(df['high'].max()),
            'session_low': float(df['low'].min())
        }