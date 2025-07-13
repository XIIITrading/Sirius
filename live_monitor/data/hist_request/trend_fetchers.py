# live_monitor/data/hist_request/trend_fetchers.py
"""
Statistical Trend specific historical data fetchers
Fetches minimal data for trend analysis
"""
import pandas as pd
from typing import Dict

from .base_fetcher import BaseHistoricalFetcher


class M1StatisticalTrendFetcher(BaseHistoricalFetcher):
    """Fetcher for M1 Statistical Trend analysis"""
    
    def __init__(self, rest_client):
        super().__init__(
            rest_client=rest_client,
            bars_needed=15,  # Minimal requirement
            timespan='1min',
            name='M1_StatisticalTrend'
        )
        self.lookback_periods = 10  # From StatisticalTrend1MinSimplified
    
    def get_minimum_bars(self) -> int:
        """Statistical trend needs at least 10 bars"""
        return self.lookback_periods
    
    def _validate_specific_data(self, df: pd.DataFrame) -> bool:
        """Statistical trend validation"""
        # Just need valid price and volume data
        if df['close'].min() <= 0:
            return False
        
        # Check for NaN values
        if df[['open', 'high', 'low', 'close', 'volume']].isnull().any().any():
            return False
        
        return True
    
    def _get_metadata(self, df: pd.DataFrame) -> Dict:
        """Get statistical trend relevant metadata"""
        # Quick trend calculation
        prices = df['close'].values
        trend_pct = ((prices[-1] - prices[0]) / prices[0]) * 100
        
        # Volume trend
        volumes = df['volume'].values
        vol_first_half = volumes[:len(volumes)//2].mean()
        vol_second_half = volumes[len(volumes)//2:].mean()
        volume_increasing = vol_second_half > vol_first_half
        
        return {
            'calculation_type': 'M1_StatisticalTrend',
            'latest_close': float(df['close'].iloc[-1]),
            'trend_percentage': float(trend_pct),
            'volume_increasing': volume_increasing,
            'bar_count': len(df)
        }


class M5StatisticalTrendFetcher(BaseHistoricalFetcher):
    """Fetcher for M5 Statistical Trend analysis"""
    
    def __init__(self, rest_client):
        super().__init__(
            rest_client=rest_client,
            bars_needed=15,  # Minimal requirement
            timespan='5min',
            name='M5_StatisticalTrend'
        )
        self.lookback_periods = 10
    
    def get_minimum_bars(self) -> int:
        """Statistical trend needs at least 10 bars"""
        return self.lookback_periods
    
    def _validate_specific_data(self, df: pd.DataFrame) -> bool:
        """Statistical trend validation"""
        if df['close'].min() <= 0:
            return False
        
        if df[['open', 'high', 'low', 'close', 'volume']].isnull().any().any():
            return False
        
        return True
    
    def _get_metadata(self, df: pd.DataFrame) -> Dict:
        """Get M5 statistical trend relevant metadata"""
        # Calculate simple volatility
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * 100
        
        # Price action quality
        prices = df['close'].values
        trend_pct = ((prices[-1] - prices[0]) / prices[0]) * 100
        
        return {
            'calculation_type': 'M5_StatisticalTrend',
            'latest_close': float(df['close'].iloc[-1]),
            'trend_percentage': float(trend_pct),
            'volatility': float(volatility),
            'bar_count': len(df)
        }


class M15StatisticalTrendFetcher(BaseHistoricalFetcher):
    """Fetcher for M15 Statistical Trend analysis"""
    
    def __init__(self, rest_client):
        super().__init__(
            rest_client=rest_client,
            bars_needed=15,  # Minimal requirement
            timespan='15min',
            name='M15_StatisticalTrend'
        )
        self.lookback_periods = 10
    
    def get_minimum_bars(self) -> int:
        """Statistical trend needs at least 10 bars"""
        return self.lookback_periods
    
    def _validate_specific_data(self, df: pd.DataFrame) -> bool:
        """Statistical trend validation"""
        if df['close'].min() <= 0:
            return False
        
        if df[['open', 'high', 'low', 'close', 'volume']].isnull().any().any():
            return False
        
        return True
    
    def _get_metadata(self, df: pd.DataFrame) -> Dict:
        """Get M15 statistical trend relevant metadata"""
        # Range analysis for 15-minute bars
        high_low_ranges = (df['high'] - df['low']).values
        avg_range = high_low_ranges.mean()
        
        # Trend metrics
        prices = df['close'].values
        trend_pct = ((prices[-1] - prices[0]) / prices[0]) * 100
        
        # Volume pattern
        volumes = df['volume'].values
        avg_volume = volumes.mean()
        
        return {
            'calculation_type': 'M15_StatisticalTrend',
            'latest_close': float(df['close'].iloc[-1]),
            'trend_percentage': float(trend_pct),
            'avg_range': float(avg_range),
            'avg_volume': float(avg_volume),
            'bar_count': len(df)
        }