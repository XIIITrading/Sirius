# live_monitor/data/hist_request/market_structure_fetchers.py
"""
Market Structure specific historical data fetchers
Fetches data optimized for fractal and structure break detection
"""
import pandas as pd
from typing import Dict

from .base_fetcher import BaseHistoricalFetcher


class M1MarketStructureFetcher(BaseHistoricalFetcher):
    """Fetcher for M1 Market Structure analysis"""
    
    def __init__(self, rest_client):
        super().__init__(
            rest_client=rest_client,
            bars_needed=200,  # More bars needed for fractal detection
            timespan='1min',
            name='M1_MarketStructure'
        )
        self.swing_length = 5  # From MarketStructureAnalyzer
    
    def get_minimum_bars(self) -> int:
        """Market structure needs at least 21 bars (from analyzer)"""
        return 21
    
    def _validate_specific_data(self, df: pd.DataFrame) -> bool:
        """Market structure specific validation"""
        # Need clean OHLC data for fractal detection
        if df['close'].min() <= 0 or df['high'].min() <= 0 or df['low'].min() <= 0:
            return False
        
        # Verify high >= low for all bars
        invalid_bars = df[df['high'] < df['low']]
        if not invalid_bars.empty:
            return False
        
        # Check that we have enough bars for swing detection
        if len(df) < self.swing_length * 2 + 1:
            return False
        
        return True
    
    def _get_metadata(self, df: pd.DataFrame) -> Dict:
        """Get market structure relevant metadata"""
        # Calculate basic structure metrics
        recent_high = df['high'].iloc[-20:].max()
        recent_low = df['low'].iloc[-20:].min()
        current_close = df['close'].iloc[-1]
        
        # Position in range
        if recent_high != recent_low:
            position_in_range = (current_close - recent_low) / (recent_high - recent_low)
        else:
            position_in_range = 0.5
        
        return {
            'calculation_type': 'M1_MarketStructure',
            'latest_close': float(current_close),
            'recent_high': float(recent_high),
            'recent_low': float(recent_low),
            'position_in_range': float(position_in_range),
            'bar_count': len(df),
            'swing_detection_ready': len(df) >= self.swing_length * 2 + 1
        }


class M5MarketStructureFetcher(BaseHistoricalFetcher):
    """Fetcher for M5 Market Structure analysis"""
    
    def __init__(self, rest_client):
        super().__init__(
            rest_client=rest_client,
            bars_needed=100,  # M5 needs fewer bars
            timespan='5min',
            name='M5_MarketStructure'
        )
        self.swing_length = 3  # From M5MarketStructureAnalyzer
    
    def get_minimum_bars(self) -> int:
        """M5 market structure needs at least 15 bars"""
        return 15
    
    def _validate_specific_data(self, df: pd.DataFrame) -> bool:
        """M5 market structure specific validation"""
        # Need clean OHLC data
        if df['close'].min() <= 0 or df['high'].min() <= 0 or df['low'].min() <= 0:
            return False
        
        # Verify high >= low for all bars
        invalid_bars = df[df['high'] < df['low']]
        if not invalid_bars.empty:
            return False
        
        # Check that we have enough bars for swing detection
        if len(df) < self.swing_length * 2 + 1:
            return False
        
        return True
    
    def _get_metadata(self, df: pd.DataFrame) -> Dict:
        """Get M5 market structure relevant metadata"""
        # Calculate volatility for 5-minute timeframe
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * 100
        
        # Trend direction
        sma_20 = df['close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['close'].mean()
        current_close = df['close'].iloc[-1]
        trend_bias = 'above_avg' if current_close > sma_20 else 'below_avg'
        
        return {
            'calculation_type': 'M5_MarketStructure',
            'latest_close': float(current_close),
            'volatility_5min': float(volatility),
            'trend_bias': trend_bias,
            'sma_20': float(sma_20) if len(df) >= 20 else None,
            'bar_count': len(df)
        }


class M15MarketStructureFetcher(BaseHistoricalFetcher):
    """Fetcher for M15 Market Structure analysis"""
    
    def __init__(self, rest_client):
        super().__init__(
            rest_client=rest_client,
            bars_needed=60,  # M15 needs even fewer bars
            timespan='15min',
            name='M15_MarketStructure'
        )
        self.swing_length = 2  # From M15MarketStructureAnalyzer
    
    def get_minimum_bars(self) -> int:
        """M15 market structure needs at least 10 bars"""
        return 10
    
    def _validate_specific_data(self, df: pd.DataFrame) -> bool:
        """M15 market structure specific validation"""
        # Need clean OHLC data
        if df['close'].min() <= 0 or df['high'].min() <= 0 or df['low'].min() <= 0:
            return False
        
        # Verify high >= low for all bars
        invalid_bars = df[df['high'] < df['low']]
        if not invalid_bars.empty:
            return False
        
        return True
    
    def _get_metadata(self, df: pd.DataFrame) -> Dict:
        """Get M15 market structure relevant metadata"""
        # Calculate session metrics
        session_high = df['high'].max()
        session_low = df['low'].min()
        session_range = session_high - session_low
        
        # Major levels
        current_close = df['close'].iloc[-1]
        distance_to_high = session_high - current_close
        distance_to_low = current_close - session_low
        
        return {
            'calculation_type': 'M15_MarketStructure',
            'latest_close': float(current_close),
            'session_high': float(session_high),
            'session_low': float(session_low),
            'session_range': float(session_range),
            'distance_to_high': float(distance_to_high),
            'distance_to_low': float(distance_to_low),
            'closer_to': 'high' if distance_to_high < distance_to_low else 'low',
            'bar_count': len(df)
        }