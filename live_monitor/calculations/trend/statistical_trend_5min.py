# modules/calculations/trend/statistical_trend_5min.py
"""
Simplified Statistical Trend Analyzer for 5-minute bars
Single timeframe analysis for position management
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Optional
from dataclasses import dataclass
from scipy import stats

@dataclass
class PositionSignal5Min:
    """Position management signal for 5-min timeframe"""
    symbol: str
    timestamp: datetime
    price: float
    signal: str  # 'BUY', 'WEAK BUY', 'SELL', 'WEAK SELL'
    confidence: float  # 0-100
    trend_strength: float  # Magnitude of move
    volatility_adjusted_strength: float  # Trend strength relative to noise
    volume_confirmation: bool  # Volume supports the trend


class StatisticalTrend5Min:
    """
    Simplified 5-minute trend analyzer using only essential metrics
    
    Key simplifications:
    1. Single 5-minute timeframe (no multi-timeframe)
    2. Focus on trend strength relative to volatility
    3. Simple volume confirmation
    4. Direct signal mapping to 4-tier system
    """
    
    def __init__(self, lookback_periods: int = 10):
        """
        Args:
            lookback_periods: Number of 5-min bars to analyze (default 10 = 50 minutes)
        """
        self.lookback_periods = lookback_periods
        
        # Match 1min thresholds exactly
        self.strength_thresholds = {
            'normal': 1.0,     # Trend >= 1x volatility for BUY/SELL
            'weak': 0.25       # Trend >= 0.25x volatility for WEAK signals
        }
        
    def analyze(self, symbol: str, bars_df: pd.DataFrame, 
                entry_time: datetime) -> PositionSignal5Min:
        """
        Simplified analysis for position management
        """
        recent_bars = bars_df[bars_df.index <= entry_time].tail(self.lookback_periods)
        
        if len(recent_bars) < self.lookback_periods:
            raise ValueError(f"Insufficient data: need {self.lookback_periods} bars")
        
        prices = recent_bars['close'].values
        volumes = recent_bars['volume'].values
        current_price = prices[-1]
        
        # Core calculations
        trend_strength = self._calculate_trend_strength(prices)
        volatility = self._calculate_volatility(prices)
        volatility_adjusted_strength = abs(trend_strength) / (volatility + 0.0001)
        volume_confirmation = self._check_volume_confirmation(prices, volumes)
        
        # Generate signal using exact same logic as 1min
        signal, confidence = self._generate_signal(
            trend_strength, 
            volatility_adjusted_strength,
            volume_confirmation
        )
        
        return PositionSignal5Min(
            symbol=symbol,
            timestamp=entry_time,
            price=current_price,
            signal=signal,
            confidence=confidence,
            trend_strength=abs(trend_strength),
            volatility_adjusted_strength=volatility_adjusted_strength,
            volume_confirmation=volume_confirmation
        )
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """
        Simple trend strength: percentage change with linear regression confirmation
        Matching 1min logic exactly
        """
        # Primary metric: total percentage change
        pct_change = (prices[-1] - prices[0]) / prices[0] * 100
        
        # Confirmation: linear regression slope significance
        x = np.arange(len(prices))
        slope, _, r_value, p_value, _ = stats.linregress(x, prices)
        
        # Match 1min thresholds: p < 0.05 and r² > 0.5
        if p_value < 0.05 and r_value**2 > 0.5:
            return pct_change
        else:
            return pct_change * (r_value**2)  # Discount by fit quality
    
    def _calculate_volatility(self, prices: np.ndarray) -> float:
        """
        Simple volatility measure: standard deviation of returns
        """
        returns = np.diff(prices) / prices[:-1]
        return returns.std() * 100  # Percentage volatility
    
    def _check_volume_confirmation(self, prices: np.ndarray, volumes: np.ndarray) -> bool:
        """
        Simple volume confirmation: is volume increasing in trend direction?
        Matching 1min logic exactly
        """
        price_direction = np.sign(prices[-1] - prices[0])
        
        # Split period in half and compare average volumes
        mid = len(volumes) // 2
        early_vol = volumes[:mid].mean()
        late_vol = volumes[mid:].mean()
        
        # Volume should increase in trend direction
        volume_increasing = late_vol > early_vol
        
        # For uptrend, we want increasing volume; for downtrend, either works
        return volume_increasing if price_direction > 0 else True
    
    def _generate_signal(self, trend_strength: float, 
                        volatility_adjusted_strength: float,
                        volume_confirmation: bool) -> tuple[str, float]:
        """
        Generate signal and confidence based on simplified metrics
        Exact match to 1min logic - only 4 signal levels: BUY, WEAK BUY, WEAK SELL, SELL
        """
        # Base confidence on volatility-adjusted strength (matching 1min)
        confidence = min(100, volatility_adjusted_strength * 25)
        
        # Boost confidence if volume confirms (matching 1min)
        if volume_confirmation:
            confidence = min(100, confidence * 1.2)
        
        # Determine signal based on 1min thresholds
        if trend_strength > 0:  # Bullish
            if volatility_adjusted_strength >= self.strength_thresholds['normal']:
                return 'BUY', confidence
            elif volatility_adjusted_strength >= self.strength_thresholds['weak']:
                return 'WEAK BUY', confidence
            else:
                # Very weak but still bullish - map to WEAK BUY with reduced confidence
                return 'WEAK BUY', confidence * 0.5
                
        else:  # Bearish
            if volatility_adjusted_strength >= self.strength_thresholds['normal']:
                return 'SELL', confidence
            elif volatility_adjusted_strength >= self.strength_thresholds['weak']:
                return 'WEAK SELL', confidence
            else:
                # Very weak but still bearish - map to WEAK SELL with reduced confidence
                return 'WEAK SELL', confidence * 0.5