# modules/calculations/trend/statistical_trend_1min_simplified.py
"""
Simplified Statistical Trend Analyzer for 1-minute bars
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict
from dataclasses import dataclass
from scipy import stats

@dataclass
class StatisticalSignal:
    """Statistical trend signal with confidence metrics"""
    symbol: str
    timestamp: datetime
    price: float
    signal: str  # 'STRONG BUY', 'BUY', 'WEAK BUY', 'NEUTRAL', 'WEAK SELL', 'SELL', 'STRONG SELL'
    confidence: float  # 0-100
    trend_strength: float  # Magnitude of move
    volatility_adjusted_strength: float  # Trend strength relative to noise
    volume_confirmation: bool  # Volume supports the trend


class StatisticalTrend1MinSimplified:
    """
    Simplified statistical trend analyzer using only essential metrics
    
    Key simplifications:
    1. Focus on trend strength relative to volatility (signal-to-noise)
    2. Simple volume confirmation 
    3. Direct confidence mapping to signal levels
    """
    
    def __init__(self, lookback_periods: int = 10):
        self.lookback_periods = lookback_periods
        
        # Simplified thresholds based on volatility-adjusted strength
        self.strength_thresholds = {
            'strong': 2.0,     # Trend > 2x volatility
            'normal': 1.0,     # Trend > 1x volatility  
            'weak': 0.5        # Trend > 0.5x volatility
        }
        
    def analyze(self, symbol: str, bars_df: pd.DataFrame, 
                entry_time: datetime) -> StatisticalSignal:
        """
        Simplified analysis focusing on trend strength vs volatility
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
        
        # Generate signal based on simplified logic
        signal, confidence = self._generate_signal(
            trend_strength, 
            volatility_adjusted_strength,
            volume_confirmation
        )
        
        return StatisticalSignal(
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
        """
        # Primary metric: total percentage change
        pct_change = (prices[-1] - prices[0]) / prices[0] * 100
        
        # Confirmation: linear regression slope significance
        x = np.arange(len(prices))
        slope, _, r_value, p_value, _ = stats.linregress(x, prices)
        
        # If regression is significant (p < 0.05) and fits well (r² > 0.5), 
        # use full strength, otherwise discount it
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
        """
        # Base confidence on volatility-adjusted strength
        confidence = min(100, volatility_adjusted_strength * 25)
        
        # Boost confidence if volume confirms
        if volume_confirmation:
            confidence = min(100, confidence * 1.2)
        
        # Determine signal based on thresholds
        if trend_strength > 0:  # Bullish
            if volatility_adjusted_strength >= self.strength_thresholds['strong']:
                return 'STRONG BUY', confidence
            elif volatility_adjusted_strength >= self.strength_thresholds['normal']:
                return 'BUY', confidence
            elif volatility_adjusted_strength >= self.strength_thresholds['weak']:
                return 'WEAK BUY', confidence
            else:
                return 'NEUTRAL', confidence * 0.5
                
        else:  # Bearish
            if volatility_adjusted_strength >= self.strength_thresholds['strong']:
                return 'STRONG SELL', confidence
            elif volatility_adjusted_strength >= self.strength_thresholds['normal']:
                return 'SELL', confidence
            elif volatility_adjusted_strength >= self.strength_thresholds['weak']:
                return 'WEAK SELL', confidence
            else:
                return 'NEUTRAL', confidence * 0.5