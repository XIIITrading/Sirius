# modules/calculations/trend/statistical_trend_5min_simplified.py
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
    signal: str  # 'STRONG TREND UP', 'TREND UP', 'RANGING', 'TREND DOWN', 'STRONG TREND DOWN'
    bias: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
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
    4. Direct signal mapping
    """
    
    def __init__(self, lookback_periods: int = 10):
        """
        Args:
            lookback_periods: Number of 5-min bars to analyze (default 10 = 50 minutes)
        """
        self.lookback_periods = lookback_periods
        
        # Thresholds for 5-minute trading
        self.strength_thresholds = {
            'strong': 1.5,     # Trend > 1.5x volatility
            'normal': 0.75,    # Trend > 0.75x volatility  
            'weak': 0.3        # Trend > 0.3x volatility
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
        
        # Generate signal
        signal, bias, confidence = self._generate_signal(
            trend_strength, 
            volatility_adjusted_strength,
            volume_confirmation
        )
        
        return PositionSignal5Min(
            symbol=symbol,
            timestamp=entry_time,
            price=current_price,
            signal=signal,
            bias=bias,
            confidence=confidence,
            trend_strength=abs(trend_strength),
            volatility_adjusted_strength=volatility_adjusted_strength,
            volume_confirmation=volume_confirmation
        )
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """
        Trend strength for 5-minute bars
        Uses both percentage change and regression for confirmation
        """
        # Primary: percentage change over period
        pct_change = (prices[-1] - prices[0]) / prices[0] * 100
        
        # Confirmation: linear regression
        x = np.arange(len(prices))
        slope, _, r_value, p_value, _ = stats.linregress(x, prices)
        
        # Weight by regression quality
        if p_value < 0.10 and r_value**2 > 0.3:  # Lower thresholds for 5-min
            return pct_change
        else:
            return pct_change * max(0.5, r_value**2)  # Minimum 50% weight
    
    def _calculate_volatility(self, prices: np.ndarray) -> float:
        """
        Volatility for 5-minute bars
        """
        returns = np.diff(prices) / prices[:-1]
        return returns.std() * 100
    
    def _check_volume_confirmation(self, prices: np.ndarray, volumes: np.ndarray) -> bool:
        """
        Volume confirmation for 5-minute trends
        """
        price_direction = np.sign(prices[-1] - prices[0])
        
        # Compare first half vs second half volume
        mid = len(volumes) // 2
        early_vol = volumes[:mid].mean()
        late_vol = volumes[mid:].mean()
        
        # Volume should increase by at least 10% in trend direction
        volume_increase = (late_vol - early_vol) / early_vol if early_vol > 0 else 0
        
        # For uptrends we want increasing volume, for downtrends either is fine
        if price_direction > 0:
            return volume_increase > 0.1
        else:
            return True  # Downtrends don't require volume confirmation
    
    def _generate_signal(self, trend_strength: float, 
                        volatility_adjusted_strength: float,
                        volume_confirmation: bool) -> tuple[str, str, float]:
        """
        Generate position signal based on 5-minute analysis
        """
        # Base confidence on volatility-adjusted strength
        confidence = min(100, volatility_adjusted_strength * 30)
        
        # Volume confirmation bonus
        if volume_confirmation:
            confidence = min(100, confidence * 1.15)
        
        # Determine signal and bias
        if trend_strength > 0:  # Bullish
            bias = 'BULLISH'
            if volatility_adjusted_strength >= self.strength_thresholds['strong']:
                signal = 'STRONG TREND UP'
            elif volatility_adjusted_strength >= self.strength_thresholds['normal']:
                signal = 'TREND UP'
            else:
                signal = 'RANGING'
                bias = 'NEUTRAL'
                confidence *= 0.6
                
        else:  # Bearish
            bias = 'BEARISH'
            if volatility_adjusted_strength >= self.strength_thresholds['strong']:
                signal = 'STRONG TREND DOWN'
            elif volatility_adjusted_strength >= self.strength_thresholds['normal']:
                signal = 'TREND DOWN'
            else:
                signal = 'RANGING'
                bias = 'NEUTRAL'
                confidence *= 0.6
        
        return signal, bias, confidence