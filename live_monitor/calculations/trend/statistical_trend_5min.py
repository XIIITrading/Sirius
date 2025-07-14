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
    bias: str  # 'BULLISH', 'BEARISH'
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
        
        # Modified thresholds for 4-tier system
        # These determine Buy vs Weak Buy and Sell vs Weak Sell
        self.strength_thresholds = {
            'strong': 1.0,     # Trend > 1.0x volatility for Buy/Sell
            'weak': 0.4        # Trend > 0.4x volatility for Weak Buy/Sell
            # Below 0.4x volatility, we assign based on direction only
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
        
        # Core calculations (unchanged)
        trend_strength = self._calculate_trend_strength(prices)
        volatility = self._calculate_volatility(prices)
        volatility_adjusted_strength = abs(trend_strength) / (volatility + 0.0001)
        volume_confirmation = self._check_volume_confirmation(prices, volumes)
        
        # Generate 4-tier signal
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
        Generate 4-tier position signal based on 5-minute analysis
        Returns: (signal, bias, confidence)
        """
        # Base confidence calculation
        # Scale volatility_adjusted_strength to 0-100 range
        # Strong trends (>1.0) map to 70-100
        # Weak trends (0.4-1.0) map to 40-70
        # Very weak trends (<0.4) map to 20-40
        if volatility_adjusted_strength >= self.strength_thresholds['strong']:
            base_confidence = 70 + min(30, (volatility_adjusted_strength - 1.0) * 30)
        elif volatility_adjusted_strength >= self.strength_thresholds['weak']:
            base_confidence = 40 + ((volatility_adjusted_strength - 0.4) / 0.6) * 30
        else:
            base_confidence = 20 + (volatility_adjusted_strength / 0.4) * 20
        
        # Volume confirmation bonus (up to 15% boost)
        if volume_confirmation:
            confidence = min(100, base_confidence * 1.15)
        else:
            confidence = base_confidence
        
        # Determine signal based on direction and strength
        if trend_strength > 0:  # Bullish direction
            bias = 'BULLISH'
            if volatility_adjusted_strength >= self.strength_thresholds['strong']:
                signal = 'BUY'
            else:  # Anything below strong threshold but positive
                signal = 'WEAK BUY'
                
        else:  # Bearish direction
            bias = 'BEARISH'
            if volatility_adjusted_strength >= self.strength_thresholds['strong']:
                signal = 'SELL'
            else:  # Anything below strong threshold but negative
                signal = 'WEAK SELL'
        
        return signal, bias, confidence