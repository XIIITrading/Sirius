# modules/calculations/trend/statistical_trend_15min_simplified.py
"""
Simplified Statistical Trend Analyzer for 15-minute bars
Single timeframe analysis for market regime detection
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Optional
from dataclasses import dataclass
from scipy import stats

@dataclass
class MarketRegimeSignal:
    """Market regime signal for daily trading decisions"""
    symbol: str
    timestamp: datetime
    price: float
    regime: str  # 'BULL MARKET', 'BEAR MARKET', 'RANGE BOUND'
    daily_bias: str  # 'LONG ONLY', 'SHORT ONLY', 'BOTH WAYS', 'STAY OUT'
    confidence: float  # 0-100
    trend_strength: float  # Magnitude of move
    volatility_adjusted_strength: float  # Trend strength relative to noise
    volatility_state: str  # 'LOW', 'NORMAL', 'HIGH', 'EXTREME'
    volume_trend: str  # 'INCREASING', 'DECREASING', 'STABLE'


class StatisticalTrend15MinSimplified:
    """
    Simplified 15-minute trend analyzer for market regime
    
    Key simplifications:
    1. Single 15-minute timeframe (no multi-timeframe)
    2. Focus on regime identification
    3. Clear daily bias generation
    4. Simple volatility assessment
    """
    
    def __init__(self, lookback_periods: int = 10):
        """
        Args:
            lookback_periods: Number of 15-min bars to analyze (default 10 = 150 minutes)
        """
        self.lookback_periods = lookback_periods
        
        # Thresholds for regime detection
        self.regime_thresholds = {
            'trending': 1.2,      # Clear trend vs volatility
            'transitioning': 0.6, # Possible trend forming
            'ranging': 0.3        # No clear trend
        }
        
    def analyze(self, symbol: str, bars_df: pd.DataFrame, 
                entry_time: datetime) -> MarketRegimeSignal:
        """
        Simplified regime analysis for daily bias
        """
        recent_bars = bars_df[bars_df.index <= entry_time].tail(self.lookback_periods)
        
        if len(recent_bars) < self.lookback_periods:
            raise ValueError(f"Insufficient data: need {self.lookback_periods} bars")
        
        prices = recent_bars['close'].values
        volumes = recent_bars['volume'].values
        highs = recent_bars['high'].values
        lows = recent_bars['low'].values
        current_price = prices[-1]
        
        # Core calculations
        trend_strength = self._calculate_trend_strength(prices)
        volatility = self._calculate_volatility(prices)
        volatility_adjusted_strength = abs(trend_strength) / (volatility + 0.0001)
        
        # Additional metrics for regime
        volatility_state = self._assess_volatility_state(highs, lows, volatility)
        volume_trend = self._assess_volume_trend(volumes)
        
        # Generate regime and bias
        regime, daily_bias, confidence = self._determine_regime(
            trend_strength,
            volatility_adjusted_strength,
            volatility_state,
            volume_trend
        )
        
        return MarketRegimeSignal(
            symbol=symbol,
            timestamp=entry_time,
            price=current_price,
            regime=regime,
            daily_bias=daily_bias,
            confidence=confidence,
            trend_strength=abs(trend_strength),
            volatility_adjusted_strength=volatility_adjusted_strength,
            volatility_state=volatility_state,
            volume_trend=volume_trend
        )
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """
        Trend strength for 15-minute bars
        More weight on overall move for longer timeframe
        """
        # Primary: percentage change
        pct_change = (prices[-1] - prices[0]) / prices[0] * 100
        
        # Confirmation: linear regression
        x = np.arange(len(prices))
        slope, _, r_value, p_value, _ = stats.linregress(x, prices)
        
        # Higher quality requirement for 15-min
        if p_value < 0.05 and r_value**2 > 0.5:
            return pct_change
        elif p_value < 0.10 and r_value**2 > 0.3:
            return pct_change * 0.8
        else:
            return pct_change * 0.6
    
    def _calculate_volatility(self, prices: np.ndarray) -> float:
        """
        Volatility for 15-minute bars
        """
        returns = np.diff(prices) / prices[:-1]
        return returns.std() * 100
    
    def _assess_volatility_state(self, highs: np.ndarray, lows: np.ndarray, 
                                volatility: float) -> str:
        """
        Assess volatility state for regime context
        """
        # Average range as percentage
        avg_range = np.mean((highs - lows) / lows) * 100
        
        # Classify based on both volatility and range
        if volatility > 2.0 or avg_range > 3.0:
            return 'EXTREME'
        elif volatility > 1.0 or avg_range > 1.5:
            return 'HIGH'
        elif volatility < 0.5 and avg_range < 0.75:
            return 'LOW'
        else:
            return 'NORMAL'
    
    def _assess_volume_trend(self, volumes: np.ndarray) -> str:
        """
        Simple volume trend assessment
        """
        # Compare first third, middle third, last third
        third = len(volumes) // 3
        early = volumes[:third].mean()
        middle = volumes[third:2*third].mean()
        late = volumes[2*third:].mean()
        
        # Trend detection
        if late > middle > early and late > early * 1.2:
            return 'INCREASING'
        elif late < middle < early and late < early * 0.8:
            return 'DECREASING'
        else:
            return 'STABLE'
    
    def _determine_regime(self, trend_strength: float,
                         volatility_adjusted_strength: float,
                         volatility_state: str,
                         volume_trend: str) -> tuple[str, str, float]:
        """
        Determine market regime and daily trading bias
        """
        # Base confidence on strength and volatility state
        confidence = min(100, volatility_adjusted_strength * 35)
        
        # Volatility adjustments
        if volatility_state == 'EXTREME':
            confidence *= 0.7  # Less confident in extreme volatility
        elif volatility_state == 'LOW':
            confidence *= 1.1  # More confident in low volatility
        
        # Volume confirmation
        if volume_trend == 'INCREASING':
            confidence = min(100, confidence * 1.1)
        
        # Determine regime
        if volatility_adjusted_strength >= self.regime_thresholds['trending']:
            if trend_strength > 0:
                regime = 'BULL MARKET'
                daily_bias = 'LONG ONLY' if confidence > 70 else 'LONG BIAS'
            else:
                regime = 'BEAR MARKET'
                daily_bias = 'SHORT ONLY' if confidence > 70 else 'SHORT BIAS'
                
        elif volatility_adjusted_strength <= self.regime_thresholds['ranging']:
            regime = 'RANGE BOUND'
            if volatility_state in ['LOW', 'NORMAL']:
                daily_bias = 'BOTH WAYS'
            else:
                daily_bias = 'STAY OUT'  # Too volatile for range trading
            confidence *= 0.8
            
        else:
            # Transitioning
            regime = 'RANGE BOUND'  # Default to range until clear trend
            if trend_strength > 0:
                daily_bias = 'LONG BIAS'
            elif trend_strength < 0:
                daily_bias = 'SHORT BIAS'
            else:
                daily_bias = 'BOTH WAYS'
            confidence *= 0.6
        
        # Override for extreme volatility
        if volatility_state == 'EXTREME' and regime == 'RANGE BOUND':
            daily_bias = 'STAY OUT'
            confidence *= 0.5
        
        return regime, daily_bias, confidence