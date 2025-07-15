# modules/calculations/trend/statistical_trend_15min.py
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
    signal: str  # 'BUY', 'WEAK BUY', 'SELL', 'WEAK SELL'
    confidence: float  # 0-100
    trend_strength: float  # Magnitude of move
    volatility_adjusted_strength: float  # Trend strength relative to noise
    volatility_state: str  # 'LOW', 'NORMAL', 'HIGH', 'EXTREME'
    volume_trend: str  # 'INCREASING', 'DECREASING', 'STABLE'
    volume_confirmation: bool  # Added to match 1min structure


class StatisticalTrend15Min:
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
        
        # Match 1min thresholds for signal generation
        self.strength_thresholds = {
            'normal': 1.0,     # Trend >= 1x volatility for BUY/SELL
            'weak': 0.25       # Trend >= 0.25x volatility for WEAK signals
        }
        
        # Keep regime thresholds for regime detection
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
        volume_confirmation = self._check_volume_confirmation(prices, volumes)
        
        # Additional metrics for regime
        volatility_state = self._assess_volatility_state(highs, lows, volatility)
        volume_trend = self._assess_volume_trend(volumes)
        
        # Generate regime and bias (keep for additional context)
        regime, daily_bias, regime_confidence = self._determine_regime(
            trend_strength,
            volatility_adjusted_strength,
            volatility_state,
            volume_trend
        )
        
        # Generate signal using exact 1min logic
        signal, signal_confidence = self._generate_signal(
            trend_strength,
            volatility_adjusted_strength,
            volume_confirmation
        )
        
        return MarketRegimeSignal(
            symbol=symbol,
            timestamp=entry_time,
            price=current_price,
            regime=regime,
            daily_bias=daily_bias,
            signal=signal,
            confidence=signal_confidence,  # Use signal confidence to match 1min
            trend_strength=abs(trend_strength),
            volatility_adjusted_strength=volatility_adjusted_strength,
            volatility_state=volatility_state,
            volume_trend=volume_trend,
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
        Note: This is kept for regime/bias context but confidence is now from signal generation
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