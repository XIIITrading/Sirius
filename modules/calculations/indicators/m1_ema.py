# modules/calculations/indicators/m1_ema.py
"""
Module: 1-Minute EMA Crossover Calculation
Purpose: Pure calculation logic for 9/21 EMA crossover on 1-minute data
Features: EMA calculation, crossover detection, trend strength
Output: Standardized calculation results
Note: This is a pure calculation module - no data fetching, no testing
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime


@dataclass
class EMAResult:
    """Result of 1-minute EMA calculation"""
    timestamp: datetime
    ema_9: float
    ema_21: float
    ema_9_prev: float
    ema_21_prev: float
    last_price: float
    is_bullish: bool
    is_crossover: bool
    crossover_type: Optional[str]  # 'bullish', 'bearish', None
    spread: float
    spread_pct: float
    trend_strength: float  # 0-100
    price_position: str  # 'above_ema9', 'below_ema9', 'between'
    signal: str  # 'BULL', 'BEAR', 'NEUTRAL'
    signal_strength: float  # 0-100
    reason: str


class M1EMACalculator:
    """
    Pure 1-minute EMA crossover calculation logic.
    No data fetching, no external dependencies.
    """
    
    def __init__(self, short_period: int = 9, long_period: int = 21):
        """
        Initialize calculator with EMA periods
        
        Args:
            short_period: Short EMA period (default 9)
            long_period: Long EMA period (default 21)
        """
        self.short_period = short_period
        self.long_period = long_period
        self.min_periods = max(short_period, long_period)
    
    def calculate(self, prices: List[float], 
                  timestamps: Optional[List[datetime]] = None) -> Optional[EMAResult]:
        """
        Calculate EMA crossover from 1-minute price series
        
        Args:
            prices: List of closing prices (oldest to newest)
            timestamps: Optional list of timestamps corresponding to prices
            
        Returns:
            EMAResult if enough data, None otherwise
        """
        if len(prices) < self.min_periods:
            return None
        
        # Calculate EMAs
        ema_short_series = self._calculate_ema_series(prices, self.short_period)
        ema_long_series = self._calculate_ema_series(prices, self.long_period)
        
        if not ema_short_series or not ema_long_series:
            return None
        
        # Get current and previous values
        current_short = ema_short_series[-1]
        current_long = ema_long_series[-1]
        prev_short = ema_short_series[-2] if len(ema_short_series) > 1 else current_short
        prev_long = ema_long_series[-2] if len(ema_long_series) > 1 else current_long
        last_price = prices[-1]
        
        # Determine trend and crossover
        is_bullish = current_short > current_long
        was_bullish = prev_short > prev_long
        is_crossover = is_bullish != was_bullish
        crossover_type = None
        
        if is_crossover:
            crossover_type = 'bullish' if is_bullish else 'bearish'
        
        # Calculate spread metrics
        spread = current_short - current_long
        spread_pct = (spread / last_price * 100) if last_price > 0 else 0
        trend_strength = min(100, abs(spread_pct) * 20)  # 5% spread = 100 strength
        
        # Determine price position
        price_position = self._get_price_position(last_price, current_short, current_long)
        
        # Generate signal
        signal, signal_strength, reason = self._generate_signal(
            is_bullish, last_price, current_short, current_long, 
            spread_pct, trend_strength, is_crossover, crossover_type
        )
        
        # Get timestamp
        timestamp = timestamps[-1] if timestamps else datetime.utcnow()
        
        return EMAResult(
            timestamp=timestamp,
            ema_9=current_short,
            ema_21=current_long,
            ema_9_prev=prev_short,
            ema_21_prev=prev_long,
            last_price=last_price,
            is_bullish=is_bullish,
            is_crossover=is_crossover,
            crossover_type=crossover_type,
            spread=spread,
            spread_pct=spread_pct,
            trend_strength=trend_strength,
            price_position=price_position,
            signal=signal,
            signal_strength=signal_strength,
            reason=reason
        )
    
    def calculate_batch(self, prices: List[float], 
                       timestamps: Optional[List[datetime]] = None) -> List[EMAResult]:
        """
        Calculate EMA results for each point in the price series
        
        Args:
            prices: List of closing prices
            timestamps: Optional list of timestamps
            
        Returns:
            List of EMAResult for each valid calculation point
        """
        results = []
        
        # Need at least min_periods to start
        if len(prices) < self.min_periods:
            return results
        
        # Calculate for each point from min_periods onwards
        for i in range(self.min_periods, len(prices) + 1):
            price_slice = prices[:i]
            time_slice = timestamps[:i] if timestamps else None
            
            result = self.calculate(price_slice, time_slice)
            if result:
                results.append(result)
        
        return results
    
    def _calculate_ema_series(self, prices: List[float], period: int) -> List[float]:
        """Calculate EMA series for given prices and period"""
        if len(prices) < period:
            return []
        
        ema = []
        multiplier = 2 / (period + 1)
        
        # Start with SMA
        sma = sum(prices[:period]) / period
        ema.append(sma)
        
        # Calculate EMA for remaining prices
        for i in range(period, len(prices)):
            ema_val = (prices[i] - ema[-1]) * multiplier + ema[-1]
            ema.append(ema_val)
        
        return ema
    
    def _get_price_position(self, price: float, ema_short: float, ema_long: float) -> str:
        """Determine price position relative to EMAs"""
        if price > ema_short and price > ema_long:
            return "above_both"
        elif price < ema_short and price < ema_long:
            return "below_both"
        elif price > ema_short:
            return "above_ema9"
        elif price < ema_short:
            return "below_ema9"
        else:
            return "at_ema9"
    
    def _generate_signal(self, is_bullish: bool, last_price: float,
                        ema_short: float, ema_long: float,
                        spread_pct: float, trend_strength: float,
                        is_crossover: bool, crossover_type: Optional[str]) -> Tuple[str, float, str]:
        """
        Generate trading signal based on EMA analysis
        
        Returns:
            (signal, strength, reason)
        """
        reasons = []
        
        # Base signal from EMA relationship
        if is_bullish:
            base_signal = 'BULL'
            reasons.append(f"EMA 9 ({ema_short:.2f}) > EMA 21 ({ema_long:.2f})")
            
            # Check price position for confirmation
            if last_price < ema_short:
                signal = 'NEUTRAL'
                reasons.append(f"But price ({last_price:.2f}) below EMA 9")
                strength = 30
            else:
                signal = 'BULL'
                strength = trend_strength
        else:
            base_signal = 'BEAR'
            reasons.append(f"EMA 9 ({ema_short:.2f}) < EMA 21 ({ema_long:.2f})")
            
            # Check price position for confirmation
            if last_price > ema_short:
                signal = 'NEUTRAL'
                reasons.append(f"But price ({last_price:.2f}) above EMA 9")
                strength = 30
            else:
                signal = 'BEAR'
                strength = trend_strength
        
        # Add spread information
        reasons.append(f"Spread: {spread_pct:.2f}%")
        
        # Boost strength for recent crossover
        if is_crossover and crossover_type:
            reasons.append(f"Fresh {crossover_type} crossover")
            strength = min(100, strength * 1.2)
        
        reason = " | ".join(reasons)
        return signal, strength, reason
    
    def get_ema_values(self, prices: List[float]) -> Optional[Dict[str, float]]:
        """
        Get just the current EMA values without full analysis
        
        Args:
            prices: List of closing prices
            
        Returns:
            Dict with ema_9 and ema_21 values, or None if insufficient data
        """
        if len(prices) < self.min_periods:
            return None
        
        ema_short_series = self._calculate_ema_series(prices, self.short_period)
        ema_long_series = self._calculate_ema_series(prices, self.long_period)
        
        if not ema_short_series or not ema_long_series:
            return None
        
        return {
            'ema_9': ema_short_series[-1],
            'ema_21': ema_long_series[-1],
            'count': len(prices)
        }


# Utility functions for working with 1-minute data
def validate_1min_data(prices: List[float], timestamps: List[datetime]) -> Tuple[bool, str]:
    """
    Validate that data is suitable for 1-minute EMA calculation
    
    Returns:
        (is_valid, error_message)
    """
    if not prices:
        return False, "No price data provided"
    
    if len(prices) < 21:  # Minimum for 21-period EMA
        return False, f"Insufficient data: {len(prices)} candles (need at least 21)"
    
    if timestamps and len(timestamps) != len(prices):
        return False, "Price and timestamp arrays must have same length"
    
    # Check for invalid prices
    if any(p <= 0 for p in prices):
        return False, "Invalid prices detected (zero or negative)"
    
    # Check timestamp spacing if provided
    if timestamps and len(timestamps) > 1:
        # Check if roughly 1-minute intervals
        intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(interval)
        
        avg_interval = sum(intervals) / len(intervals)
        if avg_interval < 50 or avg_interval > 70:  # Allow 50-70 seconds
            return False, f"Data doesn't appear to be 1-minute intervals (avg: {avg_interval:.1f}s)"
    
    return True, ""


def extract_signal_metrics(result: EMAResult) -> Dict[str, any]:
    """Extract key metrics from EMA result for display"""
    return {
        'direction': result.signal,
        'strength': result.signal_strength,
        'ema_9': result.ema_9,
        'ema_21': result.ema_21,
        'spread': result.spread,
        'spread_pct': result.spread_pct,
        'trend_strength': result.trend_strength,
        'is_crossover': result.is_crossover,
        'crossover_type': result.crossover_type,
        'price_position': result.price_position,
        'reason': result.reason
    }