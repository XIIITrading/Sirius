# modules/calculations/indicators/m5_ema.py
"""
M5 EMA Crossover Calculation Module
Handles 5-minute EMA calculations with 1-minute to 5-minute aggregation
Follows the same pattern as M1 EMA for consistency
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class M5EMAResult:
    """Result of M5 EMA calculation"""
    signal: str  # 'BULL', 'BEAR', 'NEUTRAL'
    signal_strength: float
    ema_9: float
    ema_21: float
    spread: float
    spread_pct: float
    trend_strength: float
    is_crossover: bool
    crossover_type: str  # 'bullish', 'bearish', or ''
    price_position: str  # 'above', 'below', 'at'
    reason: str
    bars_processed: int
    last_5min_close: float
    last_5min_volume: float


class M5EMACalculator:
    """
    Calculates 5-minute EMA crossover signals from 1-minute data.
    
    Process:
    1. Aggregate 1-minute bars to 5-minute bars
    2. Calculate 9 and 21 period EMAs on 5-minute data
    3. Detect crossovers and trend strength
    4. Generate trading signals
    """
    
    def __init__(self, ema_short: int = 9, ema_long: int = 21):
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.min_bars_required = ema_long + 5  # Need extra bars for stable EMA
        
    def calculate(self, bars_1min: pd.DataFrame) -> Optional[M5EMAResult]:
        """
        Main calculation method.
        
        Args:
            bars_1min: DataFrame with 1-minute OHLCV data (index=timestamp)
            
        Returns:
            M5EMAResult or None if insufficient data
        """
        try:
            # Validate input data
            if bars_1min.empty:
                logger.warning("Empty DataFrame provided")
                return None
                
            # Aggregate to 5-minute bars
            bars_5min = self._aggregate_to_5min(bars_1min)
            
            if len(bars_5min) < self.min_bars_required:
                logger.warning(f"Insufficient 5-minute bars: {len(bars_5min)} < {self.min_bars_required}")
                return None
            
            # Calculate EMAs
            ema_short_series = self._calculate_ema(bars_5min['close'], self.ema_short)
            ema_long_series = self._calculate_ema(bars_5min['close'], self.ema_long)
            
            # Get latest values
            ema_9 = ema_short_series.iloc[-1]
            ema_21 = ema_long_series.iloc[-1]
            last_close = bars_5min['close'].iloc[-1]
            last_volume = bars_5min['volume'].iloc[-1]
            
            # Calculate spread
            spread = ema_9 - ema_21
            spread_pct = (spread / ema_21) * 100 if ema_21 != 0 else 0
            
            # Determine trend and signal
            is_bullish = ema_9 > ema_21
            signal = 'BULL' if is_bullish else 'BEAR'
            
            # Check for crossover
            is_crossover, crossover_type = self._detect_crossover(
                ema_short_series, ema_long_series
            )
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(
                bars_5min, ema_short_series, ema_long_series
            )
            
            # Determine price position relative to EMA 9
            if last_close > ema_9:
                price_position = 'above'
            elif last_close < ema_9:
                price_position = 'below'
            else:
                price_position = 'at'
            
            # Calculate signal strength
            signal_strength = self._calculate_signal_strength(
                spread_pct, trend_strength, is_crossover, price_position, signal
            )
            
            # Build reason
            reason = self._build_reason(
                signal, ema_9, ema_21, spread_pct, 
                is_crossover, crossover_type, price_position
            )
            
            return M5EMAResult(
                signal=signal,
                signal_strength=signal_strength,
                ema_9=round(ema_9, 2),
                ema_21=round(ema_21, 2),
                spread=round(spread, 2),
                spread_pct=round(spread_pct, 2),
                trend_strength=round(trend_strength, 1),
                is_crossover=is_crossover,
                crossover_type=crossover_type,
                price_position=price_position,
                reason=reason,
                bars_processed=len(bars_5min),
                last_5min_close=round(last_close, 2),
                last_5min_volume=round(last_volume, 0)
            )
            
        except Exception as e:
            logger.error(f"Error in M5 EMA calculation: {e}")
            return None
    
    def _aggregate_to_5min(self, bars_1min: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate 1-minute bars to 5-minute bars.
        
        Args:
            bars_1min: 1-minute OHLCV data
            
        Returns:
            5-minute aggregated bars
        """
        # Define aggregation rules
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Resample to 5-minute bars
        bars_5min = bars_1min.resample('5T', label='right', closed='right').agg(agg_rules)
        
        # Remove any bars with no data (NaN in OHLC)
        bars_5min = bars_5min.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Ensure volume is not NaN
        bars_5min['volume'] = bars_5min['volume'].fillna(0)
        
        logger.info(f"Aggregated {len(bars_1min)} 1-min bars to {len(bars_5min)} 5-min bars")
        
        return bars_5min
    
    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    def _detect_crossover(self, ema_short: pd.Series, ema_long: pd.Series) -> Tuple[bool, str]:
        """
        Detect if a crossover occurred in the last few bars.
        
        Returns:
            (is_crossover, crossover_type)
        """
        if len(ema_short) < 3:
            return False, ''
        
        # Check last 3 bars for crossover
        for i in range(-3, -1):
            prev_short = ema_short.iloc[i]
            prev_long = ema_long.iloc[i]
            curr_short = ema_short.iloc[i + 1]
            curr_long = ema_long.iloc[i + 1]
            
            # Bullish crossover: short crosses above long
            if prev_short <= prev_long and curr_short > curr_long:
                return True, 'bullish'
            
            # Bearish crossover: short crosses below long
            if prev_short >= prev_long and curr_short < curr_long:
                return True, 'bearish'
        
        return False, ''
    
    def _calculate_trend_strength(self, bars: pd.DataFrame, 
                                 ema_short: pd.Series, 
                                 ema_long: pd.Series) -> float:
        """
        Calculate trend strength based on multiple factors.
        
        Returns:
            Trend strength 0-100
        """
        # Factor 1: EMA separation (40% weight)
        spread_pct = abs((ema_short.iloc[-1] - ema_long.iloc[-1]) / ema_long.iloc[-1] * 100)
        spread_score = min(spread_pct * 10, 40)  # Max 40 points
        
        # Factor 2: EMA alignment over recent bars (30% weight)
        recent_bars = min(10, len(ema_short))
        aligned_bars = 0
        for i in range(-recent_bars, 0):
            if (ema_short.iloc[i] > ema_long.iloc[i]) == (ema_short.iloc[-1] > ema_long.iloc[-1]):
                aligned_bars += 1
        alignment_score = (aligned_bars / recent_bars) * 30
        
        # Factor 3: Price trend consistency (30% weight)
        if len(bars) >= 10:
            recent_closes = bars['close'].iloc[-10:]
            price_trend = (recent_closes.iloc[-1] - recent_closes.iloc[0]) / recent_closes.iloc[0] * 100
            ema_trend = (ema_short.iloc[-1] > ema_long.iloc[-1])
            
            if (price_trend > 0 and ema_trend) or (price_trend < 0 and not ema_trend):
                consistency_score = 30
            else:
                consistency_score = 10
        else:
            consistency_score = 15
        
        trend_strength = spread_score + alignment_score + consistency_score
        return min(100, max(0, trend_strength))
    
    def _calculate_signal_strength(self, spread_pct: float, trend_strength: float,
                                  is_crossover: bool, price_position: str, 
                                  signal: str) -> float:
        """
        Calculate overall signal strength.
        
        Returns:
            Signal strength 0-100
        """
        # Base strength from trend
        strength = trend_strength
        
        # Adjust for crossover
        if is_crossover:
            strength = min(100, strength * 1.2)
        
        # Adjust for price position
        if signal == 'BULL' and price_position == 'below':
            strength *= 0.7  # Reduce strength if price below EMA in uptrend
        elif signal == 'BEAR' and price_position == 'above':
            strength *= 0.7  # Reduce strength if price above EMA in downtrend
        
        # Ensure minimum strength for clear trends
        if abs(spread_pct) > 0.5 and strength < 30:
            strength = 30
        
        return min(100, max(0, strength))
    
    def _build_reason(self, signal: str, ema_9: float, ema_21: float,
                     spread_pct: float, is_crossover: bool, crossover_type: str,
                     price_position: str) -> str:
        """Build human-readable reason for signal"""
        reasons = []
        
        # Primary trend
        if signal == 'BULL':
            reasons.append(f"5m EMA 9 ({ema_9:.2f}) > EMA 21 ({ema_21:.2f})")
        else:
            reasons.append(f"5m EMA 9 ({ema_9:.2f}) < EMA 21 ({ema_21:.2f})")
        
        # Spread info
        reasons.append(f"Spread: {abs(spread_pct):.2f}%")
        
        # Crossover
        if is_crossover:
            reasons.append(f"Recent {crossover_type} crossover")
        
        # Price position
        reasons.append(f"Price {price_position} EMA 9")
        
        return " | ".join(reasons)


def validate_5min_data(bars_1min: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate 1-minute data before processing.
    
    Args:
        bars_1min: 1-minute bar data
        
    Returns:
        (is_valid, error_message)
    """
    if bars_1min.empty:
        return False, "No data provided"
    
    # Check required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_columns if col not in bars_1min.columns]
    if missing:
        return False, f"Missing required columns: {missing}"
    
    # Check minimum data points (need at least 26*5 = 130 1-min bars)
    if len(bars_1min) < 130:
        return False, f"Insufficient data: {len(bars_1min)} bars, need at least 130"
    
    # Check for timezone awareness
    if bars_1min.index.tz is None:
        return False, "Timestamps must be timezone-aware"
    
    return True, ""