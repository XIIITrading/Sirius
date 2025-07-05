# modules/calculations/indicators/m15_ema.py
"""
M15 EMA Crossover Calculation Module
Handles 15-minute EMA calculations with optional 1-minute to 15-minute aggregation
Follows the same pattern as M5 EMA for consistency
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class M15EMAResult:
    """Result of M15 EMA calculation"""
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
    last_15min_close: float
    last_15min_volume: float
    timestamp: datetime


class M15EMACalculator:
    """
    Calculates 15-minute EMA crossover signals.
    
    Process:
    1. Accept 15-minute bars (or aggregate from 1-minute bars)
    2. Calculate 9 and 21 period EMAs on 15-minute data
    3. Detect crossovers and trend strength
    4. Generate trading signals with NEUTRAL override logic
    """
    
    def __init__(self, ema_short: int = 9, ema_long: int = 21):
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.min_bars_required = ema_long + 5  # Need extra bars for stable EMA
        
    def calculate(self, bars: pd.DataFrame, timeframe: str = '15min') -> Optional[M15EMAResult]:
        """
        Main calculation method.
        
        Args:
            bars: DataFrame with OHLCV data (index=timestamp)
            timeframe: '1min' or '15min' - indicates input data timeframe
            
        Returns:
            M15EMAResult or None if insufficient data
        """
        try:
            # Validate input data
            if bars.empty:
                logger.warning("Empty DataFrame provided")
                return None
            
            # Aggregate to 15-minute bars if needed
            if timeframe == '1min':
                bars_15min = self._aggregate_to_15min(bars)
            else:
                bars_15min = bars.copy()
                
            if len(bars_15min) < self.min_bars_required:
                logger.warning(f"Insufficient 15-minute bars: {len(bars_15min)} < {self.min_bars_required}")
                return None
            
            # Calculate EMAs
            ema_short_series = self._calculate_ema(bars_15min['close'], self.ema_short)
            ema_long_series = self._calculate_ema(bars_15min['close'], self.ema_long)
            
            # Get latest values
            ema_9 = ema_short_series.iloc[-1]
            ema_21 = ema_long_series.iloc[-1]
            last_close = bars_15min['close'].iloc[-1]
            last_volume = bars_15min['volume'].iloc[-1]
            last_timestamp = bars_15min.index[-1]
            
            # Ensure timestamp is UTC
            if last_timestamp.tzinfo is None:
                last_timestamp = last_timestamp.replace(tzinfo=timezone.utc)
            elif last_timestamp.tzinfo != timezone.utc:
                last_timestamp = last_timestamp.astimezone(timezone.utc)
            
            # Calculate spread
            spread = ema_9 - ema_21
            spread_pct = (spread / ema_21) * 100 if ema_21 != 0 else 0
            
            # Determine base trend
            is_bullish = ema_9 > ema_21
            
            # Check for crossover
            is_crossover, crossover_type = self._detect_crossover(
                ema_short_series, ema_long_series
            )
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(
                bars_15min, ema_short_series, ema_long_series
            )
            
            # Determine price position relative to EMA 9
            if last_close > ema_9:
                price_position = 'above'
            elif last_close < ema_9:
                price_position = 'below'
            else:
                price_position = 'at'
            
            # Determine signal with NEUTRAL override logic
            signal = self._determine_signal(
                is_bullish, price_position, last_close, ema_9
            )
            
            # Calculate signal strength
            signal_strength = self._calculate_signal_strength(
                spread_pct, trend_strength, is_crossover, price_position, signal
            )
            
            # Build reason
            reason = self._build_reason(
                signal, ema_9, ema_21, spread_pct, 
                is_crossover, crossover_type, price_position,
                is_bullish
            )
            
            return M15EMAResult(
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
                bars_processed=len(bars_15min),
                last_15min_close=round(last_close, 2),
                last_15min_volume=round(last_volume, 0),
                timestamp=last_timestamp
            )
            
        except Exception as e:
            logger.error(f"Error in M15 EMA calculation: {e}")
            return None
    
    def _aggregate_to_15min(self, bars_1min: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate 1-minute bars to 15-minute bars.
        
        Args:
            bars_1min: 1-minute OHLCV data
            
        Returns:
            15-minute aggregated bars
        """
        # Define aggregation rules
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Resample to 15-minute bars
        bars_15min = bars_1min.resample('15T', label='right', closed='right').agg(agg_rules)
        
        # Remove any bars with no data (NaN in OHLC)
        bars_15min = bars_15min.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Ensure volume is not NaN
        bars_15min['volume'] = bars_15min['volume'].fillna(0)
        
        logger.info(f"Aggregated {len(bars_1min)} 1-min bars to {len(bars_15min)} 15-min bars")
        
        return bars_15min
    
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
    
    def _determine_signal(self, is_bullish: bool, price_position: str, 
                         last_close: float, ema_9: float) -> str:
        """
        Determine signal with NEUTRAL override logic.
        
        Returns:
            'BULL', 'BEAR', or 'NEUTRAL'
        """
        if is_bullish:
            # EMA 9 > EMA 21 (bullish setup)
            if price_position == 'below':
                # Price below EMA 9 in uptrend = NEUTRAL
                return 'NEUTRAL'
            else:
                return 'BULL'
        else:
            # EMA 9 < EMA 21 (bearish setup)
            if price_position == 'above':
                # Price above EMA 9 in downtrend = NEUTRAL
                return 'NEUTRAL'
            else:
                return 'BEAR'
    
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
        
        # NEUTRAL signals get lower strength
        if signal == 'NEUTRAL':
            strength = min(30, strength * 0.5)
        
        # Ensure minimum strength for clear trends
        if abs(spread_pct) > 0.5 and signal != 'NEUTRAL' and strength < 30:
            strength = 30
        
        return min(100, max(0, strength))
    
    def _build_reason(self, signal: str, ema_9: float, ema_21: float,
                     spread_pct: float, is_crossover: bool, crossover_type: str,
                     price_position: str, is_bullish: bool) -> str:
        """Build human-readable reason for signal"""
        reasons = []
        
        # Primary trend
        if is_bullish:
            reasons.append(f"15m EMA 9 ({ema_9:.2f}) > EMA 21 ({ema_21:.2f})")
        else:
            reasons.append(f"15m EMA 9 ({ema_9:.2f}) < EMA 21 ({ema_21:.2f})")
        
        # NEUTRAL override explanation
        if signal == 'NEUTRAL':
            if is_bullish and price_position == 'below':
                reasons.append(f"But price ({price_position} EMA 9) → NEUTRAL")
            elif not is_bullish and price_position == 'above':
                reasons.append(f"But price ({price_position} EMA 9) → NEUTRAL")
        
        # Spread info
        reasons.append(f"Spread: {abs(spread_pct):.2f}%")
        
        # Crossover
        if is_crossover:
            reasons.append(f"Recent {crossover_type} crossover")
        
        # Price position (if not already mentioned for NEUTRAL)
        if signal != 'NEUTRAL':
            reasons.append(f"Price {price_position} EMA 9")
        
        return " | ".join(reasons)


def validate_15min_data(bars: pd.DataFrame, timeframe: str = '15min') -> Tuple[bool, str]:
    """
    Validate data before processing.
    
    Args:
        bars: Bar data
        timeframe: Expected timeframe ('1min' or '15min')
        
    Returns:
        (is_valid, error_message)
    """
    if bars.empty:
        return False, "No data provided"
    
    # Check required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_columns if col not in bars.columns]
    if missing:
        return False, f"Missing required columns: {missing}"
    
    # Check minimum data points
    if timeframe == '1min':
        # Need at least 26*15 = 390 1-min bars
        min_bars = 390
    else:
        # Need at least 26 15-min bars
        min_bars = 26
        
    if len(bars) < min_bars:
        return False, f"Insufficient data: {len(bars)} bars, need at least {min_bars}"
    
    # Check for timezone awareness
    if bars.index.tz is None:
        return False, "Timestamps must be timezone-aware"
    
    return True, ""