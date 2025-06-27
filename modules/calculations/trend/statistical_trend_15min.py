# modules/calculations/trend/statistical_trend_15min.py
"""
Module: 15-Minute Statistical Trend Calculation for Market Regime Analysis
Purpose: Provide broad market perspective and daily trading bias
Features: Multi-timeframe analysis (45/75/150 min), Market regime detection, Daily bias
Performance Target: <100 microseconds per calculation
Time Handling: All timestamps in UTC
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone, time
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import logging
import sys
import os
import asyncio
import time as time_module
from scipy import stats
from numba import jit

# Fix import path
current_dir = os.path.dirname(os.path.abspath(__file__))
calculations_dir = os.path.dirname(current_dir)
modules_dir = os.path.dirname(calculations_dir)
vega_root = os.path.dirname(modules_dir)

if vega_root not in sys.path:
    sys.path.insert(0, vega_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrendComponents15Min:
    """Individual trend calculation components for 15-min timeframe"""
    linear_slope: float
    linear_r_squared: float
    mann_kendall_trend: int  # -1, 0, 1
    mann_kendall_z_score: float
    mann_kendall_p_value: float
    kalman_trend: int  # -1, 0, 1
    kalman_price: float
    price_momentum: float
    price_acceleration: float
    vwap_position: float  # Price relative to VWAP
    volume_trend: float  # Volume trend direction
    range_expansion: float  # Volatility expansion/contraction


@dataclass
class StatisticalTrendResult15Min:
    """Complete statistical trend analysis result for 15-minute timeframe"""
    symbol: str
    timestamp: datetime
    current_price: float
    lookback_periods: int
    components: TrendComponents15Min
    composite_trend: float  # -1 to 1, strength of trend
    trend_direction: str  # 'bullish', 'bearish', 'neutral'
    trend_strength: float  # 0 to 100
    confidence_level: float  # 0 to 100
    calculation_time_ms: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class MarketRegimeSignal:
    """Market regime signal for daily trading decisions"""
    symbol: str
    timestamp: datetime
    price: float
    regime: str  # 'BULL MARKET', 'BEAR MARKET', 'RANGE BOUND', 'TRANSITIONING'
    daily_bias: str  # 'LONG ONLY', 'SHORT ONLY', 'BOTH WAYS', 'STAY OUT'
    strength: float
    confidence: float
    volatility_state: str  # 'LOW', 'NORMAL', 'HIGH', 'EXTREME'
    key_levels: Dict  # Important price levels
    trading_notes: str  # Specific guidance for the day
    short_trend: Dict  # 45-min (3 x 15-min)
    medium_trend: Dict  # 75-min (5 x 15-min)
    long_trend: Dict  # 150-min (10 x 15-min)


class StatisticalTrend15Min:
    """
    15-Minute Statistical Trend Calculator for market regime analysis.
    Provides daily trading bias and major trend identification.
    """
    
    def __init__(self,
                 short_lookback: int = 3,      # 45-min trend
                 medium_lookback: int = 5,      # 75-min trend
                 long_lookback: int = 10,       # 150-min trend
                 calculation_interval: int = 60):  # Calculate every minute
        """
        Initialize 15-minute trend calculator.
        
        Args:
            short_lookback: 45-min trend (3 x 15-min bars)
            medium_lookback: 75-min trend (5 x 15-min bars)
            long_lookback: 150-min trend (10 x 15-min bars)
            calculation_interval: Seconds between calculations
        """
        self.short_lookback = short_lookback
        self.medium_lookback = medium_lookback
        self.long_lookback = long_lookback
        self.calculation_interval = calculation_interval
        
        # Use long as max buffer size
        self.max_lookback = max(long_lookback, 20)  # Keep extra for calculations
        self.warmup_periods = short_lookback  # Can start after 45 minutes
        
        # Data buffers
        self.price_buffers: Dict[str, deque] = {}
        self.volume_buffers: Dict[str, deque] = {}
        self.timestamp_buffers: Dict[str, deque] = {}
        self.high_buffers: Dict[str, deque] = {}
        self.low_buffers: Dict[str, deque] = {}
        self.open_buffers: Dict[str, deque] = {}
        
        # Calculated values
        self.vwap_buffers: Dict[str, deque] = {}
        self.atr_buffers: Dict[str, deque] = {}
        
        # Kalman filter states
        self.kalman_states: Dict[str, Dict] = {}
        
        # Key levels tracking
        self.key_levels: Dict[str, Dict] = {}
        
        # WebSocket integration
        self.ws_client = None
        self.active_symbols: Dict[str, List[Callable]] = {}
        self.calculation_tasks: Dict[str, asyncio.Task] = {}
        
        # Latest results
        self.latest_signals: Dict[str, MarketRegimeSignal] = {}
        
        # Performance tracking
        self.calculation_count = 0
        self.total_calculation_time = 0
        
        logger.info(f"Initialized 15-min trend calculator: "
                   f"Short={short_lookback*15}min, Medium={medium_lookback*15}min, Long={long_lookback*15}min")
    
    def initialize_buffers(self, symbol: str):
        """Initialize data buffers for a symbol"""
        self.price_buffers[symbol] = deque(maxlen=self.max_lookback)
        self.volume_buffers[symbol] = deque(maxlen=self.max_lookback)
        self.timestamp_buffers[symbol] = deque(maxlen=self.max_lookback)
        self.high_buffers[symbol] = deque(maxlen=self.max_lookback)
        self.low_buffers[symbol] = deque(maxlen=self.max_lookback)
        self.open_buffers[symbol] = deque(maxlen=self.max_lookback)
        self.vwap_buffers[symbol] = deque(maxlen=self.max_lookback)
        self.atr_buffers[symbol] = deque(maxlen=self.max_lookback)
        self.kalman_states[symbol] = self._initialize_kalman()
        self.key_levels[symbol] = {}
        logger.info(f"Initialized 15-min buffers for {symbol}")
    
    def _initialize_kalman(self) -> Dict:
        """Initialize Kalman filter state for 15-min bars"""
        return {
            'x': 0.0,      # State estimate
            'P': 1.0,      # Error covariance
            'Q': 0.0005,   # Higher process noise for 15-min bars
            'R': 0.05,     # Higher measurement noise
            'K': 0.0       # Kalman gain
        }
    
    @staticmethod
    @jit(nopython=True)
    def _fast_linear_regression(prices: np.ndarray) -> Tuple[float, float, float]:
        """JIT-compiled linear regression"""
        n = len(prices)
        if n < 2:
            return 0.0, 0.0, 0.0
            
        x = np.arange(n, dtype=np.float64)
        x_mean = x.mean()
        y_mean = prices.mean()
        
        xy_cov = np.sum((x - x_mean) * (prices - y_mean))
        xx_var = np.sum((x - x_mean) ** 2)
        
        if xx_var == 0:
            return 0.0, y_mean, 0.0
            
        slope = xy_cov / xx_var
        intercept = y_mean - slope * x_mean
        
        y_pred = slope * x + intercept
        ss_res = np.sum((prices - y_pred) ** 2)
        ss_tot = np.sum((prices - y_mean) ** 2)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        return slope, intercept, r_squared
    
    @staticmethod
    @jit(nopython=True)
    def _fast_mann_kendall(data: np.ndarray) -> Tuple[int, float, float]:
        """Optimized Mann-Kendall test"""
        n = len(data)
        if n < 3:
            return 0, 0.0, 1.0
            
        s = 0
        for i in range(n-1):
            for j in range(i+1, n):
                diff = data[j] - data[i]
                if diff > 0:
                    s += 1
                elif diff < 0:
                    s -= 1
        
        var_s = n * (n - 1) * (2 * n + 5) / 18.0
        
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0.0
        
        p_value = 2 * (1 - 0.5 * (1 + np.sign(z) * (1 - np.exp(-2 * z * z / np.pi)) ** 0.5))
        
        if p_value < 0.05:
            trend = 1 if s > 0 else -1
        else:
            trend = 0
            
        return trend, z, p_value
    
    def _update_kalman(self, symbol: str, measurement: float) -> float:
        """Update Kalman filter with new price"""
        state = self.kalman_states[symbol]
        
        state['P'] += state['Q']
        state['K'] = state['P'] / (state['P'] + state['R'])
        state['x'] += state['K'] * (measurement - state['x'])
        state['P'] *= (1 - state['K'])
        
        return state['x']
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, 
                      closes: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(highs) < 2:
            return 0.0
            
        tr_list = []
        for i in range(1, min(len(highs), period + 1)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)
        
        return np.mean(tr_list) if tr_list else 0.0
    
    def _calculate_vwap(self, prices: np.ndarray, volumes: np.ndarray,
                       highs: np.ndarray, lows: np.ndarray) -> float:
        """Calculate VWAP for the period"""
        if len(prices) == 0 or volumes.sum() == 0:
            return prices[-1] if len(prices) > 0 else 0
            
        typical_prices = (highs + lows + prices) / 3
        vwap = np.sum(typical_prices * volumes) / np.sum(volumes)
        
        return vwap
    
    def _identify_key_levels(self, symbol: str) -> Dict:
        """Identify key support/resistance levels"""
        if len(self.high_buffers[symbol]) < self.long_lookback:
            return {}
            
        highs = np.array(list(self.high_buffers[symbol]))
        lows = np.array(list(self.low_buffers[symbol]))
        closes = np.array(list(self.price_buffers[symbol]))
        volumes = np.array(list(self.volume_buffers[symbol]))
        
        # Recent high/low
        recent_high = highs[-self.medium_lookback:].max()
        recent_low = lows[-self.medium_lookback:].min()
        
        # Volume-weighted average price
        vwap = self._calculate_vwap(closes, volumes, highs, lows)
        
        # Pivot points (simplified)
        last_high = highs[-1]
        last_low = lows[-1]
        last_close = closes[-1]
        pivot = (last_high + last_low + last_close) / 3
        
        r1 = 2 * pivot - last_low
        s1 = 2 * pivot - last_high
        
        return {
            'recent_high': recent_high,
            'recent_low': recent_low,
            'vwap': vwap,
            'pivot': pivot,
            'resistance_1': r1,
            'support_1': s1,
            'range': recent_high - recent_low
        }
    
    def update_bar(self, symbol: str, open_price: float, high: float,
                  low: float, close: float, volume: float,
                  timestamp: Optional[datetime] = None) -> Optional[MarketRegimeSignal]:
        """
        Update with new 15-minute bar data.
        
        Returns:
            MarketRegimeSignal with regime analysis
        """
        start_time = time_module.perf_counter()
        
        # Initialize if needed
        if symbol not in self.price_buffers:
            self.initialize_buffers(symbol)
        
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Update buffers
        self.price_buffers[symbol].append(close)
        self.volume_buffers[symbol].append(volume)
        self.timestamp_buffers[symbol].append(timestamp)
        self.high_buffers[symbol].append(high)
        self.low_buffers[symbol].append(low)
        self.open_buffers[symbol].append(open_price)
        
        # Calculate derived values
        if len(self.price_buffers[symbol]) >= 2:
            # ATR
            atr = self._calculate_atr(
                np.array(list(self.high_buffers[symbol])),
                np.array(list(self.low_buffers[symbol])),
                np.array(list(self.price_buffers[symbol]))
            )
            self.atr_buffers[symbol].append(atr)
            
            # VWAP
            vwap = self._calculate_vwap(
                np.array(list(self.price_buffers[symbol])),
                np.array(list(self.volume_buffers[symbol])),
                np.array(list(self.high_buffers[symbol])),
                np.array(list(self.low_buffers[symbol]))
            )
            self.vwap_buffers[symbol].append(vwap)
        
        # Need at least short lookback to generate signals
        if len(self.price_buffers[symbol]) < self.short_lookback:
            logger.debug(f"{symbol}: Warming up ({len(self.price_buffers[symbol])}/{self.short_lookback})")
            return None
        
        # Calculate regime signal
        signal = self._calculate_regime_signal(symbol, close, timestamp)
        
        # Track performance
        calculation_time = (time_module.perf_counter() - start_time) * 1000
        
        # Update tracking
        self.calculation_count += 1
        self.total_calculation_time += calculation_time
        self.latest_signals[symbol] = signal
        
        return signal
    
    def _calculate_regime_signal(self, symbol: str, price: float,
                               timestamp: datetime) -> MarketRegimeSignal:
        """Calculate complete market regime signal"""
        trends = {}
        
        # 1. Short trend (45-min / 3 bars)
        if len(self.price_buffers[symbol]) >= self.short_lookback:
            short_result = self._calculate_trend_with_lookback(
                symbol, price, self.short_lookback
            )
            trends['short'] = {
                'lookback': self.short_lookback * 15,  # Convert to minutes
                'direction': short_result.trend_direction,
                'strength': short_result.trend_strength,
                'confidence': short_result.confidence_level,
                'momentum': short_result.components.price_momentum,
                'score': short_result.composite_trend,
                'vwap_position': short_result.components.vwap_position,
                'volume_trend': short_result.components.volume_trend
            }
        
        # 2. Medium trend (75-min / 5 bars)
        if len(self.price_buffers[symbol]) >= self.medium_lookback:
            medium_result = self._calculate_trend_with_lookback(
                symbol, price, self.medium_lookback
            )
            trends['medium'] = {
                'lookback': self.medium_lookback * 15,
                'direction': medium_result.trend_direction,
                'strength': medium_result.trend_strength,
                'confidence': medium_result.confidence_level,
                'momentum': medium_result.components.price_momentum,
                'score': medium_result.composite_trend,
                'vwap_position': medium_result.components.vwap_position,
                'range_expansion': medium_result.components.range_expansion
            }
        
        # 3. Long trend (150-min / 10 bars)
        if len(self.price_buffers[symbol]) >= self.long_lookback:
            long_result = self._calculate_trend_with_lookback(
                symbol, price, self.long_lookback
            )
            trends['long'] = {
                'lookback': self.long_lookback * 15,
                'direction': long_result.trend_direction,
                'strength': long_result.trend_strength,
                'confidence': long_result.confidence_level,
                'score': long_result.composite_trend,
                'vwap_position': long_result.components.vwap_position,
                'range_expansion': long_result.components.range_expansion
            }
        
        # Identify key levels
        key_levels = self._identify_key_levels(symbol)
        
        # Analyze market regime
        regime_analysis = self._analyze_market_regime(trends, key_levels, price)
        
        return MarketRegimeSignal(
            symbol=symbol,
            timestamp=timestamp,
            price=price,
            regime=regime_analysis['regime'],
            daily_bias=regime_analysis['daily_bias'],
            strength=regime_analysis['strength'],
            confidence=regime_analysis['confidence'],
            volatility_state=regime_analysis['volatility_state'],
            key_levels=key_levels,
            trading_notes=regime_analysis['trading_notes'],
            short_trend=trends.get('short', {}),
            medium_trend=trends.get('medium', {}),
            long_trend=trends.get('long', {})
        )
    
    def _calculate_trend_with_lookback(self, symbol: str, current_price: float,
                                     lookback: int) -> StatisticalTrendResult15Min:
        """Calculate trend for specific lookback period"""
        # Get data slice
        all_prices = list(self.price_buffers[symbol])
        all_volumes = list(self.volume_buffers[symbol])
        all_timestamps = list(self.timestamp_buffers[symbol])
        all_highs = list(self.high_buffers[symbol])
        all_lows = list(self.low_buffers[symbol])
        all_vwaps = list(self.vwap_buffers[symbol])
        
        prices = np.array(all_prices[-lookback:])
        volumes = np.array(all_volumes[-lookback:])
        timestamps = all_timestamps[-lookback:]
        highs = np.array(all_highs[-lookback:])
        lows = np.array(all_lows[-lookback:])
        vwaps = np.array(all_vwaps[-lookback:]) if all_vwaps else prices
        
        warnings = []
        
        # Statistical calculations
        slope, intercept, r_squared = self._fast_linear_regression(prices)
        normalized_slope = (slope / prices[-1]) * 100 if prices[-1] != 0 else 0
        
        mk_trend, mk_z, mk_p = self._fast_mann_kendall(prices)
        
        kalman_price = self._update_kalman(symbol, current_price)
        kalman_trend = 1 if kalman_price > np.mean(prices[-min(3, lookback):]) else -1
        
        # Momentum - adjusted for 15-min bars
        momentum_periods = min(3, lookback - 1)  # 3-bar momentum
        if lookback > momentum_periods:
            momentum = ((prices[-1] - prices[-momentum_periods-1]) / 
                       prices[-momentum_periods-1] * 100)
        else:
            momentum = 0
        
        # Acceleration
        if len(prices) >= 3:
            velocity_current = prices[-1] - prices[-2]
            velocity_prev = prices[-2] - prices[-3]
            acceleration = velocity_current - velocity_prev
        else:
            acceleration = 0
        
        # VWAP position
        current_vwap = vwaps[-1] if len(vwaps) > 0 else current_price
        vwap_position = ((current_price - current_vwap) / current_vwap * 100) if current_vwap != 0 else 0
        
        # Volume trend
        if len(volumes) >= 3:
            recent_vol = np.mean(volumes[-3:])
            older_vol = np.mean(volumes[:-3])
            volume_trend = (recent_vol - older_vol) / older_vol if older_vol > 0 else 0
        else:
            volume_trend = 0
        
        # Range expansion/contraction
        recent_range = highs[-min(3, len(highs)):].max() - lows[-min(3, len(lows)):].min()
        if len(highs) >= 6:
            older_range = highs[-6:-3].max() - lows[-6:-3].min()
            range_expansion = (recent_range - older_range) / older_range if older_range > 0 else 0
        else:
            range_expansion = 0
        
        components = TrendComponents15Min(
            linear_slope=normalized_slope,
            linear_r_squared=r_squared,
            mann_kendall_trend=mk_trend,
            mann_kendall_z_score=mk_z,
            mann_kendall_p_value=mk_p,
            kalman_trend=kalman_trend,
            kalman_price=kalman_price,
            price_momentum=momentum,
            price_acceleration=acceleration,
            vwap_position=vwap_position,
            volume_trend=volume_trend,
            range_expansion=range_expansion
        )
        
        # Use 15-min optimized scoring
        composite_score = self._calculate_15min_composite_score(components, lookback)
        
        # Determine trend - higher thresholds for 15-min
        if abs(composite_score) < 0.3:  # Higher threshold
            direction = 'neutral'
        elif composite_score > 0:
            direction = 'bullish'
        else:
            direction = 'bearish'
        
        strength = abs(composite_score) * 100
        confidence = self._calculate_confidence(components)
        
        return StatisticalTrendResult15Min(
            symbol=symbol,
            timestamp=timestamps[-1],
            current_price=current_price,
            lookback_periods=lookback,
            components=components,
            composite_trend=composite_score,
            trend_direction=direction,
            trend_strength=strength,
            confidence_level=confidence,
            calculation_time_ms=0,
            warnings=warnings
        )
    
    def _calculate_15min_composite_score(self, components: TrendComponents15Min,
                                       lookback: int) -> float:
        """Calculate composite score optimized for 15-min timeframe"""
        # Different weights based on lookback
        if lookback <= self.short_lookback:
            # 45-min: Still somewhat reactive
            weights = {
                'linear': 0.25,
                'mann_kendall': 0.20,
                'kalman': 0.20,
                'momentum': 0.20,
                'vwap': 0.10,
                'volume': 0.05
            }
            sensitivity = 2.0
        elif lookback <= self.medium_lookback:
            # 75-min: Balanced
            weights = {
                'linear': 0.30,
                'mann_kendall': 0.25,
                'kalman': 0.15,
                'momentum': 0.15,
                'vwap': 0.10,
                'volume': 0.05
            }
            sensitivity = 2.5
        else:
            # 150-min: Trend focused
            weights = {
                'linear': 0.35,
                'mann_kendall': 0.30,
                'kalman': 0.15,
                'momentum': 0.10,
                'vwap': 0.05,
                'volume': 0.05
            }
            sensitivity = 3.0
        
        # Less sensitive normalization for 15-min
        slope_signal = np.tanh(components.linear_slope / sensitivity)
        momentum_signal = np.tanh(components.price_momentum / (sensitivity * 0.5))
        vwap_signal = np.tanh(components.vwap_position / 2.0)  # 2% from VWAP
        volume_signal = np.tanh(components.volume_trend / 0.5)  # 50% volume change
        
        weighted_sum = (
            weights['linear'] * slope_signal * components.linear_r_squared +
            weights['mann_kendall'] * components.mann_kendall_trend +
            weights['kalman'] * components.kalman_trend +
            weights['momentum'] * momentum_signal +
            weights['vwap'] * vwap_signal +
            weights['volume'] * volume_signal
        )
        
        return np.clip(weighted_sum, -1, 1)
    
    def _calculate_confidence(self, components: TrendComponents15Min) -> float:
        """Calculate confidence for 15-min signals"""
        signals = [
            np.sign(components.linear_slope),
            components.mann_kendall_trend,
            components.kalman_trend,
            np.sign(components.price_momentum),
            np.sign(components.vwap_position)
        ]
        
        non_zero_signals = [s for s in signals if s != 0]
        if not non_zero_signals:
            return 0
        
        agreement = abs(sum(non_zero_signals)) / len(non_zero_signals)
        confidence = agreement * 100
        
        # Boost for significant Mann-Kendall
        if components.mann_kendall_p_value < 0.01:  # Stricter for 15-min
            confidence = min(100, confidence * 1.3)
        elif components.mann_kendall_p_value < 0.05:
            confidence = min(100, confidence * 1.1)
        
        # Volume confirmation bonus
        if abs(components.volume_trend) > 0.3:
            confidence = min(100, confidence * 1.1)
        
        # R-squared factor
        confidence *= (0.6 + 0.4 * components.linear_r_squared)
        
        return min(100, confidence)
    
    def _analyze_market_regime(self, trends: Dict, key_levels: Dict, 
                             current_price: float) -> Dict:
        """Analyze market regime and generate daily trading bias"""
        short = trends.get('short', {})
        medium = trends.get('medium', {})
        long = trends.get('long', {})
        
        # Aggregate trend information
        directions = []
        strengths = []
        for trend in [short, medium, long]:
            if trend:
                if trend['direction'] == 'bullish':
                    directions.append(1)
                elif trend['direction'] == 'bearish':
                    directions.append(-1)
                else:
                    directions.append(0)
                strengths.append(trend.get('strength', 0))
        
        if not directions:
            return {
                'regime': 'UNKNOWN',
                'daily_bias': 'STAY OUT',
                'strength': 0,
                'confidence': 0,
                'volatility_state': 'UNKNOWN',
                'trading_notes': 'Insufficient data for regime analysis'
            }
        
        # Calculate aggregate metrics
        trend_alignment = sum(directions)
        avg_strength = np.mean(strengths)
        
        # Volatility assessment
        range_expansion = medium.get('range_expansion', 0)
        if abs(range_expansion) > 0.5:
            volatility_state = 'EXTREME'
        elif abs(range_expansion) > 0.3:
            volatility_state = 'HIGH'
        elif abs(range_expansion) < 0.1:
            volatility_state = 'LOW'
        else:
            volatility_state = 'NORMAL'
        
        # Price position relative to key levels
        if key_levels:
            near_resistance = abs(current_price - key_levels.get('recent_high', 0)) < key_levels.get('range', 1) * 0.1
            near_support = abs(current_price - key_levels.get('recent_low', 0)) < key_levels.get('range', 1) * 0.1
            above_vwap = current_price > key_levels.get('vwap', current_price)
        else:
            near_resistance = near_support = False
            above_vwap = True
        
        # Determine market regime
        if trend_alignment == len(directions) and avg_strength > 60:
            regime = 'BULL MARKET'
            daily_bias = 'LONG ONLY'
            trading_notes = 'Strong uptrend - Buy dips, avoid shorts'
            confidence = min(100, avg_strength * 1.2)
        elif trend_alignment == -len(directions) and avg_strength > 60:
            regime = 'BEAR MARKET'
            daily_bias = 'SHORT ONLY'
            trading_notes = 'Strong downtrend - Sell rallies, avoid longs'
            confidence = min(100, avg_strength * 1.2)
        elif abs(trend_alignment) <= 1 and avg_strength < 40:
            regime = 'RANGE BOUND'
            daily_bias = 'BOTH WAYS'
            trading_notes = f'Range trading between ${key_levels.get("recent_low", 0):.2f} - ${key_levels.get("recent_high", 0):.2f}'
            confidence = 60
        elif trend_alignment > 0:
            regime = 'BULL MARKET'
            daily_bias = 'LONG BIAS'
            trading_notes = 'Uptrend but not strong - Prefer longs, careful with size'
            confidence = avg_strength
        elif trend_alignment < 0:
            regime = 'BEAR MARKET'
            daily_bias = 'SHORT BIAS'
            trading_notes = 'Downtrend but not strong - Prefer shorts, careful with size'
            confidence = avg_strength
        else:
            regime = 'TRANSITIONING'
            daily_bias = 'STAY OUT'
            trading_notes = 'Market in transition - Wait for clear direction'
            confidence = 30
        
        # Adjust for volatility
        if volatility_state == 'EXTREME':
            trading_notes += ' | ⚠️ EXTREME volatility - Reduce position size'
            if daily_bias not in ['STAY OUT']:
                daily_bias = daily_bias.replace('ONLY', 'BIAS')
        elif volatility_state == 'LOW':
            trading_notes += ' | 💤 Low volatility - Breakout possible'
        
        # Key level notes
        if near_resistance:
            trading_notes += ' | 🔴 Near resistance'
        elif near_support:
            trading_notes += ' | 🟢 Near support'
        
        return {
            'regime': regime,
            'daily_bias': daily_bias,
            'strength': avg_strength,
            'confidence': confidence,
            'volatility_state': volatility_state,
            'trading_notes': trading_notes
        }
    
    async def start_websocket(self, symbols: List[str], callback: Optional[Callable] = None):
        """Start real-time monitoring with WebSocket"""
        from polygon import DataFetcher, PolygonWebSocketClient
        
        # Load historical data
        logger.info(f"Loading historical 15-min data for {len(symbols)} symbols...")
        fetcher = DataFetcher()
        
        for symbol in symbols:
            try:
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=3)  # Load 3 hours of 15-min bars
                
                df = fetcher.fetch_data(
                    symbol=symbol,
                    timeframe='15min',
                    start_date=start_time,
                    end_date=end_time,
                    use_cache=False
                )
                
                if not df.empty:
                    for idx, row in df.iterrows():
                        self.update_bar(
                            symbol=symbol,
                            open_price=row['open'],
                            high=row['high'],
                            low=row['low'],
                            close=row['close'],
                            volume=row['volume'],
                            timestamp=idx
                        )
                    logger.info(f"✓ Loaded {len(df)} 15-min bars for {symbol}")
                
            except Exception as e:
                logger.error(f"Error loading 15-min data for {symbol}: {e}")
        
        # Connect WebSocket
        logger.info("Connecting to WebSocket for 15-min data...")
        self.ws_client = PolygonWebSocketClient()
        await self.ws_client.connect()
        
        # Subscribe - will need to aggregate to 15-min
        await self.ws_client.subscribe(
            symbols=symbols,
            channels=['AM'],
            callback=self._handle_websocket_data
        )
        
        for symbol in symbols:
            self.active_symbols[symbol] = [callback] if callback else []
            self.calculation_tasks[symbol] = asyncio.create_task(
                self._calculation_loop(symbol)
            )
        
        logger.info(f"✓ Started 15-min regime monitoring for {symbols}")
    
    async def _handle_websocket_data(self, data: Dict):
        """Handle incoming WebSocket data - aggregate to 15-min bars"""
        # Simplified - in production you'd properly aggregate to 15-min
        try:
            event_type = data.get('event_type')
            symbol = data.get('symbol')
            
            if event_type == 'aggregate' and symbol in self.active_symbols:
                # For testing, update on each bar
                signal = self.update_bar(
                    symbol=symbol,
                    open_price=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    volume=data['volume'],
                    timestamp=datetime.fromtimestamp(data['timestamp'] / 1000, tz=timezone.utc)
                )
                
                if signal:
                    logger.info(f"15-min regime for {symbol}: {signal.regime}")
                    await self._notify_callbacks(symbol, signal)
                    
        except Exception as e:
            logger.error(f"Error handling WebSocket data: {e}")
    
    async def _calculation_loop(self, symbol: str):
        """Periodic calculation loop"""
        last_regime = None
        
        while symbol in self.active_symbols:
            try:
                await asyncio.sleep(self.calculation_interval)
                
                signal = self.latest_signals.get(symbol)
                if signal:
                    # Recalculate
                    prices = list(self.price_buffers[symbol])
                    if prices:
                        new_signal = self._calculate_regime_signal(
                            symbol, prices[-1],
                            list(self.timestamp_buffers[symbol])[-1]
                        )
                        
                        # Check for regime changes
                        if new_signal.regime != signal.regime:
                            logger.info(f"🔄 {symbol} REGIME CHANGE: {signal.regime} → {new_signal.regime}")
                            self.latest_signals[symbol] = new_signal
                            await self._notify_callbacks(symbol, new_signal)
                        elif new_signal.daily_bias != signal.daily_bias:
                            logger.info(f"📊 {symbol} bias changed: {signal.daily_bias} → {new_signal.daily_bias}")
                            self.latest_signals[symbol] = new_signal
                            await self._notify_callbacks(symbol, new_signal)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in 15-min calculation loop for {symbol}: {e}")
    
    async def _notify_callbacks(self, symbol: str, signal: MarketRegimeSignal):
        """Notify callbacks with new signal"""
        for callback in self.active_symbols.get(symbol, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(signal)
                else:
                    callback(signal)
            except Exception as e:
                logger.error(f"Callback error for {symbol}: {e}")
    
    async def stop(self):
        """Stop all calculations"""
        logger.info("Stopping 15-min trend calculator...")
        
        for task in self.calculation_tasks.values():
            task.cancel()
        
        await asyncio.gather(*self.calculation_tasks.values(), return_exceptions=True)
        
        if self.ws_client:
            await self.ws_client.disconnect()
        
        logger.info("✓ 15-min trend calculator stopped")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        avg_time = self.total_calculation_time / self.calculation_count if self.calculation_count > 0 else 0
        return {
            'total_calculations': self.calculation_count,
            'average_time_ms': avg_time,
            'active_symbols': len(self.active_symbols),
            'timeframes': {
                'short': self.short_lookback * 15,
                'medium': self.medium_lookback * 15,
                'long': self.long_lookback * 15
            }
        }


# ============= LIVE TESTING =============
async def run_15min_test():
    """Test 15-minute trend calculation"""
    print("=== Testing 15-Minute Statistical Trend for Market Regime Analysis ===\n")
    
    TEST_SYMBOLS = ['TSLA', 'AAPL', 'SPY', 'QQQ', 'NVDA']
    TEST_DURATION = 240  # 4 minutes
    
    calculator = StatisticalTrend15Min(
        short_lookback=3,    # 45-min trend
        medium_lookback=5,   # 75-min trend
        long_lookback=10,    # 150-min trend
        calculation_interval=60  # Every minute
    )
    
    update_count = 0
    
    def display_signal(signal: MarketRegimeSignal):
        nonlocal update_count
        update_count += 1
        
        print(f"\n{'='*80}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 15-MIN REGIME UPDATE #{update_count} - {signal.symbol}")
        print(f"{'='*80}")
        
        # Price and regime
        print(f"💰 Price: ${signal.price:.2f}")
        
        # Market regime with emoji
        regime_emoji = {
            'BULL MARKET': '🐂',
            'BEAR MARKET': '🐻',
            'RANGE BOUND': '📦',
            'TRANSITIONING': '🔄',
            'UNKNOWN': '❓'
        }
        
        print(f"\n🌍 MARKET REGIME: {regime_emoji.get(signal.regime, '?')} {signal.regime}")
        print(f"📅 DAILY BIAS: {signal.daily_bias}")
        print(f"💪 Regime Strength: {signal.strength:.1f}%")
        print(f"🎯 Confidence: {signal.confidence:.1f}%")
        
        # Volatility state
        vol_emoji = {
            'LOW': '😴',
            'NORMAL': '👍',
            'HIGH': '⚡',
            'EXTREME': '🌪️'
        }
        print(f"📊 Volatility: {vol_emoji.get(signal.volatility_state, '?')} {signal.volatility_state}")
        
        # Key levels
        if signal.key_levels:
            print(f"\n📍 KEY LEVELS:")
            print(f"   Resistance: ${signal.key_levels.get('recent_high', 0):.2f}")
            print(f"   Support: ${signal.key_levels.get('recent_low', 0):.2f}")
            print(f"   VWAP: ${signal.key_levels.get('vwap', 0):.2f}")
            print(f"   Pivot: ${signal.key_levels.get('pivot', 0):.2f}")
            print(f"   Range: ${signal.key_levels.get('range', 0):.2f}")
        
        # Trading notes
        print(f"\n📝 TRADING NOTES: {signal.trading_notes}")
        
        # Timeframe analysis
        print(f"\n📈 TIMEFRAME BREAKDOWN:")
        
        # Short (45-min)
        if signal.short_trend:
            s = signal.short_trend
            print(f"   45-min: {s['direction'].upper()} "
                  f"(Strength: {s['strength']:.1f}%, "
                  f"VWAP: {s['vwap_position']:+.2f}%)")
        
        # Medium (75-min)
        if signal.medium_trend:
            m = signal.medium_trend
            print(f"   75-min: {m['direction'].upper()} "
                  f"(Strength: {m['strength']:.1f}%, "
                  f"Momentum: {m['momentum']:.2f}%)")
        
        # Long (150-min)
        if signal.long_trend:
            l = signal.long_trend
            print(f"   150-min: {l['direction'].upper()} "
                  f"(Strength: {l['strength']:.1f}%, "
                  f"Score: {l['score']:.3f})")
        
        # Daily trading plan
        print(f"\n🎯 DAILY TRADING PLAN:")
        if signal.daily_bias == 'LONG ONLY':
            print("   ✅ Take ALL 1-min/5-min LONG signals")
            print("   ❌ SKIP all SHORT signals")
            print("   💡 Buy dips aggressively")
        elif signal.daily_bias == 'SHORT ONLY':
            print("   ✅ Take ALL 1-min/5-min SHORT signals")
            print("   ❌ SKIP all LONG signals")
            print("   💡 Sell rallies aggressively")
        elif signal.daily_bias == 'LONG BIAS':
            print("   ✅ Prefer LONG signals")
            print("   ⚠️  Be selective with SHORTs")
            print("   💡 Focus on high-confidence setups")
        elif signal.daily_bias == 'SHORT BIAS':
            print("   ✅ Prefer SHORT signals")
            print("   ⚠️  Be selective with LONGs")
            print("   💡 Focus on high-confidence setups")
        elif signal.daily_bias == 'BOTH WAYS':
            print("   ✅ Trade both directions")
            print("   💡 Focus on range extremes")
            print("   ⚠️  Quick exits, don't overstay")
        else:
            print("   ❌ NO TRADING - Wait for clarity")
            print("   💡 Market too uncertain")
    
    try:
        await calculator.start_websocket(TEST_SYMBOLS, display_signal)
        
        print(f"\n🚀 15-Minute Market Regime Monitor Started")
        print(f"📊 Tracking {len(TEST_SYMBOLS)} symbols")
        print(f"⏱️  Timeframes: 45-min, 75-min, 150-min")
        print(f"🔄 Updates every minute")
        print(f"⏰ Test duration: {TEST_DURATION} seconds\n")
        
        print("📖 Regime Guide:")
        print("   BULL MARKET = Strong uptrend across timeframes")
        print("   BEAR MARKET = Strong downtrend across timeframes")
        print("   RANGE BOUND = No clear trend, trading sideways")
        print("   TRANSITIONING = Trend changing, be cautious\n")
        
        print("⏳ Waiting for regime analysis...")
        
        start_time = time_module.time()
        while time_module.time() - start_time < TEST_DURATION:
            remaining = TEST_DURATION - (time_module.time() - start_time)
            print(f"\r⏳ Time remaining: {remaining:.0f}s ", end='', flush=True)
            await asyncio.sleep(1)
        
        print("\n\n🏁 Test complete!")
        
        # Summary
        stats = calculator.get_performance_stats()
        print(f"\n📊 Performance Summary:")
        print(f"  • Total calculations: {stats['total_calculations']}")
        print(f"  • Average time: {stats['average_time_ms']:.3f}ms")
        print(f"  • Regime updates: {update_count}")
        
        # Final regimes
        print(f"\n🌍 Final Market Regimes:")
        for symbol, signal in calculator.latest_signals.items():
            print(f"  • {symbol}: {signal.regime} ({signal.daily_bias})")
        
        await calculator.stop()
        print("\n✅ 15-minute regime test completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted")
        await calculator.stop()
        
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        await calculator.stop()


if __name__ == "__main__":
    print("Starting 15-Minute Statistical Trend Calculator")
    print("Provides market regime and daily trading bias\n")
    print("Features:")
    print("• Multi-timeframe: 45/75/150 minute analysis")
    print("• Market regime detection (Bull/Bear/Range)")
    print("• Daily trading bias recommendations")
    print("• Key support/resistance levels")
    print("• Volatility state monitoring")
    print("• Clear daily trading plan\n")
    
    asyncio.run(run_15min_test())