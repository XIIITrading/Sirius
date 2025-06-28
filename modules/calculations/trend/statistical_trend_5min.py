# modules/calculations/trend/statistical_trend_5min.py
"""
Module: 5-Minute Statistical Trend Calculation for Higher Timeframe Analysis
Purpose: Provide broader market perspective for short-term trading confirmation
Features: Multi-timeframe analysis (15/25/50 min), WebSocket integration, Position signals
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

# Enforce UTC for all operations
os.environ['TZ'] = 'UTC'
if hasattr(time_module, 'tzset'):
    time_module.tzset()

# Fix import path
current_dir = os.path.dirname(os.path.abspath(__file__))
calculations_dir = os.path.dirname(current_dir)
modules_dir = os.path.dirname(calculations_dir)
vega_root = os.path.dirname(modules_dir)

if vega_root not in sys.path:
    sys.path.insert(0, vega_root)

# Configure logging with UTC
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s UTC - %(name)s - %(levelname)s - %(message)s'
)
logging.Formatter.converter = time_module.gmtime  # Force UTC in logs
logger = logging.getLogger(__name__)

# UTC validation function
def ensure_utc(dt: datetime) -> datetime:
    """Ensure datetime is UTC-aware"""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    elif dt.tzinfo != timezone.utc:
        return dt.astimezone(timezone.utc)
    return dt


@dataclass
class TrendComponents5Min:
    """Individual trend calculation components for 5-min timeframe"""
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


@dataclass
class StatisticalTrendResult5Min:
    """Complete statistical trend analysis result for 5-minute timeframe"""
    symbol: str
    timestamp: datetime
    current_price: float
    lookback_periods: int
    components: TrendComponents5Min
    composite_trend: float  # -1 to 1, strength of trend
    trend_direction: str  # 'bullish', 'bearish', 'neutral'
    trend_strength: float  # 0 to 100
    confidence_level: float  # 0 to 100
    calculation_time_ms: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class PositionSignal5Min:
    """Position management signal for 5-min timeframe"""
    symbol: str
    timestamp: datetime
    price: float
    signal: str  # 'STRONG TREND UP', 'TREND UP', 'RANGING', 'TREND DOWN', 'STRONG TREND DOWN'
    bias: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    strength: float
    confidence: float
    market_state: str  # 'TRENDING', 'CONSOLIDATING', 'VOLATILE'
    recommendation: str  # Position sizing recommendation
    short_trend: Dict  # 15-min (3 x 5-min)
    medium_trend: Dict  # 25-min (5 x 5-min)
    long_trend: Dict  # 50-min (10 x 5-min)


class StatisticalTrend5Min:
    """
    5-Minute Statistical Trend Calculator for higher timeframe analysis.
    Provides market context and position management signals.
    All timestamps are in UTC.
    """
    
    def __init__(self,
                 short_lookback: int = 3,      # 15-min trend
                 medium_lookback: int = 5,      # 25-min trend
                 long_lookback: int = 10,       # 50-min trend
                 calculation_interval: int = 30):  # Calculate every 30 seconds
        """
        Initialize 5-minute trend calculator.
        
        Args:
            short_lookback: 15-min trend (3 x 5-min bars)
            medium_lookback: 25-min trend (5 x 5-min bars)
            long_lookback: 50-min trend (10 x 5-min bars)
            calculation_interval: Seconds between calculations
        """
        self.short_lookback = short_lookback
        self.medium_lookback = medium_lookback
        self.long_lookback = long_lookback
        self.calculation_interval = calculation_interval
        
        # Use long as max buffer size
        self.max_lookback = long_lookback
        self.warmup_periods = short_lookback  # Can start after 15 minutes
        
        # Data buffers
        self.price_buffers: Dict[str, deque] = {}
        self.volume_buffers: Dict[str, deque] = {}
        self.timestamp_buffers: Dict[str, deque] = {}
        self.high_buffers: Dict[str, deque] = {}
        self.low_buffers: Dict[str, deque] = {}
        
        # VWAP calculation
        self.vwap_buffers: Dict[str, deque] = {}
        
        # Kalman filter states
        self.kalman_states: Dict[str, Dict] = {}
        
        # WebSocket integration
        self.ws_client = None
        self.active_symbols: Dict[str, List[Callable]] = {}
        self.calculation_tasks: Dict[str, asyncio.Task] = {}
        
        # Latest results
        self.latest_signals: Dict[str, PositionSignal5Min] = {}
        
        # Performance tracking
        self.calculation_count = 0
        self.total_calculation_time = 0
        
        logger.info(f"Initialized 5-min trend calculator: "
                   f"Short={short_lookback*5}min, Medium={medium_lookback*5}min, Long={long_lookback*5}min")
        logger.info(f"System initialized at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    def initialize_buffers(self, symbol: str):
        """Initialize data buffers for a symbol"""
        self.price_buffers[symbol] = deque(maxlen=self.max_lookback)
        self.volume_buffers[symbol] = deque(maxlen=self.max_lookback)
        self.timestamp_buffers[symbol] = deque(maxlen=self.max_lookback)
        self.high_buffers[symbol] = deque(maxlen=self.max_lookback)
        self.low_buffers[symbol] = deque(maxlen=self.max_lookback)
        self.vwap_buffers[symbol] = deque(maxlen=self.max_lookback)
        self.kalman_states[symbol] = self._initialize_kalman()
        logger.info(f"Initialized 5-min buffers for {symbol}")
    
    def _initialize_kalman(self) -> Dict:
        """Initialize Kalman filter state for 5-min bars"""
        return {
            'x': 0.0,      # State estimate
            'P': 1.0,      # Error covariance
            'Q': 0.0002,   # Higher process noise for 5-min bars
            'R': 0.02,     # Higher measurement noise
            'K': 0.0       # Kalman gain
        }
    
    def _validate_timestamp(self, timestamp: datetime, source: str) -> datetime:
        """Validate and ensure timestamp is UTC"""
        if timestamp.tzinfo is None:
            logger.warning(f"{source}: Naive datetime received, assuming UTC")
            return timestamp.replace(tzinfo=timezone.utc)
        elif timestamp.tzinfo != timezone.utc:
            logger.warning(f"{source}: Non-UTC timezone {timestamp.tzinfo}, converting to UTC")
            return timestamp.astimezone(timezone.utc)
        return timestamp
    
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
    
    def _calculate_vwap(self, prices: np.ndarray, volumes: np.ndarray, 
                       highs: np.ndarray, lows: np.ndarray) -> float:
        """Calculate VWAP for the period"""
        if len(prices) == 0 or volumes.sum() == 0:
            return prices[-1] if len(prices) > 0 else 0
            
        # Use typical price (H+L+C)/3
        typical_prices = (highs + lows + prices) / 3
        vwap = np.sum(typical_prices * volumes) / np.sum(volumes)
        
        return vwap
    
    def update_bar(self, symbol: str, open_price: float, high: float, 
                  low: float, close: float, volume: float,
                  timestamp: Optional[datetime] = None) -> Optional[PositionSignal5Min]:
        """
        Update with new 5-minute bar data.
        
        Returns:
            PositionSignal5Min with multi-timeframe analysis
        """
        start_time = time_module.perf_counter()
        
        # Initialize if needed
        if symbol not in self.price_buffers:
            self.initialize_buffers(symbol)
        
        # Handle timestamp with UTC enforcement
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        else:
            # Ensure timestamp is UTC-aware
            if timestamp.tzinfo is None:
                # If naive datetime, assume UTC and make it aware
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            elif timestamp.tzinfo != timezone.utc:
                # If has timezone but not UTC, convert to UTC
                timestamp = timestamp.astimezone(timezone.utc)
        
        # Update buffers
        self.price_buffers[symbol].append(close)
        self.volume_buffers[symbol].append(volume)
        self.timestamp_buffers[symbol].append(timestamp)
        self.high_buffers[symbol].append(high)
        self.low_buffers[symbol].append(low)
        
        # Calculate and store VWAP
        if len(self.price_buffers[symbol]) >= 1:
            recent_vwap = self._calculate_vwap(
                np.array(list(self.price_buffers[symbol])),
                np.array(list(self.volume_buffers[symbol])),
                np.array(list(self.high_buffers[symbol])),
                np.array(list(self.low_buffers[symbol]))
            )
            self.vwap_buffers[symbol].append(recent_vwap)
        
        # Need at least short lookback to generate signals
        if len(self.price_buffers[symbol]) < self.short_lookback:
            logger.debug(f"{symbol}: Warming up ({len(self.price_buffers[symbol])}/{self.short_lookback})")
            return None
        
        # Calculate position signal
        signal = self._calculate_position_signal(symbol, close, timestamp)
        
        # Track performance
        calculation_time = (time_module.perf_counter() - start_time) * 1000
        
        # Update tracking
        self.calculation_count += 1
        self.total_calculation_time += calculation_time
        self.latest_signals[symbol] = signal
        
        return signal
    
    def _calculate_position_signal(self, symbol: str, price: float,
                                 timestamp: datetime) -> PositionSignal5Min:
        """Calculate complete position signal with all timeframes"""
        trends = {}
        
        # 1. Short trend (15-min / 3 bars)
        if len(self.price_buffers[symbol]) >= self.short_lookback:
            short_result = self._calculate_trend_with_lookback(
                symbol, price, self.short_lookback
            )
            trends['short'] = {
                'lookback': self.short_lookback * 5,  # Convert to minutes
                'direction': short_result.trend_direction,
                'strength': short_result.trend_strength,
                'confidence': short_result.confidence_level,
                'momentum': short_result.components.price_momentum,
                'score': short_result.composite_trend,
                'vwap_position': short_result.components.vwap_position
            }
        
        # 2. Medium trend (25-min / 5 bars)
        if len(self.price_buffers[symbol]) >= self.medium_lookback:
            medium_result = self._calculate_trend_with_lookback(
                symbol, price, self.medium_lookback
            )
            trends['medium'] = {
                'lookback': self.medium_lookback * 5,
                'direction': medium_result.trend_direction,
                'strength': medium_result.trend_strength,
                'confidence': medium_result.confidence_level,
                'momentum': medium_result.components.price_momentum,
                'score': medium_result.composite_trend,
                'vwap_position': medium_result.components.vwap_position
            }
        
        # 3. Long trend (50-min / 10 bars)
        if len(self.price_buffers[symbol]) >= self.long_lookback:
            long_result = self._calculate_trend_with_lookback(
                symbol, price, self.long_lookback
            )
            trends['long'] = {
                'lookback': self.long_lookback * 5,
                'direction': long_result.trend_direction,
                'strength': long_result.trend_strength,
                'confidence': long_result.confidence_level,
                'score': long_result.composite_trend,
                'vwap_position': long_result.components.vwap_position
            }
        
        # Generate position signal
        position_analysis = self._analyze_market_state(trends)
        
        return PositionSignal5Min(
            symbol=symbol,
            timestamp=timestamp,
            price=price,
            signal=position_analysis['signal'],
            bias=position_analysis['bias'],
            strength=position_analysis['strength'],
            confidence=position_analysis['confidence'],
            market_state=position_analysis['market_state'],
            recommendation=position_analysis['recommendation'],
            short_trend=trends.get('short', {}),
            medium_trend=trends.get('medium', {}),
            long_trend=trends.get('long', {})
        )
    
    def _calculate_trend_with_lookback(self, symbol: str, current_price: float,
                                     lookback: int) -> StatisticalTrendResult5Min:
        """Calculate trend for specific lookback period"""
        # Get data slice
        all_prices = list(self.price_buffers[symbol])
        all_volumes = list(self.volume_buffers[symbol])
        all_timestamps = list(self.timestamp_buffers[symbol])
        all_vwaps = list(self.vwap_buffers[symbol])
        
        prices = np.array(all_prices[-lookback:])
        volumes = np.array(all_volumes[-lookback:])
        timestamps = all_timestamps[-lookback:]
        vwaps = np.array(all_vwaps[-lookback:]) if all_vwaps else prices
        
        warnings = []
        
        # Statistical calculations
        slope, intercept, r_squared = self._fast_linear_regression(prices)
        normalized_slope = (slope / prices[-1]) * 100 if prices[-1] != 0 else 0
        
        mk_trend, mk_z, mk_p = self._fast_mann_kendall(prices)
        
        kalman_price = self._update_kalman(symbol, current_price)
        kalman_trend = 1 if kalman_price > np.mean(prices[-min(3, lookback):]) else -1
        
        # Momentum - adjusted for 5-min bars
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
        
        components = TrendComponents5Min(
            linear_slope=normalized_slope,
            linear_r_squared=r_squared,
            mann_kendall_trend=mk_trend,
            mann_kendall_z_score=mk_z,
            mann_kendall_p_value=mk_p,
            kalman_trend=kalman_trend,
            kalman_price=kalman_price,
            price_momentum=momentum,
            price_acceleration=acceleration,
            vwap_position=vwap_position
        )
        
        # Use 5-min optimized scoring
        composite_score = self._calculate_5min_composite_score(components, lookback)
        
        # Determine trend - less sensitive thresholds for 5-min
        if abs(composite_score) < 0.25:  # Higher threshold
            direction = 'neutral'
        elif composite_score > 0:
            direction = 'bullish'
        else:
            direction = 'bearish'
        
        strength = abs(composite_score) * 100
        confidence = self._calculate_confidence(components)
        
        return StatisticalTrendResult5Min(
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
    
    def _calculate_5min_composite_score(self, components: TrendComponents5Min,
                                      lookback: int) -> float:
        """Calculate composite score optimized for 5-min timeframe"""
        # Different weights based on lookback
        if lookback <= self.short_lookback:
            # 15-min: Still reactive but more stable
            weights = {
                'linear': 0.20,
                'mann_kendall': 0.20,
                'kalman': 0.25,
                'momentum': 0.25,
                'vwap': 0.10
            }
            sensitivity = 1.0  # Less sensitive than 1-min
        elif lookback <= self.medium_lookback:
            # 25-min: Balanced
            weights = {
                'linear': 0.25,
                'mann_kendall': 0.25,
                'kalman': 0.20,
                'momentum': 0.20,
                'vwap': 0.10
            }
            sensitivity = 1.5
        else:
            # 50-min: Trend focused
            weights = {
                'linear': 0.30,
                'mann_kendall': 0.30,
                'kalman': 0.20,
                'momentum': 0.10,
                'vwap': 0.10
            }
            sensitivity = 2.0
        
        # Less sensitive normalization for 5-min
        slope_signal = np.tanh(components.linear_slope / sensitivity)
        momentum_signal = np.tanh(components.price_momentum / sensitivity)
        vwap_signal = np.tanh(components.vwap_position / 1.0)  # 1% from VWAP
        
        weighted_sum = (
            weights['linear'] * slope_signal * components.linear_r_squared +
            weights['mann_kendall'] * components.mann_kendall_trend +
            weights['kalman'] * components.kalman_trend +
            weights['momentum'] * momentum_signal +
            weights['vwap'] * vwap_signal
        )
        
        return np.clip(weighted_sum, -1, 1)
    
    def _calculate_confidence(self, components: TrendComponents5Min) -> float:
        """Calculate confidence for 5-min signals"""
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
        if components.mann_kendall_p_value < 0.05:
            confidence = min(100, confidence * 1.2)
        
        # R-squared factor
        confidence *= (0.5 + 0.5 * components.linear_r_squared)
        
        return min(100, confidence)
    
    def _analyze_market_state(self, trends: Dict) -> Dict:
        """Analyze market state and generate position signals"""
        short = trends.get('short', {})
        medium = trends.get('medium', {})
        long = trends.get('long', {})
        
        # Count directional agreement
        directions = []
        for trend in [short, medium, long]:
            if trend:
                if trend['direction'] == 'bullish':
                    directions.append(1)
                elif trend['direction'] == 'bearish':
                    directions.append(-1)
                else:
                    directions.append(0)
        
        if not directions:
            return {
                'signal': 'NO DATA',
                'bias': 'NEUTRAL',
                'strength': 0,
                'confidence': 0,
                'market_state': 'UNKNOWN',
                'recommendation': 'Wait for data'
            }
        
        # Market state determination
        alignment = sum(directions)
        avg_strength = np.mean([t.get('strength', 0) for t in [short, medium, long] if t])
        avg_confidence = np.mean([t.get('confidence', 0) for t in [short, medium, long] if t])
        
        # Check for trending vs ranging
        if abs(alignment) == len(directions) and avg_strength > 40:
            market_state = 'TRENDING'
        elif avg_strength < 30:
            market_state = 'CONSOLIDATING'
        else:
            market_state = 'VOLATILE'
        
        # Generate signals
        if alignment == len(directions) and avg_strength > 50:
            # Strong trend - all timeframes aligned
            signal = 'STRONG TREND UP'
            bias = 'BULLISH'
            recommendation = 'Full position size on pullbacks'
        elif alignment == -len(directions) and avg_strength > 50:
            signal = 'STRONG TREND DOWN'
            bias = 'BEARISH'
            recommendation = 'Full short position on rallies'
        elif alignment > 0 and avg_strength > 30:
            signal = 'TREND UP'
            bias = 'BULLISH'
            recommendation = '75% position size, scale in'
        elif alignment < 0 and avg_strength > 30:
            signal = 'TREND DOWN'
            bias = 'BEARISH'
            recommendation = '75% short position, scale in'
        else:
            signal = 'RANGING'
            bias = 'NEUTRAL'
            recommendation = 'Reduce position size or stay flat'
        
        # Adjust for VWAP position
        avg_vwap_pos = np.mean([t.get('vwap_position', 0) for t in [short, medium, long] if t])
        if abs(avg_vwap_pos) > 2:  # More than 2% from VWAP
            if signal == 'RANGING':
                if avg_vwap_pos > 0:
                    recommendation = 'Consider mean reversion short'
                else:
                    recommendation = 'Consider mean reversion long'
        
        return {
            'signal': signal,
            'bias': bias,
            'strength': avg_strength,
            'confidence': avg_confidence,
            'market_state': market_state,
            'recommendation': recommendation
        }
    
    async def start_websocket(self, symbols: List[str], callback: Optional[Callable] = None):
        """Start real-time monitoring with WebSocket"""
        from polygon import DataFetcher, PolygonWebSocketClient
        
        # Load historical data with UTC timestamps
        logger.info(f"Loading historical 5-min data for {len(symbols)} symbols...")
        fetcher = DataFetcher()
        
        for symbol in symbols:
            try:
                # Use UTC explicitly
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(hours=1)  # Load 1 hour of 5-min bars
                
                df = fetcher.fetch_data(
                    symbol=symbol,
                    timeframe='5min',
                    start_date=start_time,
                    end_date=end_time,
                    use_cache=False
                )
                
                if not df.empty:
                    for idx, row in df.iterrows():
                        # Ensure index is UTC
                        timestamp = ensure_utc(idx) if isinstance(idx, datetime) else idx
                        self.update_bar(
                            symbol=symbol,
                            open_price=row['open'],
                            high=row['high'],
                            low=row['low'],
                            close=row['close'],
                            volume=row['volume'],
                            timestamp=timestamp
                        )
                    logger.info(f"✓ Loaded {len(df)} 5-min bars for {symbol}")
                
            except Exception as e:
                logger.error(f"Error loading 5-min data for {symbol}: {e}")
        
        # Connect WebSocket for 5-minute aggregates
        logger.info("Connecting to WebSocket for 5-min data...")
        self.ws_client = PolygonWebSocketClient()
        await self.ws_client.connect()
        
        # Subscribe to 5-minute aggregates
        # Note: You might need to aggregate 1-min bars if AM doesn't support 5-min directly
        await self.ws_client.subscribe(
            symbols=symbols,
            channels=['AM'],  # Will need to aggregate to 5-min
            callback=self._handle_websocket_data
        )
        
        for symbol in symbols:
            self.active_symbols[symbol] = [callback] if callback else []
            self.calculation_tasks[symbol] = asyncio.create_task(
                self._calculation_loop(symbol)
            )
        
        logger.info(f"✓ Started 5-min trend monitoring for {symbols}")
    
    async def _handle_websocket_data(self, data: Dict):
        """Handle incoming WebSocket data - aggregate to 5-min bars"""
        # This is simplified - in production you'd need proper 5-min aggregation
        try:
            event_type = data.get('event_type')
            symbol = data.get('symbol')
            
            if event_type == 'aggregate' and symbol in self.active_symbols:
                # Validate and ensure UTC timestamp
                timestamp = self._validate_timestamp(
                    datetime.fromtimestamp(data['timestamp'] / 1000, tz=timezone.utc),
                    f"WebSocket-{symbol}"
                )
                
                # For now, update on each 1-min bar
                # In production, aggregate 5 1-min bars into 1 5-min bar
                signal = self.update_bar(
                    symbol=symbol,
                    open_price=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    volume=data['volume'],
                    timestamp=timestamp
                )
                
                if signal:
                    logger.info(f"5-min signal for {symbol}: {signal.signal} ({signal.bias})")
                    await self._notify_callbacks(symbol, signal)
                    
        except Exception as e:
            logger.error(f"Error handling WebSocket data: {e}")
    
    async def _calculation_loop(self, symbol: str):
        """Periodic calculation loop"""
        while symbol in self.active_symbols:
            try:
                await asyncio.sleep(self.calculation_interval)
                
                signal = self.latest_signals.get(symbol)
                if signal:
                    # Recalculate
                    prices = list(self.price_buffers[symbol])
                    if prices:
                        new_signal = self._calculate_position_signal(
                            symbol, prices[-1],
                            list(self.timestamp_buffers[symbol])[-1]
                        )
                        
                        # Check for significant changes
                        if new_signal.signal != signal.signal or new_signal.bias != signal.bias:
                            logger.info(f"{symbol} 5-min change: {signal.signal} → {new_signal.signal}")
                            self.latest_signals[symbol] = new_signal
                            await self._notify_callbacks(symbol, new_signal)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in 5-min calculation loop for {symbol}: {e}")
    
    async def _notify_callbacks(self, symbol: str, signal: PositionSignal5Min):
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
        logger.info("Stopping 5-min trend calculator...")
        
        for task in self.calculation_tasks.values():
            task.cancel()
        
        await asyncio.gather(*self.calculation_tasks.values(), return_exceptions=True)
        
        if self.ws_client:
            await self.ws_client.disconnect()
        
        logger.info("✓ 5-min trend calculator stopped")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        avg_time = self.total_calculation_time / self.calculation_count if self.calculation_count > 0 else 0
        return {
            'total_calculations': self.calculation_count,
            'average_time_ms': avg_time,
            'active_symbols': len(self.active_symbols),
            'timeframes': {
                'short': self.short_lookback * 5,
                'medium': self.medium_lookback * 5,
                'long': self.long_lookback * 5
            }
        }


# ============= LIVE TESTING =============
async def run_5min_test():
    """Test 5-minute trend calculation"""
    print("=== Testing 5-Minute Statistical Trend for Position Management ===\n")
    
    TEST_SYMBOLS = ['TSLA', 'AAPL', 'SPY', 'QQQ', 'NVDA']
    TEST_DURATION = 180  # 3 minutes
    
    calculator = StatisticalTrend5Min(
        short_lookback=3,    # 15-min trend
        medium_lookback=5,   # 25-min trend  
        long_lookback=10,    # 50-min trend
        calculation_interval=30  # Every 30 seconds
    )
    
    update_count = 0
    
    def display_signal(signal: PositionSignal5Min):
        nonlocal update_count
        update_count += 1
        
        print(f"\n{'='*70}")
        print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}] 5-MIN UPDATE #{update_count} - {signal.symbol}")
        print(f"{'='*70}")
        
        # Price and main signal
        print(f"💰 Price: ${signal.price:.2f}")
        print(f"🕐 Data Time: {signal.timestamp.strftime('%H:%M:%S UTC')}")
        
        # Market state with emoji
        state_emoji = {
            'TRENDING': '📈',
            'CONSOLIDATING': '🔄',
            'VOLATILE': '⚡',
            'UNKNOWN': '❓'
        }
        
        print(f"\n📊 MARKET STATE: {state_emoji.get(signal.market_state, '?')} {signal.market_state}")
        print(f"📍 SIGNAL: {signal.signal}")
        print(f"🎯 BIAS: {signal.bias}")
        print(f"💪 Strength: {signal.strength:.1f}%")
        print(f"🎲 Confidence: {signal.confidence:.1f}%")
        
        # Position recommendation
        print(f"\n💡 RECOMMENDATION: {signal.recommendation}")
        
        # Timeframe breakdown
        print(f"\n📈 TIMEFRAME ANALYSIS:")
        
        # Short (15-min)
        if signal.short_trend:
            s = signal.short_trend
            print(f"   15-min: {s['direction'].upper()} "
                  f"(Strength: {s['strength']:.1f}%, "
                  f"VWAP: {s['vwap_position']:+.2f}%)")
        
        # Medium (25-min)
        if signal.medium_trend:
            m = signal.medium_trend
            print(f"   25-min: {m['direction'].upper()} "
                  f"(Strength: {m['strength']:.1f}%, "
                  f"Score: {m['score']:.3f})")
        
        # Long (50-min)
        if signal.long_trend:
            l = signal.long_trend
            print(f"   50-min: {l['direction'].upper()} "
                  f"(Strength: {l['strength']:.1f}%, "
                  f"Score: {l['score']:.3f})")
        
        # Trading guidance
        print(f"\n🎮 TRADING GUIDANCE:")
        if signal.bias == 'BULLISH':
            print("   ✅ Look for LONG entries on 1-min signals")
            print("   ❌ Avoid SHORT positions")
        elif signal.bias == 'BEARISH':
            print("   ✅ Look for SHORT entries on 1-min signals")
            print("   ❌ Avoid LONG positions")
        else:
            print("   ⚠️  Be cautious - No clear directional bias")
            print("   💡 Consider range-trading strategies")
    
    try:
        # Note: This test uses 1-min bars from WebSocket
        # In production, you'd aggregate to proper 5-min bars
        await calculator.start_websocket(TEST_SYMBOLS, display_signal)
        
        print(f"\n🚀 5-Minute Trend Monitor Started at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"📊 Tracking {len(TEST_SYMBOLS)} symbols")
        print(f"⏱️  Timeframes: 15-min, 25-min, 50-min")
        print(f"🔄 Updates every 30 seconds")
        print(f"⏰ Test duration: {TEST_DURATION} seconds")
        print(f"🌍 All timestamps in UTC\n")
        
        print("📖 Signal Guide:")
        print("   STRONG TREND UP/DOWN = Strong directional bias")
        print("   TREND UP/DOWN = Moderate directional bias")
        print("   RANGING = No clear direction\n")
        
        print("⏳ Waiting for 5-min analysis...")
        
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
        print(f"  • Updates shown: {update_count}")
        
        await calculator.stop()
        print("\n✅ 5-minute trend test completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted")
        await calculator.stop()
        
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        await calculator.stop()


if __name__ == "__main__":
    print(f"Starting 5-Minute Statistical Trend Calculator at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("Provides higher timeframe market context")
    print("All timestamps are in UTC\n")
    print("Features:")
    print("• Multi-timeframe: 15/25/50 minute analysis")
    print("• Market state detection (trending/ranging)")
    print("• Position sizing recommendations")
    print("• VWAP analysis for mean reversion")
    print("• Clear directional bias signals")
    print("• UTC timestamp enforcement\n")
    
    asyncio.run(run_5min_test())