# modules/calculations/trend/statistical_trend_1min.py
"""
Module: 1-Minute Statistical Trend Calculation Optimized for Short-Term Trading
Purpose: Generate rapid signals for trades lasting 5-60 minutes
Features: Multi-timeframe analysis (3/5/10 min), WebSocket integration, Scalping signals
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
class TrendComponents:
    """Individual trend calculation components"""
    linear_slope: float
    linear_r_squared: float
    mann_kendall_trend: int  # -1, 0, 1
    mann_kendall_z_score: float
    mann_kendall_p_value: float
    kalman_trend: int  # -1, 0, 1
    kalman_price: float
    price_momentum: float
    price_acceleration: float


@dataclass
class StatisticalTrendResult:
    """Complete statistical trend analysis result"""
    symbol: str
    timestamp: datetime
    current_price: float
    lookback_periods: int
    components: TrendComponents
    composite_trend: float  # -1 to 1, strength of trend
    trend_direction: str  # 'bullish', 'bearish', 'neutral'
    trend_strength: float  # 0 to 100
    confidence_level: float  # 0 to 100
    calculation_time_ms: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class ScalperSignal:
    """Trading signal optimized for short-term trades"""
    symbol: str
    timestamp: datetime
    price: float
    signal: str  # 'STRONG BUY', 'BUY', 'SCALP BUY', 'NEUTRAL', etc.
    confidence: float
    reason: str
    target_hold: str  # e.g., '5-15 min'
    micro_trend: Dict
    short_trend: Dict
    medium_trend: Dict
    strength: float = 0.0  # Added strength attribute for dashboard compatibility


class StatisticalTrend1Min:
    """
    1-Minute Statistical Trend Calculator optimized for short-term trading.
    Uses multiple micro-timeframes for rapid signal generation.
    """
    
    def __init__(self,
                 micro_lookback: int = 3,      # 3-min for entries
                 short_lookback: int = 5,       # 5-min for confirmation
                 medium_lookback: int = 10,     # 10-min for context
                 calculation_interval: int = 15):
        """
        Initialize trend calculator for <1 hour holds.
        
        Args:
            micro_lookback: Ultra-short trend for entry timing (default 3)
            short_lookback: Short trend for confirmation (default 5)
            medium_lookback: Medium trend for context (default 10)
            calculation_interval: Seconds between calculations
        """
        self.micro_lookback = micro_lookback
        self.short_lookback = short_lookback
        self.medium_lookback = medium_lookback
        self.calculation_interval = calculation_interval
        
        # Use medium as max buffer size
        self.max_lookback = medium_lookback
        self.warmup_periods = micro_lookback  # Can start after just 3 bars
        
        # Data buffers
        self.price_buffers: Dict[str, deque] = {}
        self.volume_buffers: Dict[str, deque] = {}
        self.timestamp_buffers: Dict[str, deque] = {}
        
        # Kalman filter states
        self.kalman_states: Dict[str, Dict] = {}
        
        # WebSocket integration
        self.ws_client = None
        self.active_symbols: Dict[str, List[Callable]] = {}
        self.calculation_tasks: Dict[str, asyncio.Task] = {}
        
        # Latest results for each timeframe
        self.latest_signals: Dict[str, ScalperSignal] = {}
        
        # Performance tracking
        self.calculation_count = 0
        self.total_calculation_time = 0
        
        logger.info(f"Initialized scalper trend calculator: "
                   f"Micro={micro_lookback}, Short={short_lookback}, Medium={medium_lookback}")
    
    def initialize_buffers(self, symbol: str):
        """Initialize data buffers for a symbol"""
        self.price_buffers[symbol] = deque(maxlen=self.max_lookback)
        self.volume_buffers[symbol] = deque(maxlen=self.max_lookback)
        self.timestamp_buffers[symbol] = deque(maxlen=self.max_lookback)
        self.kalman_states[symbol] = self._initialize_kalman()
        logger.info(f"Initialized buffers for {symbol}")
    
    def _initialize_kalman(self) -> Dict:
        """Initialize Kalman filter state"""
        return {
            'x': 0.0,      # State estimate
            'P': 1.0,      # Error covariance
            'Q': 0.00005,  # Lower process noise for 1-min bars
            'R': 0.005,    # Lower measurement noise for faster response
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
    
    def update_price(self, symbol: str, price: float, volume: float,
                    timestamp: Optional[datetime] = None) -> Optional[ScalperSignal]:
        """
        Update with new price and generate scalper signal.
        
        Returns:
            ScalperSignal with multi-timeframe analysis
        """
        start_time = time_module.perf_counter()
        
        # Initialize if needed
        if symbol not in self.price_buffers:
            self.initialize_buffers(symbol)
        
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Update buffers
        self.price_buffers[symbol].append(price)
        self.volume_buffers[symbol].append(volume)
        self.timestamp_buffers[symbol].append(timestamp)
        
        # Need at least micro lookback to generate signals
        if len(self.price_buffers[symbol]) < self.micro_lookback:
            logger.debug(f"{symbol}: Warming up ({len(self.price_buffers[symbol])}/{self.micro_lookback})")
            return None
        
        # Calculate multi-timeframe trends
        signal = self._calculate_scalper_signal(symbol, price, timestamp)
        
        # Track performance
        calculation_time = (time_module.perf_counter() - start_time) * 1000
        
        # Update tracking
        self.calculation_count += 1
        self.total_calculation_time += calculation_time
        self.latest_signals[symbol] = signal
        
        return signal
    
    def _calculate_scalper_signal(self, symbol: str, price: float, 
                                timestamp: datetime) -> ScalperSignal:
        """Calculate complete scalper signal with all timeframes"""
        trends = {}
        
        # 1. Micro trend (3-min)
        if len(self.price_buffers[symbol]) >= self.micro_lookback:
            micro_result = self._calculate_trend_with_lookback(
                symbol, price, self.micro_lookback
            )
            trends['micro'] = {
                'lookback': self.micro_lookback,
                'direction': micro_result.trend_direction,
                'strength': micro_result.trend_strength,
                'confidence': micro_result.confidence_level,
                'momentum': micro_result.components.price_momentum,
                'score': micro_result.composite_trend
            }
        
        # 2. Short trend (5-min)
        if len(self.price_buffers[symbol]) >= self.short_lookback:
            short_result = self._calculate_trend_with_lookback(
                symbol, price, self.short_lookback
            )
            trends['short'] = {
                'lookback': self.short_lookback,
                'direction': short_result.trend_direction,
                'strength': short_result.trend_strength,
                'confidence': short_result.confidence_level,
                'momentum': short_result.components.price_momentum,
                'score': short_result.composite_trend
            }
        
        # 3. Medium trend (10-min)
        if len(self.price_buffers[symbol]) >= self.medium_lookback:
            medium_result = self._calculate_trend_with_lookback(
                symbol, price, self.medium_lookback
            )
            trends['medium'] = {
                'lookback': self.medium_lookback,
                'direction': medium_result.trend_direction,
                'strength': medium_result.trend_strength,
                'confidence': medium_result.confidence_level,
                'score': medium_result.composite_trend
            }
        
        # Generate trading signal
        trading_signal = self._generate_trading_signal(trends)
        
        # Calculate overall strength (average of all timeframe strengths)
        strength_values = []
        for trend in trends.values():
            if 'strength' in trend:
                strength_values.append(trend['strength'])
        overall_strength = sum(strength_values) / len(strength_values) if strength_values else 0.0
        
        return ScalperSignal(
            symbol=symbol,
            timestamp=timestamp,
            price=price,
            signal=trading_signal['signal'],
            confidence=trading_signal['confidence'],
            reason=trading_signal['reason'],
            target_hold=trading_signal['target_hold'],
            micro_trend=trends.get('micro', {}),
            short_trend=trends.get('short', {}),
            medium_trend=trends.get('medium', {}),
            strength=overall_strength  # Set the overall strength
        )
    
    def _calculate_trend_with_lookback(self, symbol: str, current_price: float,
                                     lookback: int) -> StatisticalTrendResult:
        """Calculate trend for specific lookback period"""
        # Get data slice
        all_prices = list(self.price_buffers[symbol])
        all_volumes = list(self.volume_buffers[symbol])
        all_timestamps = list(self.timestamp_buffers[symbol])
        
        prices = np.array(all_prices[-lookback:])
        volumes = np.array(all_volumes[-lookback:])
        timestamps = all_timestamps[-lookback:]
        
        warnings = []
        
        # Statistical calculations
        slope, intercept, r_squared = self._fast_linear_regression(prices)
        normalized_slope = (slope / prices[-1]) * 100 if prices[-1] != 0 else 0
        
        mk_trend, mk_z, mk_p = self._fast_mann_kendall(prices)
        
        kalman_price = self._update_kalman(symbol, current_price)
        kalman_trend = 1 if kalman_price > np.mean(prices[-min(3, lookback):]) else -1
        
        # Momentum - ultra sensitive for scalping
        momentum_periods = min(2, lookback - 1)  # 2-bar momentum
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
        
        components = TrendComponents(
            linear_slope=normalized_slope,
            linear_r_squared=r_squared,
            mann_kendall_trend=mk_trend,
            mann_kendall_z_score=mk_z,
            mann_kendall_p_value=mk_p,
            kalman_trend=kalman_trend,
            kalman_price=kalman_price,
            price_momentum=momentum,
            price_acceleration=acceleration
        )
        
        # Use scalper-optimized scoring
        composite_score = self._calculate_scalper_composite_score(components, lookback)
        
        # Determine trend
        if abs(composite_score) < 0.15:  # Lower threshold for scalping
            direction = 'neutral'
        elif composite_score > 0:
            direction = 'bullish'
        else:
            direction = 'bearish'
        
        strength = abs(composite_score) * 100
        confidence = self._calculate_confidence(components)
        
        return StatisticalTrendResult(
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
    
    def _calculate_scalper_composite_score(self, components: TrendComponents, 
                                         lookback: int) -> float:
        """Calculate composite score optimized for scalping"""
        # Different weights based on lookback
        if lookback <= self.micro_lookback:
            # Ultra-short: momentum is everything
            weights = {
                'linear': 0.10,
                'mann_kendall': 0.10,
                'kalman': 0.20,
                'momentum': 0.60
            }
            sensitivity = 0.3  # Very sensitive
        elif lookback <= self.short_lookback:
            # Short: balanced but momentum-heavy
            weights = {
                'linear': 0.15,
                'mann_kendall': 0.15,
                'kalman': 0.25,
                'momentum': 0.45
            }
            sensitivity = 0.5
        else:
            # Medium: more balanced
            weights = {
                'linear': 0.25,
                'mann_kendall': 0.20,
                'kalman': 0.25,
                'momentum': 0.30
            }
            sensitivity = 1.0
        
        # Sensitive normalization
        slope_signal = np.tanh(components.linear_slope / sensitivity)
        momentum_signal = np.tanh(components.price_momentum / (sensitivity * 0.5))
        
        weighted_sum = (
            weights['linear'] * slope_signal * components.linear_r_squared +
            weights['mann_kendall'] * components.mann_kendall_trend +
            weights['kalman'] * components.kalman_trend +
            weights['momentum'] * momentum_signal
        )
        
        return np.clip(weighted_sum, -1, 1)
    
    def _calculate_confidence(self, components: TrendComponents) -> float:
        """Calculate confidence with scalping adjustments"""
        signals = [
            np.sign(components.linear_slope),
            components.mann_kendall_trend,
            components.kalman_trend,
            np.sign(components.price_momentum)
        ]
        
        non_zero_signals = [s for s in signals if s != 0]
        if not non_zero_signals:
            return 0
        
        agreement = abs(sum(non_zero_signals)) / len(non_zero_signals)
        confidence = agreement * 100
        
        # Boost for significant Mann-Kendall
        if components.mann_kendall_p_value < 0.05:
            confidence = min(100, confidence * 1.2)
        
        # Momentum bonus for scalping
        if abs(components.price_momentum) > 0.3:
            confidence = min(100, confidence * 1.1)
        
        confidence *= (0.5 + 0.5 * components.linear_r_squared)
        
        return min(100, confidence)
    
    def _generate_trading_signal(self, trends: Dict) -> Dict:
        """Generate actionable trading signals for short-term trades"""
        micro = trends.get('micro', {})
        short = trends.get('short', {})
        medium = trends.get('medium', {})
        
        # Check for momentum surge (key for scalping)
        micro_momentum = abs(micro.get('momentum', 0))
        momentum_surge = micro_momentum > 0.3  # 0.3% in 2 minutes
        strong_momentum = micro_momentum > 0.5  # 0.5% is strong
        
        # STRONG SIGNALS - All aligned with momentum
        if (micro.get('direction') == 'bullish' and 
            short.get('direction') == 'bullish' and
            medium.get('direction') in ['bullish', 'neutral'] and
            momentum_surge):
            confidence = min(100, (micro['confidence'] + short['confidence']) / 2)
            return {
                'signal': 'STRONG BUY',
                'confidence': confidence,
                'reason': 'All timeframes aligned with momentum',
                'target_hold': '30-60 min'
            }
            
        if (micro.get('direction') == 'bearish' and 
            short.get('direction') == 'bearish' and
            medium.get('direction') in ['bearish', 'neutral'] and
            momentum_surge):
            confidence = min(100, (micro['confidence'] + short['confidence']) / 2)
            return {
                'signal': 'STRONG SELL',
                'confidence': confidence,
                'reason': 'All timeframes aligned with momentum',
                'target_hold': '30-60 min'
            }
        
        # REGULAR SIGNALS - Micro + Short aligned
        if micro.get('direction') == 'bullish' and short.get('direction') == 'bullish':
            confidence = (micro['confidence'] + short['confidence']) / 2
            return {
                'signal': 'BUY',
                'confidence': confidence,
                'reason': 'Short-term bullish alignment',
                'target_hold': '15-30 min'
            }
            
        if micro.get('direction') == 'bearish' and short.get('direction') == 'bearish':
            confidence = (micro['confidence'] + short['confidence']) / 2
            return {
                'signal': 'SELL',
                'confidence': confidence,
                'reason': 'Short-term bearish alignment',
                'target_hold': '15-30 min'
            }
        
        # SCALP SIGNALS - Micro only with strong momentum
        if micro.get('direction') == 'bullish' and strong_momentum:
            return {
                'signal': 'SCALP BUY',
                'confidence': micro['confidence'],
                'reason': f'Strong bullish momentum ({micro_momentum:.2f}%)',
                'target_hold': '5-15 min'
            }
            
        if micro.get('direction') == 'bearish' and strong_momentum:
            return {
                'signal': 'SCALP SELL',
                'confidence': micro['confidence'],
                'reason': f'Strong bearish momentum ({micro_momentum:.2f}%)',
                'target_hold': '5-15 min'
            }
        
        # NEUTRAL
        return {
            'signal': 'NEUTRAL',
            'confidence': 0,
            'reason': f"No alignment: Micro={micro.get('direction', '?')}, "
                     f"Short={short.get('direction', '?')}, "
                     f"Medium={medium.get('direction', '?')}",
            'target_hold': 'N/A'
        }
    
    def get_batch_analysis(self, symbols: List[str]) -> Dict[str, Optional[ScalperSignal]]:
        """
        Get latest signals for multiple symbols (batch analysis).
        
        Args:
            symbols: List of symbols to get signals for
            
        Returns:
            Dictionary mapping symbols to their latest signals
        """
        results = {}
        for symbol in symbols:
            results[symbol] = self.latest_signals.get(symbol)
        return results
    
    async def start_websocket(self, symbols: List[str], callback: Optional[Callable] = None):
        """Start real-time monitoring with WebSocket"""
        from polygon import DataFetcher, PolygonWebSocketClient
        
        # Load historical data
        logger.info(f"Loading historical data for {len(symbols)} symbols...")
        fetcher = DataFetcher()
        
        for symbol in symbols:
            try:
                end_time = datetime.now()
                start_time = end_time - timedelta(minutes=15)  # Only need 15 min
                
                df = fetcher.fetch_data(
                    symbol=symbol,
                    timeframe='1min',
                    start_date=start_time,
                    end_date=end_time,
                    use_cache=False
                )
                
                if not df.empty:
                    for idx, row in df.iterrows():
                        self.update_price(
                            symbol=symbol,
                            price=row['close'],
                            volume=row['volume'],
                            timestamp=idx
                        )
                    logger.info(f"✓ Loaded {len(df)} bars for {symbol}")
                
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
        
        # Connect WebSocket
        logger.info("Connecting to WebSocket...")
        self.ws_client = PolygonWebSocketClient()
        await self.ws_client.connect()
        
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
        
        logger.info(f"✓ Started real-time scalper monitoring for {symbols}")
    
    async def _handle_websocket_data(self, data: Dict):
        """Handle incoming WebSocket data"""
        try:
            event_type = data.get('event_type')
            symbol = data.get('symbol')
            
            if event_type == 'aggregate' and symbol in self.active_symbols:
                signal = self.update_price(
                    symbol=symbol,
                    price=data['close'],
                    volume=data['volume'],
                    timestamp=datetime.fromtimestamp(data['timestamp'] / 1000, tz=timezone.utc)
                )
                
                if signal and signal.signal != 'NEUTRAL':
                    logger.info(f"New signal for {symbol}: {signal.signal}")
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
                        new_signal = self._calculate_scalper_signal(
                            symbol, prices[-1], 
                            list(self.timestamp_buffers[symbol])[-1]
                        )
                        self.latest_signals[symbol] = new_signal
                        
                        # Notify if signal changed
                        if new_signal.signal != signal.signal:
                            logger.info(f"{symbol} signal changed: {signal.signal} → {new_signal.signal}")
                            await self._notify_callbacks(symbol, new_signal)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in calculation loop for {symbol}: {e}")
    
    async def _notify_callbacks(self, symbol: str, signal: ScalperSignal):
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
        logger.info("Stopping scalper calculator...")
        
        for task in self.calculation_tasks.values():
            task.cancel()
        
        await asyncio.gather(*self.calculation_tasks.values(), return_exceptions=True)
        
        if self.ws_client:
            await self.ws_client.disconnect()
        
        logger.info("✓ Scalper calculator stopped")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        avg_time = self.total_calculation_time / self.calculation_count if self.calculation_count > 0 else 0
        return {
            'total_calculations': self.calculation_count,
            'average_time_ms': avg_time,
            'active_symbols': len(self.active_symbols),
            'timeframes': {
                'micro': self.micro_lookback,
                'short': self.short_lookback,
                'medium': self.medium_lookback
            }
        }


# ============= LIVE TESTING =============
async def run_scalper_test():
    """Test scalper-optimized trend calculation"""
    print("=== Testing 1-Minute Statistical Trend for Short-Term Trading ===\n")
    
    TEST_SYMBOLS = ['TSLA', 'AAPL', 'SPY', 'QQQ', 'NVDA']
    TEST_DURATION = 180  # 3 minutes
    
    calculator = StatisticalTrend1Min(
        micro_lookback=3,    # 3-min for entries
        short_lookback=5,    # 5-min for confirmation
        medium_lookback=10,  # 10-min for context
        calculation_interval=15
    )
    
    update_count = 0
    active_signals = {}
    
    def display_signal(signal: ScalperSignal):
        nonlocal update_count
        update_count += 1
        
        # Track signal changes
        prev_signal = active_signals.get(signal.symbol, {}).get('signal', 'NONE')
        active_signals[signal.symbol] = {
            'signal': signal.signal,
            'entry_time': signal.timestamp,
            'entry_price': signal.price
        }
        
        print(f"\n{'='*70}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] SIGNAL #{update_count} - {signal.symbol}")
        print(f"{'='*70}")
        
        # Price info
        print(f"💰 Price: ${signal.price:.2f}")
        
        # Signal with emoji
        emoji_map = {
            'STRONG BUY': '🟢🟢',
            'BUY': '🟢',
            'SCALP BUY': '🟢⚡',
            'STRONG SELL': '🔴🔴',
            'SELL': '🔴',
            'SCALP SELL': '🔴⚡',
            'NEUTRAL': '⚪'
        }
        
        emoji = emoji_map.get(signal.signal, '❓')
        
        # Signal change highlight
        if prev_signal != signal.signal and prev_signal != 'NONE':
            print(f"🔄 SIGNAL CHANGE: {prev_signal} → {signal.signal}")
        
        print(f"\n📊 TRADING SIGNAL: {emoji} {signal.signal}")
        print(f"   Confidence: {signal.confidence:.1f}%")
        print(f"   Strength: {signal.strength:.1f}%")  # Now available
        print(f"   Reason: {signal.reason}")
        print(f"   Target Hold: {signal.target_hold}")
        
        # Multi-timeframe analysis
        print(f"\n📈 TIMEFRAME ANALYSIS:")
        
        # Micro (3-min)
        if signal.micro_trend:
            m = signal.micro_trend
            print(f"   3-min: {m['direction'].upper()} "
                  f"(Strength: {m['strength']:.1f}%, "
                  f"Momentum: {m['momentum']:.2f}%)")
        
        # Short (5-min)
        if signal.short_trend:
            s = signal.short_trend
            print(f"   5-min: {s['direction'].upper()} "
                  f"(Strength: {s['strength']:.1f}%, "
                  f"Score: {s['score']:.3f})")
        
        # Medium (10-min)
        if signal.medium_trend:
            md = signal.medium_trend
            print(f"   10-min: {md['direction'].upper()} "
                  f"(Strength: {md['strength']:.1f}%, "
                  f"Score: {md['score']:.3f})")
        
        # Action required
        if signal.signal in ['STRONG BUY', 'BUY', 'SCALP BUY']:
            print(f"\n✅ ACTION: Consider LONG entry")
        elif signal.signal in ['STRONG SELL', 'SELL', 'SCALP SELL']:
            print(f"\n❌ ACTION: Consider SHORT entry")
        else:
            print(f"\n⏸️  ACTION: Wait for clear signal")
    
    try:
        await calculator.start_websocket(TEST_SYMBOLS, display_signal)
        
        print(f"\n🚀 Scalper Monitor Started")
        print(f"📊 Tracking {len(TEST_SYMBOLS)} symbols")
        print(f"⏱️  Timeframes: 3-min (entry), 5-min (confirm), 10-min (context)")
        print(f"🔄 Updates every 15 seconds + new bars")
        print(f"⏰ Test duration: {TEST_DURATION} seconds\n")
        
        print("💡 Signal Guide:")
        print("   STRONG BUY/SELL = Full position (30-60 min hold)")
        print("   BUY/SELL = 75% position (15-30 min hold)")
        print("   SCALP BUY/SELL = 50% position (5-15 min hold)")
        print("   NEUTRAL = No position\n")
        
        print("⏳ Waiting for signals...")
        
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
        print(f"  • Signal updates: {update_count}")
        
        # Active signals summary
        print(f"\n📈 Final Signals:")
        for symbol, data in active_signals.items():
            print(f"  • {symbol}: {data['signal']}")
        
        # Test batch analysis
        print(f"\n🔍 Testing batch analysis:")
        batch_results = calculator.get_batch_analysis(TEST_SYMBOLS)
        for symbol, signal in batch_results.items():
            if signal:
                print(f"  • {symbol}: {signal.signal} (Strength: {signal.strength:.1f}%)")
        
        await calculator.stop()
        print("\n✅ Scalper test completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted")
        await calculator.stop()
        
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        await calculator.stop()


if __name__ == "__main__":
    print("Starting 1-Minute Statistical Trend Calculator")
    print("Optimized for short-term trading (<1 hour holds)\n")
    print("Features:")
    print("• Multi-timeframe analysis: 3/5/10 minutes")
    print("• Rapid signal generation for scalping")
    print("• Clear entry signals with hold times")
    print("• WebSocket real-time updates\n")
    
    asyncio.run(run_scalper_test())