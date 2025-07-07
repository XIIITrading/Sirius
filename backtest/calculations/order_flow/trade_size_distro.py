# modules/calculations/order-flow/trade_size_distro.py
"""
Module: Trade Size Distribution Analysis
Purpose: Identify institutional activity by monitoring large vs small trade ratios
Features: Real-time analysis, historical baselines, institutional pattern detection
Performance Target: <100 microseconds per trade
Time Handling: All timestamps in UTC
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
import logging
import sys
import os
import asyncio
import time as time_module
from functools import lru_cache
import json

# Fix import path
current_dir = os.path.dirname(os.path.abspath(__file__))
order_flow_dir = current_dir
calculations_dir = os.path.dirname(order_flow_dir)
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
class TradeSizeComponents:
    """Detailed components of trade size analysis"""
    current_ratio: float
    volume_weighted_ratio: float
    large_direction_ratio: float
    time_weighted_ratio: float
    avg_trade_size: float
    large_volume: float
    total_volume: float
    zscore: float
    historical_mean: float
    historical_std: float
    block_sequences: int
    iceberg_candidates: int
    institutional_score: float
    dynamic_threshold: float


@dataclass
class TradeSizeSignal:
    """Complete trade size distribution signal"""
    symbol: str
    timestamp: datetime
    current_price: float
    bull_score: int  # -2 to +2
    bear_score: int  # -2 to +2
    confidence: float  # 0 to 1
    components: TradeSizeComponents
    signal_type: str  # 'ACCUMULATION', 'DISTRIBUTION', 'NEUTRAL', etc.
    signal_strength: str  # 'EXCEPTIONAL', 'STRONG', 'MODERATE', 'WEAK'
    reason: str
    calculation_time_ms: float
    trade_count: int
    warnings: List[str] = field(default_factory=list)


@dataclass
class Trade:
    """Individual trade data"""
    symbol: str
    price: float
    size: int
    timestamp: datetime
    is_buy: Optional[bool] = None  # True=buy, False=sell, None=unknown


class TradeSizeDistribution:
    """
    Trade Size Distribution Calculator for institutional activity detection.
    Monitors ratio of large trades to small trades to identify smart money flow.
    """
    
    def __init__(self,
                 large_trade_threshold: int = 1000,
                 buffer_size: int = 500,
                 historical_lookback: int = 20,
                 zscore_threshold: float = 2.0,
                 premarket_adjustment: bool = True):
        """
        Initialize trade size distribution calculator.
        
        Args:
            large_trade_threshold: Minimum shares for large trade (default 1000)
            buffer_size: Number of trades to keep in buffer (default 500)
            historical_lookback: Days for historical baseline (default 20)
            zscore_threshold: Standard deviations for significance (default 2.0)
            premarket_adjustment: Whether to adjust for pre-market conditions
        """
        self.large_trade_threshold = large_trade_threshold
        self.buffer_size = buffer_size
        self.historical_lookback = historical_lookback
        self.zscore_threshold = zscore_threshold
        self.premarket_adjustment = premarket_adjustment
        
        # Trade buffers per symbol
        self.trade_buffers: Dict[str, deque] = {}
        
        # Historical cache per symbol
        self.historical_cache: Dict[str, Dict[Tuple[int, int], Dict]] = {}
        
        # Session metrics
        self.session_metrics: Dict[str, Dict] = {}
        
        # Latest signals
        self.latest_signals: Dict[str, TradeSizeSignal] = {}
        
        # WebSocket integration
        self.ws_client = None
        self.active_symbols: Dict[str, List[Callable]] = {}
        
        # Performance tracking
        self.calculation_count = 0
        self.total_calculation_time = 0
        
        logger.info(f"Initialized TradeSizeDistribution: threshold={large_trade_threshold}, "
                   f"buffer={buffer_size}, lookback={historical_lookback}")
    
    def initialize_symbol(self, symbol: str):
        """Initialize buffers for a new symbol"""
        self.trade_buffers[symbol] = deque(maxlen=self.buffer_size)
        self.historical_cache[symbol] = {}
        self.session_metrics[symbol] = {
            'total_large_trades': 0,
            'total_trades': 0,
            'largest_trade': 0,
            'institutional_periods': []
        }
        logger.info(f"Initialized buffers for {symbol}")
    
    def process_trade(self, trade: Trade) -> Optional[TradeSizeSignal]:
        """
        Process a new trade and generate signal.
        
        Args:
            trade: Trade object with symbol, price, size, timestamp
            
        Returns:
            TradeSizeSignal if enough data, None otherwise
        """
        start_time = time_module.perf_counter()
        
        # Initialize if needed
        if trade.symbol not in self.trade_buffers:
            self.initialize_symbol(trade.symbol)
        
        # Add to buffer
        self.trade_buffers[trade.symbol].append(trade)
        
        # Update session metrics
        metrics = self.session_metrics[trade.symbol]
        metrics['total_trades'] += 1
        if trade.size >= self.large_trade_threshold:
            metrics['total_large_trades'] += 1
        metrics['largest_trade'] = max(metrics['largest_trade'], trade.size)
        
        # Need minimum trades for analysis
        if len(self.trade_buffers[trade.symbol]) < 100:
            logger.debug(f"{trade.symbol}: Warming up "
                        f"({len(self.trade_buffers[trade.symbol])}/100)")
            return None
        
        # Calculate metrics
        trades_list = list(self.trade_buffers[trade.symbol])
        
        # Core calculations
        components = self._calculate_trade_size_metrics(trades_list)
        
        # Institutional activity detection
        inst_activity = self._detect_institutional_activity(trades_list)
        components.block_sequences = inst_activity['block_sequences']
        components.iceberg_candidates = inst_activity['iceberg_candidates']
        components.institutional_score = inst_activity['institutional_score']
        
        # Historical comparison
        time_key = self._get_time_key(trade.timestamp)
        if time_key not in self.historical_cache[trade.symbol]:
            self.historical_cache[trade.symbol][time_key] = self._calculate_historical_baseline(
                trade.symbol, trade.timestamp
            )
        
        historical = self.historical_cache[trade.symbol][time_key]
        components.historical_mean = historical['mean']
        components.historical_std = historical['std']
        components.zscore = self._calculate_zscore(
            components.volume_weighted_ratio, historical
        )
        
        # Generate signal
        signal = self._calculate_trade_size_score(
            trade.symbol, trade, components, inst_activity
        )
        
        # Track performance
        calculation_time = (time_module.perf_counter() - start_time) * 1000
        signal.calculation_time_ms = calculation_time
        
        # Update tracking
        self.calculation_count += 1
        self.total_calculation_time += calculation_time
        self.latest_signals[trade.symbol] = signal
        
        # Track institutional periods
        if signal.bull_score == 2 or signal.bear_score == 2:
            metrics['institutional_periods'].append({
                'timestamp': trade.timestamp,
                'type': 'accumulation' if signal.bull_score == 2 else 'distribution',
                'signal': signal
            })
        
        return signal
    
    def _calculate_trade_size_metrics(self, trades: List[Trade]) -> TradeSizeComponents:
        """Calculate core trade size metrics"""
        # Separate large and small trades
        large_trades = [t for t in trades if t.size >= self.large_trade_threshold]
        small_trades = [t for t in trades if t.size < self.large_trade_threshold]
        
        # Basic counts
        large_count = len(large_trades)
        total_count = len(trades)
        current_ratio = large_count / total_count if total_count > 0 else 0
        
        # Volume-weighted ratio
        large_volume = sum(t.size for t in large_trades)
        total_volume = sum(t.size for t in trades)
        volume_weighted_ratio = large_volume / total_volume if total_volume > 0 else 0
        
        # Average trade size and dynamic threshold
        avg_trade_size = total_volume / total_count if total_count > 0 else 0
        dynamic_threshold = max(self.large_trade_threshold, avg_trade_size * 3)
        
        # Directional analysis (using previous price comparison)
        large_upticks = 0
        large_downticks = 0
        
        for i, trade in enumerate(trades):
            if trade.size >= self.large_trade_threshold and i > 0:
                if trade.price > trades[i-1].price:
                    large_upticks += 1
                elif trade.price < trades[i-1].price:
                    large_downticks += 1
        
        large_direction_ratio = 0
        if large_count > 0:
            large_direction_ratio = (large_upticks - large_downticks) / large_count
        
        # Time-weighted ratio (recent trades weighted more)
        time_weights = np.linspace(0.5, 1.0, len(trades))
        weighted_large_count = sum(
            w for t, w in zip(trades, time_weights)
            if t.size >= dynamic_threshold
        )
        time_weighted_ratio = weighted_large_count / sum(time_weights) if sum(time_weights) > 0 else 0
        
        return TradeSizeComponents(
            current_ratio=current_ratio,
            volume_weighted_ratio=volume_weighted_ratio,
            large_direction_ratio=large_direction_ratio,
            time_weighted_ratio=time_weighted_ratio,
            avg_trade_size=avg_trade_size,
            large_volume=large_volume,
            total_volume=total_volume,
            zscore=0.0,  # Will be set later
            historical_mean=0.0,  # Will be set later
            historical_std=0.0,  # Will be set later
            block_sequences=0,  # Will be set later
            iceberg_candidates=0,  # Will be set later
            institutional_score=0.0,  # Will be set later
            dynamic_threshold=dynamic_threshold
        )
    
    def _detect_institutional_activity(self, trades: List[Trade]) -> Dict:
        """Detect institutional trading patterns"""
        # Block trade sequences
        block_sequences = []
        for i in range(len(trades) - 5):
            window = trades[i:i+5]
            if all(t.size >= 500 for t in window):  # All medium+ trades
                prices = [t.price for t in window]
                avg_price = np.mean(prices)
                price_deviation = np.std(prices)
                
                if price_deviation < avg_price * 0.0001:  # Tight price range
                    block_sequences.append({
                        'start_idx': i,
                        'total_size': sum(t.size for t in window),
                        'avg_price': avg_price
                    })
        
        # Iceberg order detection
        price_levels = defaultdict(list)
        for trade in trades:
            price_key = round(trade.price, 2)
            price_levels[price_key].append(trade)
        
        iceberg_candidates = []
        for price, trades_at_price in price_levels.items():
            if len(trades_at_price) >= 10:  # Many trades at same price
                total_size = sum(t.size for t in trades_at_price)
                if total_size >= 5000:  # Significant total size
                    iceberg_candidates.append({
                        'price': price,
                        'trade_count': len(trades_at_price),
                        'total_size': total_size
                    })
        
        # Calculate institutional score
        institutional_score = min(
            (len(block_sequences) + len(iceberg_candidates)) / 10,
            1.0
        )
        
        return {
            'block_sequences': len(block_sequences),
            'iceberg_candidates': len(iceberg_candidates),
            'institutional_score': institutional_score,
            'details': {
                'blocks': block_sequences[:3],  # Keep top 3 for reference
                'icebergs': iceberg_candidates[:3]
            }
        }
    
    def _calculate_historical_baseline(self, symbol: str, current_time: datetime) -> Dict:
        """Calculate historical baseline for comparison"""
        # This is a placeholder - in production, would fetch from database
        # For now, return reasonable defaults
        hour = current_time.hour
        
        # Market hours adjustments
        if 9 <= hour < 10:  # First hour
            return {'mean': 0.20, 'std': 0.08, 'median': 0.18, 'percentile_90': 0.35}
        elif 15 <= hour < 16:  # Last hour
            return {'mean': 0.18, 'std': 0.07, 'median': 0.17, 'percentile_90': 0.32}
        elif 4 <= hour < 9:  # Pre-market
            return {'mean': 0.30, 'std': 0.12, 'median': 0.28, 'percentile_90': 0.50}
        else:  # Regular hours
            return {'mean': 0.15, 'std': 0.05, 'median': 0.15, 'percentile_90': 0.25}
    
    def _calculate_zscore(self, current_value: float, historical: Dict) -> float:
        """Calculate z-score vs historical baseline"""
        if historical['std'] == 0:
            return 0.0
        return (current_value - historical['mean']) / (historical['std'] + 0.0001)
    
    def _calculate_trade_size_score(self, symbol: str, trade: Trade,
                               components: TradeSizeComponents,
                               inst_activity: Dict) -> TradeSizeSignal:
        # Generate bull/bear scores and signal
        bull_score = 0
        bear_score = 0
        warnings = []
        
        # Initialize default values (this was missing!)
        signal_type = "NEUTRAL"
        signal_strength = "NEUTRAL"
        reason = "Mixed or neutral large trade activity"
        
        # Pre-market adjustments
        is_premarket = trade.timestamp.hour < 9 or trade.timestamp.hour >= 16
        zscore_threshold = 3.0 if is_premarket else self.zscore_threshold
        
        # Bull conditions
        if (components.volume_weighted_ratio > 0.3 and 
            components.zscore > zscore_threshold and 
            components.large_direction_ratio > 0.5):
            bull_score = 2  # Exceptional institutional buying
            signal_type = "STRONG ACCUMULATION"
            signal_strength = "EXCEPTIONAL"
            reason = f"Heavy institutional buying (Z={components.zscore:.2f})"
            
        elif (components.volume_weighted_ratio > 0.25 and 
            components.zscore > zscore_threshold * 0.75 and
            components.institutional_score > 0.5):
            bull_score = 2  # Strong institutional presence
            signal_type = "ACCUMULATION"
            signal_strength = "STRONG"
            reason = f"Strong institutional presence detected"
            
        elif components.volume_weighted_ratio > 0.2 and components.large_direction_ratio > 0:
            bull_score = 1  # Moderate institutional interest
            signal_type = "MODERATE ACCUMULATION"
            signal_strength = "MODERATE"
            reason = "Moderate large trade activity"
        
        # Bear conditions
        if (components.volume_weighted_ratio > 0.3 and 
            components.zscore > zscore_threshold and 
            components.large_direction_ratio < -0.5):
            bear_score = 2  # Exceptional institutional selling
            signal_type = "STRONG DISTRIBUTION"
            signal_strength = "EXCEPTIONAL"
            reason = f"Heavy institutional selling (Z={components.zscore:.2f})"
            
        elif components.volume_weighted_ratio < 0.1 and components.zscore < -1.5:
            bear_score = 1  # Absence of institutional interest
            signal_type = "RETAIL ONLY"
            signal_strength = "WEAK"
            reason = "No institutional interest"
            
        elif (components.volume_weighted_ratio > 0.25 and 
            components.large_direction_ratio < -0.3):
            bear_score = 2  # High volume distribution
            signal_type = "DISTRIBUTION"
            signal_strength = "STRONG"
            reason = "High volume distribution pattern"
        
        # Special conditions
        if components.avg_trade_size < 100:
            bear_score = max(bear_score, 1)
            warnings.append("All retail trading")
        
        # Calculate confidence
        confidence = min(abs(components.zscore), 3) / 3
        
        return TradeSizeSignal(
            symbol=symbol,
            timestamp=trade.timestamp,
            current_price=trade.price,
            bull_score=bull_score,
            bear_score=bear_score,
            confidence=confidence,
            components=components,
            signal_type=signal_type,
            signal_strength=signal_strength,
            reason=reason,
            calculation_time_ms=0,  # Will be set by caller
            trade_count=len(self.trade_buffers[symbol]),
            warnings=warnings
        )
    
    def _get_time_key(self, timestamp: datetime) -> Tuple[int, int]:
        """Get time key for caching (hour, 5-min block)"""
        return (timestamp.hour, timestamp.minute // 5)
    
    # ============= BACKTESTING FUNCTIONALITY =============
    
    async def backtest(self, symbol: str, date: datetime, 
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None) -> List[TradeSizeSignal]:
        """
        Backtest trade size distribution for a specific date/time.
        
        Args:
            symbol: Stock symbol
            date: Date to backtest
            start_time: Start time (optional)
            end_time: End time (optional)
            
        Returns:
            List of signals generated during the period
        """
        logger.info(f"Backtesting {symbol} for {date.date()}")
        
        # Import data fetcher
        try:
            from polygon import DataFetcher
            fetcher = DataFetcher()
        except ImportError:
            logger.error("DataFetcher not available for backtesting")
            return []
        
        # Set time bounds
        if start_time is None:
            start_time = date.replace(hour=4, minute=0, second=0, microsecond=0)
        if end_time is None:
            end_time = date.replace(hour=20, minute=0, second=0, microsecond=0)
        
        # Fetch trade data
        logger.info(f"Fetching trades from {start_time} to {end_time}")
        
        # Note: This assumes you have a method to fetch individual trades
        # For now, we'll simulate with 1-minute bars
        df = fetcher.fetch_data(
            symbol=symbol,
            timeframe='1min',
            start_date=start_time,
            end_date=end_time,
            use_cache=True
        )
        
        if df.empty:
            logger.warning(f"No data found for {symbol} on {date}")
            return []
        
        # Simulate trades from bars (in production, use actual trade data)
        signals = []
        for idx, row in df.iterrows():
            # Simulate multiple trades per bar
            avg_trades_per_min = 50
            for i in range(avg_trades_per_min):
                # Create synthetic trade
                size = self._generate_trade_size(row['volume'] / avg_trades_per_min)
                price_var = np.random.uniform(-0.01, 0.01)
                trade_price = row['close'] * (1 + price_var)
                trade_time = idx.to_pydatetime().replace(tzinfo=timezone.utc)
                
                trade = Trade(
                    symbol=symbol,
                    price=trade_price,
                    size=int(size),
                    timestamp=trade_time,
                    is_buy=np.random.random() > 0.5
                )
                
                signal = self.process_trade(trade)
                if signal and (signal.bull_score != 0 or signal.bear_score != 0):
                    signals.append(signal)
        
        logger.info(f"Backtest complete: {len(signals)} signals generated")
        return signals
    
    def _generate_trade_size(self, avg_size: float) -> int:
        """Generate realistic trade size distribution"""
        # 80% small trades, 15% medium, 5% large
        rand = np.random.random()
        if rand < 0.80:
            return int(np.random.uniform(10, 500))
        elif rand < 0.95:
            return int(np.random.uniform(500, 2000))
        else:
            return int(np.random.uniform(2000, 10000))
    
    # ============= WEBSOCKET FUNCTIONALITY =============
    
    async def start_websocket(self, symbols: List[str], callback: Optional[Callable] = None):
        """Start real-time monitoring with WebSocket"""
        try:
            from polygon import PolygonWebSocketClient
            
            logger.info("Connecting to WebSocket for trade data...")
            self.ws_client = PolygonWebSocketClient()
            await self.ws_client.connect()
            
            # Subscribe to trade channel
            await self.ws_client.subscribe(
                symbols=symbols,
                channels=['T'],  # Trade channel
                callback=self._handle_websocket_trade
            )
            
            for symbol in symbols:
                self.active_symbols[symbol] = [callback] if callback else []
                if symbol not in self.trade_buffers:
                    self.initialize_symbol(symbol)
            
            logger.info(f"✓ Started real-time trade monitoring for {symbols}")
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise
    
    async def _handle_websocket_trade(self, data: Dict):
        """Handle incoming trade data from WebSocket"""
        try:
            if data.get('event_type') != 'trade':
                return
            
            symbol = data.get('symbol')
            if symbol not in self.active_symbols:
                return
            
            # Create trade object
            trade = Trade(
                symbol=symbol,
                price=data['price'],
                size=data['size'],
                timestamp=datetime.fromtimestamp(data['timestamp'] / 1000, tz=timezone.utc),
                is_buy=None  # Not provided in feed
            )
            
            # Process trade
            signal = self.process_trade(trade)
            
            # Notify callbacks if significant signal
            if signal and (signal.bull_score >= 2 or signal.bear_score >= 2):
                for callback in self.active_symbols[symbol]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(signal)
                        else:
                            callback(signal)
                    except Exception as e:
                        logger.error(f"Callback error for {symbol}: {e}")
                        
        except Exception as e:
            logger.error(f"Error handling trade data: {e}")
    
    async def stop(self):
        """Stop WebSocket connection"""
        if self.ws_client:
            await self.ws_client.disconnect()
            logger.info("WebSocket disconnected")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        avg_time = self.total_calculation_time / self.calculation_count if self.calculation_count > 0 else 0
        return {
            'total_calculations': self.calculation_count,
            'average_time_ms': avg_time,
            'active_symbols': len(self.active_symbols),
            'total_institutional_periods': sum(
                len(m['institutional_periods']) 
                for m in self.session_metrics.values()
            )
        }
    
    def format_for_dashboard(self, signal: TradeSizeSignal) -> Dict:
        """Format signal for dashboard display"""
        # Color coding
        if signal.bull_score == 2:
            color = 'bright_green'
        elif signal.bull_score == 1:
            color = 'green'
        elif signal.bear_score == 2:
            color = 'bright_red'
        elif signal.bear_score == 1:
            color = 'red'
        else:
            color = 'white'
        
        # Direction arrow
        if signal.components.large_direction_ratio > 0.3:
            arrow = '↑↑'
        elif signal.components.large_direction_ratio > 0:
            arrow = '↑'
        elif signal.components.large_direction_ratio < -0.3:
            arrow = '↓↓'
        elif signal.components.large_direction_ratio < 0:
            arrow = '↓'
        else:
            arrow = '→'
        
        return {
            'main_display': f"Size: {signal.components.volume_weighted_ratio:.1%} {arrow}",
            'color': color,
            'sub_components': {
                'Z-Score': f"{signal.components.zscore:.2f}",
                'Inst Score': f"{signal.components.institutional_score:.0%}",
                'Direction': f"{signal.components.large_direction_ratio:.2f}",
                'Avg Size': f"{signal.components.avg_trade_size:.0f}"
            },
            'tooltip': signal.reason,
            'alert': signal.signal_strength == 'EXCEPTIONAL'
        }


# ============= TEST SCRIPT =============

async def run_test():
    """Test trade size distribution calculation"""
    print("=== Testing Trade Size Distribution Analysis ===\n")
    
    # Test configuration
    TEST_SYMBOL = 'AAPL'
    TEST_DURATION = 60  # seconds
    
    calculator = TradeSizeDistribution(
        large_trade_threshold=1000,
        buffer_size=500,
        historical_lookback=20,
        zscore_threshold=2.0
    )
    
    print("📊 Trade Size Distribution Monitor")
    print(f"📈 Large Trade Threshold: {calculator.large_trade_threshold} shares")
    print(f"📊 Buffer Size: {calculator.buffer_size} trades")
    print(f"🎯 Z-Score Threshold: {calculator.zscore_threshold}")
    print()
    
    # Test 1: Synthetic data test
    print("Test 1: Processing synthetic trades...")
    print("-" * 50)
    
    # Simulate institutional accumulation pattern
    base_price = 150.0
    for i in range(200):
        # Mix of trade sizes
        if i % 20 == 0:  # Large institutional trade
            size = np.random.randint(5000, 10000)
            price = base_price + np.random.uniform(0, 0.05)  # Buying pressure
        elif i % 5 == 0:  # Medium trade
            size = np.random.randint(500, 2000)
            price = base_price + np.random.uniform(-0.02, 0.02)
        else:  # Small retail trade
            size = np.random.randint(10, 500)
            price = base_price + np.random.uniform(-0.03, 0.03)
        
        trade = Trade(
            symbol=TEST_SYMBOL,
            price=price,
            size=size,
            timestamp=datetime.now(timezone.utc) + timedelta(seconds=i)
        )
        
        signal = calculator.process_trade(trade)
        
        # Display significant signals
        if signal and i >= 100 and (signal.bull_score >= 2 or signal.bear_score >= 2):
            print(f"\n🚨 INSTITUTIONAL SIGNAL DETECTED!")
            print(f"   Time: {signal.timestamp.strftime('%H:%M:%S')}")
            print(f"   Type: {signal.signal_type}")
            print(f"   Strength: {signal.signal_strength}")
            print(f"   Bull/Bear: {signal.bull_score}/{signal.bear_score}")
            print(f"   Reason: {signal.reason}")
            print(f"   Components:")
            print(f"     • VW Ratio: {signal.components.volume_weighted_ratio:.1%}")
            print(f"     • Z-Score: {signal.components.zscore:.2f}")
            print(f"     • Direction: {signal.components.large_direction_ratio:.2f}")
            print(f"     • Inst Score: {signal.components.institutional_score:.0%}")
            print()
    
    final_signal = calculator.latest_signals.get(TEST_SYMBOL)
    if final_signal:
        print("\n📊 Final Analysis:")
        dashboard_format = calculator.format_for_dashboard(final_signal)
        print(f"   Display: {dashboard_format['main_display']}")
        print(f"   Sub-components:")
        for key, value in dashboard_format['sub_components'].items():
            print(f"     • {key}: {value}")
    
    # Test 2: Performance test
    print("\n\nTest 2: Performance benchmark...")
    print("-" * 50)
    
    start_time = time_module.perf_counter()
    for i in range(1000):
        trade = Trade(
            symbol=TEST_SYMBOL,
            price=150 + np.random.uniform(-1, 1),
            size=int(np.random.lognormal(5, 1.5)),
            timestamp=datetime.now(timezone.utc)
        )
        calculator.process_trade(trade)
    
    elapsed = (time_module.perf_counter() - start_time) * 1000
    avg_time = elapsed / 1000
    
    print(f"✓ Processed 1000 trades in {elapsed:.2f}ms")
    print(f"✓ Average time per trade: {avg_time:.3f}ms")
    print(f"✓ {'PASS' if avg_time < 0.1 else 'FAIL'} performance target (<0.1ms)")
    
    # Test 3: Backtesting
    print("\n\nTest 3: Backtesting functionality...")
    print("-" * 50)
    
    backtest_date = datetime.now(timezone.utc).replace(hour=10, minute=0, second=0, microsecond=0)
    
    try:
        signals = await calculator.backtest(
            symbol=TEST_SYMBOL,
            date=backtest_date,
            start_time=backtest_date,
            end_time=backtest_date + timedelta(hours=1)
        )
        print(f"✓ Backtest generated {len(signals)} signals")
        if signals:
            print(f"   First signal: {signals[0].signal_type}")
            print(f"   Last signal: {signals[-1].signal_type}")
    except Exception as e:
        print(f"⚠️  Backtest skipped (no data source): {e}")
    
    # Test 4: WebSocket test (if available)
    print("\n\nTest 4: WebSocket integration...")
    print("-" * 50)
    
    signal_count = 0
    
    def display_signal(signal: TradeSizeSignal):
        nonlocal signal_count
        signal_count += 1
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Signal #{signal_count}")
        print(f"Symbol: {signal.symbol}")
        print(f"Signal: {signal.signal_type} ({signal.signal_strength})")
        print(f"Bull/Bear: {signal.bull_score}/{signal.bear_score}")
        print(f"VW Ratio: {signal.components.volume_weighted_ratio:.1%}")
        print(f"Z-Score: {signal.components.zscore:.2f}")
    
    try:
        await calculator.start_websocket([TEST_SYMBOL], display_signal)
        print(f"✓ WebSocket connected, monitoring {TEST_SYMBOL}")
        print(f"⏰ Running for {TEST_DURATION} seconds...")
        
        await asyncio.sleep(TEST_DURATION)
        
        await calculator.stop()
        print(f"\n✓ WebSocket test complete: {signal_count} signals received")
        
    except Exception as e:
        print(f"⚠️  WebSocket test skipped: {e}")
    
    # Final summary
    stats = calculator.get_performance_stats()
    print("\n\n📊 FINAL SUMMARY")
    print("=" * 50)
    print(f"Total calculations: {stats['total_calculations']}")
    print(f"Average time: {stats['average_time_ms']:.3f}ms")
    print(f"Institutional periods detected: {stats['total_institutional_periods']}")
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    print("Trade Size Distribution Analysis Module")
    print("Detecting institutional activity through trade size patterns\n")
    
    asyncio.run(run_test())