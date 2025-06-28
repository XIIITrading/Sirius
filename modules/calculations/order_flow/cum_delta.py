# modules/calculations/order-flow/cum_delta.py
"""
Module: Cumulative Delta Analysis
Purpose: Track buying vs selling pressure through bid/ask classification
Features: Multi-timeframe delta, efficiency metrics, divergence detection
Performance Target: <100 microseconds per trade
Time Handling: All timestamps in UTC
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Callable, Any, Deque
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

# Core Parameters
DELTA_TIMEFRAMES = {
    '1min': 60,     # seconds
    '5min': 300,
    '15min': 900,
    '30min': 1800
}

EFFICIENCY_LOOKBACK = 60  # seconds for price change
MIN_DELTA_FOR_EFFICIENCY = 1000  # shares, avoid division by small numbers
DIVERGENCE_THRESHOLD = 0.3  # 30% price vs delta mismatch


@dataclass
class Trade:
    """Individual trade data"""
    symbol: str
    price: float
    size: int
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    exchange: Optional[str] = None
    conditions: List[int] = field(default_factory=list)


@dataclass
class Quote:
    """Quote data for bid/ask"""
    symbol: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    timestamp: datetime


@dataclass
class DeltaComponents:
    """Detailed components of delta analysis"""
    delta: int
    method: str  # Classification method used
    confidence: float  # 0 to 1
    cumulative_delta: int
    delta_rate: float  # Delta per minute
    delta_volatility: float
    efficiency: float
    directional_efficiency: float
    absorption_score: float
    timeframe_deltas: Dict[str, int]
    divergences: List[Dict]


@dataclass
class DeltaSignal:
    """Complete delta signal output"""
    symbol: str
    timestamp: datetime
    current_price: float
    bull_score: int  # -2 to +2
    bear_score: int  # -2 to +2
    confidence: float  # 0 to 1
    components: DeltaComponents
    signal_type: str  # 'ACCUMULATION', 'DISTRIBUTION', 'ABSORPTION', etc.
    signal_strength: str  # 'EXCEPTIONAL', 'STRONG', 'MODERATE', 'WEAK'
    reason: str
    calculation_time_ms: float
    trade_count: int
    warnings: List[str] = field(default_factory=list)


class DeltaAggregator:
    """Multi-timeframe delta aggregation"""
    
    def __init__(self, timeframes: Dict[str, int] = None):
        self.timeframes = timeframes or DELTA_TIMEFRAMES
        self.delta_series = {tf: deque() for tf in timeframes}
        self.price_series = deque()
        self.cumulative_delta = 0
        self.session_start_delta = 0
        
    def add_trade(self, trade: Trade, delta_info: Dict):
        """Add trade to all timeframe buckets"""
        timestamp = trade.timestamp
        price = trade.price
        delta = delta_info['delta']
        
        # Update cumulative
        self.cumulative_delta += delta
        
        # Add to price series
        self.price_series.append({
            'timestamp': timestamp,
            'price': price
        })
        
        # Add to each timeframe
        for tf, seconds in self.timeframes.items():
            # Convert to nanoseconds for comparison
            timestamp_ns = int(timestamp.timestamp() * 1_000_000_000)
            
            # Create new bucket if needed
            if not self.delta_series[tf] or \
               timestamp_ns - self.delta_series[tf][-1]['timestamp'] >= seconds * 1_000_000_000:
                self.delta_series[tf].append({
                    'timestamp': timestamp_ns,
                    'delta': delta,
                    'volume': trade.size,
                    'trades': 1,
                    'high': price,
                    'low': price,
                    'close': price
                })
            else:
                # Update existing bucket
                bucket = self.delta_series[tf][-1]
                bucket['delta'] += delta
                bucket['volume'] += trade.size
                bucket['trades'] += 1
                bucket['high'] = max(bucket['high'], price)
                bucket['low'] = min(bucket['low'], price)
                bucket['close'] = price
        
        # Clean old data
        self._clean_old_data(timestamp_ns)
    
    def _clean_old_data(self, current_timestamp_ns: int):
        """Remove data outside timeframe windows"""
        # Clean price series (keep longest timeframe)
        max_seconds = max(self.timeframes.values())
        cutoff_ns = current_timestamp_ns - (max_seconds * 1_000_000_000)
        
        while self.price_series and int(self.price_series[0]['timestamp'].timestamp() * 1_000_000_000) < cutoff_ns:
            self.price_series.popleft()
        
        # Clean delta series
        for tf, seconds in self.timeframes.items():
            cutoff = current_timestamp_ns - (seconds * 1_000_000_000)
            while self.delta_series[tf] and self.delta_series[tf][0]['timestamp'] < cutoff:
                self.delta_series[tf].popleft()
    
    def get_timeframe_metrics(self, timeframe: str) -> Optional[Dict]:
        """Calculate metrics for specific timeframe"""
        if timeframe not in self.delta_series:
            return None
        
        buckets = list(self.delta_series[timeframe])
        if not buckets:
            return None
        
        # Sum delta over timeframe
        total_delta = sum(b['delta'] for b in buckets)
        total_volume = sum(b['volume'] for b in buckets)
        
        # Calculate delta rate (delta per minute)
        time_span = (buckets[-1]['timestamp'] - buckets[0]['timestamp']) / 1_000_000_000
        if time_span > 0:
            delta_rate = (total_delta / time_span) * 60
        else:
            delta_rate = 0
        
        # Delta volatility
        if len(buckets) > 1:
            bucket_deltas = [b['delta'] for b in buckets]
            delta_mean = np.mean(np.abs(bucket_deltas))
            if delta_mean > 0:
                delta_volatility = np.std(bucket_deltas) / delta_mean
            else:
                delta_volatility = 0
        else:
            delta_volatility = 0
        
        return {
            'total_delta': total_delta,
            'total_volume': total_volume,
            'delta_rate': delta_rate,
            'delta_volatility': delta_volatility,
            'bucket_count': len(buckets)
        }


class DeltaFlowAnalyzer:
    """
    Cumulative Delta analyzer for tracking buying vs selling pressure.
    Uses bid/ask classification to determine trade direction.
    """
    
    def __init__(self,
                 buffer_size: int = 1000,
                 timeframes: Dict[str, int] = None,
                 efficiency_lookback: int = 60,
                 min_delta_for_efficiency: int = 1000,
                 divergence_threshold: float = 0.3):
        """
        Initialize delta flow analyzer.
        
        Args:
            buffer_size: Number of trades to keep in buffer
            timeframes: Dictionary of timeframe names to seconds
            efficiency_lookback: Seconds to look back for efficiency calc
            min_delta_for_efficiency: Minimum delta to calculate efficiency
            divergence_threshold: Threshold for price/delta divergence
        """
        self.buffer_size = buffer_size
        self.timeframes = timeframes or DELTA_TIMEFRAMES
        self.efficiency_lookback = efficiency_lookback
        self.min_delta_for_efficiency = min_delta_for_efficiency
        self.divergence_threshold = divergence_threshold
        
        # Trade and quote buffers per symbol
        self.trade_buffers: Dict[str, deque] = {}
        self.quote_buffers: Dict[str, deque] = {}
        self.previous_trades: Dict[str, deque] = {}
        
        # Delta aggregators per symbol
        self.delta_aggregators: Dict[str, DeltaAggregator] = {}
        
        # Latest signals
        self.latest_signals: Dict[str, DeltaSignal] = {}
        
        # WebSocket integration
        self.ws_client = None
        self.active_symbols: Dict[str, List[Callable]] = {}
        
        # Performance tracking
        self.calculation_count = 0
        self.total_calculation_time = 0
        
        logger.info(f"Initialized DeltaFlowAnalyzer with {len(self.timeframes)} timeframes")
    
    def initialize_symbol(self, symbol: str):
        """Initialize buffers for a new symbol"""
        self.trade_buffers[symbol] = deque(maxlen=self.buffer_size)
        self.quote_buffers[symbol] = deque(maxlen=100)
        self.previous_trades[symbol] = deque(maxlen=10)
        self.delta_aggregators[symbol] = DeltaAggregator(self.timeframes)
        logger.info(f"Initialized buffers for {symbol}")
    
    def update_quote(self, quote: Quote):
        """Update quote data for a symbol"""
        if quote.symbol not in self.quote_buffers:
            self.initialize_symbol(quote.symbol)
        self.quote_buffers[quote.symbol].append(quote)
    
    def process_trade(self, trade: Trade) -> Optional[DeltaSignal]:
        """
        Process a new trade and generate delta signal.
        
        Args:
            trade: Trade object with symbol, price, size, timestamp
            
        Returns:
            DeltaSignal if enough data, None otherwise
        """
        start_time = time_module.perf_counter()
        
        # Initialize if needed
        if trade.symbol not in self.trade_buffers:
            self.initialize_symbol(trade.symbol)
        
        # Add to buffers
        self.trade_buffers[trade.symbol].append(trade)
        
        # Calculate delta for this trade
        delta_info = self._calculate_trade_delta(trade)
        
        # Update aggregator
        self.delta_aggregators[trade.symbol].add_trade(trade, delta_info)
        
        # Update previous trades
        self.previous_trades[trade.symbol].append(trade)
        
        # Need minimum trades for analysis
        if len(self.trade_buffers[trade.symbol]) < 100:
            logger.debug(f"{trade.symbol}: Warming up "
                        f"({len(self.trade_buffers[trade.symbol])}/100)")
            return None
        
        # Calculate comprehensive metrics
        aggregator = self.delta_aggregators[trade.symbol]
        
        # Get efficiency metrics
        efficiency_metrics = self._calculate_delta_efficiency(aggregator)
        
        # Detect divergences
        divergences = self._detect_delta_divergences(aggregator)
        
        # Get timeframe deltas
        timeframe_deltas = {}
        for tf in self.timeframes:
            metrics = aggregator.get_timeframe_metrics(tf)
            if metrics:
                timeframe_deltas[tf] = metrics['total_delta']
        
        # Create components
        components = DeltaComponents(
            delta=delta_info['delta'],
            method=delta_info['method'],
            confidence=delta_info['confidence'],
            cumulative_delta=aggregator.cumulative_delta,
            delta_rate=efficiency_metrics.get('delta_rate', 0),
            delta_volatility=efficiency_metrics.get('delta_volatility', 0),
            efficiency=efficiency_metrics.get('efficiency', 0),
            directional_efficiency=efficiency_metrics.get('directional_efficiency', 0),
            absorption_score=efficiency_metrics.get('absorption_score', 0),
            timeframe_deltas=timeframe_deltas,
            divergences=divergences
        )
        
        # Generate signal
        signal = self._calculate_delta_score(trade, components)
        
        # Track performance
        calculation_time = (time_module.perf_counter() - start_time) * 1000
        signal.calculation_time_ms = calculation_time
        
        # Update tracking
        self.calculation_count += 1
        self.total_calculation_time += calculation_time
        self.latest_signals[trade.symbol] = signal
        
        return signal
    
    def _calculate_trade_delta(self, trade: Trade) -> Dict:
        """Calculate delta contribution of a single trade"""
        delta = 0
        confidence = 1.0
        method = 'unknown'
        
        # Method 1: Direct bid/ask from trade (if available)
        if trade.bid and trade.ask:
            mid_price = (trade.bid + trade.ask) / 2
            
            if trade.price >= trade.ask:
                delta = trade.size
                method = 'ask_lift'
                confidence = 1.0
            elif trade.price > mid_price + (trade.ask - mid_price) * 0.5:
                delta = trade.size * 0.8
                method = 'near_ask'
                confidence = 0.8
            elif trade.price <= trade.bid:
                delta = -trade.size
                method = 'bid_hit'
                confidence = 1.0
            elif trade.price < mid_price - (mid_price - trade.bid) * 0.5:
                delta = -trade.size * 0.8
                method = 'near_bid'
                confidence = 0.8
            else:
                delta = 0
                method = 'midpoint'
                confidence = 0.5
        
        # Method 2: Recent quote (if available)
        elif trade.symbol in self.quote_buffers and self.quote_buffers[trade.symbol]:
            # Find closest quote within 100ms
            best_quote = None
            min_time_diff = float('inf')
            
            for quote in reversed(self.quote_buffers[trade.symbol]):
                time_diff = abs((trade.timestamp - quote.timestamp).total_seconds())
                if time_diff < 0.1 and time_diff < min_time_diff:  # 100ms
                    best_quote = quote
                    min_time_diff = time_diff
            
            if best_quote:
                mid_price = (best_quote.bid + best_quote.ask) / 2
                
                if trade.price >= best_quote.ask:
                    delta = trade.size
                    method = 'ask_lift'
                    confidence = 0.9  # Slightly lower due to time gap
                elif trade.price > mid_price:
                    delta = trade.size * 0.7
                    method = 'above_mid'
                    confidence = 0.7
                elif trade.price <= best_quote.bid:
                    delta = -trade.size
                    method = 'bid_hit'
                    confidence = 0.9
                elif trade.price < mid_price:
                    delta = -trade.size * 0.7
                    method = 'below_mid'
                    confidence = 0.7
                else:
                    delta = 0
                    method = 'midpoint'
                    confidence = 0.5
        
        # Method 3: Tick Rule (fallback)
        if method == 'unknown' and trade.symbol in self.previous_trades and self.previous_trades[trade.symbol]:
            prev_trades = list(self.previous_trades[trade.symbol])
            if prev_trades:
                prev_price = prev_trades[-1].price
                
                if trade.price > prev_price:
                    delta = trade.size
                    method = 'uptick'
                    confidence = 0.7
                elif trade.price < prev_price:
                    delta = -trade.size
                    method = 'downtick'
                    confidence = 0.7
                else:
                    # Same price - use size rule
                    avg_size = np.mean([t.size for t in prev_trades])
                    if trade.size > avg_size * 2:
                        delta = trade.size * 0.5
                        method = 'size_rule'
                        confidence = 0.5
        
        return {
            'delta': delta,
            'method': method,
            'confidence': confidence,
            'size': trade.size
        }
    
    def _calculate_delta_efficiency(self, aggregator: DeltaAggregator) -> Dict:
        """Measure how efficiently delta moves price"""
        if not aggregator.price_series or len(aggregator.price_series) < 2:
            return {
                'efficiency': 0,
                'directional_efficiency': 0.5,
                'absorption_score': 0,
                'delta_rate': 0,
                'delta_volatility': 0
            }
        
        current_time = aggregator.price_series[-1]['timestamp']
        lookback_seconds = self.efficiency_lookback
        
        # Find price change over lookback
        price_start = None
        price_end = aggregator.price_series[-1]['price']
        
        cutoff_time = current_time - timedelta(seconds=lookback_seconds)
        
        for price_point in aggregator.price_series:
            if price_point['timestamp'] >= cutoff_time:
                if price_start is None:
                    price_start = price_point['price']
                break
        
        if price_start is None:
            price_start = aggregator.price_series[0]['price']
        
        price_change = price_end - price_start
        price_change_pct = price_change / price_start if price_start != 0 else 0
        
        # Calculate delta over same period
        period_delta = 0
        for tf_data in aggregator.delta_series['1min']:
            if tf_data['timestamp'] >= int(cutoff_time.timestamp() * 1_000_000_000):
                period_delta += tf_data['delta']
        
        # Get rate and volatility from 1min timeframe
        metrics_1min = aggregator.get_timeframe_metrics('1min')
        delta_rate = metrics_1min['delta_rate'] if metrics_1min else 0
        delta_volatility = metrics_1min['delta_volatility'] if metrics_1min else 0
        
        # Efficiency metrics
        if abs(period_delta) > self.min_delta_for_efficiency:
            # Price change per 1000 shares of delta
            efficiency = (price_change / abs(period_delta)) * 1000
            
            # Directional efficiency
            if (period_delta > 0 and price_change > 0) or (period_delta < 0 and price_change < 0):
                directional_efficiency = 1.0
            elif period_delta == 0:
                directional_efficiency = 0.5
            else:
                directional_efficiency = 0.0
        else:
            efficiency = 0
            directional_efficiency = 0.5
        
        # Absorption score (high delta with low price movement)
        absorption_score = 1 - min(abs(efficiency), 1) if abs(efficiency) < 1 else 0
        
        return {
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'period_delta': period_delta,
            'efficiency': efficiency,
            'directional_efficiency': directional_efficiency,
            'absorption_score': absorption_score,
            'delta_rate': delta_rate,
            'delta_volatility': delta_volatility
        }
    
    def _detect_delta_divergences(self, aggregator: DeltaAggregator) -> List[Dict]:
        """Identify price/delta divergences across timeframes"""
        divergences = []
        
        if not aggregator.price_series or len(aggregator.price_series) < 2:
            return divergences
        
        current_price = aggregator.price_series[-1]['price']
        current_time = aggregator.price_series[-1]['timestamp']
        
        for timeframe in ['1min', '5min', '15min']:
            metrics = aggregator.get_timeframe_metrics(timeframe)
            if not metrics:
                continue
            
            # Get price change over timeframe
            tf_seconds = self.timeframes[timeframe]
            start_time = current_time - timedelta(seconds=tf_seconds)
            
            # Find price at start of timeframe
            start_price = None
            for price_point in aggregator.price_series:
                if price_point['timestamp'] >= start_time:
                    start_price = price_point['price']
                    break
            
            if not start_price:
                continue
            
            price_change_pct = (current_price - start_price) / start_price
            
            # Normalize delta to compare with price
            if metrics['total_volume'] > 0:
                normalized_delta = metrics['total_delta'] / metrics['total_volume']
            else:
                normalized_delta = 0
            
            # Check for divergence
            if abs(price_change_pct) > 0.001:  # Price moved at least 0.1%
                if normalized_delta > self.divergence_threshold and price_change_pct < -0.001:
                    divergences.append({
                        'type': 'bullish',
                        'timeframe': timeframe,
                        'strength': abs(normalized_delta),
                        'description': 'Positive delta with falling price'
                    })
                elif normalized_delta < -self.divergence_threshold and price_change_pct > 0.001:
                    divergences.append({
                        'type': 'bearish',
                        'timeframe': timeframe,
                        'strength': abs(normalized_delta),
                        'description': 'Negative delta with rising price'
                    })
        
        return divergences
    
    def _calculate_delta_score(self, trade: Trade, components: DeltaComponents) -> DeltaSignal:
        """Generate trading signal from delta analysis"""
        bull_score = 0
        bear_score = 0
        warnings = []
        
        # Analyze cumulative delta
        cum_delta = components.cumulative_delta
        efficiency = components.efficiency
        absorption = components.absorption_score
        divergences = components.divergences
        
        # Bull conditions
        if cum_delta > 5000 and components.directional_efficiency > 0.8:
            bull_score = 2
            signal_type = "STRONG ACCUMULATION"
            signal_strength = "EXCEPTIONAL"
            reason = f"Heavy buying pressure (Δ={cum_delta:+,})"
        
        elif cum_delta > 2000 and len([d for d in divergences if d['type'] == 'bullish']) > 0:
            bull_score = 2
            signal_type = "BULLISH DIVERGENCE"
            signal_strength = "STRONG"
            reason = "Positive delta despite price weakness"
        
        elif absorption > 0.7 and cum_delta > 0:
            bull_score = 1
            signal_type = "ABSORPTION"
            signal_strength = "MODERATE"
            reason = f"Absorbing selling (absorption={absorption:.0%})"
        
        elif cum_delta > 1000:
            bull_score = 1
            signal_type = "ACCUMULATION"
            signal_strength = "MODERATE"
            reason = "Steady buying pressure"
        
        # Bear conditions
        elif cum_delta < -5000 and components.directional_efficiency > 0.8:
            bear_score = 2
            signal_type = "STRONG DISTRIBUTION"
            signal_strength = "EXCEPTIONAL"
            reason = f"Heavy selling pressure (Δ={cum_delta:+,})"
        
        elif cum_delta < -2000 and len([d for d in divergences if d['type'] == 'bearish']) > 0:
            bear_score = 2
            signal_type = "BEARISH DIVERGENCE"
            signal_strength = "STRONG"
            reason = "Negative delta despite price strength"
        
        elif absorption > 0.7 and cum_delta < 0:
            bear_score = 1
            signal_type = "DISTRIBUTION"
            signal_strength = "MODERATE"
            reason = f"Absorbing buying (absorption={absorption:.0%})"
        
        elif cum_delta < -1000:
            bear_score = 1
            signal_type = "DISTRIBUTION"
            signal_strength = "MODERATE"
            reason = "Steady selling pressure"
        
        else:
            signal_type = "NEUTRAL"
            signal_strength = "WEAK"
            reason = "Balanced order flow"
        
        # Warnings
        if components.delta_volatility > 2:
            warnings.append("High delta volatility")
        
        if components.confidence < 0.7:
            warnings.append("Low classification confidence")
        
        # Calculate overall confidence
        confidence = min(components.confidence, 
                        1 - components.delta_volatility / 3,
                        components.directional_efficiency)
        
        return DeltaSignal(
            symbol=trade.symbol,
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
            trade_count=len(self.trade_buffers[trade.symbol]),
            warnings=warnings
        )
    
    # ============= BACKTESTING FUNCTIONALITY =============
    
    async def backtest(self, symbol: str, date: datetime,
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None) -> List[DeltaSignal]:
        """
        Backtest delta analysis for a specific date/time.
        
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
        
        # Note: This assumes you have a method to fetch trades with quotes
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
        
        # Simulate trades from bars
        signals = []
        for idx, row in df.iterrows():
            # Simulate bid/ask spread
            spread = row['high'] - row['low']
            typical_spread = max(0.01, spread * 0.1)  # 10% of bar range or 1 cent
            
            bid = row['close'] - typical_spread / 2
            ask = row['close'] + typical_spread / 2
            
            # Simulate multiple trades per bar
            avg_trades_per_min = 50
            for i in range(avg_trades_per_min):
                # Create synthetic trade
                price_var = np.random.uniform(-typical_spread/2, typical_spread/2)
                trade_price = row['close'] + price_var
                
                # Size distribution
                size = self._generate_trade_size(row['volume'] / avg_trades_per_min)
                
                trade_time = idx.to_pydatetime().replace(tzinfo=timezone.utc)
                
                trade = Trade(
                    symbol=symbol,
                    price=trade_price,
                    size=int(size),
                    timestamp=trade_time,
                    bid=bid,
                    ask=ask
                )
                
                signal = self.process_trade(trade)
                if signal and (signal.bull_score != 0 or signal.bear_score != 0):
                    signals.append(signal)
        
        logger.info(f"Backtest complete: {len(signals)} signals generated")
        return signals
    
    def _generate_trade_size(self, avg_size: float) -> int:
        """Generate realistic trade size distribution"""
        # Log-normal distribution for trade sizes
        return int(np.random.lognormal(np.log(avg_size), 0.8))
    
    # ============= WEBSOCKET FUNCTIONALITY =============
    
    async def start_websocket(self, symbols: List[str], callback: Optional[Callable] = None):
        """Start real-time monitoring with WebSocket"""
        try:
            from polygon import PolygonWebSocketClient
            
            logger.info("Connecting to WebSocket for trade and quote data...")
            self.ws_client = PolygonWebSocketClient()
            await self.ws_client.connect()
            
            # Subscribe to trades and quotes
            await self.ws_client.subscribe(
                symbols=symbols,
                channels=['T', 'Q'],  # Trades and Quotes
                callback=self._handle_websocket_message
            )
            
            for symbol in symbols:
                self.active_symbols[symbol] = [callback] if callback else []
                if symbol not in self.trade_buffers:
                    self.initialize_symbol(symbol)
            
            logger.info(f"✓ Started real-time delta monitoring for {symbols}")
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise
    
    async def _handle_websocket_message(self, data: Dict):
        """Handle incoming trade/quote data from WebSocket"""
        try:
            event_type = data.get('event_type')
            symbol = data.get('symbol')
            
            if symbol not in self.active_symbols:
                return
            
            if event_type == 'quote':
                # Update quote
                quote = Quote(
                    symbol=symbol,
                    bid=data['bid_price'],
                    ask=data['ask_price'],
                    bid_size=data['bid_size'],
                    ask_size=data['ask_size'],
                    timestamp=datetime.fromtimestamp(data['timestamp'] / 1000, tz=timezone.utc)
                )
                self.update_quote(quote)
                
            elif event_type == 'trade':
                # Get latest quote for this symbol
                latest_quote = None
                if symbol in self.quote_buffers and self.quote_buffers[symbol]:
                    latest_quote = self.quote_buffers[symbol][-1]
                
                # Create trade object
                trade = Trade(
                    symbol=symbol,
                    price=data['price'],
                    size=data['size'],
                    timestamp=datetime.fromtimestamp(data['timestamp'] / 1000, tz=timezone.utc),
                    bid=latest_quote.bid if latest_quote else None,
                    ask=latest_quote.ask if latest_quote else None,
                    exchange=data.get('exchange'),
                    conditions=data.get('conditions', [])
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
            logger.error(f"Error handling WebSocket data: {e}")
    
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
            'cumulative_deltas': {
                symbol: agg.cumulative_delta 
                for symbol, agg in self.delta_aggregators.items()
            }
        }
    
    def format_for_dashboard(self, signal: DeltaSignal) -> Dict:
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
        
        # Direction arrow based on cumulative delta
        cum_delta = signal.components.cumulative_delta
        if cum_delta > 5000:
            arrow = '↑↑↑'
        elif cum_delta > 1000:
            arrow = '↑↑'
        elif cum_delta > 0:
            arrow = '↑'
        elif cum_delta < -5000:
            arrow = '↓↓↓'
        elif cum_delta < -1000:
            arrow = '↓↓'
        elif cum_delta < 0:
            arrow = '↓'
        else:
            arrow = '→'
        
        return {
            'main_display': f"Δ: {cum_delta:+,} {arrow}",
            'color': color,
            'sub_components': {
                'Efficiency': f"{signal.components.efficiency:.2f}",
                'Absorption': f"{signal.components.absorption_score:.0%}",
                'Method': signal.components.method,
                'Divergences': len(signal.components.divergences)
            },
            'tooltip': signal.reason,
            'alert': signal.signal_strength == 'EXCEPTIONAL'
        }


# ============= TEST SCRIPT =============

async def run_test():
    """Test cumulative delta calculation"""
    print("=== Testing Cumulative Delta Analysis ===\n")
    
    # Test configuration
    TEST_SYMBOL = 'AAPL'
    TEST_DURATION = 120  # seconds
    
    analyzer = DeltaFlowAnalyzer(
        buffer_size=1000,
        timeframes=DELTA_TIMEFRAMES,
        efficiency_lookback=60,
        min_delta_for_efficiency=1000,
        divergence_threshold=0.3
    )
    
    print("📊 Cumulative Delta Monitor")
    print(f"📈 Timeframes: {', '.join(DELTA_TIMEFRAMES.keys())}")
    print(f"🎯 Efficiency Lookback: {analyzer.efficiency_lookback}s")
    print(f"📏 Divergence Threshold: {analyzer.divergence_threshold}")
    print()
    
    # Test 1: Synthetic data test
    print("Test 1: Processing synthetic trades with quotes...")
    print("-" * 50)
    
    # Initialize
    analyzer.initialize_symbol(TEST_SYMBOL)
    
    # Simulate market with bid/ask
    base_price = 150.0
    spread = 0.02  # 2 cent spread
    
    # Simulate accumulation pattern
    for i in range(200):
        # Update quote every 5 trades
        if i % 5 == 0:
            bid = base_price - spread/2 + np.random.uniform(-0.01, 0.01)
            ask = base_price + spread/2 + np.random.uniform(-0.01, 0.01)
            
            quote = Quote(
                symbol=TEST_SYMBOL,
                bid=bid,
                ask=ask,
                bid_size=np.random.randint(100, 1000),
                ask_size=np.random.randint(100, 1000),
                timestamp=datetime.now(timezone.utc)
            )
            analyzer.update_quote(quote)
        
        # Generate trade
        if i < 100:
            # First half: accumulation (more buys at ask)
            if np.random.random() < 0.7:  # 70% buy
                price = ask + np.random.uniform(0, 0.01)  # At or above ask
                size = np.random.randint(500, 2000)
            else:
                price = bid - np.random.uniform(0, 0.01)  # At or below bid
                size = np.random.randint(100, 500)
        else:
            # Second half: distribution (more sells at bid)
            if np.random.random() < 0.3:  # 30% buy
                price = ask + np.random.uniform(0, 0.01)
                size = np.random.randint(100, 500)
            else:
                price = bid - np.random.uniform(0, 0.01)
                size = np.random.randint(500, 2000)
        
        trade = Trade(
            symbol=TEST_SYMBOL,
            price=price,
            size=size,
            timestamp=datetime.now(timezone.utc) + timedelta(seconds=i*0.3),
            bid=bid,
            ask=ask
        )
        
        signal = analyzer.process_trade(trade)
        
        # Display significant signals
        if signal and i >= 100 and (signal.bull_score >= 2 or signal.bear_score >= 2):
            print(f"\n🚨 SIGNIFICANT DELTA SIGNAL!")
            print(f"   Time: {signal.timestamp.strftime('%H:%M:%S')}")
            print(f"   Type: {signal.signal_type}")
            print(f"   Strength: {signal.signal_strength}")
            print(f"   Bull/Bear: {signal.bull_score}/{signal.bear_score}")
            print(f"   Reason: {signal.reason}")
            print(f"   Components:")
            print(f"     • Cumulative Δ: {signal.components.cumulative_delta:+,}")
            print(f"     • Efficiency: {signal.components.efficiency:.2f}")
            print(f"     • Absorption: {signal.components.absorption_score:.0%}")
            print(f"     • Method: {signal.components.method}")
            print(f"     • Divergences: {len(signal.components.divergences)}")
            print()
        
        # Update base price with drift
        base_price += np.random.uniform(-0.02, 0.02)
    
    # Final analysis
    final_signal = analyzer.latest_signals.get(TEST_SYMBOL)
    if final_signal:
        print("\n📊 Final Delta Analysis:")
        dashboard_format = analyzer.format_for_dashboard(final_signal)
        print(f"   Display: {dashboard_format['main_display']}")
        print(f"   Timeframe Deltas:")
        for tf, delta in final_signal.components.timeframe_deltas.items():
            print(f"     • {tf}: {delta:+,}")
    
    # Test 2: Performance test
    print("\n\nTest 2: Performance benchmark...")
    print("-" * 50)
    
    start_time = time_module.perf_counter()
    for i in range(1000):
        bid = 150 + np.random.uniform(-0.5, 0.5)
        ask = bid + 0.02
        
        if i % 10 == 0:
            quote = Quote(
                symbol=TEST_SYMBOL,
                bid=bid,
                ask=ask,
                bid_size=100,
                ask_size=100,
                timestamp=datetime.now(timezone.utc)
            )
            analyzer.update_quote(quote)
        
        trade = Trade(
            symbol=TEST_SYMBOL,
            price=np.random.choice([bid, ask]),
            size=np.random.randint(100, 1000),
            timestamp=datetime.now(timezone.utc),
            bid=bid,
            ask=ask
        )
        analyzer.process_trade(trade)
    
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
        signals = await analyzer.backtest(
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
    
    # Test 4: WebSocket test
    print("\n\nTest 4: WebSocket integration...")
    print("-" * 50)
    
    signal_count = 0
    
    def display_signal(signal: DeltaSignal):
        nonlocal signal_count
        signal_count += 1
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Signal #{signal_count}")
        print(f"Symbol: {signal.symbol}")
        print(f"Signal: {signal.signal_type} ({signal.signal_strength})")
        print(f"Bull/Bear: {signal.bull_score}/{signal.bear_score}")
        print(f"Cumulative Δ: {signal.components.cumulative_delta:+,}")
        print(f"Efficiency: {signal.components.efficiency:.2f}")
    
    try:
        await analyzer.start_websocket([TEST_SYMBOL], display_signal)
        print(f"✓ WebSocket connected, monitoring {TEST_SYMBOL}")
        print(f"⏰ Running for {TEST_DURATION} seconds...")
        
        await asyncio.sleep(TEST_DURATION)
        
        await analyzer.stop()
        print(f"\n✓ WebSocket test complete: {signal_count} signals received")
        
    except Exception as e:
        print(f"⚠️  WebSocket test skipped: {e}")
    
    # Final summary
    stats = analyzer.get_performance_stats()
    print("\n\n📊 FINAL SUMMARY")
    print("=" * 50)
    print(f"Total calculations: {stats['total_calculations']}")
    print(f"Average time: {stats['average_time_ms']:.3f}ms")
    print(f"Final cumulative deltas:")
    for symbol, delta in stats['cumulative_deltas'].items():
        print(f"  • {symbol}: {delta:+,}")
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    print("Cumulative Delta Analysis Module")
    print("Tracking buying vs selling pressure through bid/ask classification\n")
    
    asyncio.run(run_test())