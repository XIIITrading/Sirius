# modules/calculations/order_flow/cum_delta.py
"""
Module: Cumulative Delta Analysis
Purpose: Track buying vs selling pressure through bid/ask classification
Features: Multi-timeframe delta, efficiency metrics, divergence detection, time series export
Performance: Optimized warmup mode for backtesting with 45-min lookback, full fidelity for live trading
Time Handling: All timestamps in UTC
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Callable, Any, Deque, Union
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

# Core Parameters - REMOVED 30min timeframe
DELTA_TIMEFRAMES = {
    '1min': 60,     # seconds
    '5min': 300,
    '15min': 900
}

EFFICIENCY_LOOKBACK = 60  # seconds for price change
MIN_DELTA_FOR_EFFICIENCY = 500  # shares, avoid division by small numbers (reduced)
DIVERGENCE_THRESHOLD = 0.3  # 30% price vs delta mismatch

# Session times (EST)
PRE_MARKET_START = (4, 0)   # 4:00 AM
MARKET_OPEN = (9, 30)       # 9:30 AM
MARKET_CLOSE = (16, 0)      # 4:00 PM
AFTER_HOURS_END = (20, 0)   # 8:00 PM

# Default warmup configuration - optimized for 45-minute lookback
DEFAULT_WARMUP_CONFIG = {
    'enabled': False,
    'max_warmup_trades': 10000,      # Maximum trades to process during warmup
    'sample_older_minutes': 10,       # Sample trades older than N minutes
    'sample_rate': 0.15,              # Keep 15% of older trades (more aggressive)
    'min_recent_minutes': 5,          # Always process full data for recent N minutes
    'use_aggregated_warmup': True,    # Use aggregated data for initial warmup
    'progress_interval': 250          # Update progress every N trades
}


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
class AggregatedBar:
    """Pre-aggregated minute bar with delta"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    delta: int
    buy_volume: int
    sell_volume: int
    neutral_volume: int


@dataclass
class DeltaTimeSeries:
    """Time series data point for visualization"""
    timestamp: datetime
    cumulative_delta: int
    period_delta: int
    period_volume: int
    vwap: float
    close_price: float
    efficiency: float
    absorption: float
    delta_rate: float
    session_type: str  # 'pre_market', 'regular', 'after_hours'


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
    time_series: List[DeltaTimeSeries] = field(default_factory=list)  # Recent history


class DeltaAggregator:
    """Multi-timeframe delta aggregation with time series export and warmup support"""
    
    def __init__(self, timeframes: Dict[str, int] = None):
        self.timeframes = timeframes or DELTA_TIMEFRAMES
        self.delta_series = {tf: deque() for tf in timeframes}
        self.price_series = deque()
        self.cumulative_delta = 0
        self.session_start_delta = 0
        self.session_start_time = None
        
        # Minute series cache for efficient export
        self.minute_cache = deque(maxlen=60)  # Reduced for 45-min lookback
        self.last_minute_timestamp = None
        
        # Warmup state
        self.is_warmed_up = False
        self.warmup_trades_processed = 0
        
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
            'price': price,
            'volume': trade.size
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
                    'open': price,
                    'close': price,
                    'vwap': price,
                    'total_value': price * trade.size
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
                bucket['total_value'] += price * trade.size
                bucket['vwap'] = bucket['total_value'] / bucket['volume'] if bucket['volume'] > 0 else price
        
        # Update minute cache for time series
        self._update_minute_cache(timestamp)
        
        # Clean old data
        self._clean_old_data(timestamp_ns)
    
    def add_aggregated_bar(self, bar: AggregatedBar):
        """Add pre-aggregated minute bar for fast warmup"""
        # Update cumulative delta
        self.cumulative_delta += bar.delta
        
        # Convert to format expected by minute cache
        minute_data = DeltaTimeSeries(
            timestamp=bar.timestamp,
            cumulative_delta=self.cumulative_delta,
            period_delta=bar.delta,
            period_volume=bar.volume,
            vwap=bar.vwap,
            close_price=bar.close,
            efficiency=self._calculate_bar_efficiency(bar),
            absorption=self._calculate_bar_absorption(bar),
            delta_rate=bar.delta,  # Delta per minute for 1-min bars
            session_type=self._get_session_type(bar.timestamp)
        )
        
        self.minute_cache.append(minute_data)
        
        # Also update price series for efficiency calculations
        self.price_series.append({
            'timestamp': bar.timestamp,
            'price': bar.close,
            'volume': bar.volume
        })
        
        # Mark as warmed up after sufficient data
        if len(self.minute_cache) >= 5:  # Reduced threshold
            self.is_warmed_up = True
    
    def _calculate_bar_efficiency(self, bar: AggregatedBar) -> float:
        """Calculate efficiency from aggregated bar"""
        if abs(bar.delta) < 100:
            return 0.0
        
        price_change = bar.close - bar.open
        if abs(bar.delta) > 0:
            efficiency = (price_change / abs(bar.delta)) * 1000
            # Directional check
            if (bar.delta > 0 and price_change > 0) or (bar.delta < 0 and price_change < 0):
                return min(abs(efficiency), 1.0)
            else:
                return -min(abs(efficiency), 1.0)
        return 0.0
    
    def _calculate_bar_absorption(self, bar: AggregatedBar) -> float:
        """Calculate absorption from aggregated bar"""
        if bar.volume == 0:
            return 0.0
        
        price_range = bar.high - bar.low
        avg_price = (bar.high + bar.low) / 2
        normalized_range = price_range / avg_price if avg_price > 0 else 0
        
        # High volume with low price range = high absorption
        volume_intensity = bar.volume / 1000  # Normalize
        if normalized_range > 0:
            absorption = min(volume_intensity / (normalized_range * 1000), 1.0)
        else:
            absorption = 1.0 if volume_intensity > 10 else 0.5
        
        return absorption
    
    def _update_minute_cache(self, current_timestamp: datetime):
        """Update the minute cache for time series export"""
        minute_timestamp = current_timestamp.replace(second=0, microsecond=0)
        
        # Check if we need to finalize the previous minute
        if self.last_minute_timestamp and minute_timestamp > self.last_minute_timestamp:
            # Get the completed minute data
            minute_data = self._get_minute_data(self.last_minute_timestamp)
            if minute_data:
                self.minute_cache.append(minute_data)
        
        self.last_minute_timestamp = minute_timestamp
    
    def _get_minute_data(self, minute_timestamp: datetime) -> Optional[DeltaTimeSeries]:
        """Get data for a specific minute"""
        # Find the minute bucket
        minute_bucket = None
        for bucket in self.delta_series['1min']:
            bucket_time = datetime.fromtimestamp(bucket['timestamp'] / 1_000_000_000, tz=timezone.utc)
            if bucket_time.replace(second=0, microsecond=0) == minute_timestamp:
                minute_bucket = bucket
                break
        
        if not minute_bucket:
            return None
        
        # Calculate efficiency for this minute
        efficiency = self._calculate_minute_efficiency(minute_bucket)
        absorption = self._calculate_minute_absorption(minute_bucket)
        
        # Determine session type
        session_type = self._get_session_type(minute_timestamp)
        
        return DeltaTimeSeries(
            timestamp=minute_timestamp,
            cumulative_delta=self.cumulative_delta,
            period_delta=minute_bucket['delta'],
            period_volume=minute_bucket['volume'],
            vwap=minute_bucket['vwap'],
            close_price=minute_bucket['close'],
            efficiency=efficiency,
            absorption=absorption,
            delta_rate=minute_bucket['delta'],  # Delta per minute
            session_type=session_type
        )
    
    def _calculate_minute_efficiency(self, bucket: Dict) -> float:
        """Calculate efficiency for a single minute"""
        if abs(bucket['delta']) < 50:  # Reduced threshold
            return 0.0
        
        price_change = bucket['close'] - bucket['open']
        if abs(bucket['delta']) > 0:
            efficiency = (price_change / abs(bucket['delta'])) * 1000
            # Directional check
            if (bucket['delta'] > 0 and price_change > 0) or (bucket['delta'] < 0 and price_change < 0):
                return min(abs(efficiency), 1.0)  # Cap at 1.0
            else:
                return -min(abs(efficiency), 1.0)  # Negative for wrong direction
        return 0.0
    
    def _calculate_minute_absorption(self, bucket: Dict) -> float:
        """Calculate absorption score for a single minute"""
        if bucket['volume'] == 0:
            return 0.0
        
        price_range = bucket['high'] - bucket['low']
        avg_price = (bucket['high'] + bucket['low']) / 2
        normalized_range = price_range / avg_price if avg_price > 0 else 0
        
        # High volume with low price range = high absorption
        volume_intensity = bucket['volume'] / 1000  # Normalize
        if normalized_range > 0:
            absorption = min(volume_intensity / (normalized_range * 1000), 1.0)
        else:
            absorption = 1.0 if volume_intensity > 10 else 0.5
        
        return absorption
    
    def _get_session_type(self, timestamp: datetime) -> str:
        """Determine market session type"""
        # Convert to EST for session determination
        hour = timestamp.hour
        minute = timestamp.minute
        time_minutes = hour * 60 + minute
        
        pre_market_start = PRE_MARKET_START[0] * 60 + PRE_MARKET_START[1]
        market_open = MARKET_OPEN[0] * 60 + MARKET_OPEN[1]
        market_close = MARKET_CLOSE[0] * 60 + MARKET_CLOSE[1]
        after_hours_end = AFTER_HOURS_END[0] * 60 + AFTER_HOURS_END[1]
        
        if time_minutes < pre_market_start or time_minutes >= after_hours_end:
            return 'closed'
        elif time_minutes < market_open:
            return 'pre_market'
        elif time_minutes < market_close:
            return 'regular'
        else:
            return 'after_hours'
    
    def _clean_old_data(self, current_timestamp_ns: int):
        """Remove data outside timeframe windows"""
        # Clean price series (keep only 15 minutes for efficiency)
        max_seconds = 900  # 15 minutes max
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
    
    def get_minute_series(self, lookback_minutes: int = 45) -> List[DeltaTimeSeries]:
        """Get minute-by-minute delta series for charting"""
        # First, ensure current minute is in cache
        if self.last_minute_timestamp:
            current_data = self._get_minute_data(self.last_minute_timestamp)
            if current_data and (not self.minute_cache or 
                                self.minute_cache[-1].timestamp != self.last_minute_timestamp):
                self.minute_cache.append(current_data)
        
        # Return requested lookback period
        if lookback_minutes >= len(self.minute_cache):
            return list(self.minute_cache)
        else:
            return list(self.minute_cache)[-lookback_minutes:]
    
    def reset_session(self, session_start: Optional[datetime] = None):
        """Reset cumulative delta for new session"""
        self.session_start_delta = self.cumulative_delta
        self.session_start_time = session_start or datetime.now(timezone.utc)
        logger.info(f"Session reset at {self.session_start_time}, delta: {self.cumulative_delta}")
    
    def get_session_delta(self) -> int:
        """Get delta since session start"""
        return self.cumulative_delta - self.session_start_delta


class DeltaFlowAnalyzer:
    """
    Cumulative Delta analyzer for tracking buying vs selling pressure.
    Uses bid/ask classification to determine trade direction.
    Includes optimized warmup mode for backtesting with 45-minute lookback.
    """
    
    def __init__(self,
                 buffer_size: int = 500,  # Reduced from 1000
                 timeframes: Dict[str, int] = None,
                 efficiency_lookback: int = 60,
                 min_delta_for_efficiency: int = 500,  # Reduced from 1000
                 divergence_threshold: float = 0.3,
                 time_series_lookback: int = 30,  # Reduced from 60
                 warmup_config: Optional[Dict[str, Any]] = None):
        """
        Initialize delta flow analyzer.
        
        Args:
            buffer_size: Number of trades to keep in buffer
            timeframes: Dictionary of timeframe names to seconds
            efficiency_lookback: Seconds to look back for efficiency calc
            min_delta_for_efficiency: Minimum delta to calculate efficiency
            divergence_threshold: Threshold for price/delta divergence
            time_series_lookback: Minutes of history to include in signals
            warmup_config: Configuration for warmup/backtest mode
        """
        self.buffer_size = buffer_size
        self.timeframes = timeframes or DELTA_TIMEFRAMES
        self.efficiency_lookback = efficiency_lookback
        self.min_delta_for_efficiency = min_delta_for_efficiency
        self.divergence_threshold = divergence_threshold
        self.time_series_lookback = time_series_lookback
        
        # Warmup configuration
        self.warmup_config = warmup_config or DEFAULT_WARMUP_CONFIG.copy()
        self.warmup_mode = False
        
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
        if self.warmup_config['enabled']:
            logger.info(f"Warmup mode enabled: max {self.warmup_config['max_warmup_trades']} trades")
    
    def initialize_symbol(self, symbol: str):
        """Initialize buffers for a new symbol"""
        self.trade_buffers[symbol] = deque(maxlen=self.buffer_size)
        self.quote_buffers[symbol] = deque(maxlen=50)  # Reduced from 100
        self.previous_trades[symbol] = deque(maxlen=10)
        self.delta_aggregators[symbol] = DeltaAggregator(self.timeframes)
        logger.info(f"Initialized buffers for {symbol}")
    
    def warmup_with_aggregated_bars(self, symbol: str, bars: pd.DataFrame,
                                   progress_callback: Optional[Callable[[int, str], None]] = None) -> bool:
        """
        Warm up analyzer with pre-aggregated minute bars for fast initialization.
        
        Args:
            symbol: Stock symbol
            bars: DataFrame with columns: delta, buy_volume, sell_volume, volume, open, high, low, close, vwap
            progress_callback: Optional progress reporting
            
        Returns:
            True if warmup successful
        """
        if symbol not in self.delta_aggregators:
            self.initialize_symbol(symbol)
        
        aggregator = self.delta_aggregators[symbol]
        total_bars = len(bars)
        
        if progress_callback:
            progress_callback(0, f"Processing {total_bars} aggregated bars...")
        
        for i, (timestamp, row) in enumerate(bars.iterrows()):
            # Create aggregated bar
            bar = AggregatedBar(
                timestamp=timestamp,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=int(row['volume']),
                vwap=row.get('vwap', row['close']),
                delta=int(row.get('delta', 0)),
                buy_volume=int(row.get('buy_volume', 0)),
                sell_volume=int(row.get('sell_volume', 0)),
                neutral_volume=int(row.get('neutral_volume', 0))
            )
            
            # Add to aggregator
            aggregator.add_aggregated_bar(bar)
            
            # Progress update
            if progress_callback and (i + 1) % 10 == 0:
                pct = int((i + 1) / total_bars * 100)
                progress_callback(pct, f"Processed {i + 1}/{total_bars} bars")
        
        aggregator.is_warmed_up = True
        logger.info(f"Warmed up {symbol} with {total_bars} aggregated bars")
        
        if progress_callback:
            progress_callback(100, "Warmup complete")
        
        return True
    
    def warmup_with_trades(self, symbol: str, 
                          trades_df: pd.DataFrame,
                          quotes_df: Optional[pd.DataFrame] = None,
                          entry_time: Optional[datetime] = None,
                          progress_callback: Optional[Callable[[int, str], None]] = None) -> Optional[DeltaSignal]:
        """
        Efficiently warm up analyzer with historical trades for backtesting.
        
        This method intelligently samples older trades while maintaining
        full fidelity for recent trades.
        
        Args:
            symbol: Stock symbol
            trades_df: DataFrame with trade data
            quotes_df: Optional DataFrame with quote data
            entry_time: Stop processing at this time
            progress_callback: Optional progress reporting
            
        Returns:
            Last signal generated or None
        """
        if symbol not in self.delta_aggregators:
            self.initialize_symbol(symbol)
        
        # Enable warmup mode
        self.warmup_mode = True
        last_signal = None
        
        try:
            # Apply intelligent sampling if configured
            if self.warmup_config['enabled'] and len(trades_df) > self.warmup_config['max_warmup_trades']:
                if progress_callback:
                    progress_callback(5, f"Sampling {len(trades_df)} trades for efficient warmup...")
                
                trades_df = self._sample_trades_for_warmup(trades_df, entry_time)
                
                if progress_callback:
                    progress_callback(10, f"Reduced to {len(trades_df)} trades")
            
            total_trades = len(trades_df)
            trades_processed = 0
            progress_interval = self.warmup_config.get('progress_interval', 250)
            
            # Process trades
            for timestamp, trade in trades_df.iterrows():
                # Stop at entry time if specified
                if entry_time and timestamp >= entry_time:
                    break
                
                # Update quotes if available
                if quotes_df is not None and not quotes_df.empty:
                    self._update_quotes_batch(symbol, timestamp, quotes_df)
                
                # Create Trade object
                trade_obj = Trade(
                    symbol=symbol,
                    price=float(trade['price']),
                    size=int(trade['size']),
                    timestamp=timestamp,
                    bid=trade.get('bid'),
                    ask=trade.get('ask')
                )
                
                # Process trade
                signal = self.process_trade(trade_obj)
                if signal:
                    last_signal = signal
                
                trades_processed += 1
                
                # Progress update
                if progress_callback and trades_processed % progress_interval == 0:
                    pct = int(trades_processed / total_trades * 100)
                    progress_callback(pct, f"Processed {trades_processed}/{total_trades} trades")
            
            # Mark as warmed up
            self.delta_aggregators[symbol].is_warmed_up = True
            
            if progress_callback:
                progress_callback(100, f"Warmup complete - processed {trades_processed} trades")
            
            logger.info(f"Warmup complete for {symbol}: {trades_processed} trades processed")
            
        finally:
            self.warmup_mode = False
        
        return last_signal
    
    def _sample_trades_for_warmup(self, trades_df: pd.DataFrame, 
                                 entry_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Intelligently sample trades to reduce processing time during warmup.
        Keep all recent trades, sample older ones.
        """
        if entry_time is None:
            entry_time = trades_df.index.max()
        
        recent_cutoff = entry_time - timedelta(minutes=self.warmup_config['min_recent_minutes'])
        
        # Split into recent and older trades
        recent_trades = trades_df[trades_df.index >= recent_cutoff]
        older_trades = trades_df[trades_df.index < recent_cutoff]
        
        logger.info(f"Warmup sampling: {len(recent_trades)} recent, {len(older_trades)} older trades")
        
        # More aggressive sampling for older trades
        if len(older_trades) > 2000:  # If more than 2000 old trades
            # Keep maximum 2000 older trades
            sample_size = min(2000, int(len(older_trades) * self.warmup_config['sample_rate']))
            sample_size = max(sample_size, 500)  # Keep at least 500 older trades
            
            if sample_size < len(older_trades):
                # Sample evenly across the time period
                older_trades = older_trades.iloc[::int(len(older_trades) / sample_size)][:sample_size]
                logger.info(f"Aggressively sampled older trades to {len(older_trades)}")
        
        # Combine and sort
        sampled_df = pd.concat([older_trades, recent_trades]).sort_index()
        
        return sampled_df
    
    def _update_quotes_batch(self, symbol: str, trade_time: datetime, quotes_df: pd.DataFrame):
        """Efficiently update quotes for a trade timestamp during warmup"""
        # Find quotes within 1 second of trade
        time_window = timedelta(seconds=1)
        mask = (quotes_df.index >= trade_time - time_window) & (quotes_df.index <= trade_time)
        recent_quotes = quotes_df[mask]
        
        if not recent_quotes.empty:
            # Just use the most recent quote
            latest_quote = recent_quotes.iloc[-1]
            quote_obj = Quote(
                symbol=symbol,
                bid=float(latest_quote['bid']),
                ask=float(latest_quote['ask']),
                bid_size=int(latest_quote.get('bid_size', 100)),
                ask_size=int(latest_quote.get('ask_size', 100)),
                timestamp=latest_quote.name
            )
            self.update_quote(quote_obj)
    
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
        
        # In warmup mode, be less strict about minimum trades
        min_trades = 50 if self.warmup_mode else 100
        
        # Need minimum trades for analysis
        if len(self.trade_buffers[trade.symbol]) < min_trades:
            if not self.warmup_mode:
                logger.debug(f"{trade.symbol}: Warming up "
                            f"({len(self.trade_buffers[trade.symbol])}/{min_trades})")
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
        
        # Get time series data
        time_series = aggregator.get_minute_series(self.time_series_lookback)
        
        # Generate signal
        signal = self._calculate_delta_score(trade, components)
        signal.time_series = time_series
        
        # Track performance
        calculation_time = (time_module.perf_counter() - start_time) * 1000
        signal.calculation_time_ms = calculation_time
        
        # Update tracking
        self.calculation_count += 1
        self.total_calculation_time += calculation_time
        self.latest_signals[trade.symbol] = signal
        
        return signal
    
    def force_complete_bar(self, symbol: str) -> Optional[DeltaSignal]:
        """Force completion of current bar and return signal"""
        if symbol not in self.delta_aggregators:
            return None
        
        aggregator = self.delta_aggregators[symbol]
        
        # Get the last trade processed
        if symbol not in self.trade_buffers or not self.trade_buffers[symbol]:
            return None
        
        last_trade = self.trade_buffers[symbol][-1]
        
        # Create a dummy trade to trigger signal generation
        return self.process_trade(last_trade)
    
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
        
        # Only check 1min, 5min, 15min (removed 30min)
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
    
    def get_delta_time_series(self, symbol: str, lookback_minutes: int = 45) -> List[DeltaTimeSeries]:
        """
        Get minute-by-minute cumulative delta for visualization.
        
        Args:
            symbol: Stock symbol
            lookback_minutes: Number of minutes to return
            
        Returns:
            List of DeltaTimeSeries objects for charting
        """
        if symbol not in self.delta_aggregators:
            return []
        
        return self.delta_aggregators[symbol].get_minute_series(lookback_minutes)
    
    def reset_session(self, symbol: Optional[str] = None, session_start: Optional[datetime] = None):
        """
        Reset cumulative delta for new trading session.
        
        Args:
            symbol: Specific symbol to reset (None for all)
            session_start: Session start time (defaults to now)
        """
        if symbol:
            if symbol in self.delta_aggregators:
                self.delta_aggregators[symbol].reset_session(session_start)
        else:
            # Reset all symbols
            for agg in self.delta_aggregators.values():
                agg.reset_session(session_start)
    
    def get_session_delta(self, symbol: str) -> int:
        """Get delta since session start for a symbol"""
        if symbol in self.delta_aggregators:
            return self.delta_aggregators[symbol].get_session_delta()
        return 0
    
    # ============= BACKTESTING FUNCTIONALITY =============
    
    async def backtest(self, symbol: str, date: datetime,
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None,
                      return_time_series: bool = True,
                      use_warmup_optimization: bool = True) -> Dict[str, Any]:
        """
        Backtest delta analysis for a specific date/time with optimization.
        
        Args:
            symbol: Stock symbol
            date: Date to backtest
            start_time: Start time (optional)
            end_time: End time (optional)
            return_time_series: Include full time series data
            use_warmup_optimization: Use optimized warmup mode
            
        Returns:
            Dictionary with signals and optionally time series data
        """
        logger.info(f"Backtesting {symbol} for {date.date()}")
        
        # Enable warmup optimization
        if use_warmup_optimization:
            self.warmup_config['enabled'] = True
        
        # Import data fetcher
        try:
            from polygon import DataFetcher
            fetcher = DataFetcher()
        except ImportError:
            logger.error("DataFetcher not available for backtesting")
            return {'signals': [], 'time_series': []}
        
        # Set time bounds
        if start_time is None:
            start_time = date.replace(hour=4, minute=0, second=0, microsecond=0)
        if end_time is None:
            end_time = date.replace(hour=20, minute=0, second=0, microsecond=0)
        
        # Reset session at market open
        market_open_time = date.replace(hour=9, minute=30, second=0, microsecond=0)
        self.reset_session(symbol, market_open_time)
        
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
            return {'signals': [], 'time_series': []}
        
        # If using warmup optimization, first do aggregated warmup
        if use_warmup_optimization and len(df) > 30:  # Reduced from 60
            logger.info("Using aggregated warmup for historical data...")
            
            # Use first 70% of data for aggregated warmup
            warmup_cutoff = int(len(df) * 0.7)
            warmup_bars = df.iloc[:warmup_cutoff]
            
            # Simulate delta for warmup bars (in real implementation, this would come from data)
            warmup_bars['delta'] = np.where(
                warmup_bars['close'] > warmup_bars['open'],
                warmup_bars['volume'] * 0.6,  # 60% buy on up bars
                -warmup_bars['volume'] * 0.6   # 60% sell on down bars
            ).astype(int)
            
            # Warm up with aggregated data
            self.warmup_with_aggregated_bars(symbol, warmup_bars)
            
            # Process remaining bars as trades
            remaining_bars = df.iloc[warmup_cutoff:]
        else:
            remaining_bars = df
        
        # Simulate trades from remaining bars
        signals = []
        for idx, row in remaining_bars.iterrows():
            # Simulate bid/ask spread
            spread = row['high'] - row['low']
            typical_spread = max(0.01, spread * 0.1)  # 10% of bar range or 1 cent
            
            bid = row['close'] - typical_spread / 2
            ask = row['close'] + typical_spread / 2
            
            # Simulate multiple trades per bar (reduced for performance)
            avg_trades_per_min = 15 if use_warmup_optimization else 30  # Further reduced
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
        
        # Get final time series
        time_series = []
        if return_time_series:
            time_series = self.get_delta_time_series(symbol, lookback_minutes=45)
        
        logger.info(f"Backtest complete: {len(signals)} signals generated, "
                   f"{len(time_series)} minute bars")
        
        return {
            'signals': signals,
            'time_series': time_series,
            'final_cumulative_delta': self.delta_aggregators[symbol].cumulative_delta,
            'session_delta': self.get_session_delta(symbol)
        }
    
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
            },
            'warmup_config': self.warmup_config
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
    
    def export_time_series_to_df(self, symbol: str, lookback_minutes: int = 45) -> pd.DataFrame:
        """
        Export time series data to pandas DataFrame for analysis/plotting.
        
        Args:
            symbol: Stock symbol
            lookback_minutes: Minutes to export (default 45)
            
        Returns:
            DataFrame with columns for all delta metrics
        """
        time_series = self.get_delta_time_series(symbol, lookback_minutes)
        
        if not time_series:
            return pd.DataFrame()
        
        data = []
        for ts in time_series:
            data.append({
                'timestamp': ts.timestamp,
                'cumulative_delta': ts.cumulative_delta,
                'period_delta': ts.period_delta,
                'period_volume': ts.period_volume,
                'vwap': ts.vwap,
                'close_price': ts.close_price,
                'efficiency': ts.efficiency,
                'absorption': ts.absorption,
                'delta_rate': ts.delta_rate,
                'session_type': ts.session_type
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df


# ============= TEST SCRIPT =============

async def run_test():
    """Test cumulative delta calculation with 45-minute lookback optimization"""
    print("=== Testing Cumulative Delta Analysis with 45-minute Lookback ===\n")
    
    # Test configuration
    TEST_SYMBOL = 'AAPL'
    TEST_DURATION = 60  # seconds
    
    # Test with warmup optimization enabled
    warmup_config = DEFAULT_WARMUP_CONFIG.copy()
    warmup_config['enabled'] = True
    warmup_config['max_warmup_trades'] = 10000
    
    analyzer = DeltaFlowAnalyzer(
        buffer_size=500,
        timeframes={'1min': 60, '5min': 300, '15min': 900},  # No 30min
        efficiency_lookback=60,
        min_delta_for_efficiency=500,
        divergence_threshold=0.3,
        time_series_lookback=30,
        warmup_config=warmup_config
    )
    
    print("📊 Cumulative Delta Monitor - 45 Minute Lookback")
    print(f"📈 Timeframes: {', '.join(analyzer.timeframes.keys())}")
    print(f"🎯 Efficiency Lookback: {analyzer.efficiency_lookback}s")
    print(f"📏 Divergence Threshold: {analyzer.divergence_threshold}")
    print(f"⚡ Warmup Mode: Enabled (max {warmup_config['max_warmup_trades']} trades)")
    print(f"🔧 Buffer Size: {analyzer.buffer_size}")
    print()
    
    # Test 1: Aggregated Warmup
    print("Test 1: Aggregated warmup with minute bars...")
    print("-" * 50)
    
    # Initialize
    analyzer.initialize_symbol(TEST_SYMBOL)
    
    # Create synthetic aggregated bars for 45 minutes
    base_price = 150.0
    current_time = datetime.now(timezone.utc).replace(hour=10, minute=0, second=0, microsecond=0)
    
    bars_data = []
    for i in range(45):  # 45 minutes of aggregated data
        bar_time = current_time - timedelta(minutes=45-i)
        
        # Simulate price movement
        open_price = base_price + np.random.uniform(-0.5, 0.5)
        close_price = open_price + np.random.uniform(-0.2, 0.2)
        high_price = max(open_price, close_price) + np.random.uniform(0, 0.1)
        low_price = min(open_price, close_price) - np.random.uniform(0, 0.1)
        
        volume = np.random.randint(50000, 150000)
        
        # Simulate delta based on price movement
        if close_price > open_price:
            delta = int(volume * np.random.uniform(0.5, 0.7))
        else:
            delta = -int(volume * np.random.uniform(0.5, 0.7))
        
        bars_data.append({
            'timestamp': bar_time,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
            'vwap': (high_price + low_price + close_price) / 3,
            'delta': delta
        })
        
        base_price = close_price
    
    # Create DataFrame
    bars_df = pd.DataFrame(bars_data)
    bars_df.set_index('timestamp', inplace=True)
    
    # Warm up with aggregated data
    def progress_callback(pct, msg):
        print(f"  [{pct:3d}%] {msg}")
    
    success = analyzer.warmup_with_aggregated_bars(TEST_SYMBOL, bars_df, progress_callback)
    print(f"\n✓ Warmup successful: {success}")
    print(f"✓ Cumulative delta after warmup: {analyzer.delta_aggregators[TEST_SYMBOL].cumulative_delta:+,}")
    
    # Test 2: Process recent trades with aggressive sampling
    print("\n\nTest 2: Processing trades with aggressive sampling...")
    print("-" * 50)
    
    # Generate trades for testing
    trades_data = []
    for i in range(30000):  # 30k trades (reduced from 50k)
        trade_time = current_time + timedelta(seconds=i*0.05)  # ~25 minutes of trades
        
        # Add some bid/ask context
        bid = base_price - 0.01
        ask = base_price + 0.01
        
        # Random walk
        if np.random.random() < 0.5:
            price = ask
            size = np.random.randint(100, 800)
        else:
            price = bid
            size = np.random.randint(100, 800)
        
        trades_data.append({
            'timestamp': trade_time,
            'price': price,
            'size': size,
            'bid': bid,
            'ask': ask
        })
        
        base_price += np.random.uniform(-0.01, 0.01)
    
    trades_df = pd.DataFrame(trades_data)
    trades_df.set_index('timestamp', inplace=True)
    
    print(f"Generated {len(trades_df)} trades")
    
    # Process with warmup optimization
    start_time = time_module.perf_counter()
    
    last_signal = analyzer.warmup_with_trades(
        TEST_SYMBOL, 
        trades_df,
        entry_time=current_time + timedelta(minutes=20),
        progress_callback=progress_callback
    )
    
    elapsed = (time_module.perf_counter() - start_time) * 1000
    
    print(f"\n✓ Processed trades in {elapsed:.2f}ms")
    print(f"✓ Average time per trade: {elapsed/len(trades_df):.3f}ms")
    
    if last_signal:
        print(f"\n📊 Final Signal:")
        print(f"   Type: {last_signal.signal_type}")
        print(f"   Strength: {last_signal.signal_strength}")
        print(f"   Cumulative Δ: {last_signal.components.cumulative_delta:+,}")
        print(f"   Efficiency: {last_signal.components.efficiency:.2f}")
        print(f"   Timeframe Deltas:")
        for tf, delta in last_signal.components.timeframe_deltas.items():
            print(f"     • {tf}: {delta:+,}")
    
    # Test 3: Time series export
    print("\n\nTest 3: Time series export (45-minute window)...")
    print("-" * 50)
    
    time_series = analyzer.get_delta_time_series(TEST_SYMBOL, lookback_minutes=45)
    
    if time_series:
        print(f"✓ Exported {len(time_series)} minute bars")
        print("\nLast 10 minutes:")
        print("Time         | Cum Delta | Period Δ | Volume | Efficiency")
        print("-" * 60)
        for ts in time_series[-10:]:
            print(f"{ts.timestamp.strftime('%H:%M')} | "
                  f"{ts.cumulative_delta:>9,} | "
                  f"{ts.period_delta:>8,} | "
                  f"{ts.period_volume:>6,} | "
                  f"{ts.efficiency:>10.2f}")
    
    # Final summary
    stats = analyzer.get_performance_stats()
    print("\n\n📊 FINAL SUMMARY")
    print("=" * 50)
    print(f"Total calculations: {stats['total_calculations']}")
    print(f"Average time: {stats['average_time_ms']:.3f}ms")
    print(f"Warmup config: {stats['warmup_config']}")
    print(f"Final cumulative deltas:")
    for symbol, delta in stats['cumulative_deltas'].items():
        print(f"  • {symbol}: {delta:+,}")
    
    print("\n✅ All tests completed with 45-minute lookback optimization!")


if __name__ == "__main__":
    print("Cumulative Delta Analysis Module - 45 Minute Lookback")
    print("Optimized for fast backtesting with intelligent trade sampling\n")
    
    asyncio.run(run_test())