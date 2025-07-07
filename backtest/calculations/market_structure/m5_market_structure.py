# modules/calculations/market_structure/m5_market_structure.py
"""
Module: Fractal-Based Market Structure Analysis (5-Minute)
Purpose: Detect market structure changes using fractals on 5-minute charts
Features: Real-time and historical data processing, BOS/CHoCH detection
Output: BULL/BEAR signals based on market structure breaks
Time Handling: All timestamps in UTC
"""

import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Deque, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging
import os
import time as time_module

# Enforce UTC for all operations
os.environ['TZ'] = 'UTC'
if hasattr(time_module, 'tzset'):
    time_module.tzset()

# Configure logging with UTC
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s UTC - %(name)s - %(levelname)s - %(message)s'
)
logging.Formatter.converter = time_module.gmtime  # Force UTC in logs
logger = logging.getLogger(__name__)


@dataclass
class Candle:
    """5-minute candle data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    trades: Optional[int] = None


@dataclass
class Fractal:
    """Fractal pivot point"""
    timestamp: datetime
    bar_index: int
    price: float
    type: str  # 'high' or 'low'
    broken: bool = False
    break_time: Optional[datetime] = None
    break_type: Optional[str] = None  # 'BOS' or 'CHoCH'


@dataclass
class MarketStructureMetrics:
    """Market structure calculation metrics"""
    current_trend: str  # 'BULL' or 'BEAR'
    last_high_fractal: Optional[float]
    last_low_fractal: Optional[float]
    last_break_type: Optional[str]  # 'BOS' or 'CHoCH'
    last_break_time: Optional[datetime]
    last_break_price: Optional[float]
    fractal_count: int
    structure_breaks: int
    trend_changes: int
    candles_processed: int


@dataclass
class MarketStructureSignal:
    """Market structure signal output"""
    symbol: str
    timestamp: datetime
    signal: str  # 'BULL' or 'BEAR'
    structure_type: str  # 'BOS' or 'CHoCH'
    strength: float  # 0-100
    metrics: Dict  # Detailed metrics
    reason: str  # Human-readable explanation


class M5MarketStructureAnalyzer:
    """
    Fractal-based market structure analyzer for 5-minute trend detection.
    All timestamps are in UTC.
    """
    
    def __init__(self, 
                 fractal_length: int = 3,      # Fewer bars for 5-min timeframe
                 buffer_size: int = 100,       # 100 candles = 500 minutes
                 min_candles_required: int = 15,  # Minimum for valid signal
                 bars_needed: int = 100):      # Number of 5-min bars to request
        """
        Initialize M5 market structure analyzer
        
        Args:
            fractal_length: Number of bars on each side for fractal detection
            buffer_size: Number of recent candles to maintain
            min_candles_required: Minimum candles needed for analysis
            bars_needed: Number of 5-minute bars to request from data source
        """
        self.fractal_length = fractal_length
        self.buffer_size = buffer_size
        self.min_candles_required = max(min_candles_required, fractal_length * 2 + 1)
        self.bars_needed = max(bars_needed, self.min_candles_required * 2)
        self.timeframe = '5-minute'
        
        # Data storage
        self.candles: Dict[str, Deque[Candle]] = {}
        self.high_fractals: Dict[str, Deque[Fractal]] = {}
        self.low_fractals: Dict[str, Deque[Fractal]] = {}
        self.current_trend: Dict[str, str] = {}  # 'BULL' or 'BEAR'
        self.structure_breaks: Dict[str, List[Dict]] = {}
        
        # Performance tracking
        self.candles_processed = 0
        self.signals_generated = 0
        self.fractals_detected = 0
        
        logger.info(f"M5 Market Structure Analyzer initialized at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info(f"Settings: timeframe={self.timeframe}, fractal_length={fractal_length}, "
                   f"buffer={buffer_size}, min_candles={min_candles_required}, bars_needed={bars_needed}")
    
    def get_required_bars(self) -> int:
        """Get the number of 5-minute bars required for analysis"""
        return self.bars_needed
    
    def process_bars_dataframe(self, symbol: str, bars_df, 
                              up_to_time: Optional[datetime] = None) -> Optional[MarketStructureSignal]:
        """
        Process a DataFrame of 5-minute bars up to a specific time.
        
        Args:
            symbol: Trading symbol
            bars_df: DataFrame with OHLCV data indexed by timestamp
            up_to_time: Process bars up to this time (exclusive)
            
        Returns:
            MarketStructureSignal if generated, None otherwise
        """
        if bars_df.empty:
            logger.warning(f"Empty DataFrame provided for {symbol}")
            return None
        
        # Convert DataFrame to list of candle dictionaries
        candles = []
        for timestamp, row in bars_df.iterrows():
            # Stop at up_to_time if specified
            if up_to_time and timestamp >= up_to_time:
                break
                
            candle_dict = {
                'timestamp': timestamp,
                'o': float(row.get('open', row.get('o', 0))),
                'h': float(row.get('high', row.get('h', 0))),
                'l': float(row.get('low', row.get('l', 0))),
                'c': float(row.get('close', row.get('c', 0))),
                'v': float(row.get('volume', row.get('v', 0)))
            }
            candles.append(candle_dict)
        
        # Process all candles
        return self.process_historical_candles(symbol, candles)
    
    def _validate_timestamp(self, timestamp: datetime, source: str) -> datetime:
        """Validate and ensure timestamp is UTC"""
        if timestamp.tzinfo is None:
            logger.warning(f"{source}: Naive datetime received, assuming UTC")
            return timestamp.replace(tzinfo=timezone.utc)
        elif timestamp.tzinfo != timezone.utc:
            logger.warning(f"{source}: Non-UTC timezone {timestamp.tzinfo}, converting to UTC")
            return timestamp.astimezone(timezone.utc)
        return timestamp
    
    def process_candle(self, symbol: str, candle_data: Dict, 
                      is_complete: bool = True) -> Optional[MarketStructureSignal]:
        """
        Process incoming 5-minute candle and generate signal if conditions met
        
        Args:
            symbol: Ticker symbol
            candle_data: Candle data (dict with OHLCV)
            is_complete: Whether candle is complete (default True)
            
        Returns:
            MarketStructureSignal if generated, None otherwise
        """
        # Initialize buffers if needed
        if symbol not in self.candles:
            self.candles[symbol] = deque(maxlen=self.buffer_size)
            self.high_fractals[symbol] = deque(maxlen=30)  # Store fewer fractals for 5-min
            self.low_fractals[symbol] = deque(maxlen=30)
            self.current_trend[symbol] = 'NEUTRAL'
            self.structure_breaks[symbol] = []
            logger.info(f"Initialized 5-minute buffers for {symbol}")
        
        # Skip incomplete candles
        if not is_complete:
            logger.debug(f"{symbol}: Skipping incomplete 5-minute candle")
            return None
        
        # Extract candle info and ensure UTC
        if 'timestamp' in candle_data:
            if isinstance(candle_data['timestamp'], datetime):
                timestamp = candle_data['timestamp']
            else:
                timestamp = datetime.fromtimestamp(candle_data['timestamp'] / 1000, tz=timezone.utc)
        else:
            timestamp = candle_data.get('t', datetime.now(timezone.utc))
            if isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
        
        timestamp = self._validate_timestamp(timestamp, f"5min-Candle-{symbol}")
        
        # Create candle object
        candle = Candle(
            timestamp=timestamp,
            open=candle_data.get('o', candle_data.get('open')),
            high=candle_data.get('h', candle_data.get('high')),
            low=candle_data.get('l', candle_data.get('low')),
            close=candle_data.get('c', candle_data.get('close')),
            volume=candle_data.get('v', candle_data.get('volume')),
            vwap=candle_data.get('vw', candle_data.get('vwap')),
            trades=candle_data.get('n', candle_data.get('trades'))
        )
        
        # Add to buffer
        self.candles[symbol].append(candle)
        self.candles_processed += 1
        
        logger.debug(f"{symbol}: Processed 5-min candle at {timestamp.strftime('%H:%M:%S UTC')} "
                   f"Close: {candle.close:.2f}, Volume: {candle.volume:,.0f}")
        
        # Check for new fractals
        self._detect_fractals(symbol)
        
        # Check for structure breaks and generate signal
        if len(self.candles[symbol]) >= self.min_candles_required:
            signal = self._check_structure_breaks(symbol)
            if signal:
                logger.info(f"{symbol}: Generated {signal.signal} signal - {signal.structure_type} on 5-min")
            return signal
        else:
            logger.debug(f"{symbol}: Need {self.min_candles_required - len(self.candles[symbol])} more 5-min candles")
            return None
    
    def process_historical_candles(self, symbol: str, candles: List[Dict]) -> Optional[MarketStructureSignal]:
        """
        Process a list of historical 5-minute candles
        
        Args:
            symbol: Ticker symbol
            candles: List of candle dictionaries
            
        Returns:
            Final signal after processing all candles
        """
        logger.info(f"{symbol}: Processing {len(candles)} historical 5-minute candles")
        
        final_signal = None
        for candle_data in candles:
            signal = self.process_candle(symbol, candle_data, is_complete=True)
            if signal:
                final_signal = signal
        
        return final_signal
    
    def _detect_fractals(self, symbol: str):
        """Detect fractal patterns in the 5-minute candle buffer"""
        candles_list = list(self.candles[symbol])
        n = len(candles_list)
        
        # Need enough candles for fractal detection
        if n < self.fractal_length * 2 + 1:
            return
        
        # Check the middle candle (fractal_length bars from the end)
        check_index = n - self.fractal_length - 1
        
        if check_index < self.fractal_length:
            return
        
        middle_candle = candles_list[check_index]
        middle_high = middle_candle.high
        middle_low = middle_candle.low
        
        # Check for high fractal
        is_high_fractal = True
        for i in range(self.fractal_length):
            # Check left side
            if candles_list[check_index - i - 1].high >= middle_high:
                is_high_fractal = False
                break
            # Check right side
            if candles_list[check_index + i + 1].high >= middle_high:
                is_high_fractal = False
                break
        
        if is_high_fractal:
            # Check if we already have this fractal
            existing = any(f.bar_index == check_index and f.type == 'high' 
                          for f in self.high_fractals[symbol])
            
            if not existing:
                fractal = Fractal(
                    timestamp=middle_candle.timestamp,
                    bar_index=check_index,
                    price=middle_high,
                    type='high'
                )
                self.high_fractals[symbol].append(fractal)
                self.fractals_detected += 1
                logger.debug(f"{symbol}: 5-min high fractal detected at {middle_high:.2f} "
                           f"({middle_candle.timestamp.strftime('%H:%M:%S UTC')})")
        
        # Check for low fractal
        is_low_fractal = True
        for i in range(self.fractal_length):
            # Check left side
            if candles_list[check_index - i - 1].low <= middle_low:
                is_low_fractal = False
                break
            # Check right side
            if candles_list[check_index + i + 1].low <= middle_low:
                is_low_fractal = False
                break
        
        if is_low_fractal:
            # Check if we already have this fractal
            existing = any(f.bar_index == check_index and f.type == 'low' 
                          for f in self.low_fractals[symbol])
            
            if not existing:
                fractal = Fractal(
                    timestamp=middle_candle.timestamp,
                    bar_index=check_index,
                    price=middle_low,
                    type='low'
                )
                self.low_fractals[symbol].append(fractal)
                self.fractals_detected += 1
                logger.debug(f"{symbol}: 5-min low fractal detected at {middle_low:.2f} "
                           f"({middle_candle.timestamp.strftime('%H:%M:%S UTC')})")
    
    def _check_structure_breaks(self, symbol: str) -> Optional[MarketStructureSignal]:
        """Check for market structure breaks and generate signals"""
        if not self.high_fractals[symbol] or not self.low_fractals[symbol]:
            return None
        
        candles_list = list(self.candles[symbol])
        current_candle = candles_list[-1]
        current_close = current_candle.close
        
        # Get most recent unbroken fractals
        recent_high = None
        recent_low = None
        
        # Find most recent unbroken high fractal
        for fractal in reversed(self.high_fractals[symbol]):
            if not fractal.broken:
                recent_high = fractal
                break
        
        # Find most recent unbroken low fractal
        for fractal in reversed(self.low_fractals[symbol]):
            if not fractal.broken:
                recent_low = fractal
                break
        
        if not recent_high or not recent_low:
            return None
        
        # Current trend
        current_trend = self.current_trend[symbol]
        signal_generated = None
        
        # Check for bullish structure break (close above recent high)
        if current_close > recent_high.price and not recent_high.broken:
            recent_high.broken = True
            recent_high.break_time = datetime.now(timezone.utc)
            
            # Determine if BOS or CHoCH
            if current_trend == 'BEAR':
                structure_type = 'CHoCH'  # Change of Character - trend reversal
                self.current_trend[symbol] = 'BULL'
            else:
                structure_type = 'BOS'  # Break of Structure - trend continuation
                self.current_trend[symbol] = 'BULL'
            
            recent_high.break_type = structure_type
            
            # Record the break
            self.structure_breaks[symbol].append({
                'timestamp': recent_high.break_time,
                'type': structure_type,
                'direction': 'BULL',
                'fractal_price': recent_high.price,
                'break_price': current_close,
                'timeframe': '5-minute'
            })
            
            # Generate signal
            signal_generated = self._generate_signal(
                symbol, 'BULL', structure_type, recent_high, current_candle
            )
        
        # Check for bearish structure break (close below recent low)
        elif current_close < recent_low.price and not recent_low.broken:
            recent_low.broken = True
            recent_low.break_time = datetime.now(timezone.utc)
            
            # Determine if BOS or CHoCH
            if current_trend == 'BULL':
                structure_type = 'CHoCH'  # Change of Character - trend reversal
                self.current_trend[symbol] = 'BEAR'
            else:
                structure_type = 'BOS'  # Break of Structure - trend continuation
                self.current_trend[symbol] = 'BEAR'
            
            recent_low.break_type = structure_type
            
            # Record the break
            self.structure_breaks[symbol].append({
                'timestamp': recent_low.break_time,
                'type': structure_type,
                'direction': 'BEAR',
                'fractal_price': recent_low.price,
                'break_price': current_close,
                'timeframe': '5-minute'
            })
            
            # Generate signal
            signal_generated = self._generate_signal(
                symbol, 'BEAR', structure_type, recent_low, current_candle
            )
        
        return signal_generated
    
    def _generate_signal(self, symbol: str, direction: str, structure_type: str,
                        broken_fractal: Fractal, current_candle: Candle) -> MarketStructureSignal:
        """Generate market structure signal"""
        # Calculate strength based on break magnitude
        break_distance = abs(current_candle.close - broken_fractal.price)
        break_pct = (break_distance / broken_fractal.price) * 100
        
        # CHoCH signals are typically stronger (trend reversal)
        # 5-minute signals get slightly higher base strength than 1-minute
        base_strength = 75 if structure_type == 'CHoCH' else 55
        
        # Add strength based on break magnitude (up to 25 points)
        strength = min(100, base_strength + min(25, break_pct * 8))
        
        # Build metrics
        metrics = self._calculate_metrics(symbol)
        
        # Build reason
        fractal_type = 'high' if direction == 'BULL' else 'low'
        action = 'above' if direction == 'BULL' else 'below'
        
        reason_parts = [
            f"5-min {structure_type}: Close {action} {fractal_type} fractal",
            f"at {broken_fractal.price:.2f}",
            f"(break: {break_pct:.1f}%)"
        ]
        
        if structure_type == 'CHoCH':
            reason_parts.append("- Trend reversal on 5-min")
        else:
            reason_parts.append("- Trend continuation on 5-min")
        
        reason = " ".join(reason_parts)
        
        self.signals_generated += 1
        
        return MarketStructureSignal(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            signal=direction,
            structure_type=structure_type,
            strength=strength,
            metrics=metrics.__dict__,
            reason=reason
        )
    
    def _calculate_metrics(self, symbol: str) -> MarketStructureMetrics:
        """Calculate comprehensive market structure metrics"""
        # Get most recent fractals
        last_high = None
        last_low = None
        
        if self.high_fractals[symbol]:
            last_high = self.high_fractals[symbol][-1].price
        
        if self.low_fractals[symbol]:
            last_low = self.low_fractals[symbol][-1].price
        
        # Get last break info
        last_break_type = None
        last_break_time = None
        last_break_price = None
        
        if self.structure_breaks[symbol]:
            last_break = self.structure_breaks[symbol][-1]
            last_break_type = last_break['type']
            last_break_time = last_break['timestamp']
            last_break_price = last_break['break_price']
        
        # Count statistics
        structure_break_count = len(self.structure_breaks[symbol])
        trend_changes = sum(1 for b in self.structure_breaks[symbol] if b['type'] == 'CHoCH')
        
        return MarketStructureMetrics(
            current_trend=self.current_trend[symbol],
            last_high_fractal=last_high,
            last_low_fractal=last_low,
            last_break_type=last_break_type,
            last_break_time=last_break_time,
            last_break_price=last_break_price,
            fractal_count=len(self.high_fractals[symbol]) + len(self.low_fractals[symbol]),
            structure_breaks=structure_break_count,
            trend_changes=trend_changes,
            candles_processed=len(self.candles[symbol])
        )
    
    def get_current_analysis(self, symbol: str) -> Optional[MarketStructureSignal]:
        """Get current analysis for a symbol without new data"""
        if symbol not in self.candles or len(self.candles[symbol]) < self.min_candles_required:
            return None
        
        metrics = self._calculate_metrics(symbol)
        
        # Return current state as a signal
        return MarketStructureSignal(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            signal=self.current_trend[symbol],
            structure_type=metrics.last_break_type or 'NONE',
            strength=50,  # Neutral strength for status check
            metrics=metrics.__dict__,
            reason=f"Current 5-min trend: {self.current_trend[symbol]}"
        )
    
    def get_statistics(self) -> Dict:
        """Get analyzer statistics"""
        return {
            'timeframe': self.timeframe,
            'candles_processed': self.candles_processed,
            'signals_generated': self.signals_generated,
            'fractals_detected': self.fractals_detected,
            'active_symbols': list(self.candles.keys()),
            'buffer_sizes': {symbol: len(candles) for symbol, candles in self.candles.items()},
            'current_trends': self.current_trend,
            'fractal_counts': {
                symbol: {
                    'high': len(self.high_fractals.get(symbol, [])),
                    'low': len(self.low_fractals.get(symbol, []))
                }
                for symbol in self.candles.keys()
            }
        }