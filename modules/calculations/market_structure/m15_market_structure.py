# modules/calculations/market-structure/m15_market_structure.py
"""
Module: Fractal-Based Market Structure Analysis (15-Minute)
Purpose: Detect market structure changes using fractals on 15-minute charts
Features: Real-time and historical data processing, BOS/CHoCH detection
Output: BULL/BEAR signals based on market structure breaks
Time Handling: All timestamps in UTC
"""

import asyncio
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Deque, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging
import time
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

# UTC validation function
def ensure_utc(dt: datetime) -> datetime:
    """Ensure datetime is UTC-aware"""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    elif dt.tzinfo != timezone.utc:
        return dt.astimezone(timezone.utc)
    return dt


@dataclass
class Candle:
    """15-minute candle data"""
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


class M15MarketStructureAnalyzer:
    """
    Fractal-based market structure analyzer for 15-minute trend detection
    All timestamps are in UTC.
    """
    
    def __init__(self, 
                 fractal_length: int = 2,      # Even fewer bars for 15-min timeframe
                 buffer_size: int = 60,        # 60 candles = 900 minutes (15 hours)
                 min_candles_required: int = 10):  # Minimum for valid signal
        """
        Initialize M15 market structure analyzer
        
        Args:
            fractal_length: Number of bars on each side for fractal detection
            buffer_size: Number of recent candles to maintain
            min_candles_required: Minimum candles needed for analysis
        """
        self.fractal_length = fractal_length
        self.buffer_size = buffer_size
        self.min_candles_required = max(min_candles_required, fractal_length * 2 + 1)
        self.timeframe = '15-minute'
        
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
        
        logger.info(f"M15 Market Structure Analyzer initialized at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info(f"Settings: timeframe={self.timeframe}, fractal_length={fractal_length}, "
                   f"buffer={buffer_size}, min_candles={min_candles_required}")
    
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
        Process incoming 15-minute candle and generate signal if conditions met
        
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
            self.high_fractals[symbol] = deque(maxlen=20)  # Store fewer fractals for 15-min
            self.low_fractals[symbol] = deque(maxlen=20)
            self.current_trend[symbol] = 'NEUTRAL'
            self.structure_breaks[symbol] = []
            logger.info(f"Initialized 15-minute buffers for {symbol}")
        
        # Skip incomplete candles
        if not is_complete:
            logger.debug(f"{symbol}: Skipping incomplete 15-minute candle")
            return None
        
        # Extract candle info and ensure UTC
        if 'timestamp' in candle_data:
            timestamp = datetime.fromtimestamp(candle_data['timestamp'] / 1000, tz=timezone.utc)
        else:
            timestamp = candle_data.get('t', datetime.now(timezone.utc))
            if isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
        
        timestamp = self._validate_timestamp(timestamp, f"15min-Candle-{symbol}")
        
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
        
        logger.debug(f"{symbol}: Processed 15-min candle at {timestamp.strftime('%H:%M:%S UTC')} "
                   f"Close: {candle.close:.2f}, Volume: {candle.volume:,.0f}")
        
        # Check for new fractals
        self._detect_fractals(symbol)
        
        # Check for structure breaks and generate signal
        if len(self.candles[symbol]) >= self.min_candles_required:
            signal = self._check_structure_breaks(symbol)
            if signal:
                logger.info(f"{symbol}: Generated {signal.signal} signal - {signal.structure_type} on 15-min")
            return signal
        else:
            logger.debug(f"{symbol}: Need {self.min_candles_required - len(self.candles[symbol])} more 15-min candles")
            return None
    
    def process_historical_candles(self, symbol: str, candles: List[Dict]) -> Optional[MarketStructureSignal]:
        """
        Process a list of historical 15-minute candles
        
        Args:
            symbol: Ticker symbol
            candles: List of candle dictionaries
            
        Returns:
            Final signal after processing all candles
        """
        logger.info(f"{symbol}: Processing {len(candles)} historical 15-minute candles")
        
        final_signal = None
        for candle_data in candles:
            signal = self.process_candle(symbol, candle_data, is_complete=True)
            if signal:
                final_signal = signal
        
        return final_signal
    
    def _detect_fractals(self, symbol: str):
        """Detect fractal patterns in the 15-minute candle buffer"""
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
                logger.debug(f"{symbol}: 15-min high fractal detected at {middle_high:.2f} "
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
                logger.debug(f"{symbol}: 15-min low fractal detected at {middle_low:.2f} "
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
                'timeframe': '15-minute'
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
                'timeframe': '15-minute'
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
        # 15-minute signals get highest base strength
        base_strength = 80 if structure_type == 'CHoCH' else 60
        
        # Add strength based on break magnitude (up to 20 points)
        strength = min(100, base_strength + min(20, break_pct * 6))
        
        # Build metrics
        metrics = self._calculate_metrics(symbol)
        
        # Build reason
        fractal_type = 'high' if direction == 'BULL' else 'low'
        action = 'above' if direction == 'BULL' else 'below'
        
        reason_parts = [
            f"15-min {structure_type}: Close {action} {fractal_type} fractal",
            f"at {broken_fractal.price:.2f}",
            f"(break: {break_pct:.1f}%)"
        ]
        
        if structure_type == 'CHoCH':
            reason_parts.append("- Major trend reversal on 15-min")
        else:
            reason_parts.append("- Strong trend continuation on 15-min")
        
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
            reason=f"Current 15-min trend: {self.current_trend[symbol]}"
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


# ============= TEST FUNCTION =============
async def test_m15_market_structure_analyzer():
    """Test M15 market structure analyzer with real-time websocket data"""
    import sys
    import os
    
    # Add parent directories to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    market_structure_dir = os.path.dirname(current_dir)
    calculations_dir = os.path.dirname(market_structure_dir)
    modules_dir = os.path.dirname(calculations_dir)
    vega_root = os.path.dirname(modules_dir)
    if vega_root not in sys.path:
        sys.path.insert(0, vega_root)
    
    from polygon import PolygonWebSocketClient, PolygonClient
    
    print("=== M15 MARKET STRUCTURE ANALYZER TEST ===")
    print("Analyzing market structure using fractals on 15-minute charts")
    print(f"Current UTC time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
    
    # Test configuration
    TEST_SYMBOLS = ['AAPL', 'TSLA', 'SPY']
    TEST_DURATION = 1200  # 20 minutes to see at least one 15-min candle
    USE_HISTORICAL = True  # Load historical data first
    
    # Create analyzer with fractal length of 2 for 15-minute
    analyzer = M15MarketStructureAnalyzer(
        fractal_length=2,
        buffer_size=60,  # 60 15-min candles = 900 minutes (15 hours)
        min_candles_required=10
    )
    
    # Track signals
    signal_history = []
    
    # Load historical data if requested
    if USE_HISTORICAL:
        print("Loading historical 15-minute candles...")
        rest_client = PolygonClient()
        
        for symbol in TEST_SYMBOLS:
            # Get last 60 15-minute candles (900 minutes of data)
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=900)
            
            try:
                candles = rest_client.get_candles(
                    symbol=symbol,
                    timespan='minute',
                    multiplier=15,  # 15-minute candles
                    from_=start_time,
                    to=end_time
                )
                
                if candles:
                    print(f"\n{symbol}: Loading {len(candles)} historical 15-minute candles")
                    signal = analyzer.process_historical_candles(symbol, candles)
                    
                    if signal:
                        signal_history.append(signal)
                        print(f"  Initial Signal: {signal.signal} - {signal.structure_type}")
                        print(f"  Reason: {signal.reason}")
                        
            except Exception as e:
                print(f"Error loading historical data for {symbol}: {e}")
    
    # Handle real-time candles
    async def handle_candle(data: Dict):
        """Process incoming 15-minute candle data"""
        symbol = data['symbol']
        
        # For 15-minute aggregates, check if this is a complete candle
        # You might need to track candle completion based on timestamps
        is_complete = data.get('complete', True)
        
        # Process candle
        signal = analyzer.process_candle(symbol, data, is_complete)
        
        if signal:
            signal_history.append(signal)
            
            # Display based on signal type
            if signal.signal == 'BULL':
                emoji = '📈'
                color_code = '\033[92m'  # Green
            else:
                emoji = '📉'
                color_code = '\033[91m'  # Red
            
            if signal.structure_type == 'CHoCH':
                structure_emoji = '🔄'  # Reversal
            else:
                structure_emoji = '➡️'   # Continuation
            
            print(f"\n{color_code}{'='*60}\033[0m")
            print(f"{emoji} {signal.symbol} - {signal.signal} Market Structure (15-MIN) {structure_emoji}")
            print(f"Type: {signal.structure_type} (Strength: {signal.strength:.0f}%)")
            print(f"Time: {signal.timestamp.strftime('%H:%M:%S.%f UTC')[:-3]}")
            print(f"Reason: {signal.reason}")
            
            # Display key metrics
            m = signal.metrics
            print(f"\nMetrics:")
            print(f"  • Current Trend: {m['current_trend']}")
            print(f"  • Last High Fractal: {m['last_high_fractal']:.2f}" if m['last_high_fractal'] else "  • Last High Fractal: None")
            print(f"  • Last Low Fractal: {m['last_low_fractal']:.2f}" if m['last_low_fractal'] else "  • Last Low Fractal: None")
            print(f"  • Total Fractals: {m['fractal_count']}")
            print(f"  • Structure Breaks: {m['structure_breaks']}")
            print(f"  • Trend Changes: {m['trend_changes']}")
    
    # Create WebSocket client
    ws_client = PolygonWebSocketClient()
    
    try:
        # Connect and authenticate
        print(f"\nConnecting to Polygon WebSocket...")
        await ws_client.connect()
        print("✓ Connected and authenticated")
        
        # Subscribe to 15-minute aggregates
        print(f"\nSubscribing to 15-minute candles for: {', '.join(TEST_SYMBOLS)}")
        
        # Subscribe to aggregates with 15-minute multiplier
        # Note: You may need to adjust this based on your WebSocket client implementation
        for symbol in TEST_SYMBOLS:
            await ws_client.subscribe_aggregates(
                symbol=symbol,
                timespan='minute',
                multiplier=15,
                callback=handle_candle
            )
        
        print("✓ Subscribed successfully to 15-minute data")
        
        print(f"\n⏰ Running for {TEST_DURATION} seconds...")
        print("Waiting for 15-minute market structure changes...\n")
        
        # Create listen task
        listen_task = asyncio.create_task(ws_client.listen())
        
        # Run for specified duration
        start_time = time.time()
        last_stats_time = start_time
        
        while time.time() - start_time < TEST_DURATION:
            await asyncio.sleep(1)
            
            # Print stats every 300 seconds (5 minutes)
            if time.time() - last_stats_time >= 300:
                stats = analyzer.get_statistics()
                print(f"\n📊 15-Min Stats: {stats['candles_processed']} candles, "
                      f"{stats['fractals_detected']} fractals, "
                      f"{stats['signals_generated']} signals")
                
                # Show current trends
                if stats['current_trends']:
                    print("Current 15-Min Trends:")
                    for sym, trend in stats['current_trends'].items():
                        fractal_info = stats['fractal_counts'][sym]
                        print(f"  {sym}: {trend} (High fractals: {fractal_info['high']}, "
                              f"Low fractals: {fractal_info['low']})")
                
                print(f"Time: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
                last_stats_time = time.time()
            
            # Show countdown
            remaining = TEST_DURATION - (time.time() - start_time)
            print(f"\r⏳ Time remaining: {remaining:.0f}s ", end='', flush=True)
        
        print("\n\n🏁 Test complete!")
        
        # Final summary
        stats = analyzer.get_statistics()
        print(f"\n📊 Final 15-Minute Statistics:")
        print(f"  • Timeframe: {stats['timeframe']}")
        print(f"  • Total candles processed: {stats['candles_processed']}")
        print(f"  • Fractals detected: {stats['fractals_detected']}")
        print(f"  • Signals generated: {stats['signals_generated']}")
        print(f"  • Symbols tracked: {', '.join(stats['active_symbols'])}")
        print(f"  • End time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Signal summary
        if signal_history:
            print(f"\n📈 15-Minute Signal Summary:")
            bull_signals = [s for s in signal_history if s.signal == 'BULL']
            bear_signals = [s for s in signal_history if s.signal == 'BEAR']
            bos_signals = [s for s in signal_history if s.structure_type == 'BOS']
            choch_signals = [s for s in signal_history if s.structure_type == 'CHoCH']
            
            print(f"  • Bullish: {len(bull_signals)} ({len(bull_signals)/len(signal_history)*100:.0f}%)")
            print(f"  • Bearish: {len(bear_signals)} ({len(bear_signals)/len(signal_history)*100:.0f}%)")
            print(f"  • BOS (Continuation): {len(bos_signals)}")
            print(f"  • CHoCH (Reversal): {len(choch_signals)}")
            
            # Show current state
            print(f"\n📍 Current 15-Minute Market Structure:")
            for symbol in TEST_SYMBOLS:
                current = analyzer.get_current_analysis(symbol)
                if current:
                    print(f"  • {symbol}: {current.signal} trend")
                    if current.metrics['last_break_type']:
                        print(f"    Last break: {current.metrics['last_break_type']} at "
                              f"{current.metrics['last_break_price']:.2f}")
        
        # Cancel listen task
        listen_task.cancel()
        await ws_client.disconnect()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        await ws_client.disconnect()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        await ws_client.disconnect()


if __name__ == "__main__":
    print(f"Starting M15 Market Structure Analyzer at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("This module detects market structure changes using fractals on 15-minute charts")
    print("All timestamps are in UTC\n")
    
    asyncio.run(test_m15_market_structure_analyzer())