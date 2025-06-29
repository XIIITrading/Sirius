# modules/calculations/indicators/ema_crossover.py
"""
Module: EMA Crossover Analysis
Purpose: Detect trend direction using 9/21 EMA crossover on 1-minute charts
Features: Real-time and historical data processing, hybrid EMA calculation
Output: BULL/BEAR/NEUTRAL signals based on EMA position and price location
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
    """1-minute candle data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    trades: Optional[int] = None


@dataclass
class EMAMetrics:
    """EMA calculation metrics"""
    ema_9: float
    ema_21: float
    ema_9_previous: float
    ema_21_previous: float
    price_vs_ema9: str  # 'above', 'below', 'at'
    ema_spread: float  # ema_9 - ema_21
    ema_spread_pct: float  # spread as % of price
    trend_strength: float  # 0-100 based on spread
    candles_processed: int
    last_crossover_time: Optional[datetime]
    last_crossover_type: Optional[str]  # 'bullish' or 'bearish'


@dataclass
class VolumeSignal:
    """Standard volume signal output"""
    symbol: str
    timestamp: datetime
    signal: str  # 'BULL', 'BEAR', 'NEUTRAL'
    strength: float  # 0-100
    metrics: Dict  # Detailed metrics
    reason: str  # Human-readable explanation


class EMAAnalyzer:
    """
    EMA crossover analyzer for 1-minute trend detection
    All timestamps are in UTC.
    """
    
    def __init__(self, 
                 buffer_size: int = 100,  # Number of candles to maintain
                 ema_short: int = 9,      # Short EMA period
                 ema_long: int = 21,      # Long EMA period
                 min_candles_required: int = 21):  # Minimum for valid signal
        """
        Initialize EMA analyzer
        
        Args:
            buffer_size: Number of recent candles to maintain
            ema_short: Short EMA period (default 9)
            ema_long: Long EMA period (default 21)
            min_candles_required: Minimum candles needed for analysis
        """
        self.buffer_size = buffer_size
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.min_candles_required = max(min_candles_required, ema_long)
        
        # Data storage
        self.candles: Dict[str, Deque[Candle]] = {}
        self.current_emas: Dict[str, Dict[str, float]] = {}
        self.last_crossover: Dict[str, Tuple[datetime, str]] = {}
        
        # For incomplete candle handling
        self.incomplete_candles: Dict[str, Dict] = {}
        
        # Performance tracking
        self.candles_processed = 0
        self.signals_generated = 0
        
        logger.info(f"EMA Analyzer initialized at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info(f"Settings: buffer={buffer_size}, EMA periods={ema_short}/{ema_long}, "
                   f"min_candles={min_candles_required}")
    
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
                      is_complete: bool = True) -> Optional[VolumeSignal]:
        """
        Process incoming candle and generate signal if conditions met
        
        Args:
            symbol: Ticker symbol
            candle_data: Candle data (dict with OHLCV)
            is_complete: Whether candle is complete (default True)
            
        Returns:
            VolumeSignal if generated, None otherwise
        """
        # Initialize buffers if needed
        if symbol not in self.candles:
            self.candles[symbol] = deque(maxlen=self.buffer_size)
            self.current_emas[symbol] = {}
            logger.info(f"Initialized buffers for {symbol}")
        
        # Handle incomplete candles (for real-time updates)
        if not is_complete:
            self.incomplete_candles[symbol] = candle_data
            logger.debug(f"{symbol}: Received incomplete candle")
            return None
        
        # Extract candle info and ensure UTC
        if 'timestamp' in candle_data:
            timestamp = datetime.fromtimestamp(candle_data['timestamp'] / 1000, tz=timezone.utc)
        else:
            timestamp = candle_data.get('t', datetime.now(timezone.utc))
            if isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
        
        timestamp = self._validate_timestamp(timestamp, f"Candle-{symbol}")
        
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
        
        logger.info(f"{symbol}: Processed candle at {timestamp.strftime('%H:%M:%S UTC')} "
                   f"Close: {candle.close:.2f}, Volume: {candle.volume:,.0f}")
        
        # Calculate EMAs
        self._calculate_emas(symbol)
        
        # Generate signal if enough data
        if len(self.candles[symbol]) >= self.min_candles_required:
            signal = self._generate_signal(symbol)
            if signal:
                logger.info(f"{symbol}: Generated {signal.signal} signal with strength {signal.strength:.0f}")
            return signal
        else:
            logger.debug(f"{symbol}: Need {self.min_candles_required - len(self.candles[symbol])} more candles")
            return None
    
    def process_historical_candles(self, symbol: str, candles: List[Dict]) -> Optional[VolumeSignal]:
        """
        Process a list of historical candles
        
        Args:
            symbol: Ticker symbol
            candles: List of candle dictionaries
            
        Returns:
            Final signal after processing all candles
        """
        logger.info(f"{symbol}: Processing {len(candles)} historical candles")
        
        final_signal = None
        for candle_data in candles:
            signal = self.process_candle(symbol, candle_data, is_complete=True)
            if signal:
                final_signal = signal
        
        return final_signal
    
    def _calculate_emas(self, symbol: str):
        """Calculate or update EMAs using hybrid approach"""
        candles_list = list(self.candles[symbol])
        
        if len(candles_list) < self.ema_long:
            return
        
        closes = [c.close for c in candles_list]
        
        # Check if we need full recalculation
        if symbol not in self.current_emas or not self.current_emas[symbol]:
            # Full calculation
            logger.debug(f"{symbol}: Performing full EMA calculation")
            ema_9 = self._calculate_ema_full(closes, self.ema_short)
            ema_21 = self._calculate_ema_full(closes, self.ema_long)
            
            self.current_emas[symbol] = {
                'ema_9': ema_9[-1],
                'ema_21': ema_21[-1],
                'ema_9_prev': ema_9[-2] if len(ema_9) > 1 else ema_9[-1],
                'ema_21_prev': ema_21[-2] if len(ema_21) > 1 else ema_21[-1]
            }
        else:
            # Incremental update
            last_close = closes[-1]
            
            # Store previous values
            self.current_emas[symbol]['ema_9_prev'] = self.current_emas[symbol]['ema_9']
            self.current_emas[symbol]['ema_21_prev'] = self.current_emas[symbol]['ema_21']
            
            # Update EMAs
            self.current_emas[symbol]['ema_9'] = self._update_ema(
                last_close, 
                self.current_emas[symbol]['ema_9'], 
                self.ema_short
            )
            self.current_emas[symbol]['ema_21'] = self._update_ema(
                last_close, 
                self.current_emas[symbol]['ema_21'], 
                self.ema_long
            )
            
            logger.debug(f"{symbol}: EMA Update - 9: {self.current_emas[symbol]['ema_9']:.2f}, "
                        f"21: {self.current_emas[symbol]['ema_21']:.2f}")
    
    def _calculate_ema_full(self, prices: List[float], period: int) -> List[float]:
        """Calculate EMA for entire price series"""
        if len(prices) < period:
            return []
        
        ema = []
        multiplier = 2 / (period + 1)
        
        # Start with SMA
        sma = sum(prices[:period]) / period
        ema.append(sma)
        
        # Calculate EMA for rest
        for i in range(period, len(prices)):
            ema_val = (prices[i] - ema[-1]) * multiplier + ema[-1]
            ema.append(ema_val)
        
        return ema
    
    def _update_ema(self, new_price: float, prev_ema: float, period: int) -> float:
        """Update EMA with new price"""
        multiplier = 2 / (period + 1)
        return (new_price - prev_ema) * multiplier + prev_ema
    
    def _generate_signal(self, symbol: str) -> Optional[VolumeSignal]:
        """Generate trading signal from current EMAs"""
        if symbol not in self.current_emas or not self.current_emas[symbol]:
            return None
        
        candles_list = list(self.candles[symbol])
        last_candle = candles_list[-1]
        
        # Get current values
        ema_9 = self.current_emas[symbol]['ema_9']
        ema_21 = self.current_emas[symbol]['ema_21']
        ema_9_prev = self.current_emas[symbol]['ema_9_prev']
        ema_21_prev = self.current_emas[symbol]['ema_21_prev']
        last_close = last_candle.close
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            symbol, ema_9, ema_21, ema_9_prev, ema_21_prev, 
            last_close, len(candles_list)
        )
        
        # Determine signal
        signal, strength, reason = self._determine_signal(
            ema_9, ema_21, last_close, metrics
        )
        
        # Check for crossover
        self._check_crossover(symbol, ema_9, ema_21, ema_9_prev, ema_21_prev)
        
        self.signals_generated += 1
        
        return VolumeSignal(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            signal=signal,
            strength=strength,
            metrics=metrics.__dict__,
            reason=reason
        )
    
    def _calculate_metrics(self, symbol: str, ema_9: float, ema_21: float,
                          ema_9_prev: float, ema_21_prev: float,
                          last_close: float, candles_count: int) -> EMAMetrics:
        """Calculate comprehensive EMA metrics"""
        # Price vs EMA9
        if last_close > ema_9:
            price_vs_ema9 = 'above'
        elif last_close < ema_9:
            price_vs_ema9 = 'below'
        else:
            price_vs_ema9 = 'at'
        
        # EMA spread
        ema_spread = ema_9 - ema_21
        ema_spread_pct = (ema_spread / last_close) * 100 if last_close > 0 else 0
        
        # Trend strength (0-100)
        # Based on spread magnitude relative to typical ranges
        trend_strength = min(100, abs(ema_spread_pct) * 20)  # 5% spread = 100 strength
        
        # Last crossover info
        last_crossover_time = None
        last_crossover_type = None
        if symbol in self.last_crossover:
            last_crossover_time, last_crossover_type = self.last_crossover[symbol]
        
        return EMAMetrics(
            ema_9=ema_9,
            ema_21=ema_21,
            ema_9_previous=ema_9_prev,
            ema_21_previous=ema_21_prev,
            price_vs_ema9=price_vs_ema9,
            ema_spread=ema_spread,
            ema_spread_pct=ema_spread_pct,
            trend_strength=trend_strength,
            candles_processed=candles_count,
            last_crossover_time=last_crossover_time,
            last_crossover_type=last_crossover_type
        )
    
    def _determine_signal(self, ema_9: float, ema_21: float, 
                         last_close: float, metrics: EMAMetrics) -> Tuple[str, float, str]:
        """
        Determine trading signal from EMAs and price position
        
        Returns:
            (signal, strength, reason)
        """
        reasons = []
        
        # Primary signal based on EMA position
        if ema_9 > ema_21:
            base_signal = 'BULL'
            reasons.append(f"9 EMA ({ema_9:.2f}) > 21 EMA ({ema_21:.2f})")
            
            # Check price position for NEUTRAL override
            if last_close < ema_9:
                signal = 'NEUTRAL'
                reasons.append(f"But price ({last_close:.2f}) below 9 EMA")
                strength = 30  # Low strength for neutral
            else:
                signal = 'BULL'
                strength = metrics.trend_strength
                
        else:  # ema_9 <= ema_21
            base_signal = 'BEAR'
            reasons.append(f"9 EMA ({ema_9:.2f}) < 21 EMA ({ema_21:.2f})")
            
            # Check price position for NEUTRAL override
            if last_close > ema_9:
                signal = 'NEUTRAL'
                reasons.append(f"But price ({last_close:.2f}) above 9 EMA")
                strength = 30
            else:
                signal = 'BEAR'
                strength = metrics.trend_strength
        
        # Add spread information
        reasons.append(f"Spread: {metrics.ema_spread:.2f} ({metrics.ema_spread_pct:.1f}%)")
        
        # Add crossover info if recent
        if metrics.last_crossover_time:
            time_since = datetime.now(timezone.utc) - metrics.last_crossover_time
            if time_since.total_seconds() < 300:  # Within 5 minutes
                reasons.append(f"Recent {metrics.last_crossover_type} crossover {time_since.seconds}s ago")
                strength = min(100, strength * 1.2)  # Boost strength for recent crossover
        
        reason = " | ".join(reasons)
        return signal, strength, reason
    
    def _check_crossover(self, symbol: str, ema_9: float, ema_21: float,
                        ema_9_prev: float, ema_21_prev: float):
        """Check and record EMA crossovers"""
        # Previous relationship
        prev_bull = ema_9_prev > ema_21_prev
        # Current relationship
        curr_bull = ema_9 > ema_21
        
        # Detect crossover
        if prev_bull != curr_bull:
            crossover_type = 'bullish' if curr_bull else 'bearish'
            self.last_crossover[symbol] = (datetime.now(timezone.utc), crossover_type)
            logger.info(f"{symbol}: {crossover_type.upper()} crossover detected!")
    
    def get_current_analysis(self, symbol: str) -> Optional[VolumeSignal]:
        """Get current analysis for a symbol without new data"""
        if symbol not in self.candles or len(self.candles[symbol]) < self.min_candles_required:
            return None
        
        return self._generate_signal(symbol)
    
    def get_statistics(self) -> Dict:
        """Get analyzer statistics"""
        return {
            'candles_processed': self.candles_processed,
            'signals_generated': self.signals_generated,
            'active_symbols': list(self.candles.keys()),
            'buffer_sizes': {symbol: len(candles) for symbol, candles in self.candles.items()},
            'current_emas': self.current_emas
        }


# ============= TEST FUNCTION =============
async def test_ema_analyzer():
    """Test EMA analyzer with real-time websocket data"""
    import sys
    import os
    
    # Add parent directories to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    modules_dir = os.path.dirname(os.path.dirname(current_dir))
    vega_root = os.path.dirname(modules_dir)
    if vega_root not in sys.path:
        sys.path.insert(0, vega_root)
    
    from polygon import PolygonWebSocketClient, PolygonRESTClient
    
    print("=== EMA CROSSOVER ANALYZER TEST ===")
    print("Analyzing 9/21 EMA crossover on 1-minute charts")
    print(f"Current UTC time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
    
    # Test configuration
    TEST_SYMBOLS = ['AAPL', 'TSLA', 'SPY']
    TEST_DURATION = 300  # 5 minutes
    USE_HISTORICAL = True  # Load historical data first
    
    # Create analyzer
    analyzer = EMAAnalyzer(
        buffer_size=100,
        ema_short=9,
        ema_long=21,
        min_candles_required=21
    )
    
    # Track signals
    signal_history = []
    
    # Load historical data if requested
    if USE_HISTORICAL:
        print("Loading historical 1-minute candles...")
        rest_client = PolygonRESTClient()
        
        for symbol in TEST_SYMBOLS:
            # Get last 100 1-minute candles
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=100)
            
            try:
                candles = rest_client.get_candles(
                    symbol=symbol,
                    timespan='minute',
                    multiplier=1,
                    from_=start_time,
                    to=end_time
                )
                
                if candles:
                    print(f"\n{symbol}: Loading {len(candles)} historical candles")
                    signal = analyzer.process_historical_candles(symbol, candles)
                    
                    if signal:
                        signal_history.append(signal)
                        print(f"  Initial Signal: {signal.signal} (Strength: {signal.strength:.0f}%)")
                        print(f"  Reason: {signal.reason}")
                        
            except Exception as e:
                print(f"Error loading historical data for {symbol}: {e}")
    
    # Handle real-time candles
    async def handle_candle(data: Dict):
        """Process incoming candle data"""
        symbol = data['symbol']
        
        # Check if this is a complete 1-minute candle
        # In real implementation, you'd track candle completion
        is_complete = data.get('complete', True)
        
        # Process candle
        signal = analyzer.process_candle(symbol, data, is_complete)
        
        if signal:
            signal_history.append(signal)
            
            # Display based on signal type
            if signal.signal == 'BULL':
                emoji = '🟢'
                color_code = '\033[92m'  # Green
            elif signal.signal == 'BEAR':
                emoji = '🔴'
                color_code = '\033[91m'  # Red
            else:
                emoji = '⚪'
                color_code = '\033[93m'  # Yellow
            
            print(f"\n{color_code}{'='*60}\033[0m")
            print(f"{emoji} {signal.symbol} - Signal: {signal.signal} (Strength: {signal.strength:.0f}%)")
            print(f"Time: {signal.timestamp.strftime('%H:%M:%S.%f UTC')[:-3]}")
            print(f"Reason: {signal.reason}")
            
            # Display key metrics
            m = signal.metrics
            print(f"\nMetrics:")
            print(f"  • 9 EMA: {m['ema_9']:.2f}")
            print(f"  • 21 EMA: {m['ema_21']:.2f}")
            print(f"  • EMA Spread: {m['ema_spread']:.2f} ({m['ema_spread_pct']:.2f}%)")
            print(f"  • Price vs 9 EMA: {m['price_vs_ema9']}")
            print(f"  • Trend Strength: {m['trend_strength']:.0f}")
            
            if m['last_crossover_time']:
                print(f"  • Last Crossover: {m['last_crossover_type']} at "
                      f"{m['last_crossover_time'].strftime('%H:%M:%S UTC')}")
    
    # Create WebSocket client
    ws_client = PolygonWebSocketClient()
    
    try:
        # Connect and authenticate
        print(f"\nConnecting to Polygon WebSocket...")
        await ws_client.connect()
        print("✓ Connected and authenticated")
        
        # Subscribe to 1-minute aggregates
        print(f"\nSubscribing to 1-minute candles for: {', '.join(TEST_SYMBOLS)}")
        await ws_client.subscribe(
            symbols=TEST_SYMBOLS,
            channels=['AM'],  # 1-minute aggregates
            callback=handle_candle
        )
        print("✓ Subscribed successfully")
        
        print(f"\n⏰ Running for {TEST_DURATION} seconds...")
        print("Waiting for 1-minute candles...\n")
        
        # Create listen task
        listen_task = asyncio.create_task(ws_client.listen())
        
        # Run for specified duration
        start_time = time.time()
        last_stats_time = start_time
        
        while time.time() - start_time < TEST_DURATION:
            await asyncio.sleep(1)
            
            # Print stats every 60 seconds
            if time.time() - last_stats_time >= 60:
                stats = analyzer.get_statistics()
                print(f"\n📊 Stats: {stats['candles_processed']} candles, "
                      f"{stats['signals_generated']} signals")
                
                # Show current EMAs
                if stats['current_emas']:
                    print("Current EMAs:")
                    for sym, emas in stats['current_emas'].items():
                        if emas:
                            print(f"  {sym}: 9 EMA={emas['ema_9']:.2f}, "
                                  f"21 EMA={emas['ema_21']:.2f}")
                
                print(f"Time: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
                last_stats_time = time.time()
            
            # Show countdown
            remaining = TEST_DURATION - (time.time() - start_time)
            print(f"\r⏳ Time remaining: {remaining:.0f}s ", end='', flush=True)
        
        print("\n\n🏁 Test complete!")
        
        # Final summary
        stats = analyzer.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"  • Total candles processed: {stats['candles_processed']}")
        print(f"  • Signals generated: {stats['signals_generated']}")
        print(f"  • Symbols tracked: {', '.join(stats['active_symbols'])}")
        print(f"  • End time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Signal summary
        if signal_history:
            print(f"\n📈 Signal Summary:")
            bull_signals = [s for s in signal_history if s.signal == 'BULL']
            bear_signals = [s for s in signal_history if s.signal == 'BEAR']
            neutral_signals = [s for s in signal_history if s.signal == 'NEUTRAL']
            
            print(f"  • Bullish: {len(bull_signals)} ({len(bull_signals)/len(signal_history)*100:.0f}%)")
            print(f"  • Bearish: {len(bear_signals)} ({len(bear_signals)/len(signal_history)*100:.0f}%)")
            print(f"  • Neutral: {len(neutral_signals)} ({len(neutral_signals)/len(signal_history)*100:.0f}%)")
            
            # Show current positions
            print(f"\n📍 Current Positions:")
            for symbol in TEST_SYMBOLS:
                current = analyzer.get_current_analysis(symbol)
                if current:
                    print(f"  • {symbol}: {current.signal} (Strength: {current.strength:.0f}%)")
        
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
    print(f"Starting EMA Analyzer at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("This module detects trend using 9/21 EMA crossover on 1-minute charts")
    print("All timestamps are in UTC\n")
    
    asyncio.run(test_ema_analyzer())