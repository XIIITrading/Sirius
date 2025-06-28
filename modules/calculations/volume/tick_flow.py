# modules/calculations/volume/tick_flow.py
"""
Module: Ultra-Fast Tick Flow Analysis
Purpose: Detect immediate buying/selling pressure from last 100-500 trades
Features: Real-time trade classification, large trade detection, momentum surges
Output: BULLISH/BEARISH/NEUTRAL signals based on order flow
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
class Trade:
    """Individual trade data"""
    timestamp: datetime
    price: float
    size: float
    is_buy: bool  # True if buy, False if sell
    is_large: bool  # True if size > threshold
    exchange: Optional[str] = None
    conditions: List[int] = field(default_factory=list)


@dataclass
class TickFlowMetrics:
    """Metrics calculated from tick flow"""
    total_trades: int
    buy_trades: int
    sell_trades: int
    buy_volume: float
    sell_volume: float
    buy_volume_pct: float
    large_buy_trades: int
    large_sell_trades: int
    avg_trade_size: float
    trade_rate: float  # trades per second
    momentum_score: float  # -100 to +100
    price_trend: str  # 'up', 'down', 'flat'
    

@dataclass
class VolumeSignal:
    """Standard volume signal output"""
    symbol: str
    timestamp: datetime
    signal: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    strength: float  # 0-100
    metrics: Dict  # Detailed metrics
    reason: str  # Human-readable explanation


class TickFlowAnalyzer:
    """
    Ultra-fast tick flow analyzer for immediate momentum detection
    All timestamps are in UTC.
    """
    
    def __init__(self, 
                 buffer_size: int = 200,  # Number of trades to analyze
                 large_trade_multiplier: float = 3.0,  # Multiple of avg size
                 momentum_threshold: float = 60.0,  # % for bull/bear signal
                 min_trades_required: int = 50):  # Minimum trades for valid signal
        """
        Initialize tick flow analyzer
        
        Args:
            buffer_size: Number of recent trades to maintain
            large_trade_multiplier: Multiplier for large trade detection
            momentum_threshold: Percentage threshold for signals
            min_trades_required: Minimum trades needed for analysis
        """
        self.buffer_size = buffer_size
        self.large_trade_multiplier = large_trade_multiplier
        self.momentum_threshold = momentum_threshold
        self.min_trades_required = min_trades_required
        
        # Data storage
        self.trades: Dict[str, Deque[Trade]] = {}
        self.last_prices: Dict[str, float] = {}
        self.avg_trade_sizes: Dict[str, float] = {}
        
        # For spread estimation (if quotes not available)
        self.bid_ask_spread: Dict[str, float] = {}
        self.spread_estimates: Dict[str, Deque[float]] = {}
        
        # Performance tracking
        self.trades_processed = 0
        self.signals_generated = 0
        
        logger.info(f"Tick Flow Analyzer initialized at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info(f"Settings: buffer={buffer_size}, large_multiplier={large_trade_multiplier}, "
                   f"momentum_threshold={momentum_threshold}%")
    
    def _validate_timestamp(self, timestamp: datetime, source: str) -> datetime:
        """Validate and ensure timestamp is UTC"""
        if timestamp.tzinfo is None:
            logger.warning(f"{source}: Naive datetime received, assuming UTC")
            return timestamp.replace(tzinfo=timezone.utc)
        elif timestamp.tzinfo != timezone.utc:
            logger.warning(f"{source}: Non-UTC timezone {timestamp.tzinfo}, converting to UTC")
            return timestamp.astimezone(timezone.utc)
        return timestamp
        
    def process_trade(self, symbol: str, trade_data: Dict) -> Optional[VolumeSignal]:
        """
        Process incoming trade and generate signal if conditions met
        
        Args:
            symbol: Ticker symbol
            trade_data: Trade data from websocket
            
        Returns:
            VolumeSignal if generated, None otherwise
        """
        # Initialize buffers if needed
        if symbol not in self.trades:
            self.trades[symbol] = deque(maxlen=self.buffer_size)
            self.spread_estimates[symbol] = deque(maxlen=100)
            self.avg_trade_sizes[symbol] = 0
            
        # Extract trade info and ensure UTC
        timestamp = datetime.fromtimestamp(trade_data['timestamp'] / 1000, tz=timezone.utc)
        timestamp = self._validate_timestamp(timestamp, f"Trade-{symbol}")
        
        price = trade_data['price']
        size = trade_data['size']
        conditions = trade_data.get('conditions', [])
        
        # Classify trade as buy or sell
        is_buy = self._classify_trade(symbol, price, size, conditions)
        
        # Update spread estimate
        self._update_spread_estimate(symbol, price)
        
        # Check if large trade
        is_large = self._is_large_trade(symbol, size)
        
        # Create trade object
        trade = Trade(
            timestamp=timestamp,
            price=price,
            size=size,
            is_buy=is_buy,
            is_large=is_large,
            exchange=trade_data.get('exchange'),
            conditions=conditions
        )
        
        # Add to buffer
        self.trades[symbol].append(trade)
        self.last_prices[symbol] = price
        self.trades_processed += 1
        
        # Update average trade size
        self._update_avg_trade_size(symbol)
        
        # Generate signal if enough data
        if len(self.trades[symbol]) >= self.min_trades_required:
            return self._generate_signal(symbol)
        
        return None
    
    def _classify_trade(self, symbol: str, price: float, size: float, 
                       conditions: List[int]) -> bool:
        """
        Classify trade as buy or sell using tick rule and other heuristics
        
        Returns:
            True if buy, False if sell
        """
        # If we have previous price, use tick rule
        if symbol in self.last_prices:
            last_price = self.last_prices[symbol]
            
            if price > last_price:
                return True  # Uptick = buy
            elif price < last_price:
                return False  # Downtick = sell
            else:
                # Price unchanged, look at conditions or use last classification
                # For now, use size as tiebreaker (larger = more likely institutional buy)
                avg_size = self.avg_trade_sizes.get(symbol, size)
                return size > avg_size
        
        # First trade - use conditions if available
        # Condition codes vary by exchange, but some common ones:
        # 12 = Intermarket sweep (aggressive)
        # 37 = Contingent trade
        if conditions:
            if 12 in conditions:  # Intermarket sweep often aggressive buying
                return True
                
        # Default to buy for first trade
        return True
    
    def _update_spread_estimate(self, symbol: str, price: float):
        """Estimate bid-ask spread from trade prices"""
        spreads = self.spread_estimates[symbol]
        
        if len(spreads) > 0:
            # Look for price changes as proxy for spread
            recent_prices = [t.price for t in list(self.trades[symbol])[-10:]]
            if recent_prices:
                price_changes = [abs(recent_prices[i] - recent_prices[i-1]) 
                               for i in range(1, len(recent_prices))]
                if price_changes:
                    # Minimum non-zero change is likely close to spread
                    non_zero_changes = [pc for pc in price_changes if pc > 0]
                    if non_zero_changes:
                        estimated_spread = min(non_zero_changes)
                        spreads.append(estimated_spread)
                        self.bid_ask_spread[symbol] = np.median(list(spreads))
    
    def _is_large_trade(self, symbol: str, size: float) -> bool:
        """Determine if trade is large relative to average"""
        avg_size = self.avg_trade_sizes.get(symbol, size)
        if avg_size > 0:
            return size > (avg_size * self.large_trade_multiplier)
        return False
    
    def _update_avg_trade_size(self, symbol: str):
        """Update rolling average trade size"""
        trades_list = list(self.trades[symbol])
        if trades_list:
            sizes = [t.size for t in trades_list]
            self.avg_trade_sizes[symbol] = np.mean(sizes)
    
    def _generate_signal(self, symbol: str) -> VolumeSignal:
        """Generate trading signal from current tick flow"""
        trades_list = list(self.trades[symbol])
        
        # Calculate metrics
        metrics = self._calculate_metrics(trades_list)
        
        # Determine signal
        signal, strength, reason = self._determine_signal(metrics)
        
        self.signals_generated += 1
        
        return VolumeSignal(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            signal=signal,
            strength=strength,
            metrics=metrics.__dict__,
            reason=reason
        )
    
    def _calculate_metrics(self, trades: List[Trade]) -> TickFlowMetrics:
        """Calculate comprehensive tick flow metrics"""
        if not trades:
            return None
            
        # Basic counts
        total_trades = len(trades)
        buy_trades = [t for t in trades if t.is_buy]
        sell_trades = [t for t in trades if not t.is_buy]
        
        # Volume calculations
        buy_volume = sum(t.size for t in buy_trades)
        sell_volume = sum(t.size for t in sell_trades)
        total_volume = buy_volume + sell_volume
        
        buy_volume_pct = (buy_volume / total_volume * 100) if total_volume > 0 else 50
        
        # Large trades
        large_buy_trades = sum(1 for t in buy_trades if t.is_large)
        large_sell_trades = sum(1 for t in sell_trades if t.is_large)
        
        # Trade rate (trades per second)
        if len(trades) > 1:
            time_span = (trades[-1].timestamp - trades[0].timestamp).total_seconds()
            trade_rate = len(trades) / time_span if time_span > 0 else 0
        else:
            trade_rate = 0
            
        # Average trade size
        avg_trade_size = total_volume / total_trades if total_trades > 0 else 0
        
        # Momentum score (-100 to +100)
        # Based on buy/sell ratio and large trade imbalance
        volume_momentum = (buy_volume_pct - 50) * 2  # -100 to +100
        
        large_trade_momentum = 0
        total_large = large_buy_trades + large_sell_trades
        if total_large > 0:
            large_buy_pct = large_buy_trades / total_large * 100
            large_trade_momentum = (large_buy_pct - 50) * 2
            
        # Weight large trades more heavily
        momentum_score = (volume_momentum * 0.7) + (large_trade_momentum * 0.3)
        
        # Price trend
        if len(trades) >= 3:
            recent_prices = [t.price for t in trades[-10:]]
            if recent_prices[-1] > recent_prices[0]:
                price_trend = 'up'
            elif recent_prices[-1] < recent_prices[0]:
                price_trend = 'down'
            else:
                price_trend = 'flat'
        else:
            price_trend = 'flat'
            
        return TickFlowMetrics(
            total_trades=total_trades,
            buy_trades=len(buy_trades),
            sell_trades=len(sell_trades),
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            buy_volume_pct=buy_volume_pct,
            large_buy_trades=large_buy_trades,
            large_sell_trades=large_sell_trades,
            avg_trade_size=avg_trade_size,
            trade_rate=trade_rate,
            momentum_score=momentum_score,
            price_trend=price_trend
        )
    
    def _determine_signal(self, metrics: TickFlowMetrics) -> Tuple[str, float, str]:
        """
        Determine trading signal from metrics
        
        Returns:
            (signal, strength, reason)
        """
        # Key factors for signal
        buy_pct = metrics.buy_volume_pct
        momentum = metrics.momentum_score
        large_trade_diff = metrics.large_buy_trades - metrics.large_sell_trades
        trade_rate = metrics.trade_rate
        
        # Calculate signal strength (0-100)
        strength = min(100, abs(momentum))
        
        # Determine signal with reason
        reasons = []
        
        # Strong bullish conditions
        if buy_pct >= self.momentum_threshold:
            if large_trade_diff > 2:
                reasons.append(f"Heavy buying {buy_pct:.0f}% with {metrics.large_buy_trades} large buys")
                signal = 'BULLISH'
                strength = min(100, strength * 1.2)  # Boost for large trades
            else:
                reasons.append(f"Strong buying pressure at {buy_pct:.0f}%")
                signal = 'BULLISH'
                
        # Strong bearish conditions
        elif buy_pct <= (100 - self.momentum_threshold):
            if large_trade_diff < -2:
                reasons.append(f"Heavy selling {100-buy_pct:.0f}% with {metrics.large_sell_trades} large sells")
                signal = 'BEARISH'
                strength = min(100, strength * 1.2)
            else:
                reasons.append(f"Strong selling pressure at {100-buy_pct:.0f}%")
                signal = 'BEARISH'
                
        # Neutral conditions
        else:
            signal = 'NEUTRAL'
            strength = 30  # Low strength for neutral
            
            if abs(large_trade_diff) >= 2:
                if large_trade_diff > 0:
                    reasons.append(f"Mixed flow but {large_trade_diff} more large buys")
                else:
                    reasons.append(f"Mixed flow but {abs(large_trade_diff)} more large sells")
            else:
                reasons.append(f"Balanced flow {buy_pct:.0f}% buy / {100-buy_pct:.0f}% sell")
        
        # Add trade rate context
        if trade_rate > 20:
            reasons.append(f"High activity: {trade_rate:.0f} trades/sec")
        elif trade_rate < 5:
            reasons.append(f"Low activity: {trade_rate:.1f} trades/sec")
            strength *= 0.8  # Reduce strength for low activity
            
        reason = " | ".join(reasons)
        return signal, strength, reason
    
    def get_current_analysis(self, symbol: str) -> Optional[VolumeSignal]:
        """Get current analysis for a symbol without new data"""
        if symbol not in self.trades or len(self.trades[symbol]) < self.min_trades_required:
            return None
            
        return self._generate_signal(symbol)
    
    def get_statistics(self) -> Dict:
        """Get analyzer statistics"""
        return {
            'trades_processed': self.trades_processed,
            'signals_generated': self.signals_generated,
            'active_symbols': list(self.trades.keys()),
            'buffer_sizes': {symbol: len(trades) for symbol, trades in self.trades.items()}
        }


# ============= TEST FUNCTION =============
async def test_tick_flow():
    """Test tick flow analyzer with real-time websocket data"""
    import sys
    import os
    
    # Add parent directories to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    modules_dir = os.path.dirname(os.path.dirname(current_dir))
    vega_root = os.path.dirname(modules_dir)
    if vega_root not in sys.path:
        sys.path.insert(0, vega_root)
    
    from polygon import PolygonWebSocketClient
    
    print("=== TICK FLOW ANALYZER TEST ===")
    print("Analyzing real-time order flow for momentum detection")
    print(f"Current UTC time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
    
    # Test configuration
    TEST_SYMBOLS = ['TSLA', 'AAPL', 'SPY']
    TEST_DURATION = 300  # 5 minutes
    
    # Create analyzer
    analyzer = TickFlowAnalyzer(
        buffer_size=200,  # Last 200 trades
        large_trade_multiplier=3.0,
        momentum_threshold=60.0,  # 60% for bull/bear signal
        min_trades_required=50
    )
    
    # Track signals
    signal_history = []
    
    async def handle_trade(data: Dict):
        """Process trade and display signal"""
        symbol = data['symbol']
        
        # Process trade
        signal = analyzer.process_trade(symbol, data)
        
        if signal:
            signal_history.append(signal)
            
            # Display based on signal type
            if signal.signal == 'BULLISH':
                emoji = '🟢'
                color_code = '\033[92m'  # Green
            elif signal.signal == 'BEARISH':
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
            print(f"  • Buy/Sell Trades: {m['buy_trades']}/{m['sell_trades']} ({m['buy_volume_pct']:.1f}% buy)")
            print(f"  • Buy/Sell Volume: {m['buy_volume']:,.0f}/{m['sell_volume']:,.0f}")
            print(f"  • Large Trades: {m['large_buy_trades']} buys, {m['large_sell_trades']} sells")
            print(f"  • Trade Rate: {m['trade_rate']:.1f} trades/sec")
            print(f"  • Momentum Score: {m['momentum_score']:+.1f}")
            print(f"  • Price Trend: {m['price_trend']}")
    
    # Create WebSocket client
    ws_client = PolygonWebSocketClient()
    
    try:
        # Connect and authenticate
        print(f"Connecting to Polygon WebSocket...")
        await ws_client.connect()
        print("✓ Connected and authenticated")
        
        # Subscribe to trades
        print(f"\nSubscribing to trades for: {', '.join(TEST_SYMBOLS)}")
        await ws_client.subscribe(
            symbols=TEST_SYMBOLS,
            channels=['T'],  # Trades only
            callback=handle_trade
        )
        print("✓ Subscribed successfully")
        
        print(f"\n⏰ Running for {TEST_DURATION} seconds...")
        print("Waiting for trades...\n")
        
        # Create listen task
        listen_task = asyncio.create_task(ws_client.listen())
        
        # Run for specified duration
        start_time = time.time()
        last_stats_time = start_time
        
        while time.time() - start_time < TEST_DURATION:
            await asyncio.sleep(1)
            
            # Print stats every 30 seconds
            if time.time() - last_stats_time >= 30:
                stats = analyzer.get_statistics()
                print(f"\n📊 Stats: {stats['trades_processed']} trades, {stats['signals_generated']} signals")
                print(f"   Time: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
                last_stats_time = time.time()
            
            # Show countdown
            remaining = TEST_DURATION - (time.time() - start_time)
            print(f"\r⏳ Time remaining: {remaining:.0f}s ", end='', flush=True)
        
        print("\n\n🏁 Test complete!")
        
        # Final summary
        stats = analyzer.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"  • Total trades processed: {stats['trades_processed']}")
        print(f"  • Signals generated: {stats['signals_generated']}")
        print(f"  • Symbols tracked: {', '.join(stats['active_symbols'])}")
        print(f"  • End time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Signal summary
        if signal_history:
            print(f"\n📈 Signal Summary:")
            bull_signals = [s for s in signal_history if s.signal == 'BULLISH']
            bear_signals = [s for s in signal_history if s.signal == 'BEARISH']
            neutral_signals = [s for s in signal_history if s.signal == 'NEUTRAL']
            
            print(f"  • Bullish: {len(bull_signals)} ({len(bull_signals)/len(signal_history)*100:.0f}%)")
            print(f"  • Bearish: {len(bear_signals)} ({len(bear_signals)/len(signal_history)*100:.0f}%)")
            print(f"  • Neutral: {len(neutral_signals)} ({len(neutral_signals)/len(signal_history)*100:.0f}%)")
            
            # Time range of signals
            if signal_history:
                first_signal_time = min(s.timestamp for s in signal_history)
                last_signal_time = max(s.timestamp for s in signal_history)
                print(f"\n  • First signal: {first_signal_time.strftime('%H:%M:%S UTC')}")
                print(f"  • Last signal: {last_signal_time.strftime('%H:%M:%S UTC')}")
        
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
    print(f"Starting Tick Flow Analyzer at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("This module detects immediate momentum from order flow")
    print("All timestamps are in UTC\n")
    
    asyncio.run(test_tick_flow())