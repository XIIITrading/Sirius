# modules/calculations/volume/volume_analysis_1min.py
"""
Module: 1-Minute Volume Analysis
Purpose: Confirm if volume is creating real price movement or absorption
Features: Buy/sell volume aggregation, absorption detection, efficiency metrics
Output: BULLISH/BEARISH/NEUTRAL signals based on volume effectiveness
Time Handling: All timestamps in UTC
"""

import asyncio
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Deque, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
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
class MinuteBar:
    """1-minute aggregated volume data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    buy_volume: float
    sell_volume: float
    trade_count: int
    buy_trade_count: int
    sell_trade_count: int
    large_buy_volume: float
    large_sell_volume: float
    vwap: float


@dataclass
class VolumeMetrics:
    """Metrics for 1-minute volume analysis"""
    buy_volume_pct: float
    sell_volume_pct: float
    volume_delta: float  # buy_volume - sell_volume
    price_change: float
    price_change_pct: float
    volume_efficiency: float  # price change per unit volume
    is_absorption: bool
    absorption_type: str  # 'buy_absorption', 'sell_absorption', 'none'
    volume_trend: str  # 'increasing', 'decreasing', 'stable'
    relative_volume: float  # vs average
    bar_strength: float  # 0-100 score


@dataclass
class VolumeSignal:
    """Standard volume signal output"""
    symbol: str
    timestamp: datetime
    signal: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    strength: float  # 0-100
    metrics: Dict  # Detailed metrics
    reason: str  # Human-readable explanation


class VolumeAnalysis1Min:
    """
    1-Minute volume analyzer for confirming price movement
    All timestamps are in UTC.
    """
    
    def __init__(self,
                 lookback_bars: int = 14,  # Bars for trend analysis
                 absorption_threshold: float = 0.1,  # Max price change % for absorption
                 efficiency_threshold: float = 0.5,  # Min efficiency for strong move
                 volume_imbalance_threshold: float = 65.0):  # % for bull/bear signal
        """
        Initialize 1-minute volume analyzer
        
        Args:
            lookback_bars: Number of bars for analysis
            absorption_threshold: Price change threshold for absorption
            efficiency_threshold: Minimum efficiency for valid moves
            volume_imbalance_threshold: Buy/sell % for signals
        """
        self.lookback_bars = lookback_bars
        self.absorption_threshold = absorption_threshold
        self.efficiency_threshold = efficiency_threshold
        self.volume_imbalance_threshold = volume_imbalance_threshold
        
        # Data storage
        self.minute_bars: Dict[str, Deque[MinuteBar]] = {}
        self.current_bar_trades: Dict[str, List[Dict]] = {}
        self.average_volumes: Dict[str, float] = {}
        
        # Large trade threshold
        self.large_trade_sizes: Dict[str, float] = {}
        
        # Track current minute per symbol
        self.current_minute: Dict[str, datetime] = {}
        
        # Performance tracking
        self.bars_processed = 0
        self.signals_generated = 0
        
        logger.info(f"1-Minute Volume Analyzer initialized at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info(f"Settings: lookback={lookback_bars}, absorption={absorption_threshold}%, "
                   f"efficiency={efficiency_threshold}, imbalance={volume_imbalance_threshold}%")
    
    def _validate_timestamp(self, timestamp: datetime, source: str) -> datetime:
        """Validate and ensure timestamp is UTC"""
        if timestamp.tzinfo is None:
            logger.warning(f"{source}: Naive datetime received, assuming UTC")
            return timestamp.replace(tzinfo=timezone.utc)
        elif timestamp.tzinfo != timezone.utc:
            logger.warning(f"{source}: Non-UTC timezone {timestamp.tzinfo}, converting to UTC")
            return timestamp.astimezone(timezone.utc)
        return timestamp
        
    def process_trade(self, symbol: str, trade_data: Dict):
        """
        Process incoming trade data and aggregate into minute bars
        
        Args:
            symbol: Ticker symbol
            trade_data: Trade data from websocket
        """
        # Initialize if needed
        if symbol not in self.minute_bars:
            self.minute_bars[symbol] = deque(maxlen=self.lookback_bars + 10)
            self.current_bar_trades[symbol] = []
            self.large_trade_sizes[symbol] = 0
            
        # Get trade time and ensure UTC
        trade_time = datetime.fromtimestamp(trade_data['timestamp'] / 1000, tz=timezone.utc)
        trade_time = self._validate_timestamp(trade_time, f"Trade-{symbol}")
        
        minute_start = trade_time.replace(second=0, microsecond=0)
        
        # Check if this is a new minute
        if symbol not in self.current_minute:
            # First trade for this symbol
            self.current_minute[symbol] = minute_start
            self.current_bar_trades[symbol] = [trade_data]
        elif minute_start > self.current_minute[symbol]:
            # New minute started - close the previous bar
            self._close_minute_bar(symbol, self.current_minute[symbol])
            self.current_minute[symbol] = minute_start
            self.current_bar_trades[symbol] = [trade_data]
        else:
            # Same minute - add to current bar
            self.current_bar_trades[symbol].append(trade_data)

    def _close_minute_bar(self, symbol: str, bar_time: datetime):
        """Close the current minute bar and start a new one"""
        trades = self.current_bar_trades[symbol]
        
        if not trades:
            return
            
        # Calculate OHLC
        prices = [t['price'] for t in trades]
        open_price = prices[0]
        high_price = max(prices)
        low_price = min(prices)
        close_price = prices[-1]
        
        # Calculate volumes
        buy_volume = 0
        sell_volume = 0
        buy_trades = 0
        sell_trades = 0
        large_buy_volume = 0
        large_sell_volume = 0
        
        # Update large trade threshold
        avg_size = np.mean([t['size'] for t in trades])
        self.large_trade_sizes[symbol] = avg_size * 3  # 3x average
        
        # Process each trade
        total_volume_price = 0
        for trade in trades:
            size = trade['size']
            price = trade['price']
            
            # Simple buy/sell classification using tick rule
            if len(self.minute_bars[symbol]) > 0:
                prev_close = self.minute_bars[symbol][-1].close
                is_buy = price >= prev_close
            else:
                is_buy = price >= open_price
                
            if is_buy:
                buy_volume += size
                buy_trades += 1
                if size > self.large_trade_sizes[symbol]:
                    large_buy_volume += size
            else:
                sell_volume += size
                sell_trades += 1
                if size > self.large_trade_sizes[symbol]:
                    large_sell_volume += size
                    
            total_volume_price += price * size
        
        # Calculate VWAP
        total_volume = buy_volume + sell_volume
        vwap = total_volume_price / total_volume if total_volume > 0 else close_price
        
        # Create minute bar
        bar = MinuteBar(
            timestamp=bar_time,  # Use the actual bar time
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=total_volume,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            trade_count=len(trades),
            buy_trade_count=buy_trades,
            sell_trade_count=sell_trades,
            large_buy_volume=large_buy_volume,
            large_sell_volume=large_sell_volume,
            vwap=vwap
        )
        
        self.minute_bars[symbol].append(bar)
        self.bars_processed += 1
        
        print(f"\n✓ Bar completed for {symbol} at {bar_time.strftime('%H:%M:%S UTC')}")
        print(f"  OHLC: {open_price:.2f}/{high_price:.2f}/{low_price:.2f}/{close_price:.2f}")
        print(f"  Volume: {total_volume:,.0f} ({buy_volume:,.0f} buy / {sell_volume:,.0f} sell)")
        
        # Update average volume
        self._update_average_volume(symbol)
        
        # Generate signal if we have enough bars
        if len(self.minute_bars[symbol]) >= 3:  # Need at least 3 bars
            signal = self._generate_signal(symbol)
            if signal:
                self.signals_generated += 1
                # Return the signal so the callback can display it
                return signal
        else:
            print(f"  Need {3 - len(self.minute_bars[symbol])} more bars for signals")
        
        return None
                
    def _update_average_volume(self, symbol: str):
        """Update rolling average volume"""
        bars = list(self.minute_bars[symbol])
        if len(bars) >= 5:
            volumes = [bar.volume for bar in bars[-20:]]  # Last 20 bars
            self.average_volumes[symbol] = np.mean(volumes)
    
    def _generate_signal(self, symbol: str) -> Optional[VolumeSignal]:
        """Generate trading signal from recent bars"""
        bars = list(self.minute_bars[symbol])
        if len(bars) < 3:
            return None
            
        # Calculate metrics for recent bars
        recent_bars = bars[-3:]  # Last 3 minutes
        metrics = self._calculate_metrics(recent_bars, symbol)
        
        # Determine signal
        signal, strength, reason = self._determine_signal(metrics, recent_bars)
        
        return VolumeSignal(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            signal=signal,
            strength=strength,
            metrics=self._metrics_to_dict(metrics),
            reason=reason
        )
    
    def _calculate_metrics(self, bars: List[MinuteBar], symbol: str) -> List[VolumeMetrics]:
        """Calculate volume metrics for bars"""
        metrics_list = []
        
        for i, bar in enumerate(bars):
            # Basic volume calculations
            total_volume = bar.volume
            buy_pct = (bar.buy_volume / total_volume * 100) if total_volume > 0 else 50
            sell_pct = 100 - buy_pct
            volume_delta = bar.buy_volume - bar.sell_volume
            
            # Price calculations
            price_change = bar.close - bar.open
            price_change_pct = (price_change / bar.open * 100) if bar.open > 0 else 0
            
            # Volume efficiency (price movement per unit volume)
            efficiency = abs(price_change_pct) / (total_volume / 1000) if total_volume > 0 else 0
            
            # Absorption detection
            is_absorption = False
            absorption_type = 'none'
            
            if total_volume > self.average_volumes.get(symbol, total_volume) * 1.5:
                # High volume bar
                if abs(price_change_pct) < self.absorption_threshold:
                    is_absorption = True
                    if buy_pct > 60:
                        absorption_type = 'buy_absorption'  # Lots of buying but price stuck
                    elif sell_pct > 60:
                        absorption_type = 'sell_absorption'  # Lots of selling but price stuck
            
            # Volume trend
            if i > 0:
                prev_volume = bars[i-1].volume
                if bar.volume > prev_volume * 1.2:
                    volume_trend = 'increasing'
                elif bar.volume < prev_volume * 0.8:
                    volume_trend = 'decreasing'
                else:
                    volume_trend = 'stable'
            else:
                volume_trend = 'stable'
            
            # Relative volume
            avg_vol = self.average_volumes.get(symbol, bar.volume)
            relative_volume = bar.volume / avg_vol if avg_vol > 0 else 1.0
            
            # Bar strength (0-100)
            # Combines volume imbalance, efficiency, and relative volume
            imbalance_score = abs(buy_pct - 50) * 2  # 0-100
            efficiency_score = min(100, efficiency * 100)
            volume_score = min(100, relative_volume * 50)
            
            bar_strength = (imbalance_score * 0.4 + efficiency_score * 0.4 + volume_score * 0.2)
            
            metrics = VolumeMetrics(
                buy_volume_pct=buy_pct,
                sell_volume_pct=sell_pct,
                volume_delta=volume_delta,
                price_change=price_change,
                price_change_pct=price_change_pct,
                volume_efficiency=efficiency,
                is_absorption=is_absorption,
                absorption_type=absorption_type,
                volume_trend=volume_trend,
                relative_volume=relative_volume,
                bar_strength=bar_strength
            )
            
            metrics_list.append(metrics)
            
        return metrics_list
    
    def _metrics_to_dict(self, metrics_list: List[VolumeMetrics]) -> Dict:
        """Convert metrics to dictionary for signal"""
        if not metrics_list:
            return {}
            
        # Average metrics across bars
        avg_metrics = {
            'buy_volume_pct': np.mean([m.buy_volume_pct for m in metrics_list]),
            'volume_efficiency': np.mean([m.volume_efficiency for m in metrics_list]),
            'relative_volume': np.mean([m.relative_volume for m in metrics_list]),
            'bar_strength': np.mean([m.bar_strength for m in metrics_list]),
            'absorption_detected': any(m.is_absorption for m in metrics_list),
            'volume_trend': metrics_list[-1].volume_trend  # Most recent
        }
        
        return avg_metrics
    
    def _determine_signal(self, metrics_list: List[VolumeMetrics], 
                         bars: List[MinuteBar]) -> Tuple[str, float, str]:
        """Determine trading signal from metrics"""
        if not metrics_list:
            return 'NEUTRAL', 0, 'Insufficient data'
            
        # Get latest metrics
        latest = metrics_list[-1]
        
        # Average metrics across recent bars
        avg_buy_pct = np.mean([m.buy_volume_pct for m in metrics_list])
        avg_efficiency = np.mean([m.volume_efficiency for m in metrics_list])
        absorption_count = sum(1 for m in metrics_list if m.is_absorption)
        
        # Price trend
        price_up = bars[-1].close > bars[0].open
        price_down = bars[-1].close < bars[0].open
        
        reasons = []
        
        # Check for absorption first (overrides other signals)
        if absorption_count >= 2:
            signal = 'NEUTRAL'
            strength = 20
            if latest.absorption_type == 'buy_absorption':
                reasons.append("Buy absorption detected - resistance above")
            elif latest.absorption_type == 'sell_absorption':
                reasons.append("Sell absorption detected - support below")
            else:
                reasons.append("Price absorption - no follow through")
                
        # Strong bullish
        elif (avg_buy_pct >= self.volume_imbalance_threshold and 
              price_up and 
              avg_efficiency >= self.efficiency_threshold):
            signal = 'BULLISH'
            strength = min(100, latest.bar_strength * 1.2)
            reasons.append(f"Strong buying {avg_buy_pct:.0f}% with price up {bars[-1].close - bars[0].open:.2f}")
            
        # Strong bearish
        elif (avg_buy_pct <= (100 - self.volume_imbalance_threshold) and 
              price_down and 
              avg_efficiency >= self.efficiency_threshold):
            signal = 'BEARISH'
            strength = min(100, latest.bar_strength * 1.2)
            reasons.append(f"Strong selling {100-avg_buy_pct:.0f}% with price down {bars[0].open - bars[-1].close:.2f}")
            
        # Weak bullish
        elif avg_buy_pct >= 55 and price_up:
            signal = 'BULLISH'
            strength = latest.bar_strength * 0.8
            reasons.append(f"Moderate buying {avg_buy_pct:.0f}%")
            
        # Weak bearish
        elif avg_buy_pct <= 45 and price_down:
            signal = 'BEARISH'
            strength = latest.bar_strength * 0.8
            reasons.append(f"Moderate selling {100-avg_buy_pct:.0f}%")
            
        # Neutral
        else:
            signal = 'NEUTRAL'
            strength = 30
            if latest.is_absorption:
                reasons.append("Volume absorption")
            else:
                reasons.append(f"Mixed signals: {avg_buy_pct:.0f}% buy")
        
        # Add context
        if latest.relative_volume > 1.5:
            reasons.append(f"High volume {latest.relative_volume:.1f}x avg")
        elif latest.relative_volume < 0.5:
            reasons.append("Low volume")
            strength *= 0.7
            
        if latest.volume_trend == 'increasing':
            reasons.append("Volume increasing")
        elif latest.volume_trend == 'decreasing':
            reasons.append("Volume decreasing")
            
        # Efficiency context
        if avg_efficiency < 0.3:
            reasons.append("Low efficiency")
            strength *= 0.8
            
        reason = " | ".join(reasons)
        return signal, strength, reason
    
    def get_current_analysis(self, symbol: str) -> Optional[VolumeSignal]:
        """Get current analysis for a symbol"""
        if symbol not in self.minute_bars or len(self.minute_bars[symbol]) < 3:
            return None
            
        return self._generate_signal(symbol)
    
    def get_latest_bar(self, symbol: str) -> Optional[MinuteBar]:
        """Get the most recent completed bar"""
        if symbol in self.minute_bars and self.minute_bars[symbol]:
            return self.minute_bars[symbol][-1]
        return None
    
    def get_statistics(self) -> Dict:
        """Get analyzer statistics"""
        return {
            'bars_processed': self.bars_processed,
            'signals_generated': self.signals_generated,
            'active_symbols': list(self.minute_bars.keys()),
            'bar_counts': {symbol: len(bars) for symbol, bars in self.minute_bars.items()}
        }


# ============= TEST FUNCTION =============
async def test_volume_1min():
    """Test 1-minute volume analyzer with real-time websocket data"""
    import sys
    import os
    
    # Add parent directories to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    modules_dir = os.path.dirname(os.path.dirname(current_dir))
    vega_root = os.path.dirname(modules_dir)
    if vega_root not in sys.path:
        sys.path.insert(0, vega_root)
    
    from polygon import PolygonWebSocketClient
    
    print("=== 1-MINUTE VOLUME ANALYZER TEST ===")
    print("Analyzing volume effectiveness and absorption patterns")
    print(f"Current UTC time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
    
    # Test configuration
    TEST_SYMBOLS = ['TSLA', 'AAPL', 'SPY']
    TEST_DURATION = 300  # 5 minutes
    
    # Create analyzer
    analyzer = VolumeAnalysis1Min(
        lookback_bars=14,
        absorption_threshold=0.1,  # 0.1% price change
        efficiency_threshold=0.5,
        volume_imbalance_threshold=65.0  # 65% for bull/bear
    )
    
    # Track signals
    signal_history = []
    last_bar_times = {}
    
    async def handle_trade(data: Dict):
        """Process trade data"""
        symbol = data['symbol']
        
        # Process trade
        analyzer.process_trade(symbol, data)
        
        # Check if we have a new completed bar
        latest_bar = analyzer.get_latest_bar(symbol)
        if latest_bar:
            bar_time = latest_bar.timestamp
            
            # Check if this is a new bar
            if symbol not in last_bar_times or bar_time > last_bar_times[symbol]:
                last_bar_times[symbol] = bar_time
                
                # Get signal for the completed bar
                signal = analyzer.get_current_analysis(symbol)
                
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
                    print(f"Time: {bar_time.strftime('%H:%M UTC')} bar completed")
                    print(f"Signal Generated: {signal.timestamp.strftime('%H:%M:%S UTC')}")
                    print(f"Reason: {signal.reason}")
                    
                    # Display bar details
                    print(f"\nBar Details:")
                    print(f"  • OHLC: {latest_bar.open:.2f} → {latest_bar.close:.2f} "
                          f"(H: {latest_bar.high:.2f}, L: {latest_bar.low:.2f})")
                    print(f"  • Volume: {latest_bar.volume:,.0f} "
                          f"({latest_bar.buy_volume:,.0f} buy / {latest_bar.sell_volume:,.0f} sell)")
                    print(f"  • Trades: {latest_bar.trade_count} "
                          f"({latest_bar.buy_trade_count} buy / {latest_bar.sell_trade_count} sell)")
                    print(f"  • VWAP: ${latest_bar.vwap:.2f}")
                    
                    # Display metrics
                    if signal.metrics:
                        m = signal.metrics
                        print(f"\nMetrics:")
                        print(f"  • Buy Volume: {m['buy_volume_pct']:.1f}%")
                        print(f"  • Efficiency: {m['volume_efficiency']:.2f}")
                        print(f"  • Relative Volume: {m['relative_volume']:.1f}x")
                        print(f"  • Bar Strength: {m['bar_strength']:.0f}")
                        if m['absorption_detected']:
                            print(f"  • ⚠️ ABSORPTION DETECTED")
    
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
        print("Waiting for minute bars to complete...")
        print("Note: New bars complete at the start of each minute (:00 seconds)\n")
        
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
                print(f"\n📊 Stats: {stats['bars_processed']} bars completed")
                print(f"   Time: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
                last_stats_time = time.time()
            
            # Show countdown
            remaining = TEST_DURATION - (time.time() - start_time)
            print(f"\r⏳ Time remaining: {remaining:.0f}s ", end='', flush=True)
        
        print("\n\n🏁 Test complete!")
        
        # Final summary
        stats = analyzer.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"  • Bars processed: {stats['bars_processed']}")
        print(f"  • Signals generated: {stats['signals_generated']}")
        print(f"  • Symbols tracked: {', '.join(stats['active_symbols'])}")
        print(f"  • End time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Signal summary
        if signal_history:
            print(f"\n📈 Signal Summary:")
            bull_signals = [s for s in signal_history if s.signal == 'BULLISH']
            bear_signals = [s for s in signal_history if s.signal == 'BEARISH']
            neutral_signals = [s for s in signal_history if s.signal == 'NEUTRAL']
            
            print(f"  • Bullish: {len(bull_signals)}")
            print(f"  • Bearish: {len(bear_signals)}")
            print(f"  • Neutral: {len(neutral_signals)}")
            
            # Check for absorption
            absorption_count = sum(1 for s in signal_history 
                                 if s.metrics and s.metrics.get('absorption_detected'))
            if absorption_count > 0:
                print(f"  • Absorption detected: {absorption_count} times")
            
            # Time range
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
    print(f"Starting 1-Minute Volume Analyzer at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("This module confirms if volume creates real price movement")
    print("All timestamps are in UTC\n")
    
    asyncio.run(test_volume_1min())