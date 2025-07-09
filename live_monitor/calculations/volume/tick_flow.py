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
from typing import Dict, List, Optional, Deque, Tuple, Union
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
        
        # Classify trade as buy or sell and get parsed conditions
        is_buy, parsed_conditions = self._classify_trade(symbol, price, size, conditions)
        
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
            conditions=parsed_conditions  # Use parsed conditions here
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
                       conditions: Union[List[int], str, int, None]) -> Tuple[bool, List[int]]:
        """
        Classify trade as buy or sell using tick rule and other heuristics
        
        Returns:
            Tuple of (is_buy, parsed_conditions)
        """
        # Handle different condition formats (string, list, etc.)
        parsed_conditions = []
        if conditions:
            if isinstance(conditions, str):
                # Parse string format: "37" or "12,37"
                try:
                    if ',' in conditions:
                        parsed_conditions = [int(c.strip()) for c in conditions.split(',') if c.strip().isdigit()]
                    elif conditions.strip().isdigit():
                        parsed_conditions = [int(conditions.strip())]
                except (ValueError, AttributeError):
                    parsed_conditions = []
            elif isinstance(conditions, list):
                # Ensure all items are integers
                try:
                    parsed_conditions = [int(c) for c in conditions]
                except (ValueError, TypeError):
                    # If conversion fails, try to extract valid integers
                    parsed_conditions = []
                    for c in conditions:
                        try:
                            parsed_conditions.append(int(c))
                        except (ValueError, TypeError):
                            continue
            elif isinstance(conditions, (int, float)):
                # Single condition as number
                parsed_conditions = [int(conditions)]
        
        # If we have previous price, use tick rule
        if symbol in self.last_prices:
            last_price = self.last_prices[symbol]
            
            if price > last_price:
                return True, parsed_conditions  # Uptick = buy
            elif price < last_price:
                return False, parsed_conditions  # Downtick = sell
            else:
                # Price unchanged, look at conditions or use last classification
                # For now, use size as tiebreaker (larger = more likely institutional buy)
                avg_size = self.avg_trade_sizes.get(symbol, size)
                return size > avg_size, parsed_conditions
        
        # First trade - use conditions if available
        # Condition codes vary by exchange, but some common ones:
        # 12 = Intermarket sweep (aggressive)
        # 37 = Contingent trade
        if parsed_conditions:
            if 12 in parsed_conditions:  # Intermarket sweep often aggressive buying
                return True, parsed_conditions
                
        # Default to buy for first trade
        return True, parsed_conditions
    
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
        
        # Get timestamp from the most recent trade
        if trades_list:
            signal_timestamp = trades_list[-1].timestamp
        else:
            # Fallback only if no trades (shouldn't happen in practice)
            signal_timestamp = datetime.now(timezone.utc)
        
        return VolumeSignal(
            symbol=symbol,
            timestamp=signal_timestamp,
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
        Enhanced signal determination for momentum trading.
        More sensitive to large trade imbalances and algo activity.
        Includes price/volume divergence detection for absorption analysis.
        
        Returns:
            (signal, strength, reason)
        """
        # Key factors for signal
        buy_pct = metrics.buy_volume_pct
        momentum = metrics.momentum_score
        large_trade_diff = metrics.large_buy_trades - metrics.large_sell_trades
        large_trade_ratio = metrics.large_buy_trades / max(1, metrics.large_sell_trades)
        trade_rate = metrics.trade_rate
        
        # Calculate base strength from volume
        volume_strength = abs(momentum)
        
        # Calculate large trade signal (more weight for algo detection)
        large_trade_strength = 0
        if metrics.large_buy_trades + metrics.large_sell_trades > 0:
            # Calculate imbalance percentage
            total_large = metrics.large_buy_trades + metrics.large_sell_trades
            large_buy_pct = (metrics.large_buy_trades / total_large) * 100
            large_trade_strength = abs(large_buy_pct - 50) * 2  # Convert to 0-100 scale
        
        # Combine signals with weights (40% large trades, 60% volume for momentum)
        combined_strength = (volume_strength * 0.6) + (large_trade_strength * 0.4)
        strength = min(100, combined_strength)
        
        # Multi-level signal determination
        reasons = []
        
        # Initial signal based on volume flow
        # STRONG BULLISH - Multiple confirmations
        if buy_pct >= 65 or (buy_pct >= 55 and large_trade_ratio >= 2.0):
            signal = 'BULLISH'
            if buy_pct >= 65:
                reasons.append(f"Strong buying {buy_pct:.0f}%")
            if large_trade_ratio >= 2.0:
                reasons.append(f"Algo buying detected ({metrics.large_buy_trades} vs {metrics.large_sell_trades})")
            strength = min(100, strength * 1.2)  # Boost for strong signals
        
        # MODERATE BULLISH - Clear but not overwhelming
        elif buy_pct >= 55 or (buy_pct >= 52 and large_trade_diff > 5):
            signal = 'BULLISH'
            reasons.append(f"Moderate buying {buy_pct:.0f}%")
            if large_trade_diff > 5:
                reasons.append(f"{large_trade_diff} more large buys")
        
        # STRONG BEARISH - Multiple confirmations
        elif buy_pct <= 35 or (buy_pct <= 45 and large_trade_ratio <= 0.5):
            signal = 'BEARISH'
            if buy_pct <= 35:
                reasons.append(f"Strong selling {100-buy_pct:.0f}%")
            if large_trade_ratio <= 0.5:
                reasons.append(f"Algo selling detected ({metrics.large_sell_trades} vs {metrics.large_buy_trades})")
            strength = min(100, strength * 1.2)
        
        # MODERATE BEARISH
        elif buy_pct <= 45 or (buy_pct <= 48 and large_trade_diff < -5):
            signal = 'BEARISH'
            reasons.append(f"Moderate selling {100-buy_pct:.0f}%")
            if large_trade_diff < -5:
                reasons.append(f"{abs(large_trade_diff)} more large sells")
        
        # NEUTRAL - But with lean indicators
        else:
            signal = 'NEUTRAL'
            
            # Add lean information
            if buy_pct > 50:
                reasons.append(f"Slight buy bias {buy_pct:.0f}%")
            else:
                reasons.append(f"Slight sell bias {100-buy_pct:.0f}%")
            
            # Note any large trade imbalance
            if abs(large_trade_diff) >= 3:
                if large_trade_diff > 0:
                    reasons.append(f"{large_trade_diff} more large buys (watch for momentum)")
                else:
                    reasons.append(f"{abs(large_trade_diff)} more large sells (watch for weakness)")
            else:
                reasons.append("Balanced large trades")
            
            # For neutral, reduce strength
            strength = strength * 0.6
        
        # CRITICAL: Check for price/volume divergence (absorption detection)
        if signal == 'BULLISH' and metrics.price_trend == 'down':
            # Bullish volume but price falling = DISTRIBUTION/ABSORPTION FAILURE
            signal = 'BEARISH'
            old_reasons = reasons.copy()
            reasons = [f"⚠️ ABSORPTION FAILURE: {buy_pct:.0f}% buying can't lift price"]
            if large_trade_diff > 0:
                reasons.append(f"Buyers trapped ({large_trade_diff} more large buys absorbed)")
            reasons.append(f"Was: {' | '.join(old_reasons)}")
            strength = min(100, strength * 1.5)  # Strong reversal signal
            
        elif signal == 'BEARISH' and metrics.price_trend == 'up':
            # Bearish volume but price rising = ACCUMULATION/SHORT SQUEEZE
            signal = 'BULLISH'
            old_reasons = reasons.copy()
            reasons = [f"⚠️ SHORT SQUEEZE: {100-buy_pct:.0f}% selling can't drop price"]
            if large_trade_diff < 0:
                reasons.append(f"Sellers trapped ({abs(large_trade_diff)} more large sells absorbed)")
            reasons.append(f"Was: {' | '.join(old_reasons)}")
            strength = min(100, strength * 1.5)  # Strong reversal signal
        
        elif signal == 'NEUTRAL':
            # For neutral signals, note any divergence
            if buy_pct > 52 and metrics.price_trend == 'down':
                reasons.insert(0, "⚠️ Buy pressure failing (distribution?)")
                signal = 'BEARISH'
                strength = min(100, strength * 1.2)
            elif buy_pct < 48 and metrics.price_trend == 'up':
                reasons.insert(0, "⚠️ Sell pressure failing (accumulation?)")
                signal = 'BULLISH'
                strength = min(100, strength * 1.2)
        
        # Add trade rate context (important for momentum)
        if trade_rate > 50:
            reasons.append(f"High momentum: {trade_rate:.0f} trades/sec")
            if signal != 'NEUTRAL':
                strength = min(100, strength * 1.1)  # Boost for high activity
        elif trade_rate < 10:
            reasons.append(f"Low activity: {trade_rate:.1f} trades/sec")
            strength *= 0.8  # Reduce for low activity
        
        # Price trend confirmation (only if not divergent)
        if signal == 'BULLISH' and metrics.price_trend == 'up':
            reasons.append("Price confirming ↗")
            strength = min(100, strength * 1.05)
        elif signal == 'BEARISH' and metrics.price_trend == 'down':
            reasons.append("Price confirming ↘")
            strength = min(100, strength * 1.05)
        
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