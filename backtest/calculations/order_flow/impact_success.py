"""
Impact Success Tracker - Consolidated Module
Purpose: Detect large orders and track net volume pressure
Output: Single line chart showing buy/sell pressure differential
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"
    UNKNOWN = "UNKNOWN"


@dataclass
class LargeOrder:
    """Large order detection"""
    order_id: str
    symbol: str
    timestamp: datetime
    price: float
    size: int
    side: OrderSide
    spread_at_execution: float
    detection_method: str            # "RATIO" or "STDEV" or "BOTH"
    size_vs_avg: float              # Multiple of average
    size_vs_stdev: float            # Number of stdevs


@dataclass
class PressurePoint:
    """Point in time for large order pressure tracking"""
    timestamp: datetime
    buy_volume: int          # Large buy order volume in this period
    sell_volume: int         # Large sell order volume in this period
    net_pressure: int        # buy_volume - sell_volume
    buy_count: int          # Number of large buy orders
    sell_count: int         # Number of large sell orders
    cumulative_pressure: int # Running total net pressure


@dataclass
class RollingStats:
    """15-minute rolling statistics for trade sizes"""
    symbol: str
    last_update: datetime
    trade_count: int
    mean_size: float
    std_size: float
    
    @property
    def has_sufficient_data(self) -> bool:
        """Check if we have enough trades for reliable STD calculation"""
        return self.trade_count >= 100


class ImpactSuccessTracker:
    """
    Consolidated tracker for large order detection and pressure analysis.
    Tracks net buy/sell pressure from large orders over time.
    """
    
    def __init__(self,
                 stats_window_minutes: int = 15,
                 min_trades_for_std: int = 100,
                 ratio_threshold: float = 1.5,
                 stdev_threshold: float = 1.25,
                 pressure_window_seconds: int = 60,  # Aggregate pressure per minute
                 history_points: int = 1800):  # 30 minutes of second-by-second data
        """
        Initialize tracker.
        
        Args:
            stats_window_minutes: Window for calculating size statistics
            min_trades_for_std: Minimum trades before using STDEV detection
            ratio_threshold: Size ratio threshold (default 1.5x average)
            stdev_threshold: Standard deviation threshold (default 1.25σ)
            pressure_window_seconds: Window for aggregating pressure
            history_points: Number of pressure points to keep
        """
        self.stats_window_minutes = stats_window_minutes
        self.min_trades_for_std = min_trades_for_std
        self.ratio_threshold = ratio_threshold
        self.stdev_threshold = stdev_threshold
        self.pressure_window_seconds = pressure_window_seconds
        self.history_points = history_points
        
        # Trade history for statistics (per symbol)
        self.trade_history: Dict[str, deque] = {}
        
        # Current statistics (per symbol)
        self.current_stats: Dict[str, RollingStats] = {}
        
        # Latest quotes for side classification
        self.latest_quotes: Dict[str, 'Quote'] = {}
        
        # Large order history (per symbol)
        self.large_orders: Dict[str, deque] = {}
        
        # Pressure history for charting (per symbol)
        self.pressure_history: Dict[str, deque] = {}
        
        # Current minute accumulator
        self.current_minute: Dict[str, Dict] = {}
        
        # Running cumulative pressure
        self.cumulative_pressure: Dict[str, int] = {}
        
        # Order ID counter
        self._order_counter = 0
        
        logger.info(f"Initialized ImpactSuccessTracker: "
                   f"stats_window={stats_window_minutes}min, "
                   f"thresholds: ratio={ratio_threshold}x, stdev={stdev_threshold}σ")
    
    def update_quote(self, symbol: str, bid: float, ask: float, timestamp: datetime):
        """Update latest quote for spread calculation"""
        from modules.calculations.order_flow.buy_sell_ratio import Quote
        
        self.latest_quotes[symbol] = Quote(
            symbol=symbol,
            bid=bid,
            ask=ask,
            timestamp=timestamp
        )
    
    def process_trade(self, trade: 'Trade') -> Optional[LargeOrder]:
        """
        Process trade and detect if it's a large order.
        Updates pressure tracking if large order detected.
        """
        symbol = trade.symbol
        
        # Initialize symbol if needed
        if symbol not in self.trade_history:
            self._initialize_symbol(symbol)
        
        # Clean old trades from history
        self._clean_old_trades(symbol, trade.timestamp)
        
        # Add trade to history
        self.trade_history[symbol].append(trade)
        
        # Update rolling statistics
        self._update_statistics(symbol)
        
        # Check if this is a large order
        large_order = self._check_large_order(trade)
        
        if large_order:
            # Add to large order history
            self.large_orders[symbol].append(large_order)
            
            # Update pressure tracking
            self._update_pressure(symbol, large_order)
            
            logger.info(f"Large order detected: {symbol} {large_order.size} @ {large_order.price} "
                       f"({large_order.detection_method}: {large_order.size_vs_avg:.1f}x avg, "
                       f"{large_order.size_vs_stdev:.1f}σ)")
        
        # Check if we need to finalize current minute
        self._check_minute_rollover(symbol, trade.timestamp)
        
        return large_order
    
    def _initialize_symbol(self, symbol: str):
        """Initialize tracking for new symbol"""
        self.trade_history[symbol] = deque()
        self.large_orders[symbol] = deque(maxlen=1000)
        self.pressure_history[symbol] = deque(maxlen=self.history_points)
        self.current_minute[symbol] = {
            'timestamp': None,
            'buy_volume': 0,
            'sell_volume': 0,
            'buy_count': 0,
            'sell_count': 0
        }
        self.cumulative_pressure[symbol] = 0
    
    def _clean_old_trades(self, symbol: str, current_time: datetime):
        """Remove trades older than stats window"""
        cutoff = current_time - timedelta(minutes=self.stats_window_minutes)
        
        while self.trade_history[symbol] and self.trade_history[symbol][0].timestamp < cutoff:
            self.trade_history[symbol].popleft()
    
    def _update_statistics(self, symbol: str):
        """Update rolling statistics for symbol"""
        trades = list(self.trade_history[symbol])
        
        if not trades:
            return
        
        # Calculate statistics
        sizes = [t.size for t in trades]
        trade_count = len(sizes)
        mean_size = sum(sizes) / trade_count if trade_count > 0 else 0
        
        # Calculate standard deviation
        if trade_count > 1:
            variance = sum((s - mean_size) ** 2 for s in sizes) / (trade_count - 1)
            std_size = np.sqrt(max(0, variance))
        else:
            std_size = 0
        
        self.current_stats[symbol] = RollingStats(
            symbol=symbol,
            last_update=trades[-1].timestamp,
            trade_count=trade_count,
            mean_size=mean_size,
            std_size=std_size
        )
    
    def _check_large_order(self, trade: 'Trade') -> Optional[LargeOrder]:
        """Check if trade qualifies as large order"""
        symbol = trade.symbol
        
        if symbol not in self.current_stats:
            return None
        
        stats = self.current_stats[symbol]
        
        # Skip if no meaningful average
        if stats.mean_size <= 0:
            return None
        
        # Calculate multiples
        size_vs_avg = trade.size / stats.mean_size
        size_vs_stdev = (trade.size - stats.mean_size) / stats.std_size if stats.std_size > 0 else 0
        
        # Check thresholds
        is_ratio_large = size_vs_avg >= self.ratio_threshold
        is_stdev_large = (stats.has_sufficient_data and 
                         stats.std_size > 0 and 
                         size_vs_stdev >= self.stdev_threshold)
        
        if not (is_ratio_large or is_stdev_large):
            return None
        
        # Determine detection method
        if is_stdev_large and is_ratio_large:
            detection_method = "BOTH"
        elif is_stdev_large:
            detection_method = "STDEV"
        else:
            detection_method = "RATIO"
        
        # Classify side
        side = self._classify_side(trade)
        
        # Get current spread
        spread = 0.0
        if symbol in self.latest_quotes:
            quote = self.latest_quotes[symbol]
            spread = quote.ask - quote.bid
        
        # Create large order
        self._order_counter += 1
        large_order = LargeOrder(
            order_id=str(self._order_counter),
            symbol=symbol,
            timestamp=trade.timestamp,
            price=trade.price,
            size=trade.size,
            side=side,
            spread_at_execution=spread,
            detection_method=detection_method,
            size_vs_avg=size_vs_avg,
            size_vs_stdev=size_vs_stdev
        )
        
        return large_order
    
    def _classify_side(self, trade: 'Trade') -> OrderSide:
        """Classify trade as buy or sell based on price vs quotes"""
        symbol = trade.symbol
        
        # Use embedded quotes if available
        if trade.bid and trade.ask:
            if trade.price >= trade.ask:
                return OrderSide.BUY
            elif trade.price <= trade.bid:
                return OrderSide.SELL
            else:
                # Inside spread - use position
                mid = (trade.bid + trade.ask) / 2
                return OrderSide.BUY if trade.price > mid else OrderSide.SELL
        
        # Use latest quotes
        if symbol in self.latest_quotes:
            quote = self.latest_quotes[symbol]
            if trade.price >= quote.ask:
                return OrderSide.BUY
            elif trade.price <= quote.bid:
                return OrderSide.SELL
            else:
                # Inside spread
                mid = (quote.bid + quote.ask) / 2
                return OrderSide.BUY if trade.price > mid else OrderSide.SELL
        
        return OrderSide.UNKNOWN
    
    def _update_pressure(self, symbol: str, large_order: LargeOrder):
        """Update pressure tracking with new large order"""
        current = self.current_minute[symbol]
        
        # Initialize timestamp if needed
        if current['timestamp'] is None:
            current['timestamp'] = large_order.timestamp.replace(second=0, microsecond=0)
        
        # Update volume and count based on side
        if large_order.side == OrderSide.BUY:
            current['buy_volume'] += large_order.size
            current['buy_count'] += 1
        elif large_order.side == OrderSide.SELL:
            current['sell_volume'] += large_order.size
            current['sell_count'] += 1
    
    def _check_minute_rollover(self, symbol: str, current_time: datetime):
        """Check if we need to finalize current minute and start new one"""
        current = self.current_minute[symbol]
        
        if current['timestamp'] is None:
            return
        
        current_minute = current_time.replace(second=0, microsecond=0)
        
        # If we've moved to a new minute, finalize the previous one
        if current_minute > current['timestamp']:
            self._finalize_minute(symbol)
            
            # Fill any gaps with zero pressure
            while current['timestamp'] < current_minute:
                current['timestamp'] += timedelta(minutes=1)
                if current['timestamp'] < current_minute:
                    self._add_pressure_point(symbol, current['timestamp'], 0, 0, 0, 0)
            
            # Reset for new minute
            current['timestamp'] = current_minute
            current['buy_volume'] = 0
            current['sell_volume'] = 0
            current['buy_count'] = 0
            current['sell_count'] = 0
    
    def _finalize_minute(self, symbol: str):
        """Finalize pressure data for current minute"""
        current = self.current_minute[symbol]
        
        if current['timestamp'] is not None:
            self._add_pressure_point(
                symbol,
                current['timestamp'],
                current['buy_volume'],
                current['sell_volume'],
                current['buy_count'],
                current['sell_count']
            )
    
    def _add_pressure_point(self, symbol: str, timestamp: datetime, 
                           buy_volume: int, sell_volume: int,
                           buy_count: int, sell_count: int):
        """Add a pressure point to history"""
        net_pressure = buy_volume - sell_volume
        self.cumulative_pressure[symbol] += net_pressure
        
        point = PressurePoint(
            timestamp=timestamp,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            net_pressure=net_pressure,
            buy_count=buy_count,
            sell_count=sell_count,
            cumulative_pressure=self.cumulative_pressure[symbol]
        )
        
        self.pressure_history[symbol].append(point)
    
    def get_chart_data(self, symbol: str, points: Optional[int] = None) -> List[Dict]:
        """Get data formatted for charting - single line of net pressure"""
        if symbol not in self.pressure_history:
            return []
        
        # Finalize current minute if needed
        if symbol in self.current_minute:
            timestamp = self.current_minute[symbol].get('timestamp')
            if timestamp:
                self._finalize_minute(symbol)
        
        history = list(self.pressure_history[symbol])
        if points:
            history = history[-points:]
        
        return [
            {
                'timestamp': point.timestamp.isoformat(),
                'net_pressure': point.net_pressure,
                'cumulative_pressure': point.cumulative_pressure,
                'buy_volume': point.buy_volume,
                'sell_volume': point.sell_volume,
                'buy_count': point.buy_count,
                'sell_count': point.sell_count
            }
            for point in history
        ]
    
    def get_current_stats(self, symbol: str) -> Dict:
        """Get current pressure statistics"""
        if symbol not in self.pressure_history:
            return {}
        
        # Get recent history (last 30 points)
        recent_history = list(self.pressure_history[symbol])[-30:]
        
        if not recent_history:
            return {}
        
        # Calculate statistics
        total_buy_volume = sum(p.buy_volume for p in recent_history)
        total_sell_volume = sum(p.sell_volume for p in recent_history)
        total_buy_count = sum(p.buy_count for p in recent_history)
        total_sell_count = sum(p.sell_count for p in recent_history)
        net_pressure = total_buy_volume - total_sell_volume
        
        # Current pressure state
        current_pressure = recent_history[-1].cumulative_pressure if recent_history else 0
        
        return {
            'total_buy_volume': total_buy_volume,
            'total_sell_volume': total_sell_volume,
            'net_pressure': net_pressure,
            'buy_order_count': total_buy_count,
            'sell_order_count': total_sell_count,
            'current_cumulative': current_pressure,
            'pressure_direction': 'BULLISH' if net_pressure > 0 else 'BEARISH' if net_pressure < 0 else 'NEUTRAL',
            'pressure_strength': abs(net_pressure),
            'interpretation': self._interpret_pressure(net_pressure, total_buy_volume + total_sell_volume)
        }
    
    def _interpret_pressure(self, net_pressure: int, total_volume: int) -> str:
        """Interpret the net pressure"""
        if total_volume == 0:
            return "No large orders detected"
        
        pressure_ratio = abs(net_pressure) / total_volume
        
        if pressure_ratio < 0.1:
            return "Balanced large order flow"
        elif net_pressure > 0:
            if pressure_ratio > 0.5:
                return "Strong buy-side pressure"
            else:
                return "Moderate buy-side pressure"
        else:
            if pressure_ratio > 0.5:
                return "Strong sell-side pressure"
            else:
                return "Moderate sell-side pressure"
    
    def get_summary_stats(self, symbol: str) -> Dict:
        """Get summary statistics for symbol"""
        if symbol not in self.current_stats:
            return {}
        
        stats = self.current_stats[symbol]
        large_order_count = len(self.large_orders.get(symbol, []))
        
        return {
            'current_stats': {
                'mean_size': stats.mean_size,
                'std_size': stats.std_size,
                'trade_count': stats.trade_count,
                'has_sufficient_data': stats.has_sufficient_data
            },
            'detection_thresholds': {
                'ratio_threshold': stats.mean_size * self.ratio_threshold,
                'stdev_threshold': stats.mean_size + (stats.std_size * self.stdev_threshold) 
                                  if stats.has_sufficient_data else None
            },
            'large_order_stats': {
                'total_detected': large_order_count,
                'recent_pressure': self.get_current_stats(symbol)
            }
        }


# For backward compatibility if needed
LargeOrderDetector = ImpactSuccessTracker