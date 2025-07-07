"""
Large Order Detection with Price Impact Analysis
Purpose: Detect orders 150% of average or 1.25 STDs above mean
Track 1-second price reaction after large orders
Window: Rolling 15-minute history for statistics
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Set
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


class ImpactStatus(Enum):
    SUCCESS = "SUCCESS"          # Price moved as expected
    FAILURE = "FAILURE"          # Price moved against or flat
    CONTESTED = "CONTESTED"      # Multiple large orders with significant competition
    INSUFFICIENT = "INSUFFICIENT" # No trades in window
    PENDING = "PENDING"          # Still collecting data


@dataclass
class LargeOrder:
    """Large order detection with impact tracking"""
    order_id: str                    # Unique identifier
    symbol: str
    timestamp: datetime
    price: float
    size: int
    side: OrderSide
    spread_at_execution: float
    detection_method: str            # "RATIO" or "STDEV"
    size_vs_avg: float              # Multiple of average
    size_vs_stdev: float            # Number of stdevs
    
    # Impact tracking
    impact_window_start: datetime
    impact_window_end: datetime
    impact_trades: List['Trade'] = field(default_factory=list)
    impact_quotes: List['Quote'] = field(default_factory=list)
    
    # Impact results
    immediate_next_price: Optional[float] = None
    vwap_1s: Optional[float] = None
    max_price_1s: Optional[float] = None
    min_price_1s: Optional[float] = None
    end_price_1s: Optional[float] = None
    trade_count_1s: int = 0
    volume_1s: int = 0
    impact_status: ImpactStatus = ImpactStatus.PENDING
    impact_magnitude: Optional[float] = None  # In spread units
    
    # Competition tracking
    competing_orders: int = 0
    competing_volume: int = 0
    
    @property
    def is_impact_complete(self) -> bool:
        return self.impact_status != ImpactStatus.PENDING


@dataclass
class RollingStats:
    """15-minute rolling statistics for trade sizes"""
    symbol: str
    last_update: datetime
    trade_count: int
    size_sum: float
    size_squared_sum: float
    mean_size: float
    std_size: float
    min_size: float
    max_size: float
    
    @property
    def has_sufficient_data(self) -> bool:
        """Check if we have enough trades for reliable STD calculation"""
        return self.trade_count >= 100


class LargeOrderDetector:
    """
    Detect large orders using rolling statistics and track price impact.
    Uses 15-minute window for statistics, 1-second window for impact.
    """
    
    def __init__(self, 
                 stats_window_minutes: int = 15,
                 impact_window_seconds: int = 1,
                 min_trades_for_std: int = 100,
                 ratio_threshold: float = 1.5,
                 stdev_threshold: float = 1.25,
                 min_competing_orders_for_contested: int = 5,  # Increased from 3
                 min_competing_volume_ratio: float = 3.0,      # Increased from 2.0
                 min_impact_for_success: float = 0.15):        # Decreased from 0.25
        """
        Initialize detector with more reactive thresholds.
        
        Args:
            stats_window_minutes: Window for calculating size statistics (default 15)
            impact_window_seconds: Window for measuring price impact (default 1)
            min_trades_for_std: Minimum trades before using STDEV detection (default 100)
            ratio_threshold: Size ratio threshold (default 1.5x average)
            stdev_threshold: Standard deviation threshold (default 1.25)
            min_competing_orders_for_contested: Min competing orders to mark as contested (default 5)
            min_competing_volume_ratio: Min ratio of competing volume to order size (default 3.0)
            min_impact_for_success: Minimum impact in spreads for success (default 0.15)
        """
        self.stats_window_minutes = stats_window_minutes
        self.impact_window_seconds = impact_window_seconds
        self.min_trades_for_std = min_trades_for_std
        self.ratio_threshold = ratio_threshold
        self.stdev_threshold = stdev_threshold
        self.min_competing_orders_for_contested = min_competing_orders_for_contested
        self.min_competing_volume_ratio = min_competing_volume_ratio
        self.min_impact_for_success = min_impact_for_success
        
        # Trade history for statistics (per symbol)
        self.trade_history: Dict[str, deque] = {}
        
        # Current statistics (per symbol)
        self.current_stats: Dict[str, RollingStats] = {}
        
        # Active large orders being tracked for impact
        self.active_impact_tracking: Dict[str, LargeOrder] = {}
        
        # Completed large orders (per symbol)
        self.completed_orders: Dict[str, deque] = {}
        
        # Quote tracking (reuse from buy_sell_ratio)
        self.latest_quotes: Dict[str, 'Quote'] = {}
        
        # Order ID counter
        self._order_counter = 0
        
        logger.info(f"Initialized LargeOrderDetector: "
                   f"stats_window={stats_window_minutes}min, "
                   f"impact_window={impact_window_seconds}s, "
                   f"thresholds: ratio={ratio_threshold}x, stdev={stdev_threshold}σ, "
                   f"min_impact={min_impact_for_success}, "
                   f"contested_min_orders={min_competing_orders_for_contested}")
    
    def update_quote(self, symbol: str, bid: float, ask: float, timestamp: datetime):
        """Update latest quote for spread calculation"""
        from modules.calculations.order_flow.buy_sell_ratio import Quote
        
        self.latest_quotes[symbol] = Quote(
            symbol=symbol,
            bid=bid,
            ask=ask,
            timestamp=timestamp
        )
        
        # Also update quotes for active impact tracking
        for order_key, large_order in self.active_impact_tracking.items():
            if large_order.symbol == symbol:
                large_order.impact_quotes.append(self.latest_quotes[symbol])
    
    def process_trade(self, trade: 'Trade') -> Optional[LargeOrder]:
        """
        Process trade and detect if it's a large order.
        Returns LargeOrder if detected, None otherwise.
        """
        symbol = trade.symbol
        
        # Initialize symbol if needed
        if symbol not in self.trade_history:
            self.trade_history[symbol] = deque()
            self.completed_orders[symbol] = deque(maxlen=1000)  # Keep last 1000
        
        # Update active impact tracking first
        self._update_impact_tracking(trade)
        
        # Clean old trades from history
        self._clean_old_trades(symbol, trade.timestamp)
        
        # Add trade to history
        self.trade_history[symbol].append(trade)
        
        # Update rolling statistics
        self._update_statistics(symbol)
        
        # Check if this is a large order
        large_order = self._check_large_order(trade)
        
        if large_order:
            # Start impact tracking
            order_key = f"{symbol}_{large_order.order_id}"
            self.active_impact_tracking[order_key] = large_order
            logger.debug(f"Large order detected: {symbol} {large_order.size} @ {large_order.price} "
                       f"({large_order.detection_method}: {large_order.size_vs_avg:.1f}x avg, "
                       f"{large_order.size_vs_stdev:.1f}σ)")
        
        return large_order
    
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
        size_sum = sum(sizes)
        size_squared_sum = sum(s * s for s in sizes)
        mean_size = size_sum / trade_count if trade_count > 0 else 0
        
        # Calculate standard deviation
        if trade_count > 1:
            variance = (size_squared_sum / trade_count) - (mean_size * mean_size)
            std_size = np.sqrt(max(0, variance))  # Ensure non-negative
        else:
            std_size = 0
        
        self.current_stats[symbol] = RollingStats(
            symbol=symbol,
            last_update=trades[-1].timestamp,
            trade_count=trade_count,
            size_sum=size_sum,
            size_squared_sum=size_squared_sum,
            mean_size=mean_size,
            std_size=std_size,
            min_size=min(sizes) if sizes else 0,
            max_size=max(sizes) if sizes else 0
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
            size_vs_stdev=size_vs_stdev,
            impact_window_start=trade.timestamp,
            impact_window_end=trade.timestamp + timedelta(seconds=self.impact_window_seconds)
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
    
    def _update_impact_tracking(self, trade: 'Trade'):
        """Update impact tracking for active large orders"""
        completed_orders = []
        
        for order_key, large_order in self.active_impact_tracking.items():
            if large_order.symbol != trade.symbol:
                continue
            
            # Check if trade is within impact window
            if trade.timestamp <= large_order.impact_window_end:
                # Add trade to impact tracking
                large_order.impact_trades.append(trade)
                
                # Update immediate price if first trade
                if large_order.immediate_next_price is None and trade.timestamp > large_order.timestamp:
                    large_order.immediate_next_price = trade.price
            
            # Check if impact window complete
            if trade.timestamp > large_order.impact_window_end and not large_order.is_impact_complete:
                self._finalize_impact(large_order)
                completed_orders.append(order_key)
        
        # Move completed orders
        for order_key in completed_orders:
            large_order = self.active_impact_tracking.pop(order_key)
            self.completed_orders[large_order.symbol].append(large_order)
    
    def _finalize_impact(self, large_order: LargeOrder):
        """Calculate final impact metrics"""
        impact_trades = [t for t in large_order.impact_trades 
                        if t.timestamp > large_order.timestamp]
        
        if not impact_trades:
            # No trades in window - check quotes
            if large_order.impact_quotes:
                # Use quote midpoint changes
                start_quote = min(large_order.impact_quotes, key=lambda q: q.timestamp)
                end_quote = max(large_order.impact_quotes, key=lambda q: q.timestamp)
                
                start_mid = (start_quote.bid + start_quote.ask) / 2
                end_mid = (end_quote.bid + end_quote.ask) / 2
                
                large_order.end_price_1s = end_mid
                large_order.impact_status = ImpactStatus.INSUFFICIENT
            else:
                large_order.impact_status = ImpactStatus.INSUFFICIENT
            return
        
        # Calculate metrics
        prices = [t.price for t in impact_trades]
        volumes = [t.size for t in impact_trades]
        
        large_order.trade_count_1s = len(impact_trades)
        large_order.volume_1s = sum(volumes)
        large_order.max_price_1s = max(prices)
        large_order.min_price_1s = min(prices)
        large_order.end_price_1s = impact_trades[-1].price
        
        # Calculate VWAP
        if large_order.volume_1s > 0:
            large_order.vwap_1s = sum(p * v for p, v in zip(prices, volumes)) / large_order.volume_1s
        
        # Determine impact success with improved logic
        self._calculate_impact_success(large_order)
    
    def _calculate_impact_success(self, large_order: LargeOrder):
        """Determine if price impact was successful with improved logic for high-volume scenarios"""
        if large_order.vwap_1s is None:
            large_order.impact_status = ImpactStatus.INSUFFICIENT
            return
        
        # Check for competing large orders
        competing_orders = []
        competing_volume = 0
        
        for o in self.active_impact_tracking.values():
            if (o.symbol == large_order.symbol and
                o.order_id != large_order.order_id and
                o.timestamp >= large_order.timestamp and
                o.timestamp <= large_order.impact_window_end):
                competing_orders.append(o)
                competing_volume += o.size
        
        # Store competition metrics
        large_order.competing_orders = len(competing_orders)
        large_order.competing_volume = competing_volume
        
        # Calculate impact magnitude
        spread = large_order.spread_at_execution
        if spread <= 0:
            spread = 0.0001  # Minimum spread for calculation
        
        price_move = large_order.vwap_1s - large_order.price
        large_order.impact_magnitude = price_move / spread
        
        # Determine status based on competition and impact
        is_heavily_contested = (
            len(competing_orders) >= self.min_competing_orders_for_contested or
            (competing_volume > large_order.size * self.min_competing_volume_ratio)
        )
        
        # For high-volume scenarios, be more lenient with success criteria
        success_threshold = self.min_impact_for_success
        if is_heavily_contested:
            success_threshold *= 0.5  # Lower threshold when heavily contested
        
        # Determine success based on side
        if large_order.side == OrderSide.BUY:
            # For buy orders, expect price to move up
            if abs(large_order.impact_magnitude) > success_threshold and price_move > 0:
                large_order.impact_status = ImpactStatus.SUCCESS
            elif is_heavily_contested and abs(large_order.impact_magnitude) > 0.05:
                # Very low threshold for contested scenarios
                large_order.impact_status = ImpactStatus.CONTESTED
            else:
                large_order.impact_status = ImpactStatus.FAILURE
        
        elif large_order.side == OrderSide.SELL:
            # For sell orders, expect price to move down
            if abs(large_order.impact_magnitude) > success_threshold and price_move < 0:
                large_order.impact_status = ImpactStatus.SUCCESS
            elif is_heavily_contested and abs(large_order.impact_magnitude) > 0.05:
                # Very low threshold for contested scenarios
                large_order.impact_status = ImpactStatus.CONTESTED
            else:
                large_order.impact_status = ImpactStatus.FAILURE
        
        else:
            large_order.impact_status = ImpactStatus.INSUFFICIENT
    
    def get_grid_data(self, symbol: str, limit: int = 50) -> List[Dict]:
        """
        Get large orders formatted for grid display.
        Includes both active and completed orders.
        """
        all_orders = []
        
        # Add active orders
        for large_order in self.active_impact_tracking.values():
            if large_order.symbol == symbol:
                all_orders.append(large_order)
        
        # Add completed orders
        if symbol in self.completed_orders:
            all_orders.extend(list(self.completed_orders[symbol]))
        
        # Sort by timestamp descending
        all_orders.sort(key=lambda o: o.timestamp, reverse=True)
        
        # Format for grid
        grid_data = []
        for order in all_orders[:limit]:
            # Format impact status with symbols
            status_symbols = {
                ImpactStatus.SUCCESS: "✓",
                ImpactStatus.FAILURE: "✗",
                ImpactStatus.CONTESTED: "⚡",
                ImpactStatus.INSUFFICIENT: "🕐",
                ImpactStatus.PENDING: "..."
            }
            
            grid_data.append({
                'timestamp': order.timestamp.strftime('%H:%M:%S.%f')[:-3],
                'price': order.price,
                'size': order.size,
                'side': order.side.value,
                'size_vs_avg': f"{order.size_vs_avg:.1f}x",
                'size_vs_std': f"{order.size_vs_stdev:.1f}σ",
                'impact': status_symbols.get(order.impact_status, "?"),
                'impact_magnitude': f"{order.impact_magnitude:.2f}" if order.impact_magnitude else "-",
                'trades_1s': order.trade_count_1s,
                'volume_1s': order.volume_1s,
                'competing': order.competing_orders
            })
        
        return grid_data
    
    def get_summary_stats(self, symbol: str) -> Dict:
        """Get summary statistics for symbol"""
        if symbol not in self.current_stats:
            return {}
        
        stats = self.current_stats[symbol]
        
        # Count large orders by status
        completed = self.completed_orders.get(symbol, [])
        status_counts = {
            'success': sum(1 for o in completed if o.impact_status == ImpactStatus.SUCCESS),
            'failure': sum(1 for o in completed if o.impact_status == ImpactStatus.FAILURE),
            'contested': sum(1 for o in completed if o.impact_status == ImpactStatus.CONTESTED),
            'insufficient': sum(1 for o in completed if o.impact_status == ImpactStatus.INSUFFICIENT)
        }
        
        # Calculate success rate (excluding contested)
        total_evaluated = status_counts['success'] + status_counts['failure']
        success_rate = status_counts['success'] / total_evaluated if total_evaluated > 0 else 0
        
        # Calculate average competition
        avg_competition = np.mean([o.competing_orders for o in completed]) if completed else 0
        
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
                'total_detected': len(completed),
                'active_tracking': len([o for o in self.active_impact_tracking.values() 
                                      if o.symbol == symbol]),
                'impact_success_rate': success_rate,
                'status_breakdown': status_counts,
                'avg_competition': avg_competition
            }
        }