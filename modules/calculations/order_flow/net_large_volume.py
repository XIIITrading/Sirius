"""
Net Large Order Volume Tracker
Purpose: Track cumulative buy volume minus sell volume for large orders
Output: Cumulative line chart data showing institutional accumulation/distribution
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging

from .large_orders import LargeOrder, OrderSide, ImpactStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VolumePoint:
    """Single point in cumulative volume series"""
    timestamp: datetime
    cumulative_net_volume: float  # Buy volume - Sell volume
    cumulative_buy_volume: float
    cumulative_sell_volume: float
    large_order_count: int
    session_high: float
    session_low: float
    rate_of_change: float  # Volume change per minute


@dataclass
class SessionStats:
    """Statistics for current session"""
    start_time: datetime
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    buy_count: int = 0
    sell_count: int = 0
    largest_buy: float = 0.0
    largest_sell: float = 0.0
    
    @property
    def net_volume(self) -> float:
        return self.buy_volume - self.sell_volume
    
    @property
    def total_volume(self) -> float:
        return self.buy_volume + self.sell_volume
    
    @property
    def buy_ratio(self) -> float:
        """Percentage of volume that is buying"""
        if self.total_volume == 0:
            return 0.5
        return self.buy_volume / self.total_volume


class NetLargeVolumeTracker:
    """
    Track cumulative net volume from large orders.
    Designed for live trading with efficient memory usage.
    """
    
    def __init__(self, 
                 history_points: int = 500,
                 session_reset_hour: Optional[int] = 9,
                 session_reset_minute: Optional[int] = 30,
                 rate_of_change_minutes: int = 5):
        """
        Initialize tracker.
        
        Args:
            history_points: Number of points to keep in chart history
            session_reset_hour: Hour to reset cumulative (None = no reset)
            session_reset_minute: Minute to reset cumulative
            rate_of_change_minutes: Period for rate of change calculation
        """
        self.history_points = history_points
        self.session_reset_hour = session_reset_hour
        self.session_reset_minute = session_reset_minute
        self.rate_of_change_minutes = rate_of_change_minutes
        
        # Current session stats per symbol
        self.sessions: Dict[str, SessionStats] = {}
        
        # Cumulative history per symbol
        self.volume_history: Dict[str, deque] = {}
        
        # Track session high/low per symbol
        self.session_highs: Dict[str, float] = {}
        self.session_lows: Dict[str, float] = {}
        
        # Last update time per symbol
        self.last_update: Dict[str, datetime] = {}
        
        logger.info(f"Initialized NetLargeVolumeTracker: "
                   f"history={history_points}, "
                   f"session_reset={'%02d:%02d' % (session_reset_hour, session_reset_minute) if session_reset_hour else 'None'}")
    
    def process_large_order(self, large_order: LargeOrder) -> Optional[VolumePoint]:
        """
        Process a completed large order and update cumulative volume.
        Only processes orders with final impact status.
        """
        # Only process completed orders
        if large_order.impact_status == ImpactStatus.PENDING:
            return None
        
        symbol = large_order.symbol
        
        # Initialize symbol if needed
        if symbol not in self.sessions:
            self._initialize_symbol(symbol, large_order.timestamp)
        
        # Check for session reset
        self._check_session_reset(symbol, large_order.timestamp)
        
        # Update session statistics
        session = self.sessions[symbol]
        
        if large_order.side == OrderSide.BUY:
            session.buy_volume += large_order.size
            session.buy_count += 1
            session.largest_buy = max(session.largest_buy, large_order.size)
        elif large_order.side == OrderSide.SELL:
            session.sell_volume += large_order.size
            session.sell_count += 1
            session.largest_sell = max(session.largest_sell, large_order.size)
        else:
            # Unknown side - skip
            return None
        
        # Update session high/low
        net_volume = session.net_volume
        self.session_highs[symbol] = max(self.session_highs.get(symbol, net_volume), net_volume)
        self.session_lows[symbol] = min(self.session_lows.get(symbol, net_volume), net_volume)
        
        # Calculate rate of change
        rate_of_change = self._calculate_rate_of_change(symbol, large_order.timestamp)
        
        # Create volume point
        volume_point = VolumePoint(
            timestamp=large_order.timestamp,
            cumulative_net_volume=net_volume,
            cumulative_buy_volume=session.buy_volume,
            cumulative_sell_volume=session.sell_volume,
            large_order_count=session.buy_count + session.sell_count,
            session_high=self.session_highs[symbol],
            session_low=self.session_lows[symbol],
            rate_of_change=rate_of_change
        )
        
        # Add to history
        self.volume_history[symbol].append(volume_point)
        self.last_update[symbol] = large_order.timestamp
        
        return volume_point
    
    def _initialize_symbol(self, symbol: str, timestamp: datetime):
        """Initialize tracking for new symbol"""
        session_start = self._get_session_start(timestamp)
        
        self.sessions[symbol] = SessionStats(start_time=session_start)
        self.volume_history[symbol] = deque(maxlen=self.history_points)
        self.session_highs[symbol] = 0.0
        self.session_lows[symbol] = 0.0
        self.last_update[symbol] = timestamp
    
    def _check_session_reset(self, symbol: str, timestamp: datetime):
        """Check if we need to reset session"""
        if self.session_reset_hour is None:
            return
        
        current_session_start = self._get_session_start(timestamp)
        
        if self.sessions[symbol].start_time < current_session_start:
            # Reset session
            logger.info(f"Session reset for {symbol} at {timestamp}")
            
            # Save final state if needed
            if self.volume_history[symbol]:
                last_point = self.volume_history[symbol][-1]
                logger.info(f"Session ended with net volume: {last_point.cumulative_net_volume:,.0f}")
            
            # Reset
            self.sessions[symbol] = SessionStats(start_time=current_session_start)
            self.session_highs[symbol] = 0.0
            self.session_lows[symbol] = 0.0
    
    def _get_session_start(self, timestamp: datetime) -> datetime:
        """Get session start time for given timestamp"""
        if self.session_reset_hour is None:
            # No session resets - use beginning of day
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        
        session_time = timestamp.replace(
            hour=self.session_reset_hour,
            minute=self.session_reset_minute,
            second=0,
            microsecond=0
        )
        
        # If timestamp is before session start, use previous day's session
        if timestamp < session_time:
            session_time -= timedelta(days=1)
        
        return session_time
    
    def _calculate_rate_of_change(self, symbol: str, current_time: datetime) -> float:
        """Calculate volume rate of change (volume per minute)"""
        if not self.volume_history[symbol]:
            return 0.0
        
        # Find point N minutes ago
        target_time = current_time - timedelta(minutes=self.rate_of_change_minutes)
        
        past_volume = 0.0
        found = False
        
        for point in reversed(self.volume_history[symbol]):
            if point.timestamp <= target_time:
                past_volume = point.cumulative_net_volume
                found = True
                break
        
        if not found:
            # Not enough history
            return 0.0
        
        # Current volume
        current_volume = self.sessions[symbol].net_volume
        
        # Calculate rate (volume per minute)
        time_diff_minutes = (current_time - target_time).total_seconds() / 60.0
        if time_diff_minutes > 0:
            return (current_volume - past_volume) / time_diff_minutes
        
        return 0.0
    
    def get_chart_data(self, symbol: str, points: Optional[int] = None) -> List[Dict]:
        """
        Get data formatted for charting.
        Returns most recent N points.
        """
        if symbol not in self.volume_history:
            return []
        
        history = list(self.volume_history[symbol])
        if points:
            history = history[-points:]
        
        return [
            {
                'timestamp': point.timestamp.isoformat(),
                'net_volume': point.cumulative_net_volume,
                'buy_volume': point.cumulative_buy_volume,
                'sell_volume': point.cumulative_sell_volume,
                'session_high': point.session_high,
                'session_low': point.session_low,
                'rate_of_change': point.rate_of_change,
                'order_count': point.large_order_count
            }
            for point in history
        ]
    
    def get_current_stats(self, symbol: str) -> Dict:
        """Get current session statistics"""
        if symbol not in self.sessions:
            return {}
        
        session = self.sessions[symbol]
        
        return {
            'session_start': session.start_time.isoformat(),
            'net_volume': session.net_volume,
            'buy_volume': session.buy_volume,
            'sell_volume': session.sell_volume,
            'buy_count': session.buy_count,
            'sell_count': session.sell_count,
            'buy_ratio': session.buy_ratio,
            'largest_buy': session.largest_buy,
            'largest_sell': session.largest_sell,
            'session_high': self.session_highs.get(symbol, 0),
            'session_low': self.session_lows.get(symbol, 0),
            'last_update': self.last_update.get(symbol, session.start_time).isoformat()
        }
    
    def get_trend_analysis(self, symbol: str, lookback_minutes: int = 30) -> Dict:
        """Analyze recent trend in net volume"""
        if symbol not in self.volume_history or not self.volume_history[symbol]:
            return {'trend': 'NEUTRAL', 'strength': 0.0}
        
        current_time = self.last_update[symbol]
        lookback_time = current_time - timedelta(minutes=lookback_minutes)
        
        # Get points in lookback window
        recent_points = [p for p in self.volume_history[symbol] 
                        if p.timestamp >= lookback_time]
        
        if len(recent_points) < 2:
            return {'trend': 'NEUTRAL', 'strength': 0.0}
        
        # Calculate trend using linear regression
        times = [(p.timestamp - recent_points[0].timestamp).total_seconds() 
                for p in recent_points]
        volumes = [p.cumulative_net_volume for p in recent_points]
        
        if len(set(times)) < 2:  # All same time
            return {'trend': 'NEUTRAL', 'strength': 0.0}
        
        # Simple linear regression
        x_mean = np.mean(times)
        y_mean = np.mean(volumes)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(times, volumes))
        denominator = sum((x - x_mean) ** 2 for x in times)
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        # Determine trend
        if slope > 10:  # Volume per second threshold
            trend = 'ACCUMULATION'
        elif slope < -10:
            trend = 'DISTRIBUTION'
        else:
            trend = 'NEUTRAL'
        
        # Calculate R-squared for strength
        if denominator > 0:
            y_pred = [slope * (x - x_mean) + y_mean for x in times]
            ss_res = sum((y - y_pred[i]) ** 2 for i, y in enumerate(volumes))
            ss_tot = sum((y - y_mean) ** 2 for y in volumes)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            strength = abs(r_squared)
        else:
            strength = 0.0
        
        return {
            'trend': trend,
            'strength': min(max(strength, 0.0), 1.0),
            'slope': slope * 60,  # Convert to per minute
            'points_analyzed': len(recent_points)
        }