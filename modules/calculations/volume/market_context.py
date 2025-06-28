# modules/calculations/volume/market_context.py
"""
Module: Market Context Analysis (15-Minute)
Purpose: Identify overall market regime and positioning using volume
Features: VWAP tracking, relative volume, cumulative delta, regime detection
Output: BULLISH/BEARISH/NEUTRAL signals based on market structure
Time Handling: All timestamps in UTC
"""

import asyncio
import numpy as np
from datetime import datetime, timezone, timedelta, time as datetime_time
from typing import Dict, List, Optional, Deque, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import logging
import time
import json
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
class SessionBar:
    """15-minute aggregated session data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    buy_volume: float
    sell_volume: float
    cumulative_volume: float
    cumulative_buy_volume: float
    cumulative_sell_volume: float
    vwap: float
    session_vwap: float
    trade_count: int


@dataclass
class MarketMetrics:
    """Metrics for market context analysis"""
    price_vs_vwap: float  # % distance from VWAP
    cumulative_delta: float  # Total buy - sell volume for session
    relative_volume: float  # vs typical volume for this time
    volume_trend: str  # 'accelerating', 'steady', 'declining'
    market_phase: str  # 'opening', 'morning', 'midday', 'afternoon', 'closing'
    institutional_activity: float  # Large trade percentage
    trend_strength: float  # 0-100 based on consistency
    regime: str  # 'trending', 'ranging', 'volatile'


@dataclass
class VolumeSignal:
    """Standard volume signal output"""
    symbol: str
    timestamp: datetime
    signal: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    strength: float  # 0-100
    metrics: Dict  # Detailed metrics
    reason: str  # Human-readable explanation


class MarketContext:
    """
    15-minute market context analyzer for regime identification
    All timestamps are in UTC.
    """
    
    def __init__(self,
                 lookback_bars: int = 10,  # 2.5 hours of 15-min bars
                 vwap_deviation_threshold: float = 0.5,  # % from VWAP
                 volume_surge_threshold: float = 1.5,  # vs average
                 delta_imbalance_threshold: float = 100000):  # Volume units
        """
        Initialize market context analyzer
        
        Args:
            lookback_bars: Number of 15-min bars to analyze
            vwap_deviation_threshold: Significant deviation from VWAP
            volume_surge_threshold: Multiple of average volume
            delta_imbalance_threshold: Cumulative delta for bias
        """
        self.lookback_bars = lookback_bars
        self.vwap_deviation_threshold = vwap_deviation_threshold
        self.volume_surge_threshold = volume_surge_threshold
        self.delta_imbalance_threshold = delta_imbalance_threshold
        
        # Data storage
        self.session_bars: Dict[str, Deque[SessionBar]] = {}
        self.current_bar_trades: Dict[str, List[Dict]] = {}
        self.current_15min: Dict[str, datetime] = {}
        
        # Session tracking
        self.session_data: Dict[str, Dict] = {}  # VWAP, cumulative volumes
        
        # Determine current DST status for market hours
        now = datetime.now(timezone.utc)
        self.is_edt = self._is_daylight_saving(now)
        
        if self.is_edt:
            self.session_start_time = datetime_time(13, 30)  # 9:30 AM EDT in UTC
            self.session_end_time = datetime_time(20, 0)     # 4:00 PM EDT in UTC
            self.market_open_minutes = 13 * 60 + 30
            self.market_close_minutes = 20 * 60
            logger.info("Using EDT market hours: 13:30-20:00 UTC (9:30 AM - 4:00 PM EDT)")
        else:
            self.session_start_time = datetime_time(14, 30)  # 9:30 AM EST in UTC
            self.session_end_time = datetime_time(21, 0)     # 4:00 PM EST in UTC
            self.market_open_minutes = 14 * 60 + 30
            self.market_close_minutes = 21 * 60
            logger.info("Using EST market hours: 14:30-21:00 UTC (9:30 AM - 4:00 PM EST)")
        
        # Historical volume profiles (load from file if exists)
        self.volume_profiles: Dict[str, Dict] = self._load_volume_profiles()
        
        # Performance tracking
        self.bars_processed = 0
        self.signals_generated = 0
        
        logger.info(f"Market Context Analyzer initialized at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    def _is_daylight_saving(self, dt: datetime) -> bool:
        """
        Determine if given datetime is during EDT (simplified).
        In reality, DST changes on specific Sundays in March and November.
        """
        # Simplified: EDT is roughly March 14 - November 7
        month = dt.month
        day = dt.day
        
        if month < 3 or month > 11:
            return False
        elif month > 3 and month < 11:
            return True
        elif month == 3:
            # Second Sunday of March
            return day >= 14
        else:  # month == 11
            # First Sunday of November
            return day < 7
    
    def _validate_timestamp(self, timestamp: datetime, source: str) -> datetime:
        """Validate and ensure timestamp is UTC"""
        if timestamp.tzinfo is None:
            logger.warning(f"{source}: Naive datetime received, assuming UTC")
            return timestamp.replace(tzinfo=timezone.utc)
        elif timestamp.tzinfo != timezone.utc:
            logger.warning(f"{source}: Non-UTC timezone {timestamp.tzinfo}, converting to UTC")
            return timestamp.astimezone(timezone.utc)
        return timestamp
    
    def _load_volume_profiles(self) -> Dict:
        """Load historical volume profiles for relative volume calculation"""
        # In production, load from database or file
        # For now, create synthetic profiles
        profiles = {}
        
        # Create time-based volume profile based on current DST status
        times = []
        if self.is_edt:
            current = datetime.now(timezone.utc).replace(hour=13, minute=30, second=0, microsecond=0)
            end = datetime.now(timezone.utc).replace(hour=20, minute=0, second=0, microsecond=0)
        else:
            current = datetime.now(timezone.utc).replace(hour=14, minute=30, second=0, microsecond=0)
            end = datetime.now(timezone.utc).replace(hour=21, minute=0, second=0, microsecond=0)
        
        while current <= end:
            times.append(current.strftime('%H:%M'))
            current += timedelta(minutes=15)
        
        # Typical volume distribution (U-shaped) - adjusted for DST
        if self.is_edt:
            base_volumes = {
                '13:30': 1.8,  # Opening surge (9:30 AM EDT)
                '13:45': 1.5,
                '14:00': 1.2,
                '14:15': 1.0,
                '14:30': 0.9,
                '14:45': 0.8,
                '15:00': 0.7,
                '15:15': 0.7,
                '15:30': 0.6,
                '15:45': 0.6,
                '16:00': 0.5,  # Lunch lull (12:00 PM EDT)
                '16:15': 0.5,
                '16:30': 0.5,
                '16:45': 0.6,
                '17:00': 0.7,
                '17:15': 0.8,
                '17:30': 0.9,
                '17:45': 1.0,
                '18:00': 1.1,
                '18:15': 1.2,
                '18:30': 1.3,
                '18:45': 1.4,
                '19:00': 1.5,
                '19:15': 1.6,
                '19:30': 1.7,
                '19:45': 1.9,  # Closing surge (3:45 PM EDT)
                '20:00': 2.0   # Close (4:00 PM EDT)
            }
        else:
            base_volumes = {
                '14:30': 1.8,  # Opening surge (9:30 AM EST)
                '14:45': 1.5,
                '15:00': 1.2,
                '15:15': 1.0,
                '15:30': 0.9,
                '15:45': 0.8,
                '16:00': 0.7,
                '16:15': 0.7,
                '16:30': 0.6,
                '16:45': 0.6,
                '17:00': 0.5,  # Lunch lull (12:00 PM EST)
                '17:15': 0.5,
                '17:30': 0.5,
                '17:45': 0.6,
                '18:00': 0.7,
                '18:15': 0.8,
                '18:30': 0.9,
                '18:45': 1.0,
                '19:00': 1.1,
                '19:15': 1.2,
                '19:30': 1.3,
                '19:45': 1.4,
                '20:00': 1.5,
                '20:15': 1.6,
                '20:30': 1.7,
                '20:45': 1.9,  # Closing surge (3:45 PM EST)
                '21:00': 2.0   # Close (4:00 PM EST)
            }
        
        return {'default': base_volumes}
    
    def process_trade(self, symbol: str, trade_data: Dict):
        """
        Process incoming trade data and aggregate into 15-minute bars
        
        Args:
            symbol: Ticker symbol
            trade_data: Trade data from websocket
        """
        # Initialize if needed
        if symbol not in self.session_bars:
            self.session_bars[symbol] = deque(maxlen=self.lookback_bars + 10)
            self.current_bar_trades[symbol] = []
            self._initialize_session(symbol)
            
        # Get trade time and ensure UTC
        trade_time = datetime.fromtimestamp(trade_data['timestamp'] / 1000, tz=timezone.utc)
        trade_time = self._validate_timestamp(trade_time, f"Trade-{symbol}")
        
        # Update DST status if date changed
        if self._is_daylight_saving(trade_time) != self.is_edt:
            self.is_edt = self._is_daylight_saving(trade_time)
            self._update_market_hours()
        
        # Check if new session
        if self._is_new_session(symbol, trade_time):
            self._initialize_session(symbol)
        
        # Get 15-minute interval
        minute = trade_time.minute
        rounded_minute = (minute // 15) * 15
        interval_start = trade_time.replace(minute=rounded_minute, second=0, microsecond=0)
        
        # Check if this is a new 15-min interval
        if symbol not in self.current_15min:
            self.current_15min[symbol] = interval_start
            self.current_bar_trades[symbol] = [trade_data]
        elif interval_start > self.current_15min[symbol]:
            # New interval - close previous bar
            self._close_15min_bar(symbol, self.current_15min[symbol])
            self.current_15min[symbol] = interval_start
            self.current_bar_trades[symbol] = [trade_data]
        else:
            # Same interval - add to current bar
            self.current_bar_trades[symbol].append(trade_data)
            
        # Update session running totals
        self._update_session_data(symbol, trade_data)
    
    def _update_market_hours(self):
        """Update market hours based on DST status"""
        if self.is_edt:
            self.session_start_time = datetime_time(13, 30)
            self.session_end_time = datetime_time(20, 0)
            self.market_open_minutes = 13 * 60 + 30
            self.market_close_minutes = 20 * 60
            logger.info("Switched to EDT market hours: 13:30-20:00 UTC")
        else:
            self.session_start_time = datetime_time(14, 30)
            self.session_end_time = datetime_time(21, 0)
            self.market_open_minutes = 14 * 60 + 30
            self.market_close_minutes = 21 * 60
            logger.info("Switched to EST market hours: 14:30-21:00 UTC")
        
        # Reload volume profiles
        self.volume_profiles = self._load_volume_profiles()
    
    def _initialize_session(self, symbol: str):
        """Initialize session data for a symbol"""
        self.session_data[symbol] = {
            'cumulative_volume': 0,
            'cumulative_buy_volume': 0,
            'cumulative_sell_volume': 0,
            'cumulative_volume_price': 0,  # For VWAP
            'session_vwap': 0,
            'session_start': datetime.now(timezone.utc),
            'last_price': 0,
            'last_update': datetime.now(timezone.utc)
        }
    
    def _is_new_session(self, symbol: str, trade_time: datetime) -> bool:
        """Check if this is a new trading session"""
        if symbol not in self.session_data:
            return True
            
        # Simple check - if more than 12 hours since last update
        last_update = self.session_data[symbol].get('last_update', trade_time)
        if (trade_time - last_update).total_seconds() > 43200:  # 12 hours
            return True
            
        # Check if we've crossed into a new trading day (in UTC)
        last_date = last_update.date()
        current_date = trade_time.date()
        if current_date > last_date and trade_time.time() >= self.session_start_time:
            return True
            
        return False
    
    def _update_session_data(self, symbol: str, trade_data: Dict):
        """Update running session totals"""
        session = self.session_data[symbol]
        
        price = trade_data['price']
        size = trade_data['size']
        
        # Simple buy/sell classification
        if session['last_price'] > 0:
            is_buy = price >= session['last_price']
        else:
            is_buy = True
            
        # Update cumulative volumes
        session['cumulative_volume'] += size
        if is_buy:
            session['cumulative_buy_volume'] += size
        else:
            session['cumulative_sell_volume'] += size
            
        # Update VWAP calculation
        session['cumulative_volume_price'] += price * size
        if session['cumulative_volume'] > 0:
            session['session_vwap'] = (session['cumulative_volume_price'] / 
                                     session['cumulative_volume'])
        
        session['last_price'] = price
        session['last_update'] = datetime.fromtimestamp(trade_data['timestamp'] / 1000, tz=timezone.utc)
    
    def _close_15min_bar(self, symbol: str, bar_time: datetime):
        """Close 15-minute bar and generate signal"""
        trades = self.current_bar_trades[symbol]
        
        if not trades:
            return
            
        # Calculate OHLC
        prices = [t['price'] for t in trades]
        open_price = prices[0]
        high_price = max(prices)
        low_price = min(prices)
        close_price = prices[-1]
        
        # Calculate volumes for this bar
        bar_buy_volume = 0
        bar_sell_volume = 0
        bar_volume_price = 0
        
        session = self.session_data[symbol]
        
        for trade in trades:
            size = trade['size']
            price = trade['price']
            
            # Classification
            if len(self.session_bars[symbol]) > 0:
                prev_close = self.session_bars[symbol][-1].close
                is_buy = price >= prev_close
            else:
                is_buy = price >= open_price
                
            if is_buy:
                bar_buy_volume += size
            else:
                bar_sell_volume += size
                
            bar_volume_price += price * size
        
        bar_volume = bar_buy_volume + bar_sell_volume
        bar_vwap = bar_volume_price / bar_volume if bar_volume > 0 else close_price
        
        # Create 15-min bar
        bar = SessionBar(
            timestamp=bar_time,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=bar_volume,
            buy_volume=bar_buy_volume,
            sell_volume=bar_sell_volume,
            cumulative_volume=session['cumulative_volume'],
            cumulative_buy_volume=session['cumulative_buy_volume'],
            cumulative_sell_volume=session['cumulative_sell_volume'],
            vwap=bar_vwap,
            session_vwap=session['session_vwap'],
            trade_count=len(trades)
        )
        
        self.session_bars[symbol].append(bar)
        self.bars_processed += 1
        
        print(f"\n✓ 15-min bar completed for {symbol} at {bar_time.strftime('%H:%M UTC')}")
        print(f"  OHLC: {open_price:.2f}/{high_price:.2f}/{low_price:.2f}/{close_price:.2f}")
        print(f"  Volume: {bar_volume:,.0f} ({bar_buy_volume:,.0f} buy / {bar_sell_volume:,.0f} sell)")
        print(f"  Session VWAP: ${session['session_vwap']:.2f}")
        print(f"  Cumulative Delta: {session['cumulative_buy_volume'] - session['cumulative_sell_volume']:+,.0f}")
        
        # Generate signal if we have enough bars
        if len(self.session_bars[symbol]) >= 2:  # Need at least 2 bars
            signal = self._generate_signal(symbol)
            if signal:
                self.signals_generated += 1
                return signal
        else:
            print(f"  Need {2 - len(self.session_bars[symbol])} more bars for signals")
            
        return None
    
    def _generate_signal(self, symbol: str) -> Optional[VolumeSignal]:
        """Generate market context signal"""
        bars = list(self.session_bars[symbol])
        if len(bars) < 2:
            return None
            
        # Calculate metrics
        metrics = self._calculate_metrics(symbol, bars)
        
        # Determine signal
        signal, strength, reason = self._determine_signal(metrics, bars[-1])
        
        return VolumeSignal(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            signal=signal,
            strength=strength,
            metrics=self._metrics_to_dict(metrics),
            reason=reason
        )
    
    def _calculate_metrics(self, symbol: str, bars: List[SessionBar]) -> MarketMetrics:
        """Calculate market context metrics"""
        latest_bar = bars[-1]
        session = self.session_data[symbol]
        
        # Price vs VWAP
        vwap = session['session_vwap']
        if vwap > 0:
            price_vs_vwap = ((latest_bar.close - vwap) / vwap) * 100
        else:
            price_vs_vwap = 0
            
        # Cumulative delta
        cumulative_delta = (session['cumulative_buy_volume'] - 
                          session['cumulative_sell_volume'])
        
        # Relative volume
        current_time = latest_bar.timestamp.strftime('%H:%M')
        typical_volume_mult = self.volume_profiles.get('default', {}).get(current_time, 1.0)
        
        # Estimate expected volume (simplified)
        if len(bars) > 4:
            avg_bar_volume = np.mean([b.volume for b in bars[-4:]])
            expected_volume = avg_bar_volume * typical_volume_mult
            relative_volume = latest_bar.volume / expected_volume if expected_volume > 0 else 1.0
        else:
            relative_volume = 1.0
            
        # Volume trend
        if len(bars) >= 3:
            recent_volumes = [b.volume for b in bars[-3:]]
            if recent_volumes[-1] > recent_volumes[0] * 1.2:
                volume_trend = 'accelerating'
            elif recent_volumes[-1] < recent_volumes[0] * 0.8:
                volume_trend = 'declining'
            else:
                volume_trend = 'steady'
        else:
            volume_trend = 'steady'
            
        # Market phase (in UTC)
        bar_time = latest_bar.timestamp
        hour = bar_time.hour
        minute = bar_time.minute
        time_minutes = hour * 60 + minute  # Convert to minutes since midnight
        
        # Check if within market hours
        if time_minutes < self.market_open_minutes or time_minutes >= self.market_close_minutes:
            market_phase = 'after_hours'
        else:
            # Calculate minutes since market open
            minutes_since_open = time_minutes - self.market_open_minutes
            
            if minutes_since_open < 15:
                market_phase = 'opening'
            elif minutes_since_open < 90:  # First 1.5 hours
                market_phase = 'morning'
            elif minutes_since_open < 270:  # Until 2 PM ET
                market_phase = 'midday'
            elif minutes_since_open < 360:  # Until 3:30 PM ET
                market_phase = 'afternoon'
            else:
                market_phase = 'closing'
            
        # Institutional activity (large trades)
        # Simplified - in reality would track actual large trades
        if latest_bar.volume > 0:
            large_volume_pct = 30.0  # Placeholder
        else:
            large_volume_pct = 0
            
        # Trend strength
        if len(bars) >= 3:
            closes = [b.close for b in bars[-3:]]
            vwaps = [b.session_vwap for b in bars[-3:]]
            
            # Consistency of price above/below VWAP
            above_vwap_count = sum(1 for c, v in zip(closes, vwaps) if c > v)
            consistency = above_vwap_count / len(closes)
            
            # Price momentum
            price_change = (closes[-1] - closes[0]) / closes[0] * 100
            
            trend_strength = min(100, abs(price_change) * 20 * consistency)
        else:
            trend_strength = 50
            
        # Market regime
        if abs(price_vs_vwap) > 1.0 and trend_strength > 70:
            regime = 'trending'
        elif relative_volume > 1.5:
            regime = 'volatile'
        else:
            regime = 'ranging'
            
        return MarketMetrics(
            price_vs_vwap=price_vs_vwap,
            cumulative_delta=cumulative_delta,
            relative_volume=relative_volume,
            volume_trend=volume_trend,
            market_phase=market_phase,
            institutional_activity=large_volume_pct,
            trend_strength=trend_strength,
            regime=regime
        )
    
    def _metrics_to_dict(self, metrics: MarketMetrics) -> Dict:
        """Convert metrics to dictionary"""
        return {
            'price_vs_vwap': metrics.price_vs_vwap,
            'cumulative_delta': metrics.cumulative_delta,
            'relative_volume': metrics.relative_volume,
            'volume_trend': metrics.volume_trend,
            'market_phase': metrics.market_phase,
            'institutional_activity': metrics.institutional_activity,
            'trend_strength': metrics.trend_strength,
            'regime': metrics.regime
        }
    
    def _determine_signal(self, metrics: MarketMetrics, 
                         latest_bar: SessionBar) -> Tuple[str, float, str]:
        """Determine market context signal"""
        reasons = []
        
        # Strong bullish context
        if (metrics.price_vs_vwap > self.vwap_deviation_threshold and
            metrics.cumulative_delta > self.delta_imbalance_threshold and
            metrics.relative_volume > 1.0):
            signal = 'BULLISH'
            strength = min(100, metrics.trend_strength * 1.2)
            reasons.append(f"Price {metrics.price_vs_vwap:+.2f}% above VWAP")
            reasons.append(f"Strong buying delta +{metrics.cumulative_delta:,.0f}")
            
        # Strong bearish context
        elif (metrics.price_vs_vwap < -self.vwap_deviation_threshold and
              metrics.cumulative_delta < -self.delta_imbalance_threshold and
              metrics.relative_volume > 1.0):
            signal = 'BEARISH'
            strength = min(100, metrics.trend_strength * 1.2)
            reasons.append(f"Price {metrics.price_vs_vwap:.2f}% below VWAP")
            reasons.append(f"Strong selling delta {metrics.cumulative_delta:,.0f}")
            
        # Moderate bullish
        elif metrics.price_vs_vwap > 0 and metrics.cumulative_delta > 0:
            signal = 'BULLISH'
            strength = metrics.trend_strength * 0.8
            reasons.append(f"Above VWAP by {metrics.price_vs_vwap:+.2f}%")
            
        # Moderate bearish
        elif metrics.price_vs_vwap < 0 and metrics.cumulative_delta < 0:
            signal = 'BEARISH'
            strength = metrics.trend_strength * 0.8
            reasons.append(f"Below VWAP by {metrics.price_vs_vwap:.2f}%")
            
        # Neutral
        else:
            signal = 'NEUTRAL'
            strength = 30
            if metrics.regime == 'ranging':
                reasons.append("Ranging near VWAP")
            else:
                reasons.append("Mixed market signals")
        
        # Add context
        reasons.append(f"{metrics.regime.title()} regime")
        
        if metrics.relative_volume > self.volume_surge_threshold:
            reasons.append(f"High volume {metrics.relative_volume:.1f}x")
        elif metrics.relative_volume < 0.5:
            reasons.append("Low volume")
            strength *= 0.7
            
        if metrics.market_phase in ['opening', 'closing']:
            reasons.append(f"{metrics.market_phase.title()} volatility")
            
        reason = " | ".join(reasons)
        return signal, strength, reason
    
    def get_current_analysis(self, symbol: str) -> Optional[VolumeSignal]:
        """Get current market context analysis"""
        if symbol not in self.session_bars or len(self.session_bars[symbol]) < 2:
            return None
            
        return self._generate_signal(symbol)
    
    def get_session_stats(self, symbol: str) -> Optional[Dict]:
        """Get session statistics"""
        if symbol not in self.session_data:
            return None
            
        session = self.session_data[symbol]
        return {
            'session_vwap': session['session_vwap'],
            'cumulative_volume': session['cumulative_volume'],
            'cumulative_delta': (session['cumulative_buy_volume'] - 
                               session['cumulative_sell_volume']),
            'delta_percentage': ((session['cumulative_buy_volume'] / 
                                session['cumulative_volume'] * 100)
                               if session['cumulative_volume'] > 0 else 50)
        }
    
    def get_statistics(self) -> Dict:
        """Get analyzer statistics"""
        return {
            'bars_processed': self.bars_processed,
            'signals_generated': self.signals_generated,
            'active_symbols': list(self.session_bars.keys()),
            'bar_counts': {symbol: len(bars) for symbol, bars in self.session_bars.items()},
            'market_hours': f"{self.session_start_time} - {self.session_end_time} UTC",
            'dst_status': 'EDT' if self.is_edt else 'EST'
        }


# ============= TEST FUNCTION =============
async def test_market_context():
    """Test market context analyzer with real-time websocket data"""
    import sys
    import os
    
    # Add parent directories to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    modules_dir = os.path.dirname(os.path.dirname(current_dir))
    vega_root = os.path.dirname(modules_dir)
    if vega_root not in sys.path:
        sys.path.insert(0, vega_root)
    
    from polygon import PolygonWebSocketClient
    
    print("=== MARKET CONTEXT ANALYZER TEST ===")
    print("Analyzing 15-minute market structure and regime")
    print(f"Current UTC time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    # Determine current market hours
    now = datetime.now(timezone.utc)
    analyzer_temp = MarketContext()  # Create temp instance to check DST
    if analyzer_temp.is_edt:
        print("Market hours: 13:30-20:00 UTC (9:30 AM - 4:00 PM EDT)")
    else:
        print("Market hours: 14:30-21:00 UTC (9:30 AM - 4:00 PM EST)")
    print()
    
    # Test configuration
    TEST_SYMBOLS = ['SPY', 'QQQ', 'IWM']  # Major indices for market context
    TEST_DURATION = 1200  # 20 minutes to get at least one 15-min bar
    
    # Create analyzer
    analyzer = MarketContext(
        lookback_bars=10,
        vwap_deviation_threshold=0.5,
        volume_surge_threshold=1.5,
        delta_imbalance_threshold=100000
    )
    
    # Track signals
    signal_history = []
    last_bar_times = {}
    
    async def handle_trade(data: Dict):
        """Process trade data"""
        symbol = data['symbol']
        
        # Process trade
        analyzer.process_trade(symbol, data)
        
        # Check for completed bars and signals
        if symbol in analyzer.session_bars and analyzer.session_bars[symbol]:
            latest_bar = analyzer.session_bars[symbol][-1]
            bar_time = latest_bar.timestamp
            
            # Check if this is a new bar
            if symbol not in last_bar_times or bar_time > last_bar_times[symbol]:
                last_bar_times[symbol] = bar_time
                
                # Get signal
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
                    
                    print(f"\n{color_code}{'='*70}\033[0m")
                    print(f"{emoji} {signal.symbol} - Signal: {signal.signal} (Strength: {signal.strength:.0f}%)")
                    print(f"Time: {bar_time.strftime('%H:%M UTC')} 15-min bar")
                    print(f"Signal Generated: {signal.timestamp.strftime('%H:%M:%S UTC')}")
                    print(f"Reason: {signal.reason}")
                    
                    # Display metrics
                    if signal.metrics:
                        m = signal.metrics
                        print(f"\nMarket Context:")
                        print(f"  • Price vs VWAP: {m['price_vs_vwap']:+.2f}%")
                        print(f"  • Cumulative Delta: {m['cumulative_delta']:+,.0f}")
                        print(f"  • Relative Volume: {m['relative_volume']:.1f}x")
                        print(f"  • Volume Trend: {m['volume_trend']}")
                        print(f"  • Market Phase: {m['market_phase']}")
                        print(f"  • Regime: {m['regime'].upper()}")
                        print(f"  • Trend Strength: {m['trend_strength']:.0f}")
                    
                    # Session stats
                    stats = analyzer.get_session_stats(symbol)
                    if stats:
                        print(f"\nSession Statistics:")
                        print(f"  • Session VWAP: ${stats['session_vwap']:.2f}")
                        print(f"  • Total Volume: {stats['cumulative_volume']:,.0f}")
                        print(f"  • Delta %: {stats['delta_percentage']:.1f}% buy")
    
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
        
        current_time = datetime.now(timezone.utc)
        print(f"\n⏰ Running for {TEST_DURATION} seconds...")
        print(f"Current UTC time: {current_time.strftime('%H:%M:%S UTC')}")
        
        # Check if market is open
        market_hour = current_time.hour
        market_minute = current_time.minute
        time_minutes = market_hour * 60 + market_minute
        
        if time_minutes < analyzer.market_open_minutes or time_minutes >= analyzer.market_close_minutes:
            print(f"\n⚠️  Market is closed (outside {analyzer.session_start_time} - {analyzer.session_end_time} UTC)")
            print("   During market hours, 15-min bars complete at :00, :15, :30, :45\n")
        else:
            print("✅ Market is open - expecting live data\n")
        
        # Create listen task
        listen_task = asyncio.create_task(ws_client.listen())
        
        # Run for specified duration
        start_time = time.time()
        last_stats_time = start_time
        
        while time.time() - start_time < TEST_DURATION:
            await asyncio.sleep(1)
            
            # Print stats every 5 minutes
            if time.time() - last_stats_time >= 300:
                stats = analyzer.get_statistics()
                print(f"\n📊 Stats: {stats['bars_processed']} 15-min bars completed")
                print(f"   Market Hours: {stats['market_hours']} ({stats['dst_status']})")
                
                # Show session stats for each symbol
                for symbol in TEST_SYMBOLS:
                    session_stats = analyzer.get_session_stats(symbol)
                    if session_stats:
                        print(f"{symbol}: VWAP ${session_stats['session_vwap']:.2f}, "
                              f"Delta {session_stats['cumulative_delta']:+,.0f}")
                
                last_stats_time = time.time()
            
            # Show countdown
            remaining = TEST_DURATION - (time.time() - start_time)
            print(f"\r⏳ Time remaining: {remaining:.0f}s ", end='', flush=True)
        
        print("\n\n🏁 Test complete!")
        
        # Final summary
        stats = analyzer.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"  • 15-min bars processed: {stats['bars_processed']}")
        print(f"  • Signals generated: {stats['signals_generated']}")
        print(f"  • Symbols tracked: {', '.join(stats['active_symbols'])}")
        print(f"  • Market Hours: {stats['market_hours']} ({stats['dst_status']})")
        
        # Signal summary
        if signal_history:
            print(f"\n📈 Signal Summary:")
            bull_signals = [s for s in signal_history if s.signal == 'BULLISH']
            bear_signals = [s for s in signal_history if s.signal == 'BEARISH']
            neutral_signals = [s for s in signal_history if s.signal == 'NEUTRAL']
            
            print(f"  • Bullish: {len(bull_signals)}")
            print(f"  • Bearish: {len(bear_signals)}")
            print(f"  • Neutral: {len(neutral_signals)}")
        
        # Final session summary
        print(f"\n📊 Final Session Summary:")
        for symbol in TEST_SYMBOLS:
            session_stats = analyzer.get_session_stats(symbol)
            if session_stats:
                print(f"\n{symbol}:")
                print(f"  • VWAP: ${session_stats['session_vwap']:.2f}")
                print(f"  • Volume: {session_stats['cumulative_volume']:,.0f}")
                print(f"  • Delta: {session_stats['cumulative_delta']:+,.0f} "
                      f"({session_stats['delta_percentage']:.1f}% buy)")
        
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
    print(f"Starting Market Context Analyzer at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("This module identifies market regime using 15-minute volume patterns")
    print("All timestamps are in UTC")
    print("\nNote: Market hours automatically adjust for Daylight Saving Time:")
    print("  • EDT (March-November): 13:30-20:00 UTC")
    print("  • EST (November-March): 14:30-21:00 UTC")
    print("\nDuring market hours, this tracks:")
    print("  • VWAP and price positioning")
    print("  • Cumulative volume delta")
    print("  • Market regime (trending/ranging/volatile)")
    print("  • Relative volume vs typical patterns")
    print("  • UTC timestamp enforcement\n")
    
    asyncio.run(test_market_context())