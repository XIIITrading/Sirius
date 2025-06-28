# modules/calculations/order_flow/micro_momentum.py
"""
Module: NBBO-Adapted Microstructure Momentum
Purpose: Detect market maker behavior and order flow using NBBO quotes + trades
Features: Quote pressure, spread dynamics, regeneration tracking, trade-quote interaction
Output: Bull/Bear signals (-2 to +2) with confidence metrics
Time Handling: All timestamps in UTC
Data Source: Polygon WebSocket NBBO quotes + trades
"""

import asyncio
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Deque, Tuple, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
import logging
import time
import os
import json

# Enforce UTC for all operations
os.environ['TZ'] = 'UTC'
if hasattr(time, 'tzset'):
    time.tzset()

# Configure logging with UTC
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s UTC - %(name)s - %(levelname)s - %(message)s'
)
logging.Formatter.converter = time.gmtime  # Force UTC in logs
logger = logging.getLogger(__name__)


@dataclass
class NBBOQuote:
    """National Best Bid and Offer data"""
    timestamp: datetime
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    spread: float
    mid_price: float
    quote_id: Optional[str] = None
    
    def __post_init__(self):
        """Calculate derived fields"""
        self.spread = self.ask_price - self.bid_price
        self.mid_price = (self.bid_price + self.ask_price) / 2


@dataclass
class QuoteChange:
    """Track changes in NBBO"""
    timestamp: datetime
    bid_price_change: float
    bid_size_change: float
    ask_price_change: float
    ask_size_change: float
    spread_change: float
    is_bid_aggressive: bool  # Bid moving up
    is_ask_aggressive: bool  # Ask moving down


@dataclass
class QuoteDepletion:
    """Track quote size depletion events"""
    timestamp: datetime
    side: str  # 'bid' or 'ask'
    initial_size: float
    depleted_size: float
    depletion_pct: float
    regeneration_time: Optional[float] = None  # Seconds to regenerate


@dataclass
class MicrostructureMetrics:
    """Comprehensive microstructure metrics"""
    # Quote Pressure
    bid_size: float
    ask_size: float
    size_imbalance: float  # -1 to +1
    size_ratio: float  # bid/ask ratio
    
    # Spread Dynamics
    current_spread: float
    spread_bps: float  # Spread in basis points
    spread_volatility: float
    is_wide_spread: bool
    
    # Quote Stability
    quote_lifetime: float  # Seconds
    quote_changes_per_min: float
    bid_stability_score: float  # 0-1
    ask_stability_score: float  # 0-1
    
    # Regeneration
    bid_regen_speed: float  # Size per second
    ask_regen_speed: float
    regen_imbalance: float  # -1 to +1
    
    # Trade-Quote Interaction
    trade_to_quote_ratio: float
    hidden_liquidity_score: float  # 0-1
    quote_hit_rate: float  # Trades hitting quotes
    
    # Quote Quality
    quote_stuffing_score: float  # 0-1
    spoofing_probability: float  # 0-1
    
    # Overall Assessment
    microstructure_quality: float  # 0-1
    signal_confidence: float  # 0-1


@dataclass
class MicroMomentumSignal:
    """Microstructure momentum signal"""
    symbol: str
    timestamp: datetime
    bull_score: int  # 0-2
    bear_score: int  # 0-2
    net_signal: int  # -2 to +2
    confidence: float  # 0-1
    metrics: MicrostructureMetrics
    reasons: List[str]
    warnings: List[str]


class NBBOState:
    """Track current and historical NBBO state"""
    
    def __init__(self, history_size: int = 1000):
        self.current_quote: Optional[NBBOQuote] = None
        self.quote_history: Deque[NBBOQuote] = deque(maxlen=history_size)
        self.quote_changes: Deque[QuoteChange] = deque(maxlen=history_size)
        self.last_update_time: Optional[datetime] = None
        self.quote_lifetime_start: Optional[datetime] = None
        
    def update(self, quote_data: Dict) -> QuoteChange:
        """Update NBBO state and return change metrics"""
        # Parse quote data - ensure UTC
        timestamp = datetime.fromtimestamp(quote_data['timestamp'] / 1000, tz=timezone.utc)
        
        new_quote = NBBOQuote(
            timestamp=timestamp,
            bid_price=quote_data['bid_price'],
            bid_size=quote_data['bid_size'],
            ask_price=quote_data['ask_price'],
            ask_size=quote_data['ask_size'],
            spread=0,  # Calculated in post_init
            mid_price=0  # Calculated in post_init
        )
        
        # Calculate changes if we have previous quote
        change = None
        if self.current_quote:
            change = QuoteChange(
                timestamp=timestamp,
                bid_price_change=new_quote.bid_price - self.current_quote.bid_price,
                bid_size_change=new_quote.bid_size - self.current_quote.bid_size,
                ask_price_change=new_quote.ask_price - self.current_quote.ask_price,
                ask_size_change=new_quote.ask_size - self.current_quote.ask_size,
                spread_change=new_quote.spread - self.current_quote.spread,
                is_bid_aggressive=new_quote.bid_price > self.current_quote.bid_price,
                is_ask_aggressive=new_quote.ask_price < self.current_quote.ask_price
            )
            self.quote_changes.append(change)
            
            # Check if quote prices changed (new quote lifetime)
            if (new_quote.bid_price != self.current_quote.bid_price or 
                new_quote.ask_price != self.current_quote.ask_price):
                self.quote_lifetime_start = timestamp
        else:
            self.quote_lifetime_start = timestamp
            
        # Update state
        self.quote_history.append(new_quote)
        self.current_quote = new_quote
        self.last_update_time = timestamp
        
        return change
    
    def get_quote_lifetime(self) -> float:
        """Get current quote lifetime in seconds"""
        if not self.quote_lifetime_start or not self.last_update_time:
            return 0.0
        return (self.last_update_time - self.quote_lifetime_start).total_seconds()
    
    def get_spread_stats(self, window_seconds: float = 60) -> Dict:
        """Calculate spread statistics over time window"""
        if not self.quote_history:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
            
        cutoff_time = self.last_update_time - timedelta(seconds=window_seconds)
        recent_quotes = [q for q in self.quote_history 
                        if q.timestamp >= cutoff_time]
        
        if not recent_quotes:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
            
        spreads = [q.spread for q in recent_quotes]
        return {
            'mean': np.mean(spreads),
            'std': np.std(spreads),
            'min': np.min(spreads),
            'max': np.max(spreads)
        }


class QuotePressureAnalyzer:
    """Analyze bid/ask pressure from NBBO"""
    
    def __init__(self):
        self.size_history: Deque[Tuple[datetime, float, float]] = deque(maxlen=100)
        self.pressure_history: Deque[Tuple[datetime, float]] = deque(maxlen=100)
        
    def update(self, quote: NBBOQuote) -> Dict:
        """Update pressure metrics"""
        # Store size history
        self.size_history.append((quote.timestamp, quote.bid_size, quote.ask_size))
        
        # Calculate size imbalance
        total_size = quote.bid_size + quote.ask_size
        if total_size > 0:
            size_imbalance = (quote.bid_size - quote.ask_size) / total_size
            size_ratio = quote.bid_size / quote.ask_size if quote.ask_size > 0 else 10
        else:
            size_imbalance = 0
            size_ratio = 1
            
        # Store pressure
        self.pressure_history.append((quote.timestamp, size_imbalance))
        
        # Calculate moving averages
        recent_pressures = [p for _, p in self.pressure_history[-20:]]
        avg_pressure = np.mean(recent_pressures) if recent_pressures else 0
        
        # Detect pressure shifts
        if len(self.pressure_history) >= 10:
            old_pressures = [p for _, p in list(self.pressure_history)[-20:-10]]
            new_pressures = [p for _, p in list(self.pressure_history)[-10:]]
            
            old_avg = np.mean(old_pressures)
            new_avg = np.mean(new_pressures)
            pressure_shift = new_avg - old_avg
        else:
            pressure_shift = 0
            
        return {
            'size_imbalance': size_imbalance,
            'size_ratio': size_ratio,
            'avg_pressure': avg_pressure,
            'pressure_shift': pressure_shift,
            'bid_size': quote.bid_size,
            'ask_size': quote.ask_size
        }


class SpreadDynamicsTracker:
    """Track spread behavior and dynamics"""
    
    def __init__(self):
        self.spread_history: Deque[Tuple[datetime, float, float]] = deque(maxlen=500)
        self.typical_spread: Optional[float] = None
        self.spread_percentiles: Dict[int, float] = {}
        
    def update(self, quote: NBBOQuote) -> Dict:
        """Update spread metrics"""
        # Calculate spread in basis points
        spread_bps = (quote.spread / quote.mid_price) * 10000 if quote.mid_price > 0 else 0
        
        # Store history
        self.spread_history.append((quote.timestamp, quote.spread, spread_bps))
        
        # Update typical spread (median of last 100)
        if len(self.spread_history) >= 20:
            recent_spreads = [s for _, s, _ in list(self.spread_history)[-100:]]
            self.typical_spread = np.median(recent_spreads)
            
            # Calculate percentiles
            self.spread_percentiles = {
                25: np.percentile(recent_spreads, 25),
                50: np.percentile(recent_spreads, 50),
                75: np.percentile(recent_spreads, 75),
                90: np.percentile(recent_spreads, 90)
            }
        
        # Determine if spread is wide
        is_wide = False
        if self.typical_spread:
            is_wide = quote.spread > self.spread_percentiles.get(75, self.typical_spread * 1.5)
            
        # Calculate spread volatility
        if len(self.spread_history) >= 10:
            recent_spreads = [s for _, s, _ in list(self.spread_history)[-20:]]
            spread_volatility = np.std(recent_spreads) / np.mean(recent_spreads) if np.mean(recent_spreads) > 0 else 0
        else:
            spread_volatility = 0
            
        return {
            'spread': quote.spread,
            'spread_bps': spread_bps,
            'is_wide': is_wide,
            'spread_volatility': spread_volatility,
            'typical_spread': self.typical_spread or quote.spread,
            'percentiles': self.spread_percentiles
        }


class QuoteRegenerationTracker:
    """Track how quickly quotes regenerate after depletion"""
    
    def __init__(self):
        self.depletion_events: Deque[QuoteDepletion] = deque(maxlen=100)
        self.size_timeline: defaultdict = defaultdict(lambda: deque(maxlen=50))
        self.regen_speeds: Dict[str, Deque[float]] = {
            'bid': deque(maxlen=20),
            'ask': deque(maxlen=20)
        }
        
    def check_depletion(self, change: QuoteChange, threshold_pct: float = 0.5):
        """Check for significant size depletion"""
        if not change:
            return
            
        # Check bid depletion
        if change.bid_size_change < 0:
            depletion_pct = abs(change.bid_size_change) / (abs(change.bid_size_change) + change.bid_size_change)
            if depletion_pct >= threshold_pct:
                self.depletion_events.append(QuoteDepletion(
                    timestamp=change.timestamp,
                    side='bid',
                    initial_size=abs(change.bid_size_change) / depletion_pct,
                    depleted_size=abs(change.bid_size_change),
                    depletion_pct=depletion_pct
                ))
                
        # Check ask depletion
        if change.ask_size_change < 0:
            depletion_pct = abs(change.ask_size_change) / (abs(change.ask_size_change) + change.ask_size_change)
            if depletion_pct >= threshold_pct:
                self.depletion_events.append(QuoteDepletion(
                    timestamp=change.timestamp,
                    side='ask',
                    initial_size=abs(change.ask_size_change) / depletion_pct,
                    depleted_size=abs(change.ask_size_change),
                    depletion_pct=depletion_pct
                ))
    
    def track_regeneration(self, quote: NBBOQuote):
        """Track size regeneration"""
        # Store current sizes
        self.size_timeline['bid'].append((quote.timestamp, quote.bid_size))
        self.size_timeline['ask'].append((quote.timestamp, quote.ask_size))
        
        # Check for regeneration after depletion events
        for event in self.depletion_events:
            if event.regeneration_time is None:
                # Check if size has recovered to 50% of original
                current_size = quote.bid_size if event.side == 'bid' else quote.ask_size
                if current_size >= event.initial_size * 0.5:
                    event.regeneration_time = (quote.timestamp - event.timestamp).total_seconds()
                    self.regen_speeds[event.side].append(
                        current_size / event.regeneration_time
                    )
    
    def get_regeneration_metrics(self) -> Dict:
        """Calculate regeneration speed metrics"""
        bid_speed = np.mean(self.regen_speeds['bid']) if self.regen_speeds['bid'] else 0
        ask_speed = np.mean(self.regen_speeds['ask']) if self.regen_speeds['ask'] else 0
        
        # Calculate imbalance (-1 to +1, positive = bids regenerate faster)
        if bid_speed + ask_speed > 0:
            regen_imbalance = (bid_speed - ask_speed) / (bid_speed + ask_speed)
        else:
            regen_imbalance = 0
            
        return {
            'bid_regen_speed': bid_speed,
            'ask_regen_speed': ask_speed,
            'regen_imbalance': regen_imbalance,
            'recent_depletions': len([e for e in self.depletion_events 
                                    if (datetime.now(timezone.utc) - e.timestamp).total_seconds() < 60])
        }


class TradeQuoteAnalyzer:
    """Analyze interaction between trades and quotes"""
    
    def __init__(self):
        self.trade_history: Deque[Dict] = deque(maxlen=100)
        self.quote_hit_history: Deque[Tuple[datetime, str, float]] = deque(maxlen=100)
        self.hidden_liquidity_events: Deque[Tuple[datetime, float]] = deque(maxlen=50)
        
    def process_trade(self, trade: Dict, current_quote: NBBOQuote) -> Dict:
        """Analyze trade relative to current quote"""
        if not current_quote:
            return {}
            
        timestamp = datetime.fromtimestamp(trade['timestamp'] / 1000, tz=timezone.utc)
        price = trade['price']
        size = trade['size']
        
        # Classify trade
        if price >= current_quote.ask_price:
            side = 'buy'
            quote_size = current_quote.ask_size
        elif price <= current_quote.bid_price:
            side = 'sell'
            quote_size = current_quote.bid_size
        else:
            side = 'mid'
            quote_size = (current_quote.bid_size + current_quote.ask_size) / 2
            
        # Check for hidden liquidity (trade larger than displayed size)
        if size > quote_size * 1.2:  # 20% buffer
            hidden_ratio = size / quote_size if quote_size > 0 else 0
            self.hidden_liquidity_events.append((timestamp, hidden_ratio))
            
        # Track quote hits
        if side in ['buy', 'sell']:
            self.quote_hit_history.append((timestamp, side, size))
            
        # Store trade
        self.trade_history.append({
            'timestamp': timestamp,
            'price': price,
            'size': size,
            'side': side,
            'quote_size': quote_size
        })
        
        return {
            'side': side,
            'hidden_liquidity': size > quote_size * 1.2,
            'size_vs_quote': size / quote_size if quote_size > 0 else 0
        }
    
    def get_trade_quote_metrics(self) -> Dict:
        """Calculate trade-quote interaction metrics"""
        if not self.trade_history:
            return {
                'trade_to_quote_ratio': 0,
                'hidden_liquidity_score': 0,
                'quote_hit_rate': 0,
                'buy_hit_rate': 0,
                'sell_hit_rate': 0
            }
            
        # Trade to quote size ratio
        trade_sizes = [t['size'] for t in self.trade_history]
        quote_sizes = [t['quote_size'] for t in self.trade_history if t['quote_size'] > 0]
        
        if quote_sizes:
            trade_to_quote_ratio = np.mean(trade_sizes) / np.mean(quote_sizes)
        else:
            trade_to_quote_ratio = 0
            
        # Hidden liquidity score (frequency of hidden liquidity events)
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=60)
        recent_hidden = [h for t, h in self.hidden_liquidity_events if t >= cutoff_time]
        hidden_liquidity_score = min(1.0, len(recent_hidden) / 10)  # Normalize to 0-1
        
        # Quote hit rates
        recent_hits = [h for h in self.quote_hit_history 
                      if h[0] >= datetime.now(timezone.utc) - timedelta(seconds=60)]
        
        total_trades = len([t for t in self.trade_history 
                          if t['timestamp'] >= datetime.now(timezone.utc) - timedelta(seconds=60)])
        
        if total_trades > 0:
            quote_hit_rate = len(recent_hits) / total_trades
            buy_hits = len([h for h in recent_hits if h[1] == 'buy'])
            sell_hits = len([h for h in recent_hits if h[1] == 'sell'])
            buy_hit_rate = buy_hits / total_trades
            sell_hit_rate = sell_hits / total_trades
        else:
            quote_hit_rate = buy_hit_rate = sell_hit_rate = 0
            
        return {
            'trade_to_quote_ratio': trade_to_quote_ratio,
            'hidden_liquidity_score': hidden_liquidity_score,
            'quote_hit_rate': quote_hit_rate,
            'buy_hit_rate': buy_hit_rate,
            'sell_hit_rate': sell_hit_rate
        }


class QuoteStabilityAnalyzer:
    """Analyze quote stability and potential spoofing"""
    
    def __init__(self):
        self.update_times: Deque[datetime] = deque(maxlen=100)
        self.price_changes: Deque[Tuple[datetime, bool]] = deque(maxlen=100)
        self.size_oscillations: defaultdict = defaultdict(lambda: deque(maxlen=20))
        
    def process_update(self, change: QuoteChange):
        """Process quote update for stability analysis"""
        if not change:
            return
            
        # Track update frequency
        self.update_times.append(change.timestamp)
        
        # Track price changes
        price_changed = (change.bid_price_change != 0 or change.ask_price_change != 0)
        self.price_changes.append((change.timestamp, price_changed))
        
        # Track size oscillations
        if abs(change.bid_size_change) > 0:
            self.size_oscillations['bid'].append(change.bid_size_change)
        if abs(change.ask_size_change) > 0:
            self.size_oscillations['ask'].append(change.ask_size_change)
    
    def get_stability_metrics(self) -> Dict:
        """Calculate stability and spoofing metrics"""
        current_time = datetime.now(timezone.utc)
        
        # Quote update frequency
        recent_updates = [t for t in self.update_times 
                         if (current_time - t).total_seconds() < 60]
        updates_per_minute = len(recent_updates)
        
        # Price stability (percentage of updates that change price)
        recent_price_changes = [pc for t, pc in self.price_changes 
                               if (current_time - t).total_seconds() < 60]
        
        if recent_price_changes:
            price_change_rate = sum(recent_price_changes) / len(recent_price_changes)
            price_stability = 1 - price_change_rate
        else:
            price_stability = 1
            
        # Size oscillation detection (potential spoofing)
        bid_oscillations = self._detect_oscillations(self.size_oscillations['bid'])
        ask_oscillations = self._detect_oscillations(self.size_oscillations['ask'])
        
        # Quote stuffing score (high update rate with low price changes)
        if updates_per_minute > 30 and price_stability > 0.8:
            stuffing_score = min(1.0, (updates_per_minute - 30) / 30)
        else:
            stuffing_score = 0
            
        # Spoofing probability
        spoofing_indicators = 0
        if bid_oscillations > 3:
            spoofing_indicators += 1
        if ask_oscillations > 3:
            spoofing_indicators += 1
        if stuffing_score > 0.5:
            spoofing_indicators += 1
            
        spoofing_probability = min(1.0, spoofing_indicators / 3)
        
        return {
            'updates_per_minute': updates_per_minute,
            'price_stability': price_stability,
            'bid_stability': 1 - (bid_oscillations / 10) if bid_oscillations < 10 else 0,
            'ask_stability': 1 - (ask_oscillations / 10) if ask_oscillations < 10 else 0,
            'stuffing_score': stuffing_score,
            'spoofing_probability': spoofing_probability
        }
    
    def _detect_oscillations(self, size_changes: Deque[float]) -> int:
        """Count directional changes in size updates"""
        if len(size_changes) < 3:
            return 0
            
        oscillations = 0
        for i in range(1, len(size_changes)):
            if size_changes[i] * size_changes[i-1] < 0:  # Sign change
                oscillations += 1
                
        return oscillations


class MicrostructureMomentum:
    """Main orchestrator for NBBO-based microstructure momentum"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize microstructure momentum analyzer
        
        Args:
            config: Configuration parameters
        """
        # Default configuration
        self.config = config or {
            'nbbo_history_size': 1000,
            'signal_threshold': 0.6,  # For bull/bear signals
            'min_confidence': 0.3,    # Minimum confidence for signals
            'spoofing_penalty': 0.5,  # Confidence reduction for spoofing
            'wide_spread_threshold': 2.0,  # Multiple of typical spread
            'depletion_threshold': 0.5,  # 50% size reduction
            'premarket_adjustment': 1.5  # Threshold multiplier for pre-market
        }
        
        # Initialize components
        self.nbbo_state = NBBOState(history_size=self.config['nbbo_history_size'])
        self.pressure_analyzer = QuotePressureAnalyzer()
        self.spread_tracker = SpreadDynamicsTracker()
        self.regen_tracker = QuoteRegenerationTracker()
        self.trade_quote_analyzer = TradeQuoteAnalyzer()
        self.stability_analyzer = QuoteStabilityAnalyzer()
        
        # Signal history
        self.signal_history: Deque[MicroMomentumSignal] = deque(maxlen=100)
        
        # Performance tracking
        self.quotes_processed = 0
        self.trades_processed = 0
        self.signals_generated = 0
        
        logger.info(f"MicrostructureMomentum initialized at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
    def process_quote(self, quote_data: Dict) -> Optional[MicroMomentumSignal]:
        """
        Process NBBO quote update
        
        Args:
            quote_data: Quote data from WebSocket
            
        Returns:
            MicroMomentumSignal if generated, None otherwise
        """
        # Update NBBO state
        change = self.nbbo_state.update(quote_data)
        quote = self.nbbo_state.current_quote
        
        # Update all analyzers
        pressure_metrics = self.pressure_analyzer.update(quote)
        spread_metrics = self.spread_tracker.update(quote)
        
        if change:
            self.regen_tracker.check_depletion(change)
            self.stability_analyzer.process_update(change)
            
        self.regen_tracker.track_regeneration(quote)
        
        # Increment counter
        self.quotes_processed += 1
        
        # Generate signal every N quotes or on significant changes
        if self.quotes_processed % 10 == 0 or self._is_significant_change(change):
            return self._generate_signal()
            
        return None
    
    def process_trade(self, trade_data: Dict) -> Optional[MicroMomentumSignal]:
        """
        Process trade data
        
        Args:
            trade_data: Trade data from WebSocket
            
        Returns:
            MicroMomentumSignal if conditions met
        """
        if not self.nbbo_state.current_quote:
            return None
            
        # Analyze trade
        self.trade_quote_analyzer.process_trade(trade_data, self.nbbo_state.current_quote)
        self.trades_processed += 1
        
        # Generate signal on significant trades
        if self._is_significant_trade(trade_data):
            return self._generate_signal()
            
        return None
    
    def _is_significant_change(self, change: QuoteChange) -> bool:
        """Determine if quote change is significant"""
        if not change:
            return False
            
        # Significant if spread changes substantially
        if abs(change.spread_change) > self.nbbo_state.current_quote.spread * 0.2:
            return True
            
        # Significant if both bid and ask move in same direction (market shift)
        if change.is_bid_aggressive and change.is_ask_aggressive:
            return True
            
        # Significant if large size change
        total_size_change = abs(change.bid_size_change) + abs(change.ask_size_change)
        current_total_size = self.nbbo_state.current_quote.bid_size + self.nbbo_state.current_quote.ask_size
        if current_total_size > 0 and total_size_change / current_total_size > 0.3:
            return True
            
        return False
    
    def _is_significant_trade(self, trade: Dict) -> bool:
        """Determine if trade is significant"""
        if not self.nbbo_state.current_quote:
            return False
            
        # Large relative to quote size
        quote_size = (self.nbbo_state.current_quote.bid_size + 
                     self.nbbo_state.current_quote.ask_size) / 2
        
        return trade['size'] > quote_size * 2
    
    def _generate_signal(self) -> MicroMomentumSignal:
        """Generate comprehensive microstructure signal"""
        # Gather all metrics
        pressure_metrics = self.pressure_analyzer.update(self.nbbo_state.current_quote)
        spread_metrics = self.spread_tracker.update(self.nbbo_state.current_quote)
        regen_metrics = self.regen_tracker.get_regeneration_metrics()
        trade_quote_metrics = self.trade_quote_analyzer.get_trade_quote_metrics()
        stability_metrics = self.stability_analyzer.get_stability_metrics()
        
        # Build comprehensive metrics
        metrics = MicrostructureMetrics(
            # Quote Pressure
            bid_size=pressure_metrics['bid_size'],
            ask_size=pressure_metrics['ask_size'],
            size_imbalance=pressure_metrics['size_imbalance'],
            size_ratio=pressure_metrics['size_ratio'],
            
            # Spread Dynamics
            current_spread=spread_metrics['spread'],
            spread_bps=spread_metrics['spread_bps'],
            spread_volatility=spread_metrics['spread_volatility'],
            is_wide_spread=spread_metrics['is_wide'],
            
            # Quote Stability
            quote_lifetime=self.nbbo_state.get_quote_lifetime(),
            quote_changes_per_min=stability_metrics['updates_per_minute'],
            bid_stability_score=stability_metrics['bid_stability'],
            ask_stability_score=stability_metrics['ask_stability'],
            
            # Regeneration
            bid_regen_speed=regen_metrics['bid_regen_speed'],
            ask_regen_speed=regen_metrics['ask_regen_speed'],
            regen_imbalance=regen_metrics['regen_imbalance'],
            
            # Trade-Quote Interaction
            trade_to_quote_ratio=trade_quote_metrics['trade_to_quote_ratio'],
            hidden_liquidity_score=trade_quote_metrics['hidden_liquidity_score'],
            quote_hit_rate=trade_quote_metrics['quote_hit_rate'],
            
            # Quote Quality
            quote_stuffing_score=stability_metrics['stuffing_score'],
            spoofing_probability=stability_metrics['spoofing_probability'],
            
            # Overall Assessment
            microstructure_quality=self._calculate_quality_score(stability_metrics),
            signal_confidence=0  # Calculated below
        )
        
        # Calculate signals
        bull_score, bear_score, reasons, warnings = self._calculate_scores(
            metrics, pressure_metrics, regen_metrics, stability_metrics
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(metrics, warnings)
        metrics.signal_confidence = confidence
        
        # Create signal
        signal = MicroMomentumSignal(
            symbol=self.nbbo_state.current_quote.mid_price,  # Symbol should be passed separately
            timestamp=datetime.now(timezone.utc),
            bull_score=bull_score,
            bear_score=bear_score,
            net_signal=bull_score - bear_score,
            confidence=confidence,
            metrics=metrics,
            reasons=reasons,
            warnings=warnings
        )
        
        self.signal_history.append(signal)
        self.signals_generated += 1
        
        return signal
    
    def _calculate_scores(self, metrics: MicrostructureMetrics, 
                         pressure_metrics: Dict, regen_metrics: Dict,
                         stability_metrics: Dict) -> Tuple[int, int, List[str], List[str]]:
        """Calculate bull/bear scores and reasons"""
        bull_score = 0
        bear_score = 0
        reasons = []
        warnings = []
        
        # Check for spoofing/stuffing first
        if metrics.spoofing_probability > 0.5:
            warnings.append(f"⚠️ Spoofing detected (prob: {metrics.spoofing_probability:.0%})")
        if metrics.quote_stuffing_score > 0.5:
            warnings.append(f"⚠️ Quote stuffing detected (score: {metrics.quote_stuffing_score:.0%})")
            
        # Quote Pressure Signals
        if metrics.size_imbalance > self.config['signal_threshold']:
            bull_score += 2
            reasons.append(f"Strong bid pressure: {metrics.size_ratio:.1f}:1 bid/ask ratio")
        elif metrics.size_imbalance > self.config['signal_threshold'] * 0.5:
            bull_score += 1
            reasons.append(f"Moderate bid pressure: {metrics.size_imbalance:.0%} imbalance")
        elif metrics.size_imbalance < -self.config['signal_threshold']:
            bear_score += 2
            reasons.append(f"Strong ask pressure: {1/metrics.size_ratio:.1f}:1 ask/bid ratio")
        elif metrics.size_imbalance < -self.config['signal_threshold'] * 0.5:
            bear_score += 1
            reasons.append(f"Moderate ask pressure: {-metrics.size_imbalance:.0%} imbalance")
            
        # Regeneration Signals
        if metrics.regen_imbalance > 0.3:
            bull_score += 1
            reasons.append(f"Bids regenerating faster ({metrics.bid_regen_speed:.0f} vs {metrics.ask_regen_speed:.0f})")
        elif metrics.regen_imbalance < -0.3:
            bear_score += 1
            reasons.append(f"Asks regenerating faster ({metrics.ask_regen_speed:.0f} vs {metrics.bid_regen_speed:.0f})")
            
        # Spread Dynamics
        if metrics.is_wide_spread:
            warnings.append("Wide spread - reduced liquidity")
            # Reduce scores
            bull_score = max(0, bull_score - 1)
            bear_score = max(0, bear_score - 1)
        
        # Hidden Liquidity
        if metrics.hidden_liquidity_score > 0.5:
            reasons.append(f"Hidden liquidity detected (score: {metrics.hidden_liquidity_score:.0%})")
            
        # Quote Hit Asymmetry
        buy_hit_rate = pressure_metrics.get('buy_hit_rate', 0)
        sell_hit_rate = pressure_metrics.get('sell_hit_rate', 0)
        
        if buy_hit_rate > sell_hit_rate * 1.5 and buy_hit_rate > 0.3:
            bull_score += 1
            reasons.append(f"Aggressive buying ({buy_hit_rate:.0%} hit rate)")
        elif sell_hit_rate > buy_hit_rate * 1.5 and sell_hit_rate > 0.3:
            bear_score += 1
            reasons.append(f"Aggressive selling ({sell_hit_rate:.0%} hit rate)")
            
        # Cap scores
        bull_score = min(2, bull_score)
        bear_score = min(2, bear_score)
        
        return bull_score, bear_score, reasons, warnings
    
    def _calculate_quality_score(self, stability_metrics: Dict) -> float:
        """Calculate overall microstructure quality (0-1)"""
        quality_factors = []
        
        # Quote stability
        quality_factors.append(stability_metrics['price_stability'])
        quality_factors.append((stability_metrics['bid_stability'] + 
                               stability_metrics['ask_stability']) / 2)
        
        # Low spoofing/stuffing
        quality_factors.append(1 - stability_metrics['spoofing_probability'])
        quality_factors.append(1 - stability_metrics['stuffing_score'])
        
        return np.mean(quality_factors)
    
    def _calculate_confidence(self, metrics: MicrostructureMetrics, 
                            warnings: List[str]) -> float:
        """Calculate signal confidence (0-1)"""
        confidence = metrics.microstructure_quality
        
        # Reduce confidence for warnings
        confidence *= (1 - metrics.spoofing_probability * self.config['spoofing_penalty'])
        confidence *= (1 - metrics.quote_stuffing_score * self.config['spoofing_penalty'])
        
        # Reduce confidence for wide spreads
        if metrics.is_wide_spread:
            confidence *= 0.8
            
        # Reduce confidence for low quote stability
        confidence *= (metrics.bid_stability_score + metrics.ask_stability_score) / 2
        
        # Ensure minimum confidence
        confidence = max(self.config['min_confidence'], confidence)
        
        return confidence
    
    def get_current_state(self) -> Dict:
        """Get current analyzer state"""
        if not self.nbbo_state.current_quote:
            return {'status': 'No data'}
            
        return {
            'current_quote': {
                'bid': f"{self.nbbo_state.current_quote.bid_price:.2f} x {self.nbbo_state.current_quote.bid_size:.0f}",
                'ask': f"{self.nbbo_state.current_quote.ask_price:.2f} x {self.nbbo_state.current_quote.ask_size:.0f}",
                'spread': self.nbbo_state.current_quote.spread,
                'mid': self.nbbo_state.current_quote.mid_price
            },
            'statistics': {
                'quotes_processed': self.quotes_processed,
                'trades_processed': self.trades_processed,
                'signals_generated': self.signals_generated
            },
            'last_signal': self.signal_history[-1] if self.signal_history else None
        }
    
    def is_premarket(self, timestamp: datetime) -> bool:
        """Check if timestamp is in pre-market hours (4:00 AM - 9:30 AM ET)"""
        # Convert to ET (assuming timestamp is UTC)
        et_hour = timestamp.hour - 5  # Simple UTC to ET conversion
        if et_hour < 0:
            et_hour += 24
            
        return 4 <= et_hour < 9.5
    
    def adjust_for_premarket(self, signal: MicroMomentumSignal) -> MicroMomentumSignal:
        """Adjust signal thresholds for pre-market conditions"""
        if self.is_premarket(signal.timestamp):
            # Pre-market typically has wider spreads, less liquidity
            signal.warnings.append("Pre-market hours - adjusted thresholds")
            signal.confidence *= 0.8  # Reduce confidence
            
        return signal


# ============= TEST FUNCTION =============
async def test_micro_momentum():
    """Test microstructure momentum with real-time Polygon data"""
    import sys
    import os
    
    # Add parent directories to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    modules_dir = os.path.dirname(os.path.dirname(current_dir))
    vega_root = os.path.dirname(modules_dir)
    if vega_root not in sys.path:
        sys.path.insert(0, vega_root)
    
    from polygon import PolygonWebSocketClient
    
    print("=== NBBO MICROSTRUCTURE MOMENTUM TEST ===")
    print("Analyzing market maker behavior using NBBO quotes + trades")
    print(f"Current UTC time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
    
    # Test configuration
    TEST_SYMBOLS = ['AAPL', 'TSLA', 'SPY']
    TEST_DURATION = 300  # 5 minutes
    
    # Create analyzer
    analyzer = MicrostructureMomentum(config={
        'nbbo_history_size': 1000,
        'signal_threshold': 0.6,
        'min_confidence': 0.3,
        'spoofing_penalty': 0.5,
        'wide_spread_threshold': 2.0,
        'depletion_threshold': 0.5,
        'premarket_adjustment': 1.5
    })
    
    # Track signals by symbol
    symbol_analyzers = {symbol: MicrostructureMomentum() for symbol in TEST_SYMBOLS}
    signal_counts = defaultdict(lambda: {'bull': 0, 'bear': 0, 'neutral': 0})
    
    # Track current symbol for each piece of data
    current_symbols = {}
    
    async def handle_quote(data: Dict):
        """Process quote update"""
        symbol = data['symbol']
        current_symbols[symbol] = symbol
        
        # Convert to expected format
        quote_data = {
            'timestamp': data['timestamp'],
            'bid_price': data.get('bid_price', data.get('bp', 0)),
            'bid_size': data.get('bid_size', data.get('bs', 0)),
            'ask_price': data.get('ask_price', data.get('ap', 0)),
            'ask_size': data.get('ask_size', data.get('as', 0))
        }
        
        # Process quote
        signal = symbol_analyzers[symbol].process_quote(quote_data)
        
        if signal:
            # Fix symbol assignment
            signal.symbol = symbol
            display_signal(signal)
            
            # Track signal type
            if signal.net_signal > 0:
                signal_counts[symbol]['bull'] += 1
            elif signal.net_signal < 0:
                signal_counts[symbol]['bear'] += 1
            else:
                signal_counts[symbol]['neutral'] += 1
    
    async def handle_trade(data: Dict):
        """Process trade"""
        symbol = data['symbol']
        
        # Process trade if we have quote data
        if symbol in symbol_analyzers:
            signal = symbol_analyzers[symbol].process_trade(data)
            
            if signal:
                signal.symbol = symbol
                display_signal(signal)
    
    def display_signal(signal: MicroMomentumSignal):
        """Display signal with formatting"""
        # Color coding
        if signal.net_signal > 0:
            emoji = '🟢'
            color_code = '\033[92m'  # Green
            direction = 'BULLISH'
        elif signal.net_signal < 0:
            emoji = '🔴'
            color_code = '\033[91m'  # Red
            direction = 'BEARISH'
        else:
            emoji = '⚪'
            color_code = '\033[93m'  # Yellow
            direction = 'NEUTRAL'
        
        print(f"\n{color_code}{'='*70}\033[0m")
        print(f"{emoji} {signal.symbol} - {direction} Signal (Net: {signal.net_signal:+d})")
        print(f"Time: {signal.timestamp.strftime('%H:%M:%S.%f UTC')[:-3]}")
        print(f"Confidence: {signal.confidence:.0%}")
        
        # Display reasons
        if signal.reasons:
            print("\nReasons:")
            for reason in signal.reasons:
                print(f"  ✓ {reason}")
        
        # Display warnings
        if signal.warnings:
            print("\nWarnings:")
            for warning in signal.warnings:
                print(f"  {warning}")
        
        # Key metrics
        m = signal.metrics
        print(f"\nKey Metrics:")
        print(f"  • Bid/Ask Size: {m.bid_size:.0f} / {m.ask_size:.0f} (Imbalance: {m.size_imbalance:+.0%})")
        print(f"  • Spread: ${m.current_spread:.4f} ({m.spread_bps:.1f} bps)")
        print(f"  • Quote Stability: Bid {m.bid_stability_score:.0%} / Ask {m.ask_stability_score:.0%}")
        print(f"  • Regeneration Speed: Bid {m.bid_regen_speed:.0f} / Ask {m.ask_regen_speed:.0f}")
        print(f"  • Hidden Liquidity: {m.hidden_liquidity_score:.0%}")
        print(f"  • Quality Score: {m.microstructure_quality:.0%}")
    
    # WebSocket handlers
    async def handle_data(data: Dict):
        """Route data to appropriate handler"""
        event_type = data.get('ev', data.get('event_type'))
        
        if event_type == 'Q':  # Quote
            await handle_quote(data)
        elif event_type == 'T':  # Trade
            await handle_trade(data)
    
    # Create WebSocket client
    ws_client = PolygonWebSocketClient()
    
    try:
        # Connect and authenticate
        print(f"Connecting to Polygon WebSocket...")
        await ws_client.connect()
        print("✓ Connected and authenticated")
        
        # Subscribe to quotes and trades
        print(f"\nSubscribing to quotes and trades for: {', '.join(TEST_SYMBOLS)}")
        await ws_client.subscribe(
            symbols=TEST_SYMBOLS,
            channels=['Q', 'T'],  # Quotes and Trades
            callback=handle_data
        )
        print("✓ Subscribed successfully")
        
        print(f"\n⏰ Running for {TEST_DURATION} seconds...")
        print("Waiting for market data...\n")
        
        # Create listen task
        listen_task = asyncio.create_task(ws_client.listen())
        
        # Run for specified duration
        start_time = time.time()
        last_stats_time = start_time
        
        while time.time() - start_time < TEST_DURATION:
            await asyncio.sleep(1)
            
            # Print stats every 30 seconds
            if time.time() - last_stats_time >= 30:
                print(f"\n📊 Progress Update:")
                for symbol, analyzer in symbol_analyzers.items():
                    state = analyzer.get_current_state()
                    if 'statistics' in state:
                        stats = state['statistics']
                        print(f"  {symbol}: {stats['quotes_processed']} quotes, "
                              f"{stats['trades_processed']} trades, "
                              f"{stats['signals_generated']} signals")
                print(f"  Time: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
                last_stats_time = time.time()
            
            # Show countdown
            remaining = TEST_DURATION - (time.time() - start_time)
            print(f"\r⏳ Time remaining: {remaining:.0f}s ", end='', flush=True)
        
        print("\n\n🏁 Test complete!")
        
        # Final summary
        print(f"\n📊 Final Summary:")
        print(f"  • Test duration: {TEST_DURATION} seconds")
        print(f"  • End time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Per-symbol summary
        print(f"\n📈 Signal Summary by Symbol:")
        for symbol in TEST_SYMBOLS:
            analyzer = symbol_analyzers[symbol]
            state = analyzer.get_current_state()
            
            if 'statistics' in state:
                stats = state['statistics']
                counts = signal_counts[symbol]
                
                print(f"\n  {symbol}:")
                print(f"    • Quotes: {stats['quotes_processed']}")
                print(f"    • Trades: {stats['trades_processed']}")
                print(f"    • Signals: {stats['signals_generated']}")
                print(f"    • Breakdown: {counts['bull']} bull, {counts['bear']} bear, {counts['neutral']} neutral")
                
                # Last quote if available
                if 'current_quote' in state:
                    q = state['current_quote']
                    print(f"    • Last Quote: Bid {q['bid']}, Ask {q['ask']}")
        
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
    print(f"Starting NBBO Microstructure Momentum at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("This module analyzes market maker behavior using NBBO quotes")
    print("All timestamps are in UTC\n")
    
    asyncio.run(test_micro_momentum())