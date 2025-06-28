# modules/calculations/order-flow/bid_ask_imbal.py
"""
Module: Bid/Ask Imbalance Analysis
Purpose: Measure buyer/seller aggressiveness by tracking trades at bid vs ask
Features: Real-time imbalance calculation, spread analysis, liquidity detection
Performance Target: <100 microseconds per trade
Time Handling: All timestamps in UTC
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
import logging
import sys
import os
import asyncio
import time as time_module
from functools import lru_cache
import json

# Fix import path
current_dir = os.path.dirname(os.path.abspath(__file__))
order_flow_dir = current_dir
calculations_dir = os.path.dirname(order_flow_dir)
modules_dir = os.path.dirname(calculations_dir)
vega_root = os.path.dirname(modules_dir)

if vega_root not in sys.path:
    sys.path.insert(0, vega_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Quote:
    """Quote data with bid/ask prices and sizes"""
    symbol: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    timestamp: datetime
    bid_levels: Optional[List[Dict]] = None  # Full depth if available
    ask_levels: Optional[List[Dict]] = None


@dataclass
class ImbalanceComponents:
    """Detailed components of bid/ask imbalance analysis"""
    raw_imbalance: float  # -1 to 1
    weighted_imbalance: float  # -1 to 1
    smoothed_imbalance: float  # -1 to 1
    aggression_ratio: float  # 0 to 1
    buy_volume: float
    sell_volume: float
    total_volume: float
    current_spread: float
    spread_ratio_1min: float
    spread_ratio_5min: float
    spread_volatility: float
    spread_trend: float
    quote_stability: float
    liquidity_state: str
    book_pressure: Optional[float] = None
    spreads_by_size: Optional[Dict[int, float]] = None


@dataclass
class BidAskSignal:
    """Complete bid/ask imbalance signal"""
    symbol: str
    timestamp: datetime
    current_price: float
    bull_score: int  # -2 to +2
    bear_score: int  # -2 to +2
    confidence: float  # 0 to 1
    components: ImbalanceComponents
    signal_type: str  # 'AGGRESSIVE_BUYING', 'AGGRESSIVE_SELLING', etc.
    signal_strength: str  # 'EXCEPTIONAL', 'STRONG', 'MODERATE', 'WEAK'
    reason: str
    calculation_time_ms: float
    trade_count: int
    warnings: List[str] = field(default_factory=list)


@dataclass
class Trade:
    """Individual trade data"""
    symbol: str
    price: float
    size: int
    timestamp: datetime
    conditions: Optional[List[str]] = None


@dataclass
class ClassifiedTrade:
    """Trade with bid/ask classification"""
    trade: Trade
    quote: Quote
    classification: str  # 'buy_aggressive', 'buy_passive', 'sell_aggressive', etc.
    spread_at_trade: float


class BidAskImbalance:
    """
    Bid/Ask Imbalance Calculator for order flow analysis.
    Tracks whether trades execute at bid (seller-initiated) or ask (buyer-initiated).
    """
    
    def __init__(self,
                 imbalance_lookback: int = 100,
                 spread_history_seconds: int = 600,
                 quote_sync_tolerance_ms: int = 100,
                 aggression_threshold: float = 0.7,
                 smoothing_alpha: float = 0.1):
        """
        Initialize bid/ask imbalance calculator.
        
        Args:
            imbalance_lookback: Number of trades for imbalance calculation
            spread_history_seconds: Seconds of spread history to maintain
            quote_sync_tolerance_ms: Max milliseconds between trade and quote
            aggression_threshold: Threshold for aggressive vs passive trades
            smoothing_alpha: EWMA smoothing factor
        """
        self.imbalance_lookback = imbalance_lookback
        self.spread_history_seconds = spread_history_seconds
        self.quote_sync_tolerance_ms = quote_sync_tolerance_ms
        self.aggression_threshold = aggression_threshold
        self.smoothing_alpha = smoothing_alpha
        
        # Data buffers per symbol
        self.classified_trades: Dict[str, deque] = {}
        self.quote_history: Dict[str, deque] = {}
        self.latest_quotes: Dict[str, Quote] = {}
        
        # Session metrics
        self.session_stats: Dict[str, Dict] = {}
        
        # Latest signals
        self.latest_signals: Dict[str, BidAskSignal] = {}
        
        # WebSocket integration
        self.ws_client = None
        self.active_symbols: Dict[str, List[Callable]] = {}
        
        # Performance tracking
        self.calculation_count = 0
        self.total_calculation_time = 0
        
        # Smoothed values
        self.smoothed_imbalances: Dict[str, float] = {}
        
        logger.info(f"Initialized BidAskImbalance: lookback={imbalance_lookback}, "
                   f"spread_history={spread_history_seconds}s, sync_tolerance={quote_sync_tolerance_ms}ms")
    
    def initialize_symbol(self, symbol: str):
        """Initialize buffers for a new symbol"""
        self.classified_trades[symbol] = deque(maxlen=self.imbalance_lookback)
        self.quote_history[symbol] = deque()
        self.session_stats[symbol] = {
            'total_buy_volume': 0,
            'total_sell_volume': 0,
            'widest_spread': 0,
            'tightest_spread': float('inf'),
            'imbalance_extremes': []
        }
        self.smoothed_imbalances[symbol] = 0.0
        logger.info(f"Initialized buffers for {symbol}")
    
    def process_quote(self, quote: Quote):
        """Process a new quote update"""
        if quote.symbol not in self.quote_history:
            self.initialize_symbol(quote.symbol)
        
        # Store latest quote
        self.latest_quotes[quote.symbol] = quote
        
        # Add to history with time-based cleanup
        self.quote_history[quote.symbol].append(quote)
        self._cleanup_quote_history(quote.symbol, quote.timestamp)
        
        # Update session spread stats
        spread = quote.ask - quote.bid
        stats = self.session_stats[quote.symbol]
        stats['widest_spread'] = max(stats['widest_spread'], spread)
        stats['tightest_spread'] = min(stats['tightest_spread'], spread)
    
    def process_trade(self, trade: Trade) -> Optional[BidAskSignal]:
        """
        Process a new trade and generate signal.
        
        Args:
            trade: Trade object with symbol, price, size, timestamp
            
        Returns:
            BidAskSignal if enough data, None otherwise
        """
        start_time = time_module.perf_counter()
        
        # Initialize if needed
        if trade.symbol not in self.classified_trades:
            self.initialize_symbol(trade.symbol)
        
        # Get synchronized quote
        quote = self._get_synchronized_quote(trade)
        if not quote:
            logger.debug(f"No synchronized quote for {trade.symbol} trade at {trade.timestamp}")
            return None
        
        # Classify trade
        classification = self._classify_trade_aggressor(trade, quote)
        classified = ClassifiedTrade(
            trade=trade,
            quote=quote,
            classification=classification,
            spread_at_trade=quote.ask - quote.bid
        )
        
        # Add to buffer
        self.classified_trades[trade.symbol].append(classified)
        
        # Update session metrics
        self._update_session_metrics(trade.symbol, classified)
        
        # Need minimum trades for analysis
        if len(self.classified_trades[trade.symbol]) < 20:
            logger.debug(f"{trade.symbol}: Warming up "
                        f"({len(self.classified_trades[trade.symbol])}/20)")
            return None
        
        # Calculate imbalance metrics
        imbalance_metrics = self._calculate_volume_imbalance(
            list(self.classified_trades[trade.symbol])
        )
        
        # Analyze spread dynamics
        spread_metrics = self._analyze_spread_dynamics(trade.symbol)
        if not spread_metrics:
            return None
        
        # Analyze depth if available
        depth_metrics = None
        if quote.bid_levels and quote.ask_levels:
            depth_metrics = self._analyze_liquidity_depth(quote)
        
        # Combine metrics
        components = ImbalanceComponents(
            raw_imbalance=imbalance_metrics['raw_imbalance'],
            weighted_imbalance=imbalance_metrics['weighted_imbalance'],
            smoothed_imbalance=imbalance_metrics['smoothed_imbalance'],
            aggression_ratio=imbalance_metrics['aggression_ratio'],
            buy_volume=imbalance_metrics['buy_volume'],
            sell_volume=imbalance_metrics['sell_volume'],
            total_volume=imbalance_metrics['total_volume'],
            current_spread=spread_metrics['current_spread'],
            spread_ratio_1min=spread_metrics['spread_ratio_1min'],
            spread_ratio_5min=spread_metrics['spread_ratio_5min'],
            spread_volatility=spread_metrics['spread_volatility'],
            spread_trend=spread_metrics['spread_trend'],
            quote_stability=spread_metrics['quote_stability'],
            liquidity_state=spread_metrics['liquidity_state']
        )
        
        if depth_metrics:
            components.book_pressure = depth_metrics['book_pressure']
            components.spreads_by_size = depth_metrics['spreads_by_size']
        
        # Generate signal
        signal = self._calculate_bid_ask_score(trade, components, depth_metrics)
        
        # Track performance
        calculation_time = (time_module.perf_counter() - start_time) * 1000
        signal.calculation_time_ms = calculation_time
        
        # Update tracking
        self.calculation_count += 1
        self.total_calculation_time += calculation_time
        self.latest_signals[trade.symbol] = signal
        
        # Track extreme imbalances
        if abs(components.smoothed_imbalance) > 0.7:
            self.session_stats[trade.symbol]['imbalance_extremes'].append({
                'timestamp': trade.timestamp,
                'imbalance': components.smoothed_imbalance,
                'price': trade.price,
                'signal': signal
            })
        
        return signal
    
    def _get_synchronized_quote(self, trade: Trade) -> Optional[Quote]:
        """Get quote synchronized with trade timestamp"""
        if trade.symbol not in self.latest_quotes:
            return None
        
        # Use latest quote if within tolerance
        latest = self.latest_quotes[trade.symbol]
        time_diff_ms = abs((trade.timestamp - latest.timestamp).total_seconds() * 1000)
        
        if time_diff_ms <= self.quote_sync_tolerance_ms:
            return latest
        
        # Search historical quotes
        if trade.symbol in self.quote_history:
            for quote in reversed(self.quote_history[trade.symbol]):
                time_diff_ms = abs((trade.timestamp - quote.timestamp).total_seconds() * 1000)
                if time_diff_ms <= self.quote_sync_tolerance_ms:
                    return quote
                if quote.timestamp < trade.timestamp - timedelta(milliseconds=self.quote_sync_tolerance_ms):
                    break
        
        return None
    
    def _classify_trade_aggressor(self, trade: Trade, quote: Quote) -> str:
        """Determine if trade was buyer or seller initiated"""
        mid_price = (quote.bid + quote.ask) / 2
        
        # Consider quote sizes for weighted classification
        total_size = quote.bid_size + quote.ask_size
        if total_size > 0:
            bid_weighted = quote.bid + (quote.bid_size / total_size) * 0.01
            ask_weighted = quote.ask - (quote.ask_size / total_size) * 0.01
        else:
            bid_weighted = quote.bid
            ask_weighted = quote.ask
        
        # Classify based on execution price
        if trade.price >= quote.ask:
            return 'buy_aggressive'  # Lifted offer
        elif trade.price > ask_weighted:
            return 'buy_passive'  # Near ask
        elif trade.price <= quote.bid:
            return 'sell_aggressive'  # Hit bid
        elif trade.price < bid_weighted:
            return 'sell_passive'  # Near bid
        else:
            return 'midpoint'  # Neutral
    
    def _calculate_volume_imbalance(self, classified_trades: List[ClassifiedTrade]) -> Dict:
        """Calculate buy/sell volume imbalance with aggression weighting"""
        # Weight aggressive trades more heavily
        weights = {
            'buy_aggressive': 1.0,
            'buy_passive': 0.7,
            'sell_aggressive': -1.0,
            'sell_passive': -0.7,
            'midpoint': 0.0,
            'unknown': 0.0
        }
        
        weighted_volume = 0
        total_volume = 0
        buy_volume = 0
        sell_volume = 0
        aggressive_volume = 0
        
        for ct in classified_trades:
            volume = ct.trade.size
            classification = ct.classification
            
            weighted_volume += volume * weights.get(classification, 0)
            total_volume += volume
            
            if 'buy' in classification:
                buy_volume += volume
            elif 'sell' in classification:
                sell_volume += volume
            
            if 'aggressive' in classification:
                aggressive_volume += volume
        
        # Calculate metrics
        if total_volume > 0:
            raw_imbalance = (buy_volume - sell_volume) / total_volume
            weighted_imbalance = weighted_volume / total_volume
            aggression_ratio = aggressive_volume / total_volume
        else:
            raw_imbalance = 0
            weighted_imbalance = 0
            aggression_ratio = 0
        
        # Apply EWMA smoothing
        symbol = classified_trades[0].trade.symbol if classified_trades else None
        if symbol:
            prev_smoothed = self.smoothed_imbalances.get(symbol, weighted_imbalance)
            smoothed = self.smoothing_alpha * weighted_imbalance + (1 - self.smoothing_alpha) * prev_smoothed
            self.smoothed_imbalances[symbol] = smoothed
        else:
            smoothed = weighted_imbalance
        
        return {
            'raw_imbalance': raw_imbalance,
            'weighted_imbalance': weighted_imbalance,
            'smoothed_imbalance': smoothed,
            'aggression_ratio': aggression_ratio,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'total_volume': total_volume
        }
    
    def _analyze_spread_dynamics(self, symbol: str) -> Optional[Dict]:
        """Analyze bid-ask spread behavior and liquidity"""
        if symbol not in self.quote_history or len(self.quote_history[symbol]) < 10:
            return None
        
        quotes = list(self.quote_history[symbol])
        spreads = [(q.ask - q.bid) for q in quotes]
        current_spread = spreads[-1]
        
        # Time-based analysis
        now = quotes[-1].timestamp
        spread_1min = [s for q, s in zip(quotes, spreads) 
                      if (now - q.timestamp).total_seconds() <= 60]
        spread_5min = [s for q, s in zip(quotes, spreads) 
                      if (now - q.timestamp).total_seconds() <= 300]
        
        if not spread_1min:
            spread_1min = spreads[-10:]
        if not spread_5min:
            spread_5min = spreads
        
        # Calculate metrics
        spread_1min_avg = np.mean(spread_1min)
        spread_5min_avg = np.mean(spread_5min)
        spread_1min_std = np.std(spread_1min)
        
        metrics = {
            'current_spread': current_spread,
            'spread_1min_avg': spread_1min_avg,
            'spread_5min_avg': spread_5min_avg,
            'spread_1min_std': spread_1min_std,
            'spread_percentile': self._percentile_rank(current_spread, spread_5min)
        }
        
        # Relative spreads
        metrics['spread_ratio_1min'] = current_spread / (spread_1min_avg + 0.0001)
        metrics['spread_ratio_5min'] = current_spread / (spread_5min_avg + 0.0001)
        
        # Spread volatility
        metrics['spread_volatility'] = spread_1min_std / (spread_1min_avg + 0.0001)
        
        # Spread trend
        if len(spread_1min) >= 10:
            x = np.arange(len(spread_1min))
            coeffs = np.polyfit(x, spread_1min, 1)
            metrics['spread_trend'] = coeffs[0] / (spread_1min_avg + 0.0001)
        else:
            metrics['spread_trend'] = 0
        
        # Quote stability
        spread_changes = sum(1 for i in range(1, len(spread_1min))
                           if spread_1min[i] != spread_1min[i-1])
        metrics['quote_stability'] = 1 - (spread_changes / max(len(spread_1min), 1))
        
        # Liquidity state
        if current_spread < spread_5min_avg * 0.8:
            metrics['liquidity_state'] = 'high'
        elif current_spread > spread_5min_avg * 1.5:
            metrics['liquidity_state'] = 'low'
        else:
            metrics['liquidity_state'] = 'normal'
        
        return metrics
    
    def _analyze_liquidity_depth(self, quote: Quote) -> Optional[Dict]:
        """Analyze full order book depth if available"""
        if not quote.bid_levels or not quote.ask_levels:
            return None
        
        # Book pressure (size imbalance)
        bid_size = sum(level['size'] for level in quote.bid_levels[:5])
        ask_size = sum(level['size'] for level in quote.ask_levels[:5])
        book_pressure = (bid_size - ask_size) / (bid_size + ask_size + 1)
        
        # Top of book sizes
        top_bid_size = quote.bid_levels[0]['size'] if quote.bid_levels else 0
        top_ask_size = quote.ask_levels[0]['size'] if quote.ask_levels else 0
        
        # Weighted spreads at different depths
        spreads_by_size = {}
        cumulative_sizes = [1000, 5000, 10000, 25000]
        
        for target_size in cumulative_sizes:
            bid_price = self._calculate_vwap_price(quote.bid_levels, target_size, 'bid')
            ask_price = self._calculate_vwap_price(quote.ask_levels, target_size, 'ask')
            spreads_by_size[target_size] = ask_price - bid_price
        
        return {
            'book_pressure': book_pressure,
            'top_bid_size': top_bid_size,
            'top_ask_size': top_ask_size,
            'size_ratio': top_bid_size / (top_ask_size + 1),
            'spreads_by_size': spreads_by_size,
            'bid_levels': len(quote.bid_levels),
            'ask_levels': len(quote.ask_levels)
        }
    
    def _calculate_vwap_price(self, levels: List[Dict], target_size: int, side: str) -> float:
        """Calculate VWAP for given size"""
        cumulative_size = 0
        cumulative_value = 0
        
        for level in levels:
            size_at_level = min(level['size'], target_size - cumulative_size)
            cumulative_value += level['price'] * size_at_level
            cumulative_size += size_at_level
            
            if cumulative_size >= target_size:
                break
        
        if cumulative_size > 0:
            return cumulative_value / cumulative_size
        else:
            return levels[0]['price'] if levels else 0
    
    def _calculate_bid_ask_score(self, trade: Trade, components: ImbalanceComponents,
                                depth_metrics: Optional[Dict] = None) -> BidAskSignal:
        """Generate bull/bear scores from imbalance data"""
        bull_score = 0
        bear_score = 0
        warnings = []
        
        # Extract key metrics
        imb = components.smoothed_imbalance
        aggr = components.aggression_ratio
        spread_ratio = components.spread_ratio_1min
        liquidity = components.liquidity_state
        spread_trend = components.spread_trend
        
        # Adjust for pre-market
        is_premarket = trade.timestamp.hour < 9 or trade.timestamp.hour >= 16
        if is_premarket:
            # Adjust thresholds for pre-market
            imb_threshold = 0.7  # vs 0.6 normal
            aggr_threshold = 0.5  # vs 0.7 normal
        else:
            imb_threshold = 0.6
            aggr_threshold = 0.7
        
        # Bull conditions
        if imb > imb_threshold and aggr > aggr_threshold and liquidity == 'high':
            bull_score = 2  # Aggressive buying with tight spreads
            signal_type = "AGGRESSIVE_BUYING"
            signal_strength = "EXCEPTIONAL"
            reason = f"Heavy buying pressure (imb={imb:.2f}) with tight spreads"
        elif imb > 0.5 and spread_ratio < 0.8:
            bull_score = 2  # Strong buying with tightening spreads
            signal_type = "STRONG_BUYING"
            signal_strength = "STRONG"
            reason = f"Strong buying with tightening spreads ({spread_ratio:.2f}x)"
        elif imb > 0.3 and aggr > 0.6:
            bull_score = 1  # Moderate buying pressure
            signal_type = "MODERATE_BUYING"
            signal_strength = "MODERATE"
            reason = "Moderate buying pressure detected"
        elif spread_trend < -0.1 and imb > 0:
            bull_score = 1  # Spreads tightening with buying
            signal_type = "LIQUIDITY_IMPROVING"
            signal_strength = "MODERATE"
            reason = "Improving liquidity with buying interest"
        else:
            signal_type = "NEUTRAL"
            signal_strength = "NEUTRAL"
            reason = "Mixed order flow signals"
        
        # Bear conditions
        if imb < -imb_threshold and aggr > aggr_threshold and liquidity == 'low':
            bear_score = 2  # Aggressive selling with wide spreads
            signal_type = "AGGRESSIVE_SELLING"
            signal_strength = "EXCEPTIONAL"
            reason = f"Heavy selling pressure (imb={imb:.2f}) with wide spreads"
        elif imb < -0.5 and spread_ratio > 1.5:
            bear_score = 2  # Strong selling with widening spreads
            signal_type = "STRONG_SELLING"
            signal_strength = "STRONG"
            reason = f"Strong selling with widening spreads ({spread_ratio:.2f}x)"
        elif imb < -0.3 and aggr > 0.6:
            bear_score = 1  # Moderate selling pressure
            signal_type = "MODERATE_SELLING"
            signal_strength = "MODERATE"
            reason = "Moderate selling pressure detected"
        elif spread_ratio > 2.0:
            bear_score = 1  # Liquidity exodus
            signal_type = "LIQUIDITY_CRISIS"
            signal_strength = "MODERATE"
            reason = f"Liquidity drying up (spread {spread_ratio:.1f}x normal)"
        
        # Depth-based adjustments
        if depth_metrics:
            book_pressure = depth_metrics['book_pressure']
            if book_pressure > 0.5 and imb > 0:
                bull_score = min(bull_score + 1, 2)
            elif book_pressure < -0.5 and imb < 0:
                bear_score = min(bear_score + 1, 2)
        
        # Special patterns
        if imb > 0.3 and spread_ratio > 1.5:
            # Buying into wide spreads - desperation
            bull_score = max(bull_score - 1, 0)
            warnings.append("Desperate buying into wide spreads")
        elif imb < -0.3 and spread_ratio < 0.7:
            # Selling into tight spreads - hidden distribution
            bear_score = min(bear_score + 1, 2)
            signal_type = "HIDDEN_DISTRIBUTION"
            warnings.append("Hidden distribution detected")
        
        # Calculate confidence
        confidence = min(abs(imb) + aggr, 1.0)
        
        # Pre-market warning
        if is_premarket:
            warnings.append("Pre-market conditions")
        
        return BidAskSignal(
            symbol=trade.symbol,
            timestamp=trade.timestamp,
            current_price=trade.price,
            bull_score=bull_score,
            bear_score=bear_score,
            confidence=confidence,
            components=components,
            signal_type=signal_type,
            signal_strength=signal_strength,
            reason=reason,
            calculation_time_ms=0,  # Will be set by caller
            trade_count=len(self.classified_trades.get(trade.symbol, [])),
            warnings=warnings
        )
    
    def _cleanup_quote_history(self, symbol: str, current_time: datetime):
        """Remove old quotes beyond history window"""
        cutoff_time = current_time - timedelta(seconds=self.spread_history_seconds)
        
        while (self.quote_history[symbol] and 
               self.quote_history[symbol][0].timestamp < cutoff_time):
            self.quote_history[symbol].popleft()
    
    def _update_session_metrics(self, symbol: str, classified: ClassifiedTrade):
        """Update session-wide statistics"""
        stats = self.session_stats[symbol]
        
        if 'buy' in classified.classification:
            stats['total_buy_volume'] += classified.trade.size
        elif 'sell' in classified.classification:
            stats['total_sell_volume'] += classified.trade.size
    
    def _percentile_rank(self, value: float, data: List[float]) -> float:
        """Calculate percentile rank of value in data"""
        if not data:
            return 50.0
        sorted_data = sorted(data)
        count_below = sum(1 for x in sorted_data if x < value)
        return (count_below / len(data)) * 100
    
    # ============= WEBSOCKET FUNCTIONALITY =============
    
    async def start_websocket(self, symbols: List[str], callback: Optional[Callable] = None):
        """Start real-time monitoring with WebSocket"""
        try:
            from polygon import PolygonWebSocketClient
            
            logger.info("Connecting to WebSocket for trade and quote data...")
            self.ws_client = PolygonWebSocketClient()
            await self.ws_client.connect()
            
            # Subscribe to both trade and quote channels
            await self.ws_client.subscribe(
                symbols=symbols,
                channels=['T', 'Q'],  # Trade and Quote channels
                callback=self._handle_websocket_message
            )
            
            for symbol in symbols:
                self.active_symbols[symbol] = [callback] if callback else []
                if symbol not in self.classified_trades:
                    self.initialize_symbol(symbol)
            
            logger.info(f"✓ Started real-time bid/ask monitoring for {symbols}")
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise
    
    async def _handle_websocket_message(self, data: Dict):
        """Handle incoming trade or quote data from WebSocket"""
        try:
            event_type = data.get('event_type')
            symbol = data.get('symbol')
            
            if symbol not in self.active_symbols:
                return
            
            if event_type == 'quote':
                # Process quote update
                quote = Quote(
                    symbol=symbol,
                    bid=data['bid_price'],
                    ask=data['ask_price'],
                    bid_size=data['bid_size'],
                    ask_size=data['ask_size'],
                    timestamp=datetime.fromtimestamp(data['timestamp'] / 1000, tz=timezone.utc)
                )
                self.process_quote(quote)
                
            elif event_type == 'trade':
                # Process trade
                trade = Trade(
                    symbol=symbol,
                    price=data['price'],
                    size=data['size'],
                    timestamp=datetime.fromtimestamp(data['timestamp'] / 1000, tz=timezone.utc),
                    conditions=data.get('conditions', [])
                )
                
                signal = self.process_trade(trade)
                
                # Notify callbacks if significant signal
                if signal and (signal.bull_score >= 2 or signal.bear_score >= 2):
                    for callback in self.active_symbols[symbol]:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(signal)
                            else:
                                callback(signal)
                        except Exception as e:
                            logger.error(f"Callback error for {symbol}: {e}")
                            
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    async def stop(self):
        """Stop WebSocket connection"""
        if self.ws_client:
            await self.ws_client.disconnect()
            logger.info("WebSocket disconnected")
    
    def get_session_summary(self, symbol: str) -> Dict:
        """Provide session-wide statistics for a symbol"""
        if symbol not in self.session_stats:
            return {}
        
        stats = self.session_stats[symbol]
        total_volume = stats['total_buy_volume'] + stats['total_sell_volume']
        
        if total_volume > 0:
            session_imbalance = (
                (stats['total_buy_volume'] - stats['total_sell_volume']) / total_volume
            )
        else:
            session_imbalance = 0
        
        return {
            'session_imbalance': session_imbalance,
            'spread_range': (
                stats['tightest_spread'],
                stats['widest_spread']
            ),
            'extreme_periods': len(stats['imbalance_extremes']),
            'total_volume': total_volume,
            'buy_volume': stats['total_buy_volume'],
            'sell_volume': stats['total_sell_volume']
        }
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        avg_time = self.total_calculation_time / self.calculation_count if self.calculation_count > 0 else 0
        return {
            'total_calculations': self.calculation_count,
            'average_time_ms': avg_time,
            'active_symbols': len(self.active_symbols),
            'total_extreme_imbalances': sum(
                len(m['imbalance_extremes']) 
                for m in self.session_stats.values()
            )
        }
    
    def format_for_dashboard(self, signal: BidAskSignal) -> Dict:
        """Format signal for dashboard display"""
        # Color coding
        if signal.bull_score == 2:
            color = 'bright_green'
        elif signal.bull_score == 1:
            color = 'green'
        elif signal.bear_score == 2:
            color = 'bright_red'
        elif signal.bear_score == 1:
            color = 'red'
        else:
            color = 'white'
        
        # Create imbalance bar
        imb = signal.components.smoothed_imbalance
        bar_length = 20
        center = bar_length // 2
        position = int(center + (imb * center))
        bar = ['-'] * bar_length
        bar[center] = '|'
        bar[max(0, min(position, bar_length-1))] = '●'
        
        # Spread indicator
        spread_ratio = signal.components.spread_ratio_1min
        spread_icon = '⟷' if spread_ratio > 1.5 else '→←'
        
        return {
            'main_display': f"Imb: {''.join(bar)} {imb:+.1%}",
            'color': color,
            'sub_components': {
                'Aggression': f"{signal.components.aggression_ratio:.0%}",
                'Spread': f"{spread_ratio:.2f}x {spread_icon}",
                'Liquidity': signal.components.liquidity_state.upper(),
                'Volume': f"B:{signal.components.buy_volume/1000:.1f}k/S:{signal.components.sell_volume/1000:.1f}k"
            },
            'tooltip': signal.reason,
            'alert': signal.signal_strength == 'EXCEPTIONAL',
            'warnings': signal.warnings
        }


# ============= TEST SCRIPT =============

async def run_test():
    """Test bid/ask imbalance calculation"""
    print("=== Testing Bid/Ask Imbalance Analysis ===\n")
    
    # Test configuration
    TEST_SYMBOL = 'AAPL'
    TEST_DURATION = 60  # seconds
    
    calculator = BidAskImbalance(
        imbalance_lookback=100,
        spread_history_seconds=600,
        quote_sync_tolerance_ms=100,
        aggression_threshold=0.7
    )
    
    print("📊 Bid/Ask Imbalance Monitor")
    print(f"📈 Lookback: {calculator.imbalance_lookback} trades")
    print(f"📊 Spread History: {calculator.spread_history_seconds} seconds")
    print(f"🎯 Quote Sync Tolerance: {calculator.quote_sync_tolerance_ms}ms")
    print()
    
    # Test 1: Synthetic data test
    print("Test 1: Processing synthetic trades with quotes...")
    print("-" * 50)
    
    # Initialize symbol
    calculator.initialize_symbol(TEST_SYMBOL)
    
    # Simulate order flow with quotes
    base_price = 150.0
    base_spread = 0.02
    current_time = datetime.now(timezone.utc)
    
    # Simulate accumulation pattern
    for i in range(200):
        # Generate quote
        if i < 50:  # Normal market
            spread = base_spread
        elif i < 100:  # Tightening spreads (bullish)
            spread = base_spread * (1 - (i-50)/100)
        elif i < 150:  # Aggressive buying
            spread = base_spread * 0.5
        else:  # Widening spreads
            spread = base_spread * (1 + (i-150)/50)
        
        bid = base_price - spread/2 + (i * 0.001)  # Slight uptrend
        ask = bid + spread
        
        quote = Quote(
            symbol=TEST_SYMBOL,
            bid=bid,
            ask=ask,
            bid_size=np.random.randint(100, 1000),
            ask_size=np.random.randint(100, 1000),
            timestamp=current_time + timedelta(milliseconds=i*100)
        )
        
        calculator.process_quote(quote)
        
        # Generate trade
        if i < 100:  # Mixed trading
            if np.random.random() > 0.5:
                price = ask  # Buy at ask
                size = np.random.randint(100, 1000)
            else:
                price = bid  # Sell at bid
                size = np.random.randint(100, 500)
        else:  # Aggressive buying phase
            if np.random.random() > 0.2:
                price = ask + np.random.uniform(0, 0.01)  # Aggressive buys
                size = np.random.randint(500, 2000)
            else:
                price = bid
                size = np.random.randint(50, 200)
        
        trade = Trade(
            symbol=TEST_SYMBOL,
            price=price,
            size=size,
            timestamp=current_time + timedelta(milliseconds=i*100 + 50)
        )
        
        signal = calculator.process_trade(trade)
        
        # Display significant signals
        if signal and i >= 50 and (signal.bull_score >= 2 or signal.bear_score >= 2):
            print(f"\n🚨 SIGNIFICANT IMBALANCE DETECTED!")
            print(f"   Time: {signal.timestamp.strftime('%H:%M:%S.%f')[:-3]}")
            print(f"   Type: {signal.signal_type}")
            print(f"   Strength: {signal.signal_strength}")
            print(f"   Bull/Bear: {signal.bull_score}/{signal.bear_score}")
            print(f"   Reason: {signal.reason}")
            print(f"   Components:")
            print(f"     • Imbalance: {signal.components.smoothed_imbalance:+.2f}")
            print(f"     • Aggression: {signal.components.aggression_ratio:.0%}")
            print(f"     • Spread: {signal.components.spread_ratio_1min:.2f}x")
            print(f"     • Liquidity: {signal.components.liquidity_state}")
            if signal.warnings:
                print(f"   ⚠️  Warnings: {', '.join(signal.warnings)}")
            print()
    
    # Final analysis
    final_signal = calculator.latest_signals.get(TEST_SYMBOL)
    if final_signal:
        print("\n📊 Final Analysis:")
        dashboard_format = calculator.format_for_dashboard(final_signal)
        print(f"   Display: {dashboard_format['main_display']}")
        print(f"   Color: {dashboard_format['color']}")
        print(f"   Sub-components:")
        for key, value in dashboard_format['sub_components'].items():
            print(f"     • {key}: {value}")
        
        # Session summary
        summary = calculator.get_session_summary(TEST_SYMBOL)
        print(f"\n📈 Session Summary:")
        print(f"   Session Imbalance: {summary['session_imbalance']:+.2%}")
        print(f"   Spread Range: {summary['spread_range'][0]:.4f} - {summary['spread_range'][1]:.4f}")
        print(f"   Extreme Periods: {summary['extreme_periods']}")
        print(f"   Total Volume: {summary['total_volume']:,}")
    
    # Test 2: Performance test
    print("\n\nTest 2: Performance benchmark...")
    print("-" * 50)
    
    # Reset calculator
    calculator = BidAskImbalance()
    calculator.initialize_symbol(TEST_SYMBOL)
    
    # Pre-generate quotes
    quotes = []
    for i in range(1000):
        quotes.append(Quote(
            symbol=TEST_SYMBOL,
            bid=150 - 0.01,
            ask=150 + 0.01,
            bid_size=np.random.randint(100, 1000),
            ask_size=np.random.randint(100, 1000),
            timestamp=datetime.now(timezone.utc) + timedelta(milliseconds=i*10)
        ))
    
    # Process quotes first
    for quote in quotes:
        calculator.process_quote(quote)
    
    # Benchmark trade processing
    start_time = time_module.perf_counter()
    for i in range(1000):
        trade = Trade(
            symbol=TEST_SYMBOL,
            price=150 + np.random.uniform(-0.05, 0.05),
            size=np.random.randint(100, 1000),
            timestamp=quotes[i].timestamp + timedelta(milliseconds=5)
        )
        calculator.process_trade(trade)
    
    elapsed = (time_module.perf_counter() - start_time) * 1000
    avg_time = elapsed / 1000
    
    print(f"✓ Processed 1000 trades in {elapsed:.2f}ms")
    print(f"✓ Average time per trade: {avg_time:.3f}ms")
    print(f"✓ {'PASS' if avg_time < 0.1 else 'FAIL'} performance target (<0.1ms)")
    
    # Performance stats
    stats = calculator.get_performance_stats()
    print(f"\n📊 Performance Statistics:")
    print(f"   Total calculations: {stats['total_calculations']}")
    print(f"   Average calculation time: {stats['average_time_ms']:.3f}ms")
    
    # Test 3: WebSocket test
    print("\n\nTest 3: WebSocket integration...")
    print("-" * 50)
    
    signal_count = 0
    
    def display_signal(signal: BidAskSignal):
        nonlocal signal_count
        signal_count += 1
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Signal #{signal_count}")
        print(f"Symbol: {signal.symbol}")
        print(f"Signal: {signal.signal_type} ({signal.signal_strength})")
        print(f"Bull/Bear: {signal.bull_score}/{signal.bear_score}")
        print(f"Imbalance: {signal.components.smoothed_imbalance:+.2%}")
        print(f"Spread: {signal.components.spread_ratio_1min:.2f}x")
    
    try:
        await calculator.start_websocket([TEST_SYMBOL], display_signal)
        print(f"✓ WebSocket connected, monitoring {TEST_SYMBOL}")
        print(f"⏰ Running for {TEST_DURATION} seconds...")
        
        await asyncio.sleep(TEST_DURATION)
        
        await calculator.stop()
        print(f"\n✓ WebSocket test complete: {signal_count} signals received")
        
    except Exception as e:
        print(f"⚠️  WebSocket test skipped: {e}")
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    print("Bid/Ask Imbalance Analysis Module")
    print("Measuring order flow aggression through bid/ask execution analysis\n")
    
    asyncio.run(run_test())