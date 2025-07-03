"""
Simplified Delta Tracker with Continuous Weighting
Purpose: Track buying vs selling pressure at 1-minute intervals
Output: Percentage from -1 (all selling) to +1 (all buying)
Window: Rolling 30-minute history for charting
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Individual trade data"""
    symbol: str
    price: float
    size: int
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None


@dataclass
class Quote:
    """Quote data for classification"""
    symbol: str
    bid: float
    ask: float
    timestamp: datetime
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None


@dataclass
class MinuteBar:
    """1-minute aggregated data"""
    symbol: str
    timestamp: datetime  # Start of minute
    weighted_pressure: float  # -1 to +1 based on continuous weighting
    positive_volume: float  # Volume with positive weight
    negative_volume: float  # Volume with negative weight
    total_volume: float
    trade_count: int
    buy_sell_ratio: float  # Backward compatibility
    high: float
    low: float
    close: float
    avg_spread: float
    classified_trades: int  # Number of trades with valid quotes


class SimpleDeltaTracker:
    """
    Delta tracker using continuous pressure weighting.
    Trades are weighted based on their position relative to bid/ask spread.
    """
    
    def __init__(self, window_minutes: int = 30, quote_sync_tolerance_ms: int = 500):
        """
        Initialize tracker.
        
        Args:
            window_minutes: Number of minutes to keep in history (default 30)
            quote_sync_tolerance_ms: Max milliseconds between trade and quote for sync (increased default)
        """
        self.window_minutes = window_minutes
        self.quote_sync_tolerance_ms = quote_sync_tolerance_ms
        
        # Current minute accumulator per symbol
        self.current_minute: Dict[str, Dict] = {}
        
        # Historical bars per symbol
        self.minute_bars: Dict[str, deque] = {}
        
        # Quote history for synchronization
        self.quote_history: Dict[str, deque] = {}
        self.latest_quotes: Dict[str, Quote] = {}
        
        # Track sync stats
        self.sync_stats = {
            'total_trades': 0,
            'synced_trades': 0,
            'failed_syncs': 0
        }
        
        logger.info(f"Initialized SimpleDeltaTracker with {window_minutes}-minute window, "
                   f"quote sync tolerance: {quote_sync_tolerance_ms}ms")
    
    def update_quote(self, symbol: str, bid: float, ask: float, timestamp: datetime):
        """Update quote data for trade classification"""
        quote = Quote(
            symbol=symbol,
            bid=bid,
            ask=ask,
            timestamp=timestamp
        )
        
        # Store latest quote
        self.latest_quotes[symbol] = quote
        
        # Add to history
        if symbol not in self.quote_history:
            self.quote_history[symbol] = deque(maxlen=50000)  # Increased capacity
        
        self.quote_history[symbol].append(quote)
        
        # Clean old quotes periodically
        if len(self.quote_history[symbol]) > 40000:
            self._cleanup_old_quotes(symbol, timestamp)
    
    def _cleanup_old_quotes(self, symbol: str, current_time: datetime):
        """Remove quotes older than 10 minutes"""
        cutoff = current_time - timedelta(minutes=10)
        while self.quote_history[symbol] and self.quote_history[symbol][0].timestamp < cutoff:
            self.quote_history[symbol].popleft()
    
    def _get_synchronized_quote(self, trade: Trade) -> Optional[Quote]:
        """Get quote synchronized with trade timestamp - handles future quotes"""
        if trade.symbol not in self.quote_history:
            return None
        
        # First check if trade has embedded bid/ask
        if trade.bid and trade.ask:
            return Quote(
                symbol=trade.symbol,
                bid=trade.bid,
                ask=trade.ask,
                timestamp=trade.timestamp
            )
        
        best_quote = None
        best_time_diff = float('inf')
        
        # Search for the closest quote in time (before or after trade)
        for quote in reversed(self.quote_history[trade.symbol]):
            # Calculate absolute time difference
            time_diff = abs((trade.timestamp - quote.timestamp).total_seconds() * 1000)
            
            # If this quote is closer in time, use it
            if time_diff < best_time_diff and time_diff <= self.quote_sync_tolerance_ms:
                best_quote = quote
                best_time_diff = time_diff
            
            # Stop searching if we've gone too far back
            if quote.timestamp < trade.timestamp - timedelta(milliseconds=self.quote_sync_tolerance_ms * 2):
                break
        
        return best_quote
    
    def _calculate_trade_weight(self, trade: Trade, quote: Quote) -> float:
        """
        Calculate continuous weight for trade based on position in spread.
        Returns value between -2 and +2 (can exceed ±1 for trades outside spread).
        """
        spread = quote.ask - quote.bid
        
        # Handle zero or negative spread
        if spread <= 0:
            # Use simple classification
            if trade.price >= quote.ask:
                return 1.0
            elif trade.price <= quote.bid:
                return -1.0
            else:
                return 0.0
        
        # Calculate position in spread
        mid_price = (quote.bid + quote.ask) / 2
        
        # If trade is outside the spread, calculate extended weight
        if trade.price > quote.ask:
            # Above ask - strong buying pressure
            excess = (trade.price - quote.ask) / spread
            return min(1.0 + excess, 2.0)  # Cap at 2.0
        
        elif trade.price < quote.bid:
            # Below bid - strong selling pressure
            excess = (quote.bid - trade.price) / spread
            return max(-1.0 - excess, -2.0)  # Cap at -2.0
        
        else:
            # Inside spread - linear interpolation
            # At bid = -1, at mid = 0, at ask = +1
            position = (trade.price - quote.bid) / spread  # 0 to 1
            return (position * 2) - 1  # Convert to -1 to +1
    
    def process_trade(self, trade: Trade) -> Optional[MinuteBar]:
        """
        Process a trade and return completed minute bar if minute rolled over.
        """
        symbol = trade.symbol
        
        # Initialize symbol if needed
        if symbol not in self.minute_bars:
            self.minute_bars[symbol] = deque(maxlen=self.window_minutes)
            self.current_minute[symbol] = self._new_minute_accumulator(trade.timestamp)
        
        # Get current minute
        minute_start = self._get_minute_start(trade.timestamp)
        current_acc = self.current_minute[symbol]
        
        # Check if we need to roll to new minute
        completed_bar = None
        if minute_start > current_acc['minute_start']:
            # Complete current minute
            completed_bar = self._complete_minute(symbol)
            
            # Start new minute
            self.current_minute[symbol] = self._new_minute_accumulator(trade.timestamp)
            current_acc = self.current_minute[symbol]
        
        # Get synchronized quote and calculate weight
        quote = self._get_synchronized_quote(trade)
        self.sync_stats['total_trades'] += 1
        
        if quote:
            self.sync_stats['synced_trades'] += 1
            weight = self._calculate_trade_weight(trade, quote)
            
            # Add weighted volume
            weighted_volume = trade.size * weight
            current_acc['weighted_volume'] += weighted_volume
            current_acc['classified_trades'] += 1
            
            # Track positive/negative volumes
            if weight > 0:
                current_acc['positive_volume'] += trade.size * abs(weight)
            else:
                current_acc['negative_volume'] += trade.size * abs(weight)
            
            # Track spread
            spread = quote.ask - quote.bid
            current_acc['spread_sum'] += spread
            current_acc['spread_count'] += 1
            
            # Debug log for extreme weights
            if abs(weight) > 1.5:
                logger.debug(f"Extreme weight {weight:.2f}: Trade at {trade.price:.2f}, "
                           f"Bid={quote.bid:.2f}, Ask={quote.ask:.2f}")
        else:
            self.sync_stats['failed_syncs'] += 1
            # For unsynced trades, still count volume but with zero weight
            current_acc['unclassified_volume'] += trade.size
        
        # Always update basic stats
        current_acc['trade_count'] += 1
        current_acc['total_volume'] += trade.size
        
        # Update price range
        if current_acc['high'] == 0 or trade.price > current_acc['high']:
            current_acc['high'] = trade.price
        if current_acc['low'] == float('inf') or trade.price < current_acc['low']:
            current_acc['low'] = trade.price
        current_acc['close'] = trade.price
        
        return completed_bar
    
    def _new_minute_accumulator(self, timestamp: datetime) -> Dict:
        """Create new minute accumulator with continuous weighting support"""
        minute_start = self._get_minute_start(timestamp)
        return {
            'minute_start': minute_start,
            'weighted_volume': 0.0,  # Sum of (volume * weight)
            'positive_volume': 0.0,  # Volume with positive weight
            'negative_volume': 0.0,  # Volume with negative weight
            'unclassified_volume': 0,  # Volume without quotes
            'total_volume': 0,
            'trade_count': 0,
            'classified_trades': 0,
            'spread_sum': 0.0,
            'spread_count': 0,
            'high': 0,
            'low': float('inf'),
            'close': 0
        }
    
    def _get_minute_start(self, timestamp: datetime) -> datetime:
        """Get start of minute for timestamp"""
        return timestamp.replace(second=0, microsecond=0)
    
    def _complete_minute(self, symbol: str) -> MinuteBar:
        """Complete current minute and return bar"""
        acc = self.current_minute[symbol]
        
        # Calculate weighted pressure ratio
        if acc['total_volume'] > 0:
            # Weighted pressure: sum of weighted volumes / total volume
            weighted_pressure = acc['weighted_volume'] / acc['total_volume']
            # Clamp to [-1, 1] range
            weighted_pressure = max(-1.0, min(1.0, weighted_pressure))
        else:
            weighted_pressure = 0.0
        
        # Calculate average spread
        avg_spread = acc['spread_sum'] / acc['spread_count'] if acc['spread_count'] > 0 else 0.0
        
        # Handle price edge cases
        if acc['high'] == 0:
            acc['high'] = acc['close']
        if acc['low'] == float('inf'):
            acc['low'] = acc['close']
        
        # Create bar
        bar = MinuteBar(
            symbol=symbol,
            timestamp=acc['minute_start'],
            weighted_pressure=weighted_pressure,
            positive_volume=acc['positive_volume'],
            negative_volume=acc['negative_volume'],
            total_volume=acc['total_volume'],
            trade_count=acc['trade_count'],
            buy_sell_ratio=weighted_pressure,  # For backward compatibility
            high=acc['high'],
            low=acc['low'],
            close=acc['close'],
            avg_spread=avg_spread,
            classified_trades=acc['classified_trades']
        )
        
        # Add to history
        self.minute_bars[symbol].append(bar)
        
        # Log sync rate periodically
        if self.sync_stats['total_trades'] % 1000 == 0:
            sync_rate = (self.sync_stats['synced_trades'] / 
                        max(1, self.sync_stats['total_trades'])) * 100
            logger.info(f"Quote sync rate: {sync_rate:.1f}% "
                       f"({self.sync_stats['synced_trades']}/{self.sync_stats['total_trades']})")
        
        return bar
    
    def get_chart_data(self, symbol: str) -> List[Dict]:
        """
        Get data formatted for charting.
        """
        if symbol not in self.minute_bars:
            return []
        
        return [
            {
                'timestamp': bar.timestamp.isoformat(),
                'buy_sell_ratio': bar.weighted_pressure,  # Use weighted pressure
                'volume': bar.total_volume,
                'positive_volume': bar.positive_volume,
                'negative_volume': bar.negative_volume,
                'classified_trades': bar.classified_trades,
                'classification_rate': bar.classified_trades / max(1, bar.trade_count)
            }
            for bar in self.minute_bars[symbol]
        ]
    
    def get_latest_ratio(self, symbol: str) -> Optional[float]:
        """Get the most recent buy/sell ratio"""
        if symbol in self.minute_bars and self.minute_bars[symbol]:
            return self.minute_bars[symbol][-1].weighted_pressure
        return None
    
    def get_summary_stats(self, symbol: str) -> Dict:
        """Get summary statistics for the symbol"""
        if symbol not in self.minute_bars or not self.minute_bars[symbol]:
            return {}
        
        bars = list(self.minute_bars[symbol])
        ratios = [bar.weighted_pressure for bar in bars]
        
        # Calculate classification rate
        total_trades = sum(bar.trade_count for bar in bars)
        classified_trades = sum(bar.classified_trades for bar in bars)
        classification_rate = classified_trades / max(1, total_trades)
        
        return {
            'current_ratio': bars[-1].weighted_pressure,
            'avg_ratio': np.mean(ratios),
            'max_ratio': max(ratios),
            'min_ratio': min(ratios),
            'total_volume': sum(bar.total_volume for bar in bars),
            'minutes_tracked': len(bars),
            'classification_rate': classification_rate,
            'sync_stats': self.sync_stats.copy()
        }