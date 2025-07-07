# modules/calculations/volume/m1_bid_ask_analysis.py
"""
Module: M1 Bid/Ask Volume Analyzer
Purpose: Aggregate trades into 1-minute volume bars with bid/ask classification
Features: Above/Below bid/ask classification, 14-bar lookback, volume imbalance analysis
Output: Bid/ask volume breakdowns and directional signals
Time Handling: All timestamps in UTC
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque
from typing import Dict, List, Deque, Optional, Tuple
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s UTC - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Individual trade data with bid/ask context"""
    timestamp: datetime
    price: float
    size: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    classification: Optional[str] = None  # ABOVE_ASK, BELOW_BID, AT_ASK, AT_BID, BETWEEN


@dataclass
class BidAskVolumeBar:
    """M1 Volume Bar with detailed bid/ask breakdown"""
    timestamp: datetime  # Bar start time (UTC)
    total_volume: float
    
    # Aggressive volumes
    above_ask_volume: float  # Aggressive buying
    below_bid_volume: float  # Aggressive selling
    
    # Passive volumes
    at_ask_volume: float     # Passive buying
    at_bid_volume: float     # Passive selling
    
    # Neutral volume
    between_spread_volume: float  # Between bid/ask
    unknown_volume: float    # No quote data available
    
    # Derived metrics
    aggressive_buy_ratio: float  # Above ask / total aggressive
    aggressive_sell_ratio: float # Below bid / total aggressive
    buy_pressure: float     # (Above ask + At ask) / total with quotes
    
    # Standard OHLC
    trade_count: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    
    # Spread metrics
    avg_spread: float
    avg_spread_bps: float  # Basis points


class M1VolumeAnalyzer:
    """
    Enhanced M1 volume bar analyzer with bid/ask classification
    All timestamps must be in UTC
    """
    
    def __init__(self, lookback_bars: int = 14):
        """
        Initialize M1 bid/ask volume analyzer
        
        Args:
            lookback_bars: Number of completed bars to maintain for analysis
        """
        self.lookback_bars = lookback_bars
        self.bars: Dict[str, Deque[BidAskVolumeBar]] = {}  # Symbol -> completed bars
        self.current_bar: Dict[str, Dict] = {}  # Symbol -> accumulator
        self.last_trade: Dict[str, Trade] = {}  # Symbol -> last trade for fallback classification
        self.trades_processed = 0
        self.bars_completed = 0  # Track total bars completed
        
        logger.info(f"M1 Bid/Ask Analyzer initialized with {lookback_bars} bar lookback")
    
    def process_trade_with_context(self, symbol: str, timestamp: datetime, 
                                 price: float, size: float,
                                 bid: Optional[float] = None,
                                 ask: Optional[float] = None) -> Optional[Dict]:
        """
        Process a single trade with bid/ask context
        
        Step 1: Normalize timestamp to minute boundary
        Step 2: Classify trade (Above Ask, Below Bid, etc.)
        Step 3: Add to appropriate minute bar
        Step 4: Return bar completion info when minute changes
        """
        # Handle pandas Timestamp
        if hasattr(timestamp, 'to_pydatetime'):
            timestamp = timestamp.to_pydatetime()
        
        # Ensure UTC
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        elif timestamp.tzinfo != timezone.utc:
            timestamp = timestamp.astimezone(timezone.utc)
        
        # Initialize if needed
        if symbol not in self.bars:
            self.bars[symbol] = deque(maxlen=self.lookback_bars)
            self.current_bar[symbol] = None
        
        # Step 2: Classify trade
        classification = self._classify_trade_with_quotes(price, bid, ask, symbol)
        
        # Create trade object
        trade = Trade(
            timestamp=timestamp,
            price=price,
            size=size,
            bid=bid,
            ask=ask,
            classification=classification
        )
        
        # Store last trade
        self.last_trade[symbol] = trade
        
        # Step 3 & 4: Process into bars
        bar_completed = self._add_trade_to_bars(symbol, trade)
        
        self.trades_processed += 1
        
        if bar_completed:
            return {
                'bar_completed': True,
                'bar_timestamp': bar_completed.timestamp,
                'bars_completed_total': self.bars_completed,
                'bars_in_memory': len(self.bars[symbol])
            }
        
        return None
    
    def process_trade(self, symbol: str, timestamp: datetime, 
                     price: float, size: float) -> Optional[Dict]:
        """
        Process trade without bid/ask context (backward compatibility)
        Falls back to tick rule classification
        """
        return self.process_trade_with_context(symbol, timestamp, price, size)
    
    def _classify_trade_with_quotes(self, price: float, bid: Optional[float], 
                                  ask: Optional[float], symbol: str) -> str:
        """
        Classify trade based on bid/ask quotes
        
        Returns classification: ABOVE_ASK, BELOW_BID, AT_ASK, AT_BID, BETWEEN, UNKNOWN
        """
        if bid is None or ask is None:
            # No quote data - try tick rule as fallback
            if symbol in self.last_trade and self.last_trade[symbol].price is not None:
                last_price = self.last_trade[symbol].price
                if price > last_price:
                    return "UNKNOWN_UPTICK"
                elif price < last_price:
                    return "UNKNOWN_DOWNTICK"
            return "UNKNOWN"
        
        # Classify based on bid/ask
        if price > ask:
            return "ABOVE_ASK"
        elif price < bid:
            return "BELOW_BID"
        elif price == ask:
            return "AT_ASK"
        elif price == bid:
            return "AT_BID"
        else:
            return "BETWEEN"
    
    def _add_trade_to_bars(self, symbol: str, trade: Trade) -> Optional[BidAskVolumeBar]:
        """
        Add trade to appropriate bar based on minute timestamp
        
        Returns completed bar if a new minute started, None otherwise
        """
        # Get bar timestamp (minute boundary)
        bar_time = self._get_bar_time(trade.timestamp)
        
        # First trade - start new bar
        if self.current_bar[symbol] is None:
            self._start_new_bar(symbol, bar_time, trade)
            return None
        
        current_bar_time = self.current_bar[symbol]['bar_time']
        
        # Same minute - add to current bar
        if bar_time == current_bar_time:
            self._update_current_bar(symbol, trade)
            return None
        
        # New minute - complete previous bar and start new one
        elif bar_time > current_bar_time:
            # Complete previous bar
            completed_bar = self._complete_bar(symbol)
            if completed_bar:
                self.bars[symbol].append(completed_bar)
                self.bars_completed += 1
                logger.debug(f"Completed {symbol} bar at {completed_bar.timestamp}: "
                           f"Above Ask: {completed_bar.above_ask_volume}, "
                           f"Below Bid: {completed_bar.below_bid_volume}")
            
            # Start new bar
            self._start_new_bar(symbol, bar_time, trade)
            
            return completed_bar
        
        else:
            # Trade is before current bar (shouldn't happen in sequential processing)
            logger.warning(f"Out of order trade: {trade.timestamp} before current bar {current_bar_time}")
            return None
    
    def _get_bar_time(self, timestamp: datetime) -> datetime:
        """
        Get M1 bar start time (minute boundary)
        Ensures all trades in same minute go to same bar
        """
        return datetime(
            year=timestamp.year,
            month=timestamp.month,
            day=timestamp.day,
            hour=timestamp.hour,
            minute=timestamp.minute,
            second=0,
            microsecond=0,
            tzinfo=timestamp.tzinfo
        )
    
    def _start_new_bar(self, symbol: str, bar_time: datetime, trade: Trade):
        """Start a new bar accumulator"""
        self.current_bar[symbol] = {
            'bar_time': bar_time,
            'open_price': trade.price,
            'high_price': trade.price,
            'low_price': trade.price,
            'close_price': trade.price,
            'above_ask_volume': 0,
            'below_bid_volume': 0,
            'at_ask_volume': 0,
            'at_bid_volume': 0,
            'between_spread_volume': 0,
            'unknown_volume': 0,
            'trade_count': 0,  # Start at 0, will be incremented
            'spread_sum': 0,
            'spread_count': 0
        }
        
        # Add volume based on classification
        self._add_volume_to_bar(self.current_bar[symbol], trade)
    
    def _update_current_bar(self, symbol: str, trade: Trade):
        """Update current bar with new trade"""
        bar = self.current_bar[symbol]
        
        # Update OHLC
        bar['high_price'] = max(bar['high_price'], trade.price)
        bar['low_price'] = min(bar['low_price'], trade.price)
        bar['close_price'] = trade.price
        
        # Add volume based on classification
        self._add_volume_to_bar(bar, trade)
    
    def _add_volume_to_bar(self, bar: Dict, trade: Trade):
        """
        Step 3: Aggregate volumes based on classification
        """
        bar['trade_count'] += 1
        
        # Aggressive volumes (most important for analysis)
        if trade.classification == "ABOVE_ASK":
            bar['above_ask_volume'] += trade.size
        elif trade.classification == "BELOW_BID":
            bar['below_bid_volume'] += trade.size
        
        # Passive volumes
        elif trade.classification == "AT_ASK":
            bar['at_ask_volume'] += trade.size
        elif trade.classification == "AT_BID":
            bar['at_bid_volume'] += trade.size
        
        # Neutral
        elif trade.classification == "BETWEEN":
            bar['between_spread_volume'] += trade.size
        else:  # UNKNOWN
            bar['unknown_volume'] += trade.size
        
        # Track spread if available
        if trade.bid is not None and trade.ask is not None:
            spread = trade.ask - trade.bid
            bar['spread_sum'] += spread
            bar['spread_count'] += 1
    
    def _complete_bar(self, symbol: str) -> Optional[BidAskVolumeBar]:
        """
        Step 4: Complete bar and determine winner
        """
        bar = self.current_bar[symbol]
        if bar['trade_count'] == 0:
            return None
        
        # Calculate totals
        total_volume = (bar['above_ask_volume'] + bar['below_bid_volume'] + 
                       bar['at_ask_volume'] + bar['at_bid_volume'] + 
                       bar['between_spread_volume'] + bar['unknown_volume'])
        
        # Determine winner based on aggressive flow
        total_aggressive = bar['above_ask_volume'] + bar['below_bid_volume']
        
        if total_aggressive > 0:
            aggressive_buy_ratio = (bar['above_ask_volume'] / total_aggressive * 100)
            aggressive_sell_ratio = (bar['below_bid_volume'] / total_aggressive * 100)
        else:
            aggressive_buy_ratio = 50
            aggressive_sell_ratio = 50
        
        # Total buy/sell pressure including passive
        total_buy = bar['above_ask_volume'] + bar['at_ask_volume']
        total_sell = bar['below_bid_volume'] + bar['at_bid_volume']
        total_directional = total_buy + total_sell
        buy_pressure = (total_buy / total_directional * 100) if total_directional > 0 else 50
        
        # Spread metrics
        avg_spread = (bar['spread_sum'] / bar['spread_count']) if bar['spread_count'] > 0 else 0
        avg_price = (bar['high_price'] + bar['low_price'] + bar['close_price']) / 3
        avg_spread_bps = (avg_spread / avg_price * 10000) if avg_price > 0 else 0
        
        return BidAskVolumeBar(
            timestamp=bar['bar_time'],
            total_volume=total_volume,
            above_ask_volume=bar['above_ask_volume'],
            below_bid_volume=bar['below_bid_volume'],
            at_ask_volume=bar['at_ask_volume'],
            at_bid_volume=bar['at_bid_volume'],
            between_spread_volume=bar['between_spread_volume'],
            unknown_volume=bar['unknown_volume'],
            aggressive_buy_ratio=aggressive_buy_ratio,
            aggressive_sell_ratio=aggressive_sell_ratio,
            buy_pressure=buy_pressure,
            trade_count=bar['trade_count'],
            open_price=bar['open_price'],
            high_price=bar['high_price'],
            low_price=bar['low_price'],
            close_price=bar['close_price'],
            avg_spread=avg_spread,
            avg_spread_bps=avg_spread_bps
        )
    
    def get_current_analysis(self, symbol: str) -> Optional[Dict]:
        """
        Get current analysis based on completed bars
        This is the main method to call for getting analysis results
        """
        if symbol not in self.bars or len(self.bars[symbol]) == 0:
            return None
        
        # Only analyze if we have at least some bars (don't require full 14)
        return self._analyze_volume_trend(symbol)
    
    def _analyze_volume_trend(self, symbol: str) -> Dict:
        """
        Analyze all available bars (up to 14) for bid/ask volume trends
        
        Returns:
            Dict with comprehensive bid/ask volume analysis
        """
        bars_list = list(self.bars[symbol])
        
        # Calculate aggregates
        total_above_ask = sum(bar.above_ask_volume for bar in bars_list)
        total_below_bid = sum(bar.below_bid_volume for bar in bars_list)
        total_at_ask = sum(bar.at_ask_volume for bar in bars_list)
        total_at_bid = sum(bar.at_bid_volume for bar in bars_list)
        total_aggressive = total_above_ask + total_below_bid
        
        # Overall ratios
        aggressive_buy_ratio = (total_above_ask / total_aggressive * 100) if total_aggressive > 0 else 50
        
        # Recent vs older comparison (last 5 bars vs rest)
        if len(bars_list) >= 5:
            recent_bars = bars_list[-5:]
            older_bars = bars_list[:-5] if len(bars_list) > 5 else []
        else:
            recent_bars = bars_list
            older_bars = []
        
        recent_aggressive_buy = sum(b.above_ask_volume for b in recent_bars)
        recent_aggressive_sell = sum(b.below_bid_volume for b in recent_bars)
        recent_aggressive_total = recent_aggressive_buy + recent_aggressive_sell
        recent_buy_ratio = (recent_aggressive_buy / recent_aggressive_total * 100) if recent_aggressive_total > 0 else 50
        
        if older_bars:
            older_aggressive_buy = sum(b.above_ask_volume for b in older_bars)
            older_aggressive_sell = sum(b.below_bid_volume for b in older_bars)
            older_aggressive_total = older_aggressive_buy + older_aggressive_sell
            older_buy_ratio = (older_aggressive_buy / older_aggressive_total * 100) if older_aggressive_total > 0 else 50
        else:
            older_buy_ratio = recent_buy_ratio
        
        # Volume acceleration
        recent_avg_volume = np.mean([b.total_volume for b in recent_bars]) if recent_bars else 0
        older_avg_volume = np.mean([b.total_volume for b in older_bars]) if older_bars else recent_avg_volume
        volume_acceleration = ((recent_avg_volume / older_avg_volume) - 1) * 100 if older_avg_volume > 0 else 0
        
        # Price trend
        price_change = ((bars_list[-1].close_price / bars_list[0].open_price) - 1) * 100 if bars_list else 0
        
        # Average spread
        avg_spread_bps = np.mean([b.avg_spread_bps for b in bars_list]) if bars_list else 0
        
        # Generate signal based on aggressive flow
        signal = 'NEUTRAL'
        strength = 50
        reasons = []
        
        if aggressive_buy_ratio > 65:
            if recent_buy_ratio > older_buy_ratio:
                signal = 'BULLISH'
                strength = min(100, aggressive_buy_ratio)
                reasons.append(f"Strong aggressive buying {aggressive_buy_ratio:.0f}% with increasing momentum")
            else:
                signal = 'BULLISH'
                strength = min(80, aggressive_buy_ratio)
                reasons.append(f"Aggressive buying {aggressive_buy_ratio:.0f}% but momentum slowing")
        elif aggressive_buy_ratio < 35:
            if recent_buy_ratio < older_buy_ratio:
                signal = 'BEARISH'
                strength = min(100, 100 - aggressive_buy_ratio)
                reasons.append(f"Heavy aggressive selling {100-aggressive_buy_ratio:.0f}% with increasing momentum")
            else:
                signal = 'BEARISH'
                strength = min(80, 100 - aggressive_buy_ratio)
                reasons.append(f"Aggressive selling {100-aggressive_buy_ratio:.0f}% but momentum slowing")
        else:
            reasons.append(f"Balanced aggressive flow {aggressive_buy_ratio:.0f}% buy")
        
        # Add spread context
        if avg_spread_bps < 10:
            reasons.append(f"Tight spreads {avg_spread_bps:.1f}bps")
        elif avg_spread_bps > 20:
            reasons.append(f"Wide spreads {avg_spread_bps:.1f}bps")
        
        # Add volume context
        if volume_acceleration > 50:
            reasons.append(f"Volume surging +{volume_acceleration:.0f}%")
            strength = min(100, strength * 1.1)
        elif volume_acceleration < -30:
            reasons.append(f"Volume declining {volume_acceleration:.0f}%")
            strength = strength * 0.9
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now(timezone.utc),
            'signal': signal,
            'strength': float(strength),
            'timeframe': 'M1',
            'bars_analyzed': len(bars_list),
            'reason': ' | '.join(reasons),
            'metrics': {
                'total_above_ask_volume': total_above_ask,
                'total_below_bid_volume': total_below_bid,
                'total_at_ask_volume': total_at_ask,
                'total_at_bid_volume': total_at_bid,
                'aggressive_buy_ratio': aggressive_buy_ratio,
                'recent_buy_ratio': recent_buy_ratio,
                'older_buy_ratio': older_buy_ratio,
                'volume_acceleration': volume_acceleration,
                'avg_spread_bps': avg_spread_bps,
                'total_trades': sum(b.trade_count for b in bars_list),
                'price_change_pct': price_change
            }
        }
    
    def get_current_bars(self, symbol: str) -> List[BidAskVolumeBar]:
        """Get current completed bars for a symbol"""
        if symbol in self.bars:
            return list(self.bars[symbol])
        return []
    
    def force_complete_bar(self, symbol: str) -> Optional[Dict]:
        """
        Force complete current bar and return full analysis
        Use this at the end of processing or when you need final results
        """
        if symbol not in self.current_bar or self.current_bar[symbol] is None:
            # No current bar to complete, just return analysis if we have bars
            return self.get_current_analysis(symbol)
        
        # Complete the current bar
        completed_bar = self._complete_bar(symbol)
        if completed_bar:
            self.bars[symbol].append(completed_bar)
            self.bars_completed += 1
            self.current_bar[symbol] = None
            logger.debug(f"Force completed bar for {symbol} at {completed_bar.timestamp}")
        
        # Return full analysis
        return self.get_current_analysis(symbol)