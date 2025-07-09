"""
Buy/Sell Ratio Calculator using Pre-aligned Trades from TradeQuoteAligner
Purpose: Calculate buying vs selling pressure at 1-minute intervals
Output: Percentage from -1 (all selling) to +1 (all buying)
Window: Rolling 30-minute history for charting
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MinuteBar:
    """1-minute aggregated data"""
    symbol: str
    timestamp: datetime  # Start of minute
    weighted_pressure: float  # -1 to +1 based on buy/sell volume
    positive_volume: float  # Buy volume
    negative_volume: float  # Sell volume
    total_volume: float
    trade_count: int
    buy_sell_ratio: float  # Backward compatibility
    high: float
    low: float
    close: float
    avg_spread: float
    classified_trades: int  # Number of trades with buy/sell classification
    avg_confidence: float  # Average confidence from alignment


class BuySellRatioCalculator:
    """
    Calculate buy/sell ratios from pre-aligned trade data.
    Simply aggregates the already-classified trades into minute bars.
    """
    
    def __init__(self, window_minutes: int = 30, min_confidence: float = 0.5):
        self.window_minutes = window_minutes
        self.min_confidence = min_confidence
        logger.info(f"Initialized BuySellRatioCalculator with {window_minutes}-minute window")
    
    def process_aligned_trades(self, aligned_df: pd.DataFrame, symbol: str) -> List[MinuteBar]:
        """
        Process pre-aligned trades into minute bars.
        
        The aligned_df already contains:
        - trade_side: 'buy', 'sell', 'unknown', 'midpoint'
        - confidence: 0-1 score
        - All quote data pre-synchronized
        """
        if aligned_df.empty:
            logger.warning("Empty aligned trades DataFrame")
            return []
        
        # Filter by confidence
        if 'confidence' in aligned_df.columns:
            df = aligned_df[aligned_df['confidence'] >= self.min_confidence].copy()
            logger.info(f"Using {len(df)}/{len(aligned_df)} trades with confidence >= {self.min_confidence}")
        else:
            df = aligned_df.copy()
        
        if df.empty:
            return []
        
        # CRITICAL FIX: Set index to trade_time
        df = df.set_index('trade_time')
        
        # Now we can group by minute
        df['minute'] = df.index.floor('1min')
        
        minute_bars = []
        
        for minute, group in df.groupby('minute'):
            # Calculate volumes by side
            buy_trades = group[group['trade_side'] == 'buy']
            sell_trades = group[group['trade_side'] == 'sell']
            
            buy_volume = buy_trades['trade_size'].sum() if len(buy_trades) > 0 else 0
            sell_volume = sell_trades['trade_size'].sum() if len(sell_trades) > 0 else 0
            total_volume = group['trade_size'].sum()
            
            # Calculate weighted pressure (-1 to +1)
            if total_volume > 0:
                # Simple calculation: (buy - sell) / total
                weighted_pressure = (buy_volume - sell_volume) / total_volume
            else:
                weighted_pressure = 0.0
            
            # Other metrics
            avg_spread = group['spread'].mean() if 'spread' in group.columns else 0
            avg_confidence = group['confidence'].mean() if 'confidence' in group.columns else 1.0
            classified_trades = len(group[group['trade_side'].isin(['buy', 'sell'])])
            
            # minute is already a Timestamp and should be timezone-aware
            bar = MinuteBar(
                symbol=symbol,
                timestamp=minute.to_pydatetime() if hasattr(minute, 'to_pydatetime') else minute,
                weighted_pressure=weighted_pressure,
                positive_volume=float(buy_volume),
                negative_volume=float(sell_volume),
                total_volume=float(total_volume),
                trade_count=len(group),
                buy_sell_ratio=weighted_pressure,  # Backward compatibility
                high=group['trade_price'].max(),
                low=group['trade_price'].min(),
                close=group['trade_price'].iloc[-1],
                avg_spread=float(avg_spread),
                classified_trades=classified_trades,
                avg_confidence=float(avg_confidence)
            )
            
            minute_bars.append(bar)
        
        # Sort by timestamp and keep only last window_minutes
        minute_bars.sort(key=lambda x: x.timestamp)
        if len(minute_bars) > self.window_minutes:
            minute_bars = minute_bars[-self.window_minutes:]
        
        return minute_bars
    
    def get_chart_data(self, minute_bars: List[MinuteBar]) -> List[Dict]:
        """Convert minute bars to chart data format"""
        return [
            {
                'timestamp': bar.timestamp.isoformat(),
                'buy_sell_ratio': bar.weighted_pressure,
                'volume': bar.total_volume,
                'positive_volume': bar.positive_volume,
                'negative_volume': bar.negative_volume,
                'classified_trades': bar.classified_trades,
                'classification_rate': bar.classified_trades / max(1, bar.trade_count),
                'avg_confidence': bar.avg_confidence
            }
            for bar in minute_bars
        ]
    
    def get_summary_stats(self, minute_bars: List[MinuteBar]) -> Dict:
        """Get summary statistics from minute bars"""
        if not minute_bars:
            return {}
        
        ratios = [bar.weighted_pressure for bar in minute_bars]
        
        # Calculate classification rate
        total_trades = sum(bar.trade_count for bar in minute_bars)
        classified_trades = sum(bar.classified_trades for bar in minute_bars)
        classification_rate = classified_trades / max(1, total_trades)
        
        # Average confidence
        avg_confidence = sum(bar.avg_confidence * bar.trade_count for bar in minute_bars) / max(1, total_trades)
        
        return {
            'current_ratio': minute_bars[-1].weighted_pressure,
            'avg_ratio': sum(ratios) / len(ratios),
            'max_ratio': max(ratios),
            'min_ratio': min(ratios),
            'total_volume': sum(bar.total_volume for bar in minute_bars),
            'minutes_tracked': len(minute_bars),
            'classification_rate': classification_rate,
            'avg_confidence': avg_confidence
        }