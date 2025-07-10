# live_monitor/dashboard/components/chart/data/chart_data_aggregator.py
"""
Aggregates minute bars into higher timeframes
"""
import logging
from typing import Dict, List, Optional, Deque
from datetime import datetime, timedelta
from collections import defaultdict, deque

from .models import Bar, TimeframeType, TIMEFRAME_MAPPINGS, ChartDataUpdate  # Add ChartDataUpdate!

logger = logging.getLogger(__name__)


class ChartDataAggregator:
    """
    Aggregates 1-minute bars into higher timeframes
    
    Maintains rolling windows of bars for each timeframe and
    efficiently builds higher timeframe bars from minute data.
    """
    
    def __init__(self, max_bars: int = 500):
        """
        Args:
            max_bars: Maximum bars to keep per timeframe
        """
        self.max_bars = max_bars
        
        # Storage: {timeframe: deque of Bars}
        self.bars: Dict[TimeframeType, Deque[Bar]] = {
            tf: deque(maxlen=max_bars) 
            for tf in TIMEFRAME_MAPPINGS.keys()
        }
        
        # Track incomplete bars for each timeframe
        self.incomplete_bars: Dict[TimeframeType, Optional[Bar]] = {
            tf: None for tf in TIMEFRAME_MAPPINGS.keys()
        }
        
        # Track last processed minute for each timeframe
        self.last_processed: Dict[TimeframeType, Optional[datetime]] = {
            tf: None for tf in TIMEFRAME_MAPPINGS.keys()
        }
    
    def add_minute_bar(self, minute_bar: Bar) -> Dict[TimeframeType, ChartDataUpdate]:
        """
        Add a new minute bar and return updates for all affected timeframes
        
        Args:
            minute_bar: The incoming 1-minute bar
            
        Returns:
            Dictionary of timeframe -> ChartDataUpdate for updated timeframes
        """
        updates = {}
        
        # Always add to 1-minute timeframe
        self.bars['1m'].append(minute_bar)
        updates['1m'] = ChartDataUpdate(
            symbol='',  # Will be set by handler
            timeframe='1m',
            bars=[minute_bar],
            is_update=True,
            latest_bar_complete=True
        )
        
        # Process higher timeframes
        for timeframe in ['5m', '15m', '30m', '1h', '4h', '1d']:
            update = self._process_timeframe(timeframe, minute_bar)
            if update:
                updates[timeframe] = update
        
        return updates
    
    def _process_timeframe(self, timeframe: TimeframeType, minute_bar: Bar) -> Optional[ChartDataUpdate]:
        """
        Process a minute bar for a specific timeframe
        
        Returns:
            ChartDataUpdate if there's an update, None otherwise
        """
        minutes = TIMEFRAME_MAPPINGS[timeframe]
        bar_timestamp = minute_bar.timestamp
        
        # Calculate the timeframe boundary
        if timeframe == '1d':
            # Daily bars align to market open (9:30 AM ET)
            bar_start = bar_timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
            if bar_timestamp.time() < bar_start.time():
                bar_start -= timedelta(days=1)
        else:
            # Align to timeframe boundaries
            total_minutes = bar_timestamp.hour * 60 + bar_timestamp.minute
            aligned_minutes = (total_minutes // minutes) * minutes
            bar_start = bar_timestamp.replace(
                hour=aligned_minutes // 60,
                minute=aligned_minutes % 60,
                second=0,
                microsecond=0
            )
        
        # Check if we need to complete the previous bar
        incomplete = self.incomplete_bars[timeframe]
        
        if incomplete and bar_timestamp >= bar_start + timedelta(minutes=minutes):
            # Complete the previous bar
            self.bars[timeframe].append(incomplete)
            self.incomplete_bars[timeframe] = None
            
            # Start new bar
            new_bar = Bar(
                timestamp=bar_start,
                open=minute_bar.open,
                high=minute_bar.high,
                low=minute_bar.low,
                close=minute_bar.close,
                volume=minute_bar.volume,
                trades=minute_bar.trades,
                vwap=minute_bar.vwap
            )
            self.incomplete_bars[timeframe] = new_bar
            
            return ChartDataUpdate(
                symbol='',
                timeframe=timeframe,
                bars=[incomplete],  # Return the completed bar
                is_update=True,
                latest_bar_complete=True
            )
        
        elif incomplete and incomplete.timestamp == bar_start:
            # Update existing incomplete bar
            incomplete.high = max(incomplete.high, minute_bar.high)
            incomplete.low = min(incomplete.low, minute_bar.low)
            incomplete.close = minute_bar.close
            incomplete.volume += minute_bar.volume
            incomplete.trades += minute_bar.trades
            
            # Update VWAP if available
            if minute_bar.vwap and incomplete.vwap:
                # Simple volume-weighted average
                total_value = (incomplete.vwap * (incomplete.volume - minute_bar.volume) + 
                             minute_bar.vwap * minute_bar.volume)
                incomplete.vwap = total_value / incomplete.volume if incomplete.volume > 0 else minute_bar.vwap
            
            return ChartDataUpdate(
                symbol='',
                timeframe=timeframe,
                bars=[incomplete],
                is_update=True,
                latest_bar_complete=False
            )
        
        else:
            # Start a new incomplete bar
            new_bar = Bar(
                timestamp=bar_start,
                open=minute_bar.open,
                high=minute_bar.high,
                low=minute_bar.low,
                close=minute_bar.close,
                volume=minute_bar.volume,
                trades=minute_bar.trades,
                vwap=minute_bar.vwap
            )
            self.incomplete_bars[timeframe] = new_bar
            
            return ChartDataUpdate(
                symbol='',
                timeframe=timeframe,
                bars=[new_bar],
                is_update=True,
                latest_bar_complete=False
            )
    
    def get_bars(self, timeframe: TimeframeType, count: Optional[int] = None) -> List[Bar]:
        """
        Get historical bars for a timeframe
        
        Args:
            timeframe: The timeframe to get bars for
            count: Number of bars to return (None for all)
            
        Returns:
            List of bars (oldest to newest)
        """
        bars = list(self.bars[timeframe])
        
        # Add incomplete bar if exists
        if self.incomplete_bars[timeframe]:
            bars.append(self.incomplete_bars[timeframe])
        
        if count and len(bars) > count:
            return bars[-count:]
        
        return bars
    
    def clear(self, timeframe: Optional[TimeframeType] = None):
        """Clear data for specific timeframe or all timeframes"""
        if timeframe:
            self.bars[timeframe].clear()
            self.incomplete_bars[timeframe] = None
            self.last_processed[timeframe] = None
        else:
            for tf in TIMEFRAME_MAPPINGS.keys():
                self.bars[tf].clear()
                self.incomplete_bars[tf] = None
                self.last_processed[tf] = None