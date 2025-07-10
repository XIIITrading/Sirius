# live_monitor/dashboard/components/chart/data/models.py
"""
Chart-specific data models
"""
from typing import Dict, List, Optional, Literal
from datetime import datetime
from dataclasses import dataclass, field


# Standardized timeframe types
TimeframeType = Literal['1m', '5m', '15m', '30m', '1h', '4h', '1d']

# Timeframe conversions (display -> internal)
TIMEFRAME_MAPPINGS = {
    '1m': 1,
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1h': 60,
    '4h': 240,
    '1d': 1440,
}


@dataclass
class Bar:
    """Single OHLCV bar"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    trades: int = 0  # Number of trades in this bar
    vwap: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for signals"""
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'trades': self.trades,
            'vwap': self.vwap
        }


@dataclass
class ChartDataUpdate:
    """Update message for chart widget"""
    symbol: str
    timeframe: TimeframeType
    bars: List[Bar]
    is_update: bool  # True for partial update, False for full refresh
    latest_bar_complete: bool = False  # True when a bar just completed