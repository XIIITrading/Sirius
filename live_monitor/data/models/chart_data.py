# live_monitor/data/models/chart_data.py
"""
Chart data models
"""
from typing import TypedDict, List, Literal
from datetime import datetime


class ChartBar(TypedDict):
    """OHLCV bar for charting"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    

class ChartUpdate(TypedDict):
    """Chart update message"""
    symbol: str
    timeframe: Literal['1min', '5min', '15min', '1hour', '1day']
    bars: List[ChartBar]
    is_update: bool  # True for partial update, False for full refresh