# live_monitor/data/models/market_data.py
"""
Market data models for WebSocket messages and UI updates
"""
from typing import TypedDict, Optional, List, Literal
from datetime import datetime


class TradeData(TypedDict):
    """Raw trade data from WebSocket"""
    event_type: Literal['trade']
    symbol: str
    timestamp: int  # milliseconds
    price: float
    size: int
    conditions: List[int]
    exchange: Optional[str]


class QuoteData(TypedDict):
    """Raw quote data from WebSocket"""
    event_type: Literal['quote']
    symbol: str
    timestamp: int
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int
    exchange: Optional[str]


class AggregateData(TypedDict):
    """Aggregate bar data from WebSocket"""
    event_type: Literal['aggregate']
    symbol: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float]
    transactions: Optional[int]


class MarketDataUpdate(TypedDict):
    """Combined market data for UI updates"""
    symbol: str
    timestamp: datetime
    last_price: float
    bid: float
    ask: float
    spread: float
    mid_price: float
    volume: int
    
    # Optional fields
    last_size: Optional[int]
    bid_size: Optional[int]
    ask_size: Optional[int]
    

class TickerCalculationData(TypedDict):
    """Data format for TickerCalculations widget"""
    last_price: float
    bid: float
    ask: float
    spread: float
    mid_price: float
    volume: int
    
    # Daily stats (to be calculated)
    change: Optional[float]
    change_percent: Optional[float]
    day_high: Optional[float]
    day_low: Optional[float]
    day_open: Optional[float]
    
    # Additional market data
    last_update: str  # Formatted time
    market_state: Literal['open', 'closed', 'pre', 'post']