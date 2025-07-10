# live_monitor/data/models/signals.py
"""
Trading signal models
"""
from typing import TypedDict, Literal, Optional
from datetime import datetime


class EntrySignal(TypedDict):
    """Entry signal for PointCallEntry grid"""
    time: str  # Formatted time
    signal_type: Literal['LONG', 'SHORT']
    price: str  # Formatted price
    signal: str  # Signal description
    strength: Literal['Strong', 'Medium', 'Weak']
    notes: Optional[str]
    
    # Internal fields
    timestamp: datetime
    symbol: str


class ExitSignal(TypedDict):
    """Exit signal for PointCallExit grid"""
    time: str
    exit_type: Literal['TARGET', 'STOP', 'TRAIL']
    price: str
    pnl: str  # Formatted P&L percentage
    signal: str
    urgency: Literal['Urgent', 'Warning', 'Normal']
    
    # Internal fields
    timestamp: datetime
    symbol: str
    pnl_value: float  # Actual P&L value