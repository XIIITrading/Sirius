# journal/trades/models.py
"""Data models for trades and executions."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from decimal import Decimal


@dataclass
class Execution:
    """Represents a single execution/fill."""
    event: str
    side: str  # B/S
    symbol: str
    shares: int
    price: Decimal
    route: str
    time: datetime
    account: str
    note: Optional[str] = None
    
    @property
    def value(self) -> Decimal:
        """Calculate the value of this execution."""
        return self.shares * self.price


@dataclass
class Trade:
    """Represents a complete trade with entry and exit."""
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: Decimal
    exit_price: Optional[Decimal]
    qty: int
    net_pl: Optional[Decimal]
    executions: List[Execution]
    
    def to_dict(self) -> dict:
        """Convert trade to dictionary for database storage."""
        return {
            'symbol': self.symbol,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'entry_price': float(self.entry_price),
            'exit_price': float(self.exit_price) if self.exit_price else None,
            'qty': self.qty,
            'net_pl': float(self.net_pl) if self.net_pl else None,
        }