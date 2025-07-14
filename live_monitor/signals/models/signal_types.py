# live_monitor/signals/models/signal_types.py
"""
Signal type definitions and data models
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any


class SignalCategory(Enum):
    """Standard signal categories"""
    BULLISH = "Bullish"           # 25 to 50
    WEAK_BULLISH = "Weak Bullish" # 0 to 24
    WEAK_BEARISH = "Weak Bearish" # -24 to 0
    BEARISH = "Bearish"           # -50 to -25
    
    @classmethod
    def from_value(cls, value: float) -> 'SignalCategory':
        """Get category from signal value"""
        if value >= 25:
            return cls.BULLISH
        elif value > 0:
            return cls.WEAK_BULLISH
        elif value > -25:
            return cls.WEAK_BEARISH
        else:
            return cls.BEARISH


@dataclass
class StandardSignal:
    """Standardized signal output"""
    value: float  # -50 to 50
    category: SignalCategory
    source: str  # e.g., "M1_EMA"
    confidence: float  # 0-1
    direction: str  # 'LONG' or 'SHORT'
    strength: str  # 'Strong', 'Medium', 'Weak'
    metadata: Dict[str, Any] = field(default_factory=dict)
        
    @property
    def should_generate_exit(self) -> bool:
        """Determine if this signal should generate an exit"""
        # Generate exits when signal weakens or reverses
        return (abs(self.value) < 10 or  # Signal weakening
                (self.value > 0 and self.metadata.get('previous_value', 0) < -10) or  # Reversal
                (self.value < 0 and self.metadata.get('previous_value', 0) > 10))