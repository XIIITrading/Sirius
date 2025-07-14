# live_monitor/signals/processors/base_processor.py
"""
Base class for all signal processors
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from ..models.signal_types import StandardSignal, SignalCategory
from ..models.constants import (
    MIN_SIGNAL_VALUE, MAX_SIGNAL_VALUE,
    STRONG_SIGNAL_THRESHOLD, MEDIUM_SIGNAL_THRESHOLD
)


class BaseSignalProcessor(ABC):
    """Abstract base class for all signal processors"""
    
    def __init__(self, source_name: str):
        self.source_name = source_name
        self.previous_signal: Optional[StandardSignal] = None
    
    @abstractmethod
    def process(self, result: Any) -> StandardSignal:
        """Process indicator result into StandardSignal"""
        pass
    
    @abstractmethod
    def calculate_confidence(self, result: Any, signal_value: float) -> float:
        """Calculate confidence level for the signal"""
        pass
    
    def get_previous_value(self) -> float:
        """Get previous signal value for comparison"""
        return self.previous_signal.value if self.previous_signal else 0
    
    def update_previous_signal(self, signal: StandardSignal):
        """Store signal for next comparison"""
        self.previous_signal = signal
    
    def clamp_value(self, value: float) -> float:
        """Ensure value stays within valid range"""
        return max(MIN_SIGNAL_VALUE, min(MAX_SIGNAL_VALUE, value))
    
    def get_strength_label(self, value: float) -> str:
        """Determine strength label from value"""
        abs_value = abs(value)
        if abs_value >= STRONG_SIGNAL_THRESHOLD:
            return 'Strong'
        elif abs_value >= MEDIUM_SIGNAL_THRESHOLD:
            return 'Medium'
        else:
            return 'Weak'
    
    def get_direction(self, value: float) -> str:
        """Get direction from value"""
        return 'LONG' if value > 0 else 'SHORT'
    
    def create_signal(self, value: float, confidence: float, 
                     metadata: Dict[str, Any]) -> StandardSignal:
        """Create a StandardSignal with common processing"""
        clamped_value = self.clamp_value(value)
        
        signal = StandardSignal(
            value=round(clamped_value, 1),
            category=SignalCategory.from_value(clamped_value),
            source=self.source_name,
            confidence=confidence,
            direction=self.get_direction(clamped_value),
            strength=self.get_strength_label(clamped_value),
            metadata=metadata
        )
        
        self.update_previous_signal(signal)
        return signal