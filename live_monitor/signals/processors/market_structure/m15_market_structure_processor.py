"""
M15 Market Structure Signal Processor
"""

from typing import Dict, Any
from ..base_processor import BaseSignalProcessor
from ...models.signal_types import StandardSignal
from ...models.constants import *
from live_monitor.calculations.market_structure.m15_market_structure import MarketStructureSignal


class M15MarketStructureProcessor(BaseSignalProcessor):
    """Process M15 Market Structure signals"""
    
    def __init__(self):
        super().__init__("M15_MARKET_STRUCTURE")
    
    def process(self, result: MarketStructureSignal) -> StandardSignal:
        """Convert Market Structure result to standard signal"""
        # M15 gets highest signal values due to timeframe significance
        if result.signal == 'BULL':
            base_value = 45 if result.structure_type == 'CHoCH' else 30
        else:  # BEAR
            base_value = -45 if result.structure_type == 'CHoCH' else -30
        
        # Adjust based on strength (0-100)
        signal_value = base_value * (result.strength / 100)
        
        # CHoCH signals on M15 are very significant
        if result.structure_type == 'CHoCH':
            signal_value = signal_value * 1.2
        
        # Create metadata
        metadata = {
            'structure_type': result.structure_type,
            'direction': result.signal,
            'strength': result.strength,
            'reason': result.reason,
            'original_signal': result.signal,
            'timeframe': '15-minute',
            'current_trend': result.metrics.get('current_trend'),
            'last_break_type': result.metrics.get('last_break_type'),
            'fractal_count': result.metrics.get('fractal_count', 0),
            'structure_breaks': result.metrics.get('structure_breaks', 0),
            'previous_value': self.get_previous_value()
        }
        
        # Use strength as confidence (0-100 -> 0-1)
        confidence = result.strength / 100.0
        
        # Create and return signal
        return self.create_signal(signal_value, confidence, metadata)
    
    def calculate_confidence(self, result: MarketStructureSignal, signal_value: float) -> float:
        """Use strength from result as confidence"""
        return result.strength / 100.0