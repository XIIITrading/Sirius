# live_monitor/signals/processors/market_structure/m1_market_structure_processor.py
"""
M1 Market Structure Signal Processor
"""

from typing import Dict, Any
from ..base_processor import BaseSignalProcessor
from ...models.signal_types import StandardSignal
from ...models.constants import *
from live_monitor.calculations.market_structure.m1_market_structure import MarketStructureSignal


class M1MarketStructureProcessor(BaseSignalProcessor):
    """Process M1 Market Structure signals"""
    
    def __init__(self):
        super().__init__("M1_MARKET_STRUCTURE")
    
    def process(self, result: MarketStructureSignal) -> StandardSignal:
        """Convert Market Structure result to standard signal"""
        # Map 4-tier signal to numeric value
        signal_map = {
            'BUY': 40,          # Strong bullish
            'WEAK BUY': 15,     # Weak bullish  
            'WEAK SELL': -15,   # Weak bearish
            'SELL': -40         # Strong bearish
        }
        
        signal_value = signal_map.get(result.signal, 0)
        
        # Adjust based on structure type
        if result.structure_type == 'CHoCH':
            # Change of Character signals are typically stronger
            if abs(signal_value) == 40:
                signal_value = signal_value * 1.1  # Boost strong CHoCH signals
        
        # Create metadata
        metadata = {
            'structure_type': result.structure_type,  # 'BOS' or 'CHoCH'
            'direction': result.direction,  # Original 'BULL' or 'BEAR'
            'strength': result.strength,
            'reason': result.reason,
            'original_signal': result.signal,
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