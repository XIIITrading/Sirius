# live_monitor/signals/processors/trend/m15_trend_processor.py
"""
M15 Statistical Trend Signal Processor
"""

from typing import Dict, Any
from ..base_processor import BaseSignalProcessor
from ...models.signal_types import StandardSignal  # ADD THIS IMPORT
from ...models.constants import *
from live_monitor.calculations.trend.statistical_trend_15min import MarketRegimeSignal


class M15TrendProcessor(BaseSignalProcessor):
    """Process M15 Statistical Trend Market Regime signals"""
    
    def __init__(self):
        super().__init__("STATISTICAL_TREND_15M")
    
    def process(self, result: MarketRegimeSignal) -> StandardSignal:
        """Process 15-minute statistical trend market regime signal"""
        # Map 4-tier signal to numeric value
        signal_map = {
            'BUY': 40,          # Strong bullish
            'WEAK BUY': 15,     # Weak bullish
            'WEAK SELL': -15,   # Weak bearish
            'SELL': -40         # Strong bearish
        }
        
        signal_value = signal_map.get(result.signal, 0)
        
        # Create metadata
        metadata = {
            'regime': result.regime,
            'daily_bias': result.daily_bias,
            'trend_strength': round(result.trend_strength, 2),
            'volatility_adjusted_strength': round(result.volatility_adjusted_strength, 2),
            'volatility_state': result.volatility_state,
            'volume_trend': result.volume_trend,
            'confidence': round(result.confidence, 1),
            'original_signal': result.signal,
            'previous_value': self.get_previous_value()
        }
        
        # Convert confidence to 0-1 scale
        confidence = result.confidence / 100.0
        
        # Create and return signal
        return self.create_signal(signal_value, confidence, metadata)
    
    def calculate_confidence(self, result: MarketRegimeSignal, signal_value: float) -> float:
        """Use confidence from result directly"""
        return result.confidence / 100.0