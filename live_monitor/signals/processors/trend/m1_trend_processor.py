# live_monitor/signals/processors/trend/m1_trend_processor.py
"""
M1 Statistical Trend Signal Processor
"""

from typing import Dict, Any
from ..base_processor import BaseSignalProcessor
from ...models.signal_types import StandardSignal  # ADD THIS IMPORT
from ...models.constants import *
from live_monitor.calculations.trend.statistical_trend_1min import StatisticalSignal


class M1TrendProcessor(BaseSignalProcessor):
    """Process M1 Statistical Trend signals"""
    
    def __init__(self):
        super().__init__("STATISTICAL_TREND")
    
    def process(self, result: StatisticalSignal) -> StandardSignal:
        """Convert Statistical Trend result to standard signal"""
        vol_adj_strength = result.volatility_adjusted_strength
        
        # Determine base magnitude based on signal and volatility-adjusted strength
        if result.signal == 'BUY':
            if vol_adj_strength >= VERY_STRONG_VOL_ADJ_THRESHOLD:
                magnitude = 40 + min(10, (vol_adj_strength - 2.0) * 5)
            elif vol_adj_strength >= NORMAL_VOL_ADJ_THRESHOLD:
                magnitude = 25 + (vol_adj_strength - 1.0) * 15
            else:
                magnitude = 25
                
        elif result.signal == 'WEAK BUY':
            if vol_adj_strength >= DECENT_VOL_ADJ_THRESHOLD:
                magnitude = 10 + (vol_adj_strength - 0.5) * 28
            else:
                magnitude = vol_adj_strength * 20
                
        elif result.signal == 'WEAK SELL':
            if vol_adj_strength >= DECENT_VOL_ADJ_THRESHOLD:
                magnitude = 10 + (vol_adj_strength - 0.5) * 28
            else:
                magnitude = vol_adj_strength * 20
                
        elif result.signal == 'SELL':
            if vol_adj_strength >= VERY_STRONG_VOL_ADJ_THRESHOLD:
                magnitude = 40 + min(10, (vol_adj_strength - 2.0) * 5)
            elif vol_adj_strength >= NORMAL_VOL_ADJ_THRESHOLD:
                magnitude = 25 + (vol_adj_strength - 1.0) * 15
            else:
                magnitude = 25
        else:
            magnitude = 0
        
        # Apply direction
        if result.signal in ['SELL', 'WEAK SELL']:
            value = -magnitude
        else:
            value = magnitude
        
        # Apply volume confirmation boost
        if result.volume_confirmation:
            value = value * VOLUME_CONFIRMATION_BOOST
        
        # Convert confidence from 0-100 to 0-1
        confidence = result.confidence / 100.0
        
        # Build metadata
        metadata = self._build_metadata(result)
        
        # Create and return signal
        return self.create_signal(value, confidence, metadata)
    
    def calculate_confidence(self, result: StatisticalSignal, signal_value: float) -> float:
        """Use confidence from result directly"""
        return result.confidence / 100.0
    
    def _build_metadata(self, result: StatisticalSignal) -> Dict[str, Any]:
        """Build metadata dictionary"""
        return {
            'original_signal': result.signal,
            'trend_strength': result.trend_strength,
            'volatility_adjusted_strength': result.volatility_adjusted_strength,
            'volume_confirmation': result.volume_confirmation,
            'price': result.price,
            'previous_value': self.get_previous_value()
        }