# live_monitor/signals/processors/ema/m5_ema_processor.py
"""
M5 EMA Signal Processor
"""

from typing import Dict, Any
from ..base_processor import BaseSignalProcessor
from ...models.signal_types import StandardSignal  # ADD THIS IMPORT
from ...models.constants import *
from live_monitor.calculations.indicators.m5_ema import M5EMAResult


class M5EMAProcessor(BaseSignalProcessor):
    """Process M5 EMA signals"""
    
    def __init__(self):
        super().__init__("M5_EMA")
    
    def process(self, result: M5EMAResult) -> StandardSignal:
        """Convert M5 EMA result to standard signal"""
        # Base conversion: map strength to magnitude
        magnitude = result.signal_strength / 2  # 0-100 → 0-50
        
        # Enhance magnitude for special conditions
        if result.is_crossover:
            if result.crossover_type == 'bullish' and result.signal == 'BULL':
                magnitude = min(50, magnitude * CROSSOVER_MAGNITUDE_BOOST)
            elif result.crossover_type == 'bearish' and result.signal == 'BEAR':
                magnitude = min(50, magnitude * CROSSOVER_MAGNITUDE_BOOST)
            else:
                magnitude *= CROSSOVER_AGAINST_TREND_PENALTY
        
        # Fine-tune based on spread percentage
        if abs(result.spread_pct) < TIGHT_SPREAD_THRESHOLD:
            magnitude *= TIGHT_SPREAD_MAGNITUDE_PENALTY
        elif abs(result.spread_pct) > NORMAL_SPREAD_MAX:
            magnitude = max(STRONG_SIGNAL_THRESHOLD, magnitude)
        
        # Apply direction
        value = magnitude if result.signal == 'BULL' else -magnitude
        
        # Adjust for price position relative to EMAs
        if result.signal == 'BULL' and result.price_position == 'below':
            value *= PRICE_POSITION_MISALIGNMENT_PENALTY
        elif result.signal == 'BEAR' and result.price_position == 'above':
            value *= PRICE_POSITION_MISALIGNMENT_PENALTY
        
        # Calculate confidence
        confidence = self.calculate_confidence(result, value)
        
        # Build metadata
        metadata = self._build_metadata(result)
        
        # Create and return signal
        return self.create_signal(value, confidence, metadata)
    
    def calculate_confidence(self, result: M5EMAResult, signal_value: float) -> float:
        """Calculate confidence level for M5 EMA signal"""
        confidence = BASE_CONFIDENCE
        
        # Boost for crossovers
        if result.is_crossover:
            confidence += CROSSOVER_CONFIDENCE_BOOST
        
        # Boost for strong trends
        if result.trend_strength > STRONG_TREND_THRESHOLD:
            confidence += STRONG_TREND_CONFIDENCE_BOOST
        elif result.trend_strength < WEAK_TREND_THRESHOLD:
            confidence += WEAK_TREND_CONFIDENCE_PENALTY
        
        # Boost for good spread
        if NORMAL_SPREAD_MIN <= abs(result.spread_pct) <= NORMAL_SPREAD_MAX:
            confidence += GOOD_SPREAD_CONFIDENCE_BOOST
        elif abs(result.spread_pct) > EXTENDED_SPREAD_THRESHOLD:
            confidence += EXTENDED_SPREAD_CONFIDENCE_PENALTY
        
        # Boost for price position alignment
        if ((result.signal == 'BULL' and result.price_position == 'above') or
            (result.signal == 'BEAR' and result.price_position == 'below')):
            confidence += POSITION_ALIGNMENT_CONFIDENCE_BOOST
        
        # Additional M5-specific confidence factors
        if result.bars_processed >= 50:  # Good amount of 5-min bars
            confidence += 0.05
        
        if result.last_5min_volume > 0:  # Has volume data
            confidence += 0.05
        
        # Ensure signal strength matches confidence
        if abs(signal_value) >= STRONG_SIGNAL_THRESHOLD and confidence < MINIMUM_STRONG_SIGNAL_CONFIDENCE:
            confidence = MINIMUM_STRONG_SIGNAL_CONFIDENCE
        
        return min(1.0, max(0.1, confidence))
    
    def _build_metadata(self, result: M5EMAResult) -> Dict[str, Any]:
        """Build metadata dictionary"""
        return {
            'ema_9': result.ema_9,
            'ema_21': result.ema_21,
            'spread_pct': result.spread_pct,
            'is_crossover': result.is_crossover,
            'crossover_type': result.crossover_type,
            'price_position': result.price_position,
            'trend_strength': result.trend_strength,
            'previous_value': self.get_previous_value(),
            'reason': result.reason,
            'last_5min_close': result.last_5min_close,
            'last_5min_volume': result.last_5min_volume,
            'bars_processed': result.bars_processed
        }