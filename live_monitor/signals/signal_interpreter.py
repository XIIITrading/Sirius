# live_monitor/signals/signal_interpreter.py
"""
Signal Interpreter Module
Converts various indicator outputs to standardized trading signals
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional
from enum import Enum
from datetime import datetime

from PyQt6.QtCore import QObject, pyqtSignal

from live_monitor.calculations.indicators.m1_ema import M1EMAResult
from live_monitor.calculations.indicators.m5_ema import M5EMAResult
from live_monitor.calculations.indicators.m15_ema import M15EMAResult
from live_monitor.calculations.trend.statistical_trend_1min import StatisticalSignal
from live_monitor.calculations.trend.statistical_trend_5min import PositionSignal5Min
from live_monitor.calculations.trend.statistical_trend_15min import MarketRegimeSignal
from live_monitor.data.models.signals import EntrySignal, ExitSignal

logger = logging.getLogger(__name__)


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


class SignalInterpreter(QObject):
    """
    Converts various indicator outputs to standardized signals
    Manages signal state and generates entry/exit signals
    """
    
    # Signals that match PolygonDataManager's signal format
    entry_signal_generated = pyqtSignal(dict)  # EntrySignal format
    exit_signal_generated = pyqtSignal(dict)   # ExitSignal format
    
    def __init__(self):
        super().__init__()
        self.current_symbol = None
        self.current_price = 0.0
        
        # Track previous signals to detect changes
        self.previous_signals: Dict[str, StandardSignal] = {}
        
        # Track active positions for exit generation
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        
        # Track which sources can generate entries
        self.active_entry_sources = {
            'M1_EMA': True,
            'M5_EMA': True,
            'M15_EMA': True,
            'STATISTICAL_TREND': True
        }
    
    def set_active_entry_sources(self, sources: Dict[str, bool]):
        """Update which sources can generate entry signals"""
        self.active_entry_sources.update(sources)
        logger.info(f"Active entry sources updated: {self.active_entry_sources}")
    
    def set_symbol_context(self, symbol: str, current_price: float):
        """Update current trading context"""
        if symbol != self.current_symbol:
            # Clear state on symbol change
            self.previous_signals.clear()
            self.active_positions.clear()
            
        self.current_symbol = symbol
        self.current_price = current_price
    
    def process_m1_ema(self, result: M1EMAResult) -> StandardSignal:
        """
        Convert M1 EMA result to standard signal
        
        Scale mapping:
        - signal_strength (0-100) → magnitude (0-50)
        - Apply direction based on BULL/BEAR
        - Adjust for special conditions
        """
        # Base conversion: map strength to magnitude
        magnitude = result.signal_strength / 2  # 0-100 → 0-50
        
        # Enhance magnitude for special conditions
        if result.is_crossover:
            # Crossovers are stronger signals
            if result.crossover_type == 'bullish' and result.signal == 'BULL':
                magnitude = min(50, magnitude * 1.2)
            elif result.crossover_type == 'bearish' and result.signal == 'BEAR':
                magnitude = min(50, magnitude * 1.2)
            else:
                # Crossover against trend - reduce strength
                magnitude *= 0.8
        
        # Fine-tune based on spread percentage
        if abs(result.spread_pct) < 0.1:  # Very tight spread
            magnitude *= 0.5  # Reduce to weak zone
        elif abs(result.spread_pct) > 1.0:  # Strong spread
            magnitude = max(25, magnitude)  # Ensure at least strong signal
        
        # Apply direction
        value = magnitude if result.signal == 'BULL' else -magnitude
        
        # Adjust for price position relative to EMAs
        if result.signal == 'BULL' and result.price_position == 'below':
            # Price below EMA in uptrend - reduce bullish signal
            value *= 0.7
        elif result.signal == 'BEAR' and result.price_position == 'above':
            # Price above EMA in downtrend - reduce bearish signal
            value *= 0.7
        
        # Ensure value stays in range
        value = max(-50, min(50, value))
        
        # Determine category
        category = SignalCategory.from_value(value)
        
        # Determine strength for display
        if abs(value) >= 25:
            strength = 'Strong'
        elif abs(value) >= 10:
            strength = 'Medium'
        else:
            strength = 'Weak'
        
        # Calculate confidence
        confidence = self._calculate_m1_ema_confidence(result, value)
        
        # Get previous value for comparison
        prev_signal = self.previous_signals.get('M1_EMA')
        previous_value = prev_signal.value if prev_signal else 0
        
        signal = StandardSignal(
            value=round(value, 1),
            category=category,
            source="M1_EMA",
            confidence=confidence,
            direction='LONG' if value > 0 else 'SHORT',
            strength=strength,
            metadata={
                'ema_9': result.ema_9,
                'ema_21': result.ema_21,
                'spread_pct': result.spread_pct,
                'is_crossover': result.is_crossover,
                'crossover_type': result.crossover_type,
                'price_position': result.price_position,
                'trend_strength': result.trend_strength,
                'previous_value': previous_value,
                'reason': result.reason
            }
        )
        
        # Store for next comparison
        self.previous_signals['M1_EMA'] = signal
        
        # Check if we should generate trading signals
        self._check_and_generate_signals(signal)
        
        return signal
    
    def process_m5_ema(self, result: M5EMAResult) -> StandardSignal:
        """
        Convert M5 EMA result to standard signal
        
        Scale mapping:
        - signal_strength (0-100) → magnitude (0-50)
        - Apply direction based on BULL/BEAR
        - Adjust for special conditions
        """
        # Base conversion: map strength to magnitude
        magnitude = result.signal_strength / 2  # 0-100 → 0-50
        
        # Enhance magnitude for special conditions
        if result.is_crossover:
            # Crossovers are stronger signals
            if result.crossover_type == 'bullish' and result.signal == 'BULL':
                magnitude = min(50, magnitude * 1.2)
            elif result.crossover_type == 'bearish' and result.signal == 'BEAR':
                magnitude = min(50, magnitude * 1.2)
            else:
                # Crossover against trend - reduce strength
                magnitude *= 0.8
        
        # Fine-tune based on spread percentage
        if abs(result.spread_pct) < 0.1:  # Very tight spread
            magnitude *= 0.5  # Reduce to weak zone
        elif abs(result.spread_pct) > 1.0:  # Strong spread
            magnitude = max(25, magnitude)  # Ensure at least strong signal
        
        # Apply direction
        value = magnitude if result.signal == 'BULL' else -magnitude
        
        # Adjust for price position relative to EMAs
        if result.signal == 'BULL' and result.price_position == 'below':
            # Price below EMA in uptrend - reduce bullish signal
            value *= 0.7
        elif result.signal == 'BEAR' and result.price_position == 'above':
            # Price above EMA in downtrend - reduce bearish signal
            value *= 0.7
        
        # Ensure value stays in range
        value = max(-50, min(50, value))
        
        # Determine category
        category = SignalCategory.from_value(value)
        
        # Determine strength for display
        if abs(value) >= 25:
            strength = 'Strong'
        elif abs(value) >= 10:
            strength = 'Medium'
        else:
            strength = 'Weak'
        
        # Calculate confidence
        confidence = self._calculate_m5_ema_confidence(result, value)
        
        # Get previous value for comparison
        prev_signal = self.previous_signals.get('M5_EMA')
        previous_value = prev_signal.value if prev_signal else 0
        
        signal = StandardSignal(
            value=round(value, 1),
            category=category,
            source="M5_EMA",
            confidence=confidence,
            direction='LONG' if value > 0 else 'SHORT',
            strength=strength,
            metadata={
                'ema_9': result.ema_9,
                'ema_21': result.ema_21,
                'spread_pct': result.spread_pct,
                'is_crossover': result.is_crossover,
                'crossover_type': result.crossover_type,
                'price_position': result.price_position,
                'trend_strength': result.trend_strength,
                'previous_value': previous_value,
                'reason': result.reason,
                'last_5min_close': result.last_5min_close,
                'last_5min_volume': result.last_5min_volume,
                'bars_processed': result.bars_processed
            }
        )
        
        # Store for next comparison
        self.previous_signals['M5_EMA'] = signal
        
        # Check if we should generate trading signals
        self._check_and_generate_signals(signal)
        
        return signal
    
    def process_m15_ema(self, result: M15EMAResult) -> StandardSignal:
        """
        Convert M15 EMA result to standard signal
        
        Scale mapping:
        - signal_strength (0-100) → magnitude (0-50)
        - Apply direction based on BULL/BEAR/NEUTRAL
        - Adjust for special conditions
        """
        # Base conversion: map strength to magnitude
        magnitude = result.signal_strength / 2  # 0-100 → 0-50
        
        # Handle NEUTRAL signal specially
        if result.signal == 'NEUTRAL':
            # NEUTRAL signals get very low magnitude
            magnitude = min(10, magnitude * 0.3)
            # Direction is based on underlying trend
            value = magnitude if result.ema_9 > result.ema_21 else -magnitude
        else:
            # Enhance magnitude for special conditions
            if result.is_crossover:
                # Crossovers are stronger signals
                if result.crossover_type == 'bullish' and result.signal == 'BULL':
                    magnitude = min(50, magnitude * 1.2)
                elif result.crossover_type == 'bearish' and result.signal == 'BEAR':
                    magnitude = min(50, magnitude * 1.2)
                else:
                    # Crossover against trend - reduce strength
                    magnitude *= 0.8
            
            # Fine-tune based on spread percentage
            if abs(result.spread_pct) < 0.1:  # Very tight spread
                magnitude *= 0.5  # Reduce to weak zone
            elif abs(result.spread_pct) > 1.0:  # Strong spread
                magnitude = max(25, magnitude)  # Ensure at least strong signal
            
            # Apply direction
            value = magnitude if result.signal == 'BULL' else -magnitude
        
        # Ensure value stays in range
        value = max(-50, min(50, value))
        
        # Determine category
        category = SignalCategory.from_value(value)
        
        # Determine strength for display
        if result.signal == 'NEUTRAL':
            strength = 'Neutral'
        elif abs(value) >= 25:
            strength = 'Strong'
        elif abs(value) >= 10:
            strength = 'Medium'
        else:
            strength = 'Weak'
        
        # Calculate confidence
        confidence = self._calculate_m15_ema_confidence(result, value)
        
        # Get previous value for comparison
        prev_signal = self.previous_signals.get('M15_EMA')
        previous_value = prev_signal.value if prev_signal else 0
        
        # Determine direction for display (even for NEUTRAL)
        direction = 'LONG' if value > 0 else 'SHORT'
        
        signal = StandardSignal(
            value=round(value, 1),
            category=category,
            source="M15_EMA",
            confidence=confidence,
            direction=direction,
            strength=strength,
            metadata={
                'ema_9': result.ema_9,
                'ema_21': result.ema_21,
                'spread_pct': result.spread_pct,
                'is_crossover': result.is_crossover,
                'crossover_type': result.crossover_type,
                'price_position': result.price_position,
                'trend_strength': result.trend_strength,
                'previous_value': previous_value,
                'reason': result.reason,
                'last_15min_close': result.last_15min_close,
                'last_15min_volume': result.last_15min_volume,
                'bars_processed': result.bars_processed,
                'signal_type': result.signal  # Store original BULL/BEAR/NEUTRAL
            }
        )
        
        # Store for next comparison
        self.previous_signals['M15_EMA'] = signal
        
        # Check if we should generate trading signals
        self._check_and_generate_signals(signal)
        
        return signal
    
    def process_statistical_trend(self, result: StatisticalSignal) -> StandardSignal:
        """
        Convert Statistical Trend result to standard signal
        
        Scale mapping based on volatility-adjusted strength:
        - BUY with high vol-adj strength (>2.0) → +40 to +50
        - BUY with normal vol-adj strength (1.0-2.0) → +25 to +40
        - WEAK BUY with good vol-adj strength (0.5-1.0) → +10 to +24
        - WEAK BUY with low vol-adj strength (<0.5) → 0 to +10
        - Same pattern for SELL signals (negative values)
        """
        vol_adj_strength = result.volatility_adjusted_strength
        
        # Determine base magnitude based on signal and volatility-adjusted strength
        if result.signal == 'BUY':
            if vol_adj_strength >= 2.0:
                # Very strong trend relative to volatility
                magnitude = 40 + min(10, (vol_adj_strength - 2.0) * 5)
            elif vol_adj_strength >= 1.0:
                # Trend equals or exceeds volatility
                magnitude = 25 + (vol_adj_strength - 1.0) * 15
            else:
                # Should not happen for BUY signal, but handle gracefully
                magnitude = 25
                
        elif result.signal == 'WEAK BUY':
            if vol_adj_strength >= 0.5:
                # Decent trend relative to volatility
                magnitude = 10 + (vol_adj_strength - 0.5) * 28
            else:
                # Weak trend
                magnitude = vol_adj_strength * 20
                
        elif result.signal == 'WEAK SELL':
            if vol_adj_strength >= 0.5:
                # Decent trend relative to volatility
                magnitude = 10 + (vol_adj_strength - 0.5) * 28
            else:
                # Weak trend
                magnitude = vol_adj_strength * 20
                
        elif result.signal == 'SELL':
            if vol_adj_strength >= 2.0:
                # Very strong trend relative to volatility
                magnitude = 40 + min(10, (vol_adj_strength - 2.0) * 5)
            elif vol_adj_strength >= 1.0:
                # Trend equals or exceeds volatility
                magnitude = 25 + (vol_adj_strength - 1.0) * 15
            else:
                # Should not happen for SELL signal, but handle gracefully
                magnitude = 25
        else:
            # Unknown signal type
            magnitude = 0
        
        # Apply direction
        if result.signal in ['SELL', 'WEAK SELL']:
            value = -magnitude
        else:
            value = magnitude
        
        # Apply volume confirmation boost
        if result.volume_confirmation:
            value = value * 1.1  # 10% boost for volume confirmation
        
        # Ensure value stays in range
        value = max(-50, min(50, value))
        
        # Determine category
        category = SignalCategory.from_value(value)
        
        # Determine strength for display
        if abs(value) >= 25:
            strength = 'Strong'
        elif abs(value) >= 10:
            strength = 'Medium'
        else:
            strength = 'Weak'
        
        # Convert confidence from 0-100 to 0-1
        confidence = result.confidence / 100.0
        
        # Get previous value for comparison
        prev_signal = self.previous_signals.get('STATISTICAL_TREND')
        previous_value = prev_signal.value if prev_signal else 0
        
        signal = StandardSignal(
            value=round(value, 1),
            category=category,
            source="STATISTICAL_TREND",
            confidence=confidence,
            direction='LONG' if value > 0 else 'SHORT',
            strength=strength,
            metadata={
                'original_signal': result.signal,
                'trend_strength': result.trend_strength,
                'volatility_adjusted_strength': result.volatility_adjusted_strength,
                'volume_confirmation': result.volume_confirmation,
                'price': result.price,
                'previous_value': previous_value
            }
        )
        
        # Store for next comparison
        self.previous_signals['STATISTICAL_TREND'] = signal
        
        # Check if we should generate trading signals
        self._check_and_generate_signals(signal)
        
        return signal
    
    def process_m5_statistical_trend(self, result: PositionSignal5Min) -> StandardSignal:
        """
        Convert M5 Statistical Trend result to standard signal
        Uses the same logic as M1 but with M5 timeframe context
        """
        # Use the same processing logic as the 1-minute version
        vol_adj_strength = result.volatility_adjusted_strength
        
        # Determine base magnitude based on signal and volatility-adjusted strength
        if result.signal == 'BUY':
            if vol_adj_strength >= 2.0:
                magnitude = 40 + min(10, (vol_adj_strength - 2.0) * 5)
            elif vol_adj_strength >= 1.0:
                magnitude = 25 + (vol_adj_strength - 1.0) * 15
            else:
                magnitude = 25
        elif result.signal == 'WEAK BUY':
            if vol_adj_strength >= 0.5:
                magnitude = 10 + (vol_adj_strength - 0.5) * 28
            else:
                magnitude = vol_adj_strength * 20
        elif result.signal == 'WEAK SELL':
            if vol_adj_strength >= 0.5:
                magnitude = 10 + (vol_adj_strength - 0.5) * 28
            else:
                magnitude = vol_adj_strength * 20
        elif result.signal == 'SELL':
            if vol_adj_strength >= 2.0:
                magnitude = 40 + min(10, (vol_adj_strength - 2.0) * 5)
            elif vol_adj_strength >= 1.0:
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
            value = value * 1.1
        
        # Ensure value stays in range
        value = max(-50, min(50, value))
        
        # Determine category
        category = SignalCategory.from_value(value)
        
        # Determine strength for display
        if abs(value) >= 25:
            strength = 'Strong'
        elif abs(value) >= 10:
            strength = 'Medium'
        else:
            strength = 'Weak'
        
        # Get previous value for comparison
        prev_signal = self.previous_signals.get('STATISTICAL_TREND_5M')
        previous_value = prev_signal.value if prev_signal else 0
        
        signal = StandardSignal(
            value=round(value, 1),
            category=category,
            source="STATISTICAL_TREND_5M",
            confidence=result.confidence / 100.0,
            direction='LONG' if value > 0 else 'SHORT',
            strength=strength,
            metadata={
                'original_signal': result.signal,
                'bias': result.bias,
                'trend_strength': result.trend_strength,
                'volatility_adjusted_strength': result.volatility_adjusted_strength,
                'volume_confirmation': result.volume_confirmation,
                'price': result.price,
                'previous_value': previous_value,
                'timeframe': '5min'
            }
        )
        
        # Store for next comparison
        self.previous_signals['STATISTICAL_TREND_5M'] = signal
        
        # Check if we should generate trading signals
        self._check_and_generate_signals(signal)
        
        return signal
    
    def process_m15_statistical_trend(self, result: MarketRegimeSignal) -> StandardSignal:
        """Process 15-minute statistical trend market regime signal"""
        try:
            # Map 4-tier signal to numeric value
            signal_map = {
                'BUY': 40,          # Strong bullish
                'WEAK BUY': 15,     # Weak bullish
                'WEAK SELL': -15,   # Weak bearish
                'SELL': -40         # Strong bearish
            }
            
            signal_value = signal_map.get(result.signal, 0)
            
            # Determine category
            category = SignalCategory.from_value(signal_value)
            
            # Determine strength
            if abs(signal_value) >= 25:
                strength = 'Strong'
            elif abs(signal_value) >= 10:
                strength = 'Medium'
            else:
                strength = 'Weak'
            
            # Create metadata combining regime and daily bias info
            metadata = {
                'regime': result.regime,
                'daily_bias': result.daily_bias,
                'trend_strength': round(result.trend_strength, 2),
                'volatility_adjusted_strength': round(result.volatility_adjusted_strength, 2),
                'volatility_state': result.volatility_state,
                'volume_trend': result.volume_trend,
                'confidence': round(result.confidence, 1),
                'original_signal': result.signal
            }
            
            # Get previous value for comparison
            prev_signal = self.previous_signals.get('STATISTICAL_TREND_15M')
            previous_value = prev_signal.value if prev_signal else 0
            metadata['previous_value'] = previous_value
            
            # Create the StandardSignal
            signal = StandardSignal(
                value=round(signal_value, 1),
                category=category,
                source='STATISTICAL_TREND_15M',
                confidence=result.confidence / 100.0,  # Convert to 0-1 scale
                direction='LONG' if signal_value > 0 else 'SHORT',
                strength=strength,
                metadata=metadata
            )
            
            # Store for next comparison
            self.previous_signals['STATISTICAL_TREND_15M'] = signal
            
            # Check if we should generate trading signals
            self._check_and_generate_signals(signal)
            
            logger.info(f"M15 Statistical Trend processed: {result.signal} -> {signal_value}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error processing M15 statistical trend: {e}", exc_info=True)
            raise
    
    def _calculate_m1_ema_confidence(self, result: M1EMAResult, signal_value: float) -> float:
        """Calculate confidence level for M1 EMA signal"""
        confidence = 0.5  # Base confidence
        
        # Boost for crossovers
        if result.is_crossover:
            confidence += 0.2
        
        # Boost for strong trends
        if result.trend_strength > 70:
            confidence += 0.2
        elif result.trend_strength < 30:
            confidence -= 0.1
        
        # Boost for good spread
        if 0.3 <= abs(result.spread_pct) <= 1.5:
            confidence += 0.1
        elif abs(result.spread_pct) > 2.0:
            confidence -= 0.1  # Too extended
        
        # Boost for price position alignment
        if (result.signal == 'BULL' and result.price_position == 'above') or \
           (result.signal == 'BEAR' and result.price_position == 'below'):
            confidence += 0.1
        
        # Ensure signal strength matches confidence
        if abs(signal_value) >= 25 and confidence < 0.6:
            confidence = 0.6  # Minimum confidence for strong signals
        
        return min(1.0, max(0.1, confidence))
    
    def _calculate_m5_ema_confidence(self, result: M5EMAResult, signal_value: float) -> float:
        """Calculate confidence level for M5 EMA signal"""
        confidence = 0.5  # Base confidence
        
        # Boost for crossovers
        if result.is_crossover:
            confidence += 0.2
        
        # Boost for strong trends
        if result.trend_strength > 70:
            confidence += 0.2
        elif result.trend_strength < 30:
            confidence -= 0.1
        
        # Boost for good spread
        if 0.3 <= abs(result.spread_pct) <= 1.5:
            confidence += 0.1
        elif abs(result.spread_pct) > 2.0:
            confidence -= 0.1  # Too extended
        
        # Boost for price position alignment
        if (result.signal == 'BULL' and result.price_position == 'above') or \
           (result.signal == 'BEAR' and result.price_position == 'below'):
            confidence += 0.1
        
        # Additional M5-specific confidence factors
        # Boost if we have sufficient data
        if result.bars_processed >= 50:  # Good amount of 5-min bars
            confidence += 0.05
        
        # Boost for volume confirmation (if volume is above average)
        if result.last_5min_volume > 0:  # Has volume data
            confidence += 0.05
        
        # Ensure signal strength matches confidence
        if abs(signal_value) >= 25 and confidence < 0.6:
            confidence = 0.6  # Minimum confidence for strong signals
        
        return min(1.0, max(0.1, confidence))
    
    def _calculate_m15_ema_confidence(self, result: M15EMAResult, signal_value: float) -> float:
        """Calculate confidence level for M15 EMA signal"""
        confidence = 0.5  # Base confidence
        
        # Lower confidence for NEUTRAL signals
        if result.signal == 'NEUTRAL':
            confidence = 0.3  # Start lower for neutral
        
        # Boost for crossovers
        if result.is_crossover and result.signal != 'NEUTRAL':
            confidence += 0.2
        
        # Boost for strong trends
        if result.trend_strength > 70:
            confidence += 0.2
        elif result.trend_strength < 30:
            confidence -= 0.1
        
        # Boost for good spread (but not for NEUTRAL)
        if result.signal != 'NEUTRAL':
            if 0.3 <= abs(result.spread_pct) <= 1.5:
                confidence += 0.1
            elif abs(result.spread_pct) > 2.0:
                confidence -= 0.1  # Too extended
        
        # Boost for price position alignment (only for non-NEUTRAL)
        if result.signal != 'NEUTRAL':
            if (result.signal == 'BULL' and result.price_position == 'above') or \
               (result.signal == 'BEAR' and result.price_position == 'below'):
                confidence += 0.1
        
        # Additional M15-specific confidence factors
        # Boost if we have sufficient data
        if result.bars_processed >= 30:  # Good amount of 15-min bars
            confidence += 0.05
        
        # Boost for volume confirmation
        if result.last_15min_volume > 0:
            confidence += 0.05
        
        # Ensure signal strength matches confidence
        if abs(signal_value) >= 25 and confidence < 0.6:
            confidence = 0.6  # Minimum confidence for strong signals
        
        return min(1.0, max(0.1, confidence))
    
    def _check_and_generate_signals(self, signal: StandardSignal):
        """Check if we should generate entry or exit signals"""
        if not self.current_symbol:
            return
        
        # Check if this source is active for entries
        source_key = signal.source
        if not self.active_entry_sources.get(source_key, True):
            logger.debug(f"Entry generation disabled for {signal.source}")
            return
        
        # ALWAYS generate/update entry signal for active sources
        self._generate_entry_signal(signal)
        
        # Check for exit signal
        if signal.should_generate_exit:
            # Check if we have an active position to exit
            if signal.source in self.active_positions:
                self._generate_exit_signal(signal)
    
    def _generate_entry_signal(self, signal: StandardSignal):
        """Generate entry signal for the grid"""
        # Build signal description based on source
        if signal.source == 'M1_EMA':
            if signal.metadata.get('is_crossover'):
                signal_desc = f"M1 EMA {signal.metadata['crossover_type'].title()} Crossover"
            else:
                signal_desc = f"M1 EMA {signal.direction} Signal"
        elif signal.source == 'M5_EMA':
            if signal.metadata.get('is_crossover'):
                signal_desc = f"M5 EMA {signal.metadata['crossover_type'].title()} Crossover"
            else:
                signal_desc = f"M5 EMA {signal.direction} Signal"
        elif signal.source == 'M15_EMA':
            if signal.metadata.get('signal_type') == 'NEUTRAL':
                signal_desc = f"M15 EMA NEUTRAL (Price vs Trend Conflict)"
            elif signal.metadata.get('is_crossover'):
                signal_desc = f"M15 EMA {signal.metadata['crossover_type'].title()} Crossover"
            else:
                signal_desc = f"M15 EMA {signal.direction} Signal"
        elif signal.source == 'STATISTICAL_TREND':
            vol_adj = signal.metadata.get('volatility_adjusted_strength', 0)
            if vol_adj >= 2.0:
                signal_desc = f"M1 Trend STRONG {signal.direction}"
            elif vol_adj >= 1.0:
                signal_desc = f"M1 Trend {signal.direction}"
            else:
                signal_desc = f"M1 Trend WEAK {signal.direction}"
            if signal.metadata.get('volume_confirmation'):
                signal_desc += " (Vol Confirm)"
        elif signal.source == 'STATISTICAL_TREND_5M':
            vol_adj = signal.metadata.get('volatility_adjusted_strength', 0)
            original_signal = signal.metadata.get('original_signal', '')
            
            # Create descriptive signal
            signal_desc = f"M5 Trend: {original_signal}"
            signal_desc += f" (Vol-Adj: {vol_adj:.2f})"
            
            if signal.metadata.get('volume_confirmation'):
                signal_desc += " ✓Vol"
        elif signal.source == 'STATISTICAL_TREND_15M':
            # Create descriptive signal for M15 Statistical Trend
            regime = signal.metadata.get('regime', '')
            daily_bias = signal.metadata.get('daily_bias', '')
            original_signal = signal.metadata.get('original_signal', '')
            
            signal_desc = f"M15 Trend: {original_signal}"
            
            # Add regime if it's a clear market state
            if regime in ['BULL MARKET', 'BEAR MARKET']:
                signal_desc += f" ({regime})"
            
            # Add daily bias if it's significant
            if daily_bias in ['LONG ONLY', 'SHORT ONLY', 'STAY OUT']:
                signal_desc += f" [{daily_bias}]"
        else:
            signal_desc = f"{signal.source} {signal.direction} Signal"
        
        # Add additional context
        if signal.source in ['M1_EMA', 'M5_EMA', 'M15_EMA'] and 'spread_pct' in signal.metadata:
            signal_desc += f" (Spread: {abs(signal.metadata['spread_pct']):.2f}%)"
        elif signal.source in ['STATISTICAL_TREND', 'STATISTICAL_TREND_5M', 'STATISTICAL_TREND_15M']:
            if 'volatility_adjusted_strength' in signal.metadata:
                signal_desc += f" (Vol-Adj: {signal.metadata['volatility_adjusted_strength']:.2f})"
        
        entry_signal: EntrySignal = {
            'time': datetime.now().strftime("%H:%M:%S"),
            'signal_type': signal.direction,
            'price': f"{self.current_price:.2f}",
            'signal': signal_desc,
            'strength': signal.strength,
            'notes': f"Confidence: {signal.confidence:.0%} | Signal: {signal.value:+.1f}",
            'timestamp': datetime.now(),
            'symbol': self.current_symbol
        }
        
        # Track active position
        self.active_positions[signal.source] = {
            'entry_price': self.current_price,
            'entry_time': datetime.now(),
            'direction': signal.direction,
            'signal_value': signal.value
        }
        
        # Emit the signal
        self.entry_signal_generated.emit(entry_signal)
        logger.info(f"Generated entry signal: {signal.direction} at {self.current_price:.2f}")
    
    def _generate_exit_signal(self, signal: StandardSignal):
        """Generate exit signal for the grid"""
        position = self.active_positions.get(signal.source)
        if not position:
            return
        
        # Calculate P&L
        entry_price = position['entry_price']
        if position['direction'] == 'LONG':
            pnl_pct = ((self.current_price - entry_price) / entry_price) * 100
        else:  # SHORT
            pnl_pct = ((entry_price - self.current_price) / entry_price) * 100
        
        # Determine exit type and urgency
        if abs(signal.value) < 5:
            exit_type = 'TRAIL'
            urgency = 'Warning'
            signal_desc = "Signal Weakening - Trail Stop"
        elif signal.metadata.get('previous_value', 0) * signal.value < 0:
            exit_type = 'STOP'
            urgency = 'Urgent'
            signal_desc = "Signal Reversal - Exit Position"
        else:
            exit_type = 'TARGET'
            urgency = 'Normal'
            signal_desc = "Signal Target - Consider Exit"
        
        exit_signal: ExitSignal = {
            'time': datetime.now().strftime("%H:%M:%S"),
            'exit_type': exit_type,
            'price': f"{self.current_price:.2f}",
            'pnl': f"{pnl_pct:+.2f}%",
            'signal': signal_desc,
            'urgency': urgency,
            'timestamp': datetime.now(),
            'symbol': self.current_symbol,
            'pnl_value': pnl_pct
        }
        
        # Remove from active positions
        del self.active_positions[signal.source]
        
        # Emit the signal
        self.exit_signal_generated.emit(exit_signal)
        logger.info(f"Generated exit signal: {exit_type} at {self.current_price:.2f} (P&L: {pnl_pct:+.2f}%)")
    
    def process_combined_signals(self, signals: List[StandardSignal]) -> StandardSignal:
        """
        Combine multiple signals into a composite signal
        Future enhancement for multiple indicators
        """
        if not signals:
            raise ValueError("No signals to combine")
        
        # Weight signals by confidence
        total_weight = sum(s.confidence for s in signals)
        if total_weight == 0:
            total_weight = 1
        
        # Weighted average of signal values
        weighted_value = sum(s.value * s.confidence for s in signals) / total_weight
        
        # Average confidence
        avg_confidence = sum(s.confidence for s in signals) / len(signals)
        
        # Determine direction and strength
        category = SignalCategory.from_value(weighted_value)
        direction = 'LONG' if weighted_value > 0 else 'SHORT'
        
        if abs(weighted_value) >= 25:
            strength = 'Strong'
        elif abs(weighted_value) >= 10:
            strength = 'Medium'
        else:
            strength = 'Weak'
        
        return StandardSignal(
            value=round(weighted_value, 1),
            category=category,
            source="Combined",
            confidence=avg_confidence,
            direction=direction,
            strength=strength,
            metadata={
                'sources': [s.source for s in signals],
                'individual_values': {s.source: s.value for s in signals}
            }
        )