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
    def should_generate_entry(self) -> bool:
        """Determine if this signal should generate an entry"""
        # Only generate entries for non-weak signals with good confidence
        return (abs(self.value) >= 15 and self.confidence >= 0.5)
    
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
    
    def _check_and_generate_signals(self, signal: StandardSignal):
        """Check if we should generate entry or exit signals"""
        if not self.current_symbol:
            return
        
        # Check for entry signal
        if signal.should_generate_entry:
            prev_signal = self.previous_signals.get(signal.source)
            
            # Only generate if this is a new signal or significant change
            if not prev_signal or abs(prev_signal.value) < 15:
                self._generate_entry_signal(signal)
        
        # Check for exit signal
        if signal.should_generate_exit:
            # Check if we have an active position to exit
            if signal.source in self.active_positions:
                self._generate_exit_signal(signal)
    
    def _generate_entry_signal(self, signal: StandardSignal):
        """Generate entry signal for the grid"""
        # Build signal description
        if signal.source == 'M1_EMA':
            if signal.metadata.get('is_crossover'):
                signal_desc = f"M1 EMA {signal.metadata['crossover_type'].title()} Crossover"
            else:
                signal_desc = f"M1 EMA {signal.direction} Signal"
        else:
            signal_desc = f"{signal.source} {signal.direction} Signal"
        
        # Add spread info
        if 'spread_pct' in signal.metadata:
            signal_desc += f" (Spread: {abs(signal.metadata['spread_pct']):.2f}%)"
        
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