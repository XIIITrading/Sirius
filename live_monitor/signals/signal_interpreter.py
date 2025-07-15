"""
Signal Interpreter Module - Main orchestrator
Converts various indicator outputs to standardized trading signals
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from PyQt6.QtCore import QObject, pyqtSignal

from .models.signal_types import StandardSignal
from .processors.factory import ProcessorFactory
from .processors.base_processor import BaseSignalProcessor
from .utils import generate_ema_description, generate_trend_description, generate_market_structure_description
from live_monitor.data.models.signals import EntrySignal, ExitSignal

logger = logging.getLogger(__name__)


class SignalInterpreter(QObject):
    """
    Orchestrates signal processing and manages signal state
    Manages signal state and generates entry/exit signals
    """
    
    # Signals that match PolygonDataManager's signal format
    entry_signal_generated = pyqtSignal(dict)  # EntrySignal format
    exit_signal_generated = pyqtSignal(dict)   # ExitSignal format
    
    def __init__(self):
        super().__init__()
        self.current_symbol = None
        self.current_price = 0.0
        
        # Initialize processors
        self.processors: Dict[str, BaseSignalProcessor] = {}
        self._initialize_processors()
        
        # Track active positions for exit generation
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        
        # Track which sources can generate entries
        self.active_entry_sources = {
            'M1_EMA': True,
            'M5_EMA': True,
            'M15_EMA': True,
            'STATISTICAL_TREND': True,
            'STATISTICAL_TREND_5M': True,
            'STATISTICAL_TREND_15M': True,
            'M1_MARKET_STRUCTURE': True,
            'M5_MARKET_STRUCTURE': True,
            'M15_MARKET_STRUCTURE': True,
        }
    
    def _initialize_processors(self):
        """Initialize all signal processors"""
        processor_types = ProcessorFactory.get_available_processors()
        
        for processor_type in processor_types:
            try:
                self.processors[processor_type] = ProcessorFactory.create_processor(processor_type)
                logger.info(f"Initialized processor: {processor_type}")
            except Exception as e:
                logger.error(f"Failed to initialize processor {processor_type}: {e}")
    
    def register_new_processor(self, name: str, processor_class):
        """Register and initialize a new processor"""
        try:
            ProcessorFactory.register_processor(name, processor_class)
            self.processors[name] = ProcessorFactory.create_processor(name)
            self.active_entry_sources[name] = True
            logger.info(f"Registered new processor: {name}")
        except Exception as e:
            logger.error(f"Failed to register processor {name}: {e}")
    
    def set_active_entry_sources(self, sources: Dict[str, bool]):
        """Update which sources can generate entry signals"""
        self.active_entry_sources.update(sources)
        logger.info(f"Active entry sources updated: {self.active_entry_sources}")
    
    def set_symbol_context(self, symbol: str, current_price: float):
        """Update current trading context"""
        if symbol != self.current_symbol:
            # Clear state on symbol change
            for processor in self.processors.values():
                processor.previous_signal = None
            self.active_positions.clear()
            
        self.current_symbol = symbol
        self.current_price = current_price
    
    def process_signal(self, source: str, result: Any) -> Optional[StandardSignal]:
        """Process any signal through the appropriate processor"""
        if source not in self.processors:
            logger.error(f"No processor found for source: {source}")
            return None
        
        try:
            processor = self.processors[source]
            signal = processor.process(result)
            
            # Check if we should generate trading signals
            self._check_and_generate_signals(signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error processing signal for {source}: {e}", exc_info=True)
            return None
    
    # Convenience methods for backward compatibility
    def process_m1_ema(self, result):
        """Process M1 EMA signal"""
        return self.process_signal('M1_EMA', result)
    
    def process_m5_ema(self, result):
        """Process M5 EMA signal"""
        return self.process_signal('M5_EMA', result)
    
    def process_m15_ema(self, result):
        """Process M15 EMA signal"""
        return self.process_signal('M15_EMA', result)
    
    def process_statistical_trend(self, result):
        """Process M1 Statistical Trend signal"""
        return self.process_signal('STATISTICAL_TREND', result)
    
    def process_m5_statistical_trend(self, result):
        """Process M5 Statistical Trend signal"""
        return self.process_signal('STATISTICAL_TREND_5M', result)
    
    def process_m15_statistical_trend(self, result):
        """Process M15 Statistical Trend signal"""
        return self.process_signal('STATISTICAL_TREND_15M', result)
    
    def process_m1_market_structure(self, result):
        """Process M1 Market Structure signal"""
        return self.process_signal('M1_MARKET_STRUCTURE', result)
    
    def process_m5_market_structure(self, result):
        """Process M5 Market Structure signal"""
        return self.process_signal('M5_MARKET_STRUCTURE', result)

    def process_m15_market_structure(self, result):
        """Process M15 Market Structure signal"""
        return self.process_signal('M15_MARKET_STRUCTURE', result)
    
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
        from .models.signal_types import SignalCategory
        category = SignalCategory.from_value(weighted_value)
        
        # Use the base processor methods for consistency
        if self.processors:
            base_processor = next(iter(self.processors.values()))
            direction = base_processor.get_direction(weighted_value)
            strength = base_processor.get_strength_label(weighted_value)
        else:
            direction = 'LONG' if weighted_value > 0 else 'SHORT'
            strength = 'Strong' if abs(weighted_value) >= 25 else 'Medium' if abs(weighted_value) >= 10 else 'Weak'
        
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
        if signal.source in ['M1_EMA', 'M5_EMA', 'M15_EMA']:
            signal_desc = generate_ema_description(
                signal.source,
                signal.direction,
                signal.metadata.get('crossover_type'),
                signal.metadata.get('is_crossover', False),
                signal.metadata.get('signal_type')
            )
            
            # Add spread info
            if 'spread_pct' in signal.metadata:
                signal_desc += f" (Spread: {abs(signal.metadata['spread_pct']):.2f}%)"
                
        elif signal.source in ['STATISTICAL_TREND', 'STATISTICAL_TREND_5M', 'STATISTICAL_TREND_15M']:
            signal_desc = generate_trend_description(
                signal.source,
                signal.metadata.get('original_signal', signal.direction),
                signal.metadata.get('volatility_adjusted_strength', 0),
                signal.metadata.get('volume_confirmation', False),
                signal.metadata.get('regime'),
                signal.metadata.get('daily_bias')
            )

        elif signal.source in ['M1_MARKET_STRUCTURE', 'M5_MARKET_STRUCTURE', 'M15_MARKET_STRUCTURE']:
            signal_desc = generate_market_structure_description(
                signal.source,
                signal.metadata.get('original_signal', signal.direction),
                signal.metadata.get('structure_type', 'Unknown'),
                signal.metadata.get('direction', signal.direction)
            )
        else:
            signal_desc = f"{signal.source} {signal.direction} Signal"
        
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