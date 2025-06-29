# backtest/adapters/base.py
"""
Base adapter interface for wrapping live calculations in backtest framework.
Provides standardized interface for all calculation adapters.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class StandardSignal:
    """Standardized signal format for all calculations"""
    name: str                          # Calculation name
    timestamp: datetime                # Signal timestamp (UTC)
    direction: str                     # 'BULLISH', 'BEARISH', 'NEUTRAL'
    strength: float                    # 0-100 signal strength
    confidence: float                  # 0-100 confidence level
    bull_score: int = 0               # -2 to +2 for compatibility
    bear_score: int = 0               # -2 to +2 for compatibility
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'name': self.name,
            'timestamp': self.timestamp.isoformat(),
            'direction': self.direction,
            'strength': self.strength,
            'confidence': self.confidence,
            'bull_score': self.bull_score,
            'bear_score': self.bear_score,
            'metadata': self.metadata
        }


class CalculationAdapter(ABC):
    """
    Base adapter for wrapping live calculations for backtesting.
    Each calculation type (trend, volume, order flow) will have its own adapter.
    """
    
    def __init__(self, calculation_class: Any, config: Dict[str, Any], name: str):
        """
        Initialize adapter with live calculation class.
        
        Args:
            calculation_class: The live calculation class to wrap
            config: Configuration parameters for the calculation
            name: Friendly name for this calculation
        """
        self.calculation_class = calculation_class
        self.config = config
        self.name = name
        self.calculation = None
        self.is_initialized = False
        self.signal_history: List[StandardSignal] = []
        self.requires_trades = False  # Override in subclasses that need tick data
        self.warmup_periods = 0  # Override with calculation's warmup requirement
        
    def initialize(self, symbol: str) -> None:
        """Initialize the calculation instance"""
        try:
            self.calculation = self.calculation_class(**self.config)
            if hasattr(self.calculation, 'initialize_symbol'):
                self.calculation.initialize_symbol(symbol)
            if hasattr(self.calculation, 'warmup_periods'):
                self.warmup_periods = self.calculation.warmup_periods
            self.is_initialized = True
            logger.info(f"Initialized {self.name} for {symbol}")
        except Exception as e:
            logger.error(f"Failed to initialize {self.name}: {e}")
            raise
            
    @abstractmethod
    def feed_historical_data(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Feed historical bar data to prime the calculation.
        
        Args:
            data: DataFrame with OHLCV data, index is UTC datetime
            symbol: Stock symbol
        """
        pass
        
    @abstractmethod
    def process_bar(self, bar_data: Dict[str, Any], symbol: str) -> Optional[StandardSignal]:
        """
        Process a single bar and return standardized signal.
        
        Args:
            bar_data: Dict with keys: timestamp, open, high, low, close, volume
            symbol: Stock symbol
            
        Returns:
            StandardSignal if generated, None otherwise
        """
        pass
        
    def process_trades(self, trades: List[Dict[str, Any]], symbol: str) -> Optional[StandardSignal]:
        """
        Process tick/trade data if calculation requires it.
        Override in subclasses that use tick data.
        
        Args:
            trades: List of trade dictionaries
            symbol: Stock symbol
            
        Returns:
            StandardSignal if generated, None otherwise
        """
        return None
        
    def process_quotes(self, quotes: List[Dict[str, Any]], symbol: str) -> Optional[StandardSignal]:
        """
        Process quote data if calculation requires it.
        Override in subclasses that use quote data.
        
        Args:
            quotes: List of quote dictionaries
            symbol: Stock symbol
            
        Returns:
            StandardSignal if generated, None otherwise
        """
        return None
        
    def get_latest_signal(self) -> Optional[StandardSignal]:
        """Get the most recent signal"""
        return self.signal_history[-1] if self.signal_history else None
        
    def get_signal_at_time(self, timestamp: datetime) -> Optional[StandardSignal]:
        """Get signal at or before specific time"""
        for signal in reversed(self.signal_history):
            if signal.timestamp <= timestamp:
                return signal
        return None
        
    def _create_neutral_signal(self, timestamp: datetime) -> StandardSignal:
        """Create a neutral signal when no direction detected"""
        return StandardSignal(
            name=self.name,
            timestamp=timestamp,
            direction='NEUTRAL',
            strength=0.0,
            confidence=0.0,
            metadata={}
        )
        
    def _map_signal_direction(self, raw_signal: Any) -> str:
        """
        Map calculation-specific signal to standard direction.
        Override if calculation uses different signal format.
        """
        # Handle common signal formats
        if hasattr(raw_signal, 'signal'):
            signal_text = raw_signal.signal.upper()
            if 'BUY' in signal_text or 'BULL' in signal_text:
                return 'BULLISH'
            elif 'SELL' in signal_text or 'BEAR' in signal_text:
                return 'BEARISH'
                
        # Handle bull/bear score format
        if hasattr(raw_signal, 'bull_score') and hasattr(raw_signal, 'bear_score'):
            if raw_signal.bull_score > raw_signal.bear_score:
                return 'BULLISH'
            elif raw_signal.bear_score > raw_signal.bull_score:
                return 'BEARISH'
                
        return 'NEUTRAL'
        
    def reset(self) -> None:
        """Reset adapter state for new backtest"""
        self.signal_history.clear()
        self.is_initialized = False
        self.calculation = None