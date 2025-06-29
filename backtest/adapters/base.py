# backtest/adapters/base.py
"""
Base adapter interface for wrapping calculations.
Provides standardized interface without modifying original calculations.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import logging

from core.signal_aggregator import StandardSignal

logger = logging.getLogger(__name__)


class CalculationAdapter(ABC):
    """
    Abstract base adapter for calculation integration.
    Wraps existing calculations without modification.
    """
    
    def __init__(self, calculation_class: type, config: Optional[Dict] = None, name: str = ""):
        """
        Initialize adapter.
        
        Args:
            calculation_class: The calculation class to wrap
            config: Configuration for the calculation
            name: Human-readable name
        """
        self.calculation_class = calculation_class
        self.config = config or {}
        self.name = name
        self.calculation = None
        self.last_signal = None
        self.symbol = None
        
        # Requirements flags
        self.requires_trades = False
        self.requires_quotes = False
        self.warmup_periods = 0
        
    def initialize(self, symbol: str) -> None:
        """Initialize the calculation for a specific symbol"""
        self.symbol = symbol
        try:
            # Create calculation instance
            if self.config:
                self.calculation = self.calculation_class(**self.config)
            else:
                self.calculation = self.calculation_class()
            logger.info(f"Initialized {self.name} for {symbol}")
        except Exception as e:
            logger.error(f"Failed to initialize {self.name}: {e}")
            raise
            
    @abstractmethod
    def feed_historical_data(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Feed historical bar data to warm up the calculation.
        
        Args:
            data: DataFrame with OHLCV data (UTC timezone index)
            symbol: Stock symbol
        """
        pass
        
    @abstractmethod
    def process_bar(self, bar_data: Dict, symbol: str) -> Optional[StandardSignal]:
        """
        Process a new bar and return signal if generated.
        
        Args:
            bar_data: Dict with 'timestamp', 'open', 'high', 'low', 'close', 'volume'
            symbol: Stock symbol
            
        Returns:
            StandardSignal if generated, None otherwise
        """
        pass
        
    def process_trades(self, trades: List[Dict], symbol: str) -> Optional[StandardSignal]:
        """
        Process trade data (for calculations that need it).
        Override in adapters that use trade data.
        
        Args:
            trades: List of trade dictionaries
            symbol: Stock symbol
            
        Returns:
            StandardSignal if generated
        """
        return None
        
    def process_quotes(self, quotes: List[Dict], symbol: str) -> Optional[StandardSignal]:
        """
        Process quote data (for calculations that need it).
        Override in adapters that use quote data.
        
        Args:
            quotes: List of quote dictionaries
            symbol: Stock symbol
            
        Returns:
            StandardSignal if generated
        """
        return None
        
    def get_signal_at_time(self, timestamp: datetime) -> Optional[StandardSignal]:
        """
        Get the signal at a specific time.
        Returns the last signal if available.
        
        Args:
            timestamp: Time to get signal for (UTC)
            
        Returns:
            StandardSignal or None
        """
        return self.last_signal
        
    def _create_signal(self, direction: str, strength: float, 
                      confidence: float, metadata: Dict[str, Any],
                      timestamp: Optional[datetime] = None) -> StandardSignal:
        """
        Helper to create a standardized signal.
        
        Args:
            direction: 'BULLISH', 'BEARISH', or 'NEUTRAL'
            strength: 0-100
            confidence: 0-100
            metadata: Additional calculation-specific data
            timestamp: Signal timestamp (defaults to now)
            
        Returns:
            StandardSignal instance
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
            
        signal = StandardSignal(
            name=self.name,
            timestamp=timestamp,
            direction=direction,
            strength=strength,
            confidence=confidence,
            metadata=metadata
        )
        
        self.last_signal = signal
        return signal
        
    def _create_neutral_signal(self, timestamp: Optional[datetime] = None) -> StandardSignal:
        """Create a neutral signal when no clear direction"""
        return self._create_signal(
            direction='NEUTRAL',
            strength=50.0,
            confidence=50.0,
            metadata={'reason': 'No clear signal'},
            timestamp=timestamp
        )
        
    def _map_signal_direction(self, original_signal: str) -> str:
        """
        Map calculation-specific signals to standard format.
        
        Args:
            original_signal: Signal from calculation (e.g., 'LONG', 'UP', 'BUY')
            
        Returns:
            Standardized direction: 'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        # Common mappings
        bullish_terms = ['BULLISH', 'LONG', 'BUY', 'UP', 'UPTREND', 'POSITIVE']
        bearish_terms = ['BEARISH', 'SHORT', 'SELL', 'DOWN', 'DOWNTREND', 'NEGATIVE']
        
        signal_upper = original_signal.upper()
        
        if signal_upper in bullish_terms:
            return 'BULLISH'
        elif signal_upper in bearish_terms:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
            
    def reset(self) -> None:
        """Reset the adapter and calculation state"""
        self.last_signal = None
        if hasattr(self.calculation, 'reset'):
            self.calculation.reset()
        else:
            # Reinitialize if no reset method
            self.initialize(self.symbol)