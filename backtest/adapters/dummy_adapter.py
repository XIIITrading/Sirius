# backtest/adapters/dummy_adapter.py
"""
Dummy adapter for testing the backtest system.
Generates random signals for development.
"""

import random
from datetime import datetime, timezone
from typing import Dict, Optional
import pandas as pd

from adapters.base import CalculationAdapter, StandardSignal


class DummyCalculation:
    """Dummy calculation that generates random signals"""
    def __init__(self):
        self.last_price = 100
        
    def process(self, price: float) -> Dict:
        """Generate random signal"""
        directions = ['BULLISH', 'BEARISH', 'NEUTRAL']
        direction = random.choice(directions)
        strength = random.uniform(40, 90)
        self.last_price = price
        
        return {
            'direction': direction,
            'strength': strength,
            'price': price
        }


class DummyAdapter(CalculationAdapter):
    """Adapter for dummy calculation"""
    
    def __init__(self):
        super().__init__(
            calculation_class=DummyCalculation,
            config={},
            name="Dummy Test Signal"
        )
        
    def feed_historical_data(self, data: pd.DataFrame, symbol: str) -> None:
        """Process historical data"""
        if not data.empty:
            # Just use last price
            last_price = data.iloc[-1]['close']
            if self.calculation:
                self.calculation.last_price = last_price
                
    def process_bar(self, bar_data: Dict, symbol: str) -> Optional[StandardSignal]:
        """Process bar and generate signal"""
        if not self.calculation:
            return None
            
        # Get signal from dummy calculation
        result = self.calculation.process(bar_data['close'])
        
        # Convert to standard signal
        return self._create_signal(
            direction=result['direction'],
            strength=result['strength'],
            confidence=random.uniform(60, 90),  # Random confidence
            metadata={
                'price': result['price'],
                'source': 'dummy'
            },
            timestamp=bar_data['timestamp']
        )