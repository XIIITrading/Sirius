# backtest/adapters/indicators/m1_ema_back_adapter.py
"""
Adapter for EMA Crossover Analysis calculation.
Wraps the 9/21 EMA crossover analyzer for backtesting integration.
"""

import sys
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional
import pandas as pd
import logging

# Fix the path setup to properly find modules directory
current_file = os.path.abspath(__file__)
indicators_dir = os.path.dirname(current_file)  # .../backtest/adapters/indicators
adapters_dir = os.path.dirname(indicators_dir)  # .../backtest/adapters
backtest_dir = os.path.dirname(adapters_dir)   # .../backtest
sirius_dir = os.path.dirname(backtest_dir)     # .../Sirius

# Add both Sirius root and backtest to path
if sirius_dir not in sys.path:
    sys.path.insert(0, sirius_dir)
if backtest_dir not in sys.path:
    sys.path.insert(0, backtest_dir)

# Now imports should work with the correct filename
from adapters.base import CalculationAdapter, StandardSignal
from modules.calculations.indicators.m1_ema import EMAAnalyzer, VolumeSignal as EMASignal

logger = logging.getLogger(__name__)


class M1EMABackAdapter(CalculationAdapter):
    """
    Adapter for 1-minute EMA Crossover backtesting.
    Converts 1-minute bar data to EMA signals.
    """
    
    def __init__(self, buffer_size: int = 100):
        """
        Initialize EMA adapter.
        
        Args:
            buffer_size: Number of candles to maintain in analyzer
        """
        # Initialize with EMAAnalyzer
        super().__init__(
            calculation_class=EMAAnalyzer,
            config={
                'buffer_size': buffer_size,
                'ema_short': 9,
                'ema_long': 21,
                'min_candles_required': 21
            },
            name="1-Min EMA Crossover"
        )
        
        # This calculation only needs bar data
        self.requires_trades = False
        self.requires_quotes = False
        self.warmup_periods = 21  # Need at least 21 bars for valid signal
        
        logger.info(f"M1EMABackAdapter initialized with buffer_size={buffer_size}")
        
    def feed_historical_data(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Feed historical bar data to warm up the calculation.
        
        Args:
            data: DataFrame with OHLCV data (UTC timezone index)
            symbol: Stock symbol
        """
        if data.empty:
            logger.warning(f"No historical data provided for {symbol}")
            return
            
        if not self.calculation:
            logger.error("Calculation not initialized")
            return
            
        logger.info(f"Feeding {len(data)} historical bars to EMA analyzer for {symbol}")
        
        # Convert DataFrame to list of candle dictionaries
        candles = []
        for timestamp, row in data.iterrows():
            # Ensure timestamp is UTC
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(timezone.utc)
            elif timestamp.tzinfo != timezone.utc:
                timestamp = timestamp.tz_convert(timezone.utc)
                
            candle_dict = {
                'timestamp': int(timestamp.timestamp() * 1000),  # milliseconds
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
                'vwap': row.get('vwap', (row['high'] + row['low'] + row['close']) / 3),
                'trades': row.get('transactions', 0)
            }
            candles.append(candle_dict)
            
        # Process historical candles
        signal = self.calculation.process_historical_candles(symbol, candles)
        
        # Store last signal if generated
        if signal:
            self.last_signal = self._convert_ema_signal(signal)
            logger.info(f"Historical data processed, last signal: {self.last_signal.direction}")
        else:
            logger.info("Historical data processed, no signal generated yet")
            
    def process_bar(self, bar_data: Dict, symbol: str) -> Optional[StandardSignal]:
        """
        Process a new bar and return signal if generated.
        
        Args:
            bar_data: Dict with 'timestamp', 'open', 'high', 'low', 'close', 'volume'
            symbol: Stock symbol
            
        Returns:
            StandardSignal if generated, None otherwise
        """
        if not self.calculation:
            logger.error("Calculation not initialized")
            return None
            
        # Convert bar data to format expected by EMA analyzer
        candle_data = {
            'timestamp': int(bar_data['timestamp'].timestamp() * 1000),
            'open': bar_data['open'],
            'high': bar_data['high'],
            'low': bar_data['low'],
            'close': bar_data['close'],
            'volume': bar_data['volume'],
            'vwap': bar_data.get('vwap', (bar_data['high'] + bar_data['low'] + bar_data['close']) / 3),
            'trades': bar_data.get('transactions', 0)
        }
        
        # Process candle
        ema_signal = self.calculation.process_candle(symbol, candle_data, is_complete=True)
        
        if ema_signal:
            # Convert to standard signal
            standard_signal = self._convert_ema_signal(ema_signal)
            self.last_signal = standard_signal
            return standard_signal
            
        return None
        
    def _convert_ema_signal(self, ema_signal: EMASignal) -> StandardSignal:
        """
        Convert EMA signal to standard signal format.
        
        Args:
            ema_signal: Signal from EMA analyzer (VolumeSignal type)
            
        Returns:
            StandardSignal with mapped direction
        """
        # Map signal directions
        direction_map = {
            'BULL': 'BULLISH',
            'BEAR': 'BEARISH',
            'NEUTRAL': 'NEUTRAL'
        }
        
        direction = direction_map.get(ema_signal.signal, 'NEUTRAL')
        
        # Extract key metrics for metadata
        metrics = ema_signal.metrics
        
        # ENHANCED DEBUG LOGGING
        logger.info(f"=== EMA Signal Debug ===")
        logger.info(f"Signal: {ema_signal.signal} -> {direction}")
        logger.info(f"EMA 9: {metrics.get('ema_9', 0):.2f}")
        logger.info(f"EMA 21: {metrics.get('ema_21', 0):.2f}")
        logger.info(f"EMA Spread: {metrics.get('ema_spread', 0):.2f}")
        logger.info(f"Price vs EMA9: {metrics.get('price_vs_ema9', 'unknown')}")
        logger.info(f"Reason: {ema_signal.reason}")
        logger.info(f"========================")
        
        metadata = {
            'ema_9': metrics.get('ema_9', 0),
            'ema_21': metrics.get('ema_21', 0),
            'ema_spread': metrics.get('ema_spread', 0),
            'ema_spread_pct': metrics.get('ema_spread_pct', 0),
            'price_vs_ema9': metrics.get('price_vs_ema9', 'unknown'),
            'trend_strength': metrics.get('trend_strength', 0),
            'last_crossover_type': metrics.get('last_crossover_type'),
            'reason': ema_signal.reason
        }
        
        return StandardSignal(
            name=self.name,
            timestamp=ema_signal.timestamp,
            direction=direction,
            strength=ema_signal.strength,
            confidence=ema_signal.strength,  # Use strength as confidence for EMA
            metadata=metadata
        )
        
    def get_signal_at_time(self, timestamp: datetime) -> Optional[StandardSignal]:
        """
        Get the signal at a specific time.
        For EMA, we return the last calculated signal.
        
        Args:
            timestamp: Time to get signal for (UTC)
            
        Returns:
            StandardSignal or None
        """
        # For bar-based calculations, we return the last signal
        # In production, you might want to check if the signal is still valid
        if self.last_signal:
            # Update timestamp to requested time
            return StandardSignal(
                name=self.last_signal.name,
                timestamp=timestamp,
                direction=self.last_signal.direction,
                strength=self.last_signal.strength,
                confidence=self.last_signal.confidence,
                metadata=self.last_signal.metadata
            )
        return None
        
    def reset(self) -> None:
        """Reset the adapter and calculation state"""
        self.last_signal = None
        # EMA analyzer doesn't have a reset method, so reinitialize
        if self.symbol:
            self.initialize(self.symbol)


# Standalone test
if __name__ == "__main__":
    import asyncio
    from datetime import timedelta
    import numpy as np
    
    async def test_adapter():
        """Test the M1 EMA adapter with sample data"""
        print("=== Testing 1-Minute EMA Backtest Adapter ===")
        
        # Create adapter
        adapter = M1EMABackAdapter()
        adapter.initialize("AAPL")
        
        # Create sample data
        now = datetime.now(timezone.utc)
        timestamps = pd.date_range(
            start=now - timedelta(hours=2),
            end=now,
            freq='1min',
            tz=timezone.utc
        )
        
        # Generate synthetic price data
        base_price = 150
        prices = base_price + np.cumsum(np.random.randn(len(timestamps)) * 0.1)
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': prices + np.random.randn(len(prices)) * 0.05,
            'high': prices + abs(np.random.randn(len(prices))) * 0.1,
            'low': prices - abs(np.random.randn(len(prices))) * 0.1,
            'close': prices,
            'volume': np.random.randint(1000, 10000, len(prices))
        }, index=timestamps)
        
        # Ensure high/low relationships
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        print(f"Created {len(df)} bars of test data")
        
        # Feed historical data
        adapter.feed_historical_data(df, "AAPL")
        
        # Process a new bar
        new_bar = {
            'timestamp': now + timedelta(minutes=1),
            'open': prices[-1],
            'high': prices[-1] + 0.1,
            'low': prices[-1] - 0.1,
            'close': prices[-1] + 0.05,
            'volume': 5000
        }
        
        signal = adapter.process_bar(new_bar, "AAPL")
        
        if signal:
            print(f"\nGenerated Signal:")
            print(f"  Direction: {signal.direction}")
            print(f"  Strength: {signal.strength:.1f}")
            print(f"  Confidence: {signal.confidence:.1f}")
            print(f"  EMA 9: {signal.metadata['ema_9']:.2f}")
            print(f"  EMA 21: {signal.metadata['ema_21']:.2f}")
            print(f"  Spread: {signal.metadata['ema_spread_pct']:.2f}%")
            print(f"  Reason: {signal.metadata['reason']}")
        else:
            print("No signal generated (might need more data)")
            
        # Get signal at specific time
        check_signal = adapter.get_signal_at_time(now)
        if check_signal:
            print(f"\nSignal at {now.strftime('%H:%M:%S UTC')}: {check_signal.direction}")
            
    asyncio.run(test_adapter())