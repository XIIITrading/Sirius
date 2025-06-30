# backtest/plugins/m1_ema/adapter.py
"""
M1 EMA Crossover Adapter for backtesting.
Uses shared data cache from the engine.
"""

import sys
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional
import pandas as pd
import logging

# Add paths for imports
current_file = os.path.abspath(__file__)
plugin_dir = os.path.dirname(current_file)
plugins_dir = os.path.dirname(plugin_dir)
backtest_dir = os.path.dirname(plugins_dir)
sirius_dir = os.path.dirname(backtest_dir)

if sirius_dir not in sys.path:
    sys.path.insert(0, sirius_dir)
if backtest_dir not in sys.path:
    sys.path.insert(0, backtest_dir)

from adapters.base import CalculationAdapter, StandardSignal
from modules.calculations.indicators.m1_ema import EMAAnalyzer, VolumeSignal as EMASignal
from .aggregator import M1EMAggregator

logger = logging.getLogger(__name__)


class M1EMABackAdapter(CalculationAdapter):
    """
    Adapter for 1-minute EMA Crossover backtesting.
    Works with shared data cache from BacktestEngine.
    """
    
    def __init__(self, buffer_size: int = 100):
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
        
        # Data requirements
        self.requires_trades = False
        self.requires_quotes = False
        self.warmup_periods = 21
        
        # Shared data cache reference
        self.data_cache = None
        
        # Initialize aggregator
        self.aggregator = M1EMAggregator()
        
        logger.info(f"M1EMABackAdapter initialized with buffer_size={buffer_size}")
    
    def get_data_requirements(self) -> Dict:
        """Declare data requirements for shared cache system"""
        return {
            'bars': {
                'timeframe': '1min',
                'lookback_minutes': 120  # 2 hours of 1-minute data
            },
            'trades': None,  # Not needed
            'quotes': None   # Not needed
        }
    
    def set_data_cache(self, cache: Dict):
        """Receive shared data cache from engine"""
        self.data_cache = cache
        logger.debug(f"Data cache set with keys: {list(cache.keys())}")
    
    def feed_historical_data(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Feed historical bar data from shared cache.
        Note: 'data' parameter is legacy, we use self.data_cache
        """
        if not self.data_cache:
            logger.error("No data cache available")
            return
            
        # Get 1-minute bars from shared cache
        bars_1min = self.data_cache.get('1min_bars')
        if bars_1min is None or bars_1min.empty:
            logger.warning(f"No 1-minute bars in data cache for {symbol}")
            return
        
        # Use aggregator to validate and clean bars
        bars = self.aggregator.aggregate(bars_1min)
        
        if bars.empty:
            logger.warning(f"No valid 1-minute bars after validation")
            return
            
        # Get entry time to prevent look-ahead
        entry_time = self.data_cache.get('entry_time')
        if not entry_time:
            logger.error("No entry_time in data cache")
            return
        
        logger.info(f"Processing {len(bars)} historical 1-minute bars for {symbol}")
        
        # Convert to candle format and process
        candles = []
        for timestamp, row in bars.iterrows():
            # Stop at entry time to prevent look-ahead
            if timestamp >= entry_time:
                break
                
            # Ensure timestamp is UTC
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(timezone.utc)
            elif timestamp.tzinfo != timezone.utc:
                timestamp = timestamp.tz_convert(timezone.utc)
                
            candle_dict = {
                'timestamp': int(timestamp.timestamp() * 1000),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'vwap': float(row.get('vwap', (row['high'] + row['low'] + row['close']) / 3)),
                'trades': int(row.get('transactions', 0))
            }
            candles.append(candle_dict)
        
        logger.info(f"Prepared {len(candles)} candles for processing")
        
        # Process historical candles
        signal = self.calculation.process_historical_candles(symbol, candles)
        
        if signal:
            self.last_signal = self._convert_ema_signal(signal)
            logger.info(f"Historical processing complete, last signal: {self.last_signal.direction}")
        else:
            logger.info("Historical processing complete, no signal generated")
    
    def process_bar(self, bar_data: Dict, symbol: str) -> Optional[StandardSignal]:
        """Process a single bar (not used in backtest mode with shared cache)"""
        if not self.calculation:
            logger.error("Calculation not initialized")
            return None
            
        # Convert bar to candle format
        candle_data = {
            'timestamp': int(bar_data['timestamp'].timestamp() * 1000),
            'open': float(bar_data['open']),
            'high': float(bar_data['high']),
            'low': float(bar_data['low']),
            'close': float(bar_data['close']),
            'volume': float(bar_data['volume']),
            'vwap': float(bar_data.get('vwap', (bar_data['high'] + bar_data['low'] + bar_data['close']) / 3)),
            'trades': int(bar_data.get('transactions', 0))
        }
        
        # Process candle
        ema_signal = self.calculation.process_candle(symbol, candle_data, is_complete=True)
        
        if ema_signal:
            standard_signal = self._convert_ema_signal(ema_signal)
            self.last_signal = standard_signal
            return standard_signal
            
        return None
    
    def _convert_ema_signal(self, ema_signal: EMASignal) -> StandardSignal:
        """Convert EMA signal to standard signal format"""
        # Map directions
        direction_map = {
            'BULL': 'BULLISH',
            'BEAR': 'BEARISH',
            'NEUTRAL': 'NEUTRAL'
        }
        
        direction = direction_map.get(ema_signal.signal, 'NEUTRAL')
        metrics = ema_signal.metrics
        
        # Build metadata - ensure all values are clean
        metadata = {
            'ema_9': float(metrics.get('ema_9', 0)),
            'ema_21': float(metrics.get('ema_21', 0)),
            'ema_spread': float(metrics.get('ema_spread', 0)),
            'ema_spread_pct': float(metrics.get('ema_spread_pct', 0)),
            'price_vs_ema9': str(metrics.get('price_vs_ema9', 'unknown')),
            'trend_strength': float(metrics.get('trend_strength', 0)),
            'last_crossover_type': metrics.get('last_crossover_type'),
            'reason': str(ema_signal.reason)
        }
        
        return StandardSignal(
            name=self.name,
            timestamp=ema_signal.timestamp,
            direction=direction,
            strength=float(ema_signal.strength),
            confidence=float(ema_signal.strength),  # Use strength as confidence
            metadata=metadata
        )
    
    def get_signal_at_time(self, timestamp: datetime) -> Optional[StandardSignal]:
        """Get the signal at entry time"""
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