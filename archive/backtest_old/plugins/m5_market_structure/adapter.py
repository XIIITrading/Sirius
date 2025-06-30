# backtest/plugins/m5_market_structure/adapter.py
"""
M5 Market Structure Adapter for backtesting.
Uses shared data cache from the engine and fractal-based market structure analysis on 5-minute timeframe.
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
from modules.calculations.market_structure.m5_market_structure import (
    M5MarketStructureAnalyzer, 
    MarketStructureSignal
)
from .aggregator import M5MarketStructureAggregator

logger = logging.getLogger(__name__)


class M5MarketStructureBackAdapter(CalculationAdapter):
    """
    Adapter for 5-minute Market Structure backtesting.
    Works with shared data cache from BacktestEngine.
    """
    
    def __init__(self, 
                 fractal_length: int = 3,      # Fewer bars for 5-min timeframe
                 buffer_size: int = 100,       # 100 candles = 500 minutes
                 min_candles_required: int = 15):
        """
        Initialize M5 market structure adapter.
        
        Args:
            fractal_length: Number of bars on each side for fractal detection
            buffer_size: Number of 5-minute candles to maintain in buffer
            min_candles_required: Minimum candles needed for analysis
        """
        super().__init__(
            calculation_class=M5MarketStructureAnalyzer,
            config={
                'fractal_length': fractal_length,
                'buffer_size': buffer_size,
                'min_candles_required': min_candles_required
            },
            name="5-Min Market Structure"
        )
        
        # Data requirements
        self.requires_trades = False
        self.requires_quotes = False
        self.warmup_periods = max(min_candles_required * 5, fractal_length * 10 + 5)  # Convert to minutes
        
        # Shared data cache reference
        self.data_cache = None
        
        # Initialize aggregator
        self.aggregator = M5MarketStructureAggregator()
        
        logger.info(f"M5MarketStructureBackAdapter initialized with fractal_length={fractal_length}, "
                   f"buffer_size={buffer_size}, min_candles={min_candles_required}")
    
    def get_data_requirements(self) -> Dict:
        """Declare data requirements for shared cache system"""
        return {
            'bars': {
                'timeframe': '1min',
                'lookback_minutes': 600  # 10 hours of 1-minute data for 5-min aggregation
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
        
        # Use aggregator to create 5-minute bars
        bars_5min = self.aggregator.aggregate(bars_1min)
        
        if bars_5min.empty:
            logger.warning(f"No valid 5-minute bars after aggregation")
            return
            
        # Get entry time to prevent look-ahead
        entry_time = self.data_cache.get('entry_time')
        if not entry_time:
            logger.error("No entry_time in data cache")
            return
        
        logger.info(f"Processing {len(bars_5min)} historical 5-minute bars for {symbol}")
        
        # Convert to candle format and process
        candles = []
        for timestamp, row in bars_5min.iterrows():
            # Stop at entry time to prevent look-ahead
            if timestamp >= entry_time:
                break
                
            # Ensure timestamp is UTC
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(timezone.utc)
            elif timestamp.tzinfo != timezone.utc:
                timestamp = timestamp.tz_convert(timezone.utc)
                
            candle_dict = {
                't': timestamp,  # M5MarketStructureAnalyzer expects 't' key
                'o': float(row['open']),
                'h': float(row['high']),
                'l': float(row['low']),
                'c': float(row['close']),
                'v': float(row['volume']),
                'vw': float(row.get('vwap', (row['high'] + row['low'] + row['close']) / 3)),
                'n': int(row.get('transactions', 0))
            }
            candles.append(candle_dict)
        
        logger.info(f"Prepared {len(candles)} 5-minute candles for processing")
        
        # Process historical candles
        signal = self.calculation.process_historical_candles(symbol, candles)
        
        if signal:
            self.last_signal = self._convert_market_structure_signal(signal)
            logger.info(f"Historical processing complete, last signal: {self.last_signal.direction} "
                       f"({signal.structure_type})")
        else:
            logger.info("Historical processing complete, no signal generated")
    
    def process_bar(self, bar_data: Dict, symbol: str) -> Optional[StandardSignal]:
        """Process a single bar (not used in backtest mode with shared cache)"""
        if not self.calculation:
            logger.error("Calculation not initialized")
            return None
            
        # Convert bar to candle format
        candle_data = {
            't': bar_data['timestamp'],
            'o': float(bar_data['open']),
            'h': float(bar_data['high']),
            'l': float(bar_data['low']),
            'c': float(bar_data['close']),
            'v': float(bar_data['volume']),
            'vw': float(bar_data.get('vwap', (bar_data['high'] + bar_data['low'] + bar_data['close']) / 3)),
            'n': int(bar_data.get('transactions', 0))
        }
        
        # Process candle
        ms_signal = self.calculation.process_candle(symbol, candle_data, is_complete=True)
        
        if ms_signal:
            standard_signal = self._convert_market_structure_signal(ms_signal)
            self.last_signal = standard_signal
            return standard_signal
            
        return None
    
    def _convert_market_structure_signal(self, ms_signal: MarketStructureSignal) -> StandardSignal:
        """Convert Market Structure signal to standard signal format"""
        # Map directions
        direction_map = {
            'BULL': 'BULLISH',
            'BEAR': 'BEARISH',
            'NEUTRAL': 'NEUTRAL'
        }
        
        direction = direction_map.get(ms_signal.signal, 'NEUTRAL')
        metrics = ms_signal.metrics
        
        # Build metadata - ensure all values are clean
        metadata = {
            'structure_type': ms_signal.structure_type,  # BOS or CHoCH
            'current_trend': metrics.get('current_trend', 'NEUTRAL'),
            'last_high_fractal': float(metrics.get('last_high_fractal', 0)) if metrics.get('last_high_fractal') else None,
            'last_low_fractal': float(metrics.get('last_low_fractal', 0)) if metrics.get('last_low_fractal') else None,
            'last_break_type': metrics.get('last_break_type'),
            'last_break_price': float(metrics.get('last_break_price', 0)) if metrics.get('last_break_price') else None,
            'fractal_count': int(metrics.get('fractal_count', 0)),
            'structure_breaks': int(metrics.get('structure_breaks', 0)),
            'trend_changes': int(metrics.get('trend_changes', 0)),
            'candles_processed': int(metrics.get('candles_processed', 0)),
            'timeframe': '5-minute',
            'reason': str(ms_signal.reason)
        }
        
        # Convert last_break_time to ISO format if present
        if metrics.get('last_break_time'):
            break_time = metrics['last_break_time']
            if hasattr(break_time, 'isoformat'):
                metadata['last_break_time'] = break_time.isoformat()
        
        return StandardSignal(
            name=self.name,
            timestamp=ms_signal.timestamp,
            direction=direction,
            strength=float(ms_signal.strength),
            confidence=float(ms_signal.strength),  # Use strength as confidence
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