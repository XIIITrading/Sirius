# backtest/plugins/m1_market_structure/adapter.py
"""
M1 Market Structure Adapter for backtesting.
Uses shared data cache from the engine and fractal-based market structure analysis.
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
from modules.calculations.market_structure.m1_market_structure import (
    MarketStructureAnalyzer, 
    MarketStructureSignal
)
from .aggregator import M1MarketStructureAggregator

logger = logging.getLogger(__name__)


class M1MarketStructureBackAdapter(CalculationAdapter):
    """
    Adapter for 1-minute Market Structure backtesting.
    Works with shared data cache from BacktestEngine.
    """
    
    def __init__(self, 
                 fractal_length: int = 5,
                 buffer_size: int = 200,
                 min_candles_required: int = 21):
        """
        Initialize market structure adapter.
        
        Args:
            fractal_length: Number of bars on each side for fractal detection
            buffer_size: Number of candles to maintain in buffer
            min_candles_required: Minimum candles needed for analysis
        """
        super().__init__(
            calculation_class=MarketStructureAnalyzer,
            config={
                'fractal_length': fractal_length,
                'buffer_size': buffer_size,
                'min_candles_required': min_candles_required
            },
            name="1-Min Market Structure"
        )
        
        # Data requirements
        self.requires_trades = False
        self.requires_quotes = False
        self.warmup_periods = max(min_candles_required, fractal_length * 2 + 1)
        
        # Shared data cache reference
        self.data_cache = None
        
        # Initialize aggregator
        self.aggregator = M1MarketStructureAggregator()
        
        # Debug mode flag
        self.debug_mode = logger.isEnabledFor(logging.DEBUG)
        
        logger.info(f"M1MarketStructureBackAdapter initialized with fractal_length={fractal_length}, "
                   f"buffer_size={buffer_size}, min_candles={min_candles_required}")
    
    def get_data_requirements(self) -> Dict:
        """Declare data requirements for shared cache system"""
        return {
            'bars': {
                'timeframe': '1min',
                'lookback_minutes': 200  # Match buffer_size for fractal detection
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
        Feed historical bar data from shared cache with detailed debugging.
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
        
        # DEBUG: Log data range and entry time
        if self.debug_mode:
            logger.info(f"\n{'='*80}")
            logger.info(f"M1 MARKET STRUCTURE ADAPTER DEBUG - {symbol}")
            logger.info(f"Entry time: {entry_time}")
            logger.info(f"Data range: {bars.index[0]} to {bars.index[-1]}")
            logger.info(f"Total bars available: {len(bars)}")
            logger.info(f"{'='*80}")
        
        # Convert to candle format and process
        candles = []
        candles_processed = 0
        
        for timestamp, row in bars.iterrows():
            # Stop at entry time to prevent look-ahead
            if timestamp >= entry_time:
                if self.debug_mode:
                    logger.debug(f"Stopping at entry time: {timestamp} >= {entry_time}")
                break
                
            # Ensure timestamp is UTC
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(timezone.utc)
            elif timestamp.tzinfo != timezone.utc:
                timestamp = timestamp.tz_convert(timezone.utc)
                
            candle_dict = {
                't': timestamp,  # MarketStructureAnalyzer expects 't' key
                'o': float(row['open']),
                'h': float(row['high']),
                'l': float(row['low']),
                'c': float(row['close']),
                'v': float(row['volume']),
                'vw': float(row.get('vwap', (row['high'] + row['low'] + row['close']) / 3)),
                'n': int(row.get('transactions', 0))
            }
            candles.append(candle_dict)
            candles_processed += 1
            
            # DEBUG: Log every 10th candle
            if self.debug_mode and candles_processed % 10 == 0:
                logger.debug(f"Processed {candles_processed} candles, last: {timestamp} "
                            f"OHLC: {row['open']:.2f}/{row['high']:.2f}/{row['low']:.2f}/{row['close']:.2f}")
        
        logger.info(f"Prepared {len(candles)} candles for processing (stopped before entry)")
        
        # DEBUG: Show last 5 candles
        if self.debug_mode and len(candles) >= 5:
            logger.info("\nLast 5 candles before entry:")
            for i in range(-5, 0):
                c = candles[i]
                logger.info(f"  {c['t'].strftime('%H:%M:%S')}: "
                           f"O:{c['o']:.2f} H:{c['h']:.2f} L:{c['l']:.2f} C:{c['c']:.2f}")
        
        # Process historical candles
        logger.info("\nProcessing historical candles...")
        signal = self.calculation.process_historical_candles(symbol, candles)
        
        if signal:
            self.last_signal = self._convert_market_structure_signal(signal)
            
            if self.debug_mode:
                logger.info(f"\nHistorical processing complete:")
                logger.info(f"  Last signal: {self.last_signal.direction}")
                logger.info(f"  Structure type: {self.last_signal.metadata.get('structure_type')}")
                logger.info(f"  Current trend: {self.last_signal.metadata.get('current_trend')}")
                logger.info(f"  Last high fractal: {self.last_signal.metadata.get('last_high_fractal')}")
                logger.info(f"  Last low fractal: {self.last_signal.metadata.get('last_low_fractal')}")
                logger.info(f"  Reason: {self.last_signal.metadata.get('reason')}")
                
                # DEBUG: Get analyzer state
                if hasattr(self.calculation, 'get_statistics'):
                    stats = self.calculation.get_statistics()
                    logger.info(f"\nAnalyzer statistics:")
                    logger.info(f"  Candles processed: {stats.get('candles_processed', 0)}")
                    logger.info(f"  Signals generated: {stats.get('signals_generated', 0)}")
                    logger.info(f"  Current trends: {stats.get('current_trends', {})}")
        else:
            logger.warning("Historical processing complete, no signal generated")
            # Try to get current state anyway
            if self.debug_mode and hasattr(self.calculation, 'get_current_analysis'):
                current = self.calculation.get_current_analysis(symbol)
                if current:
                    logger.info(f"\nCurrent analysis shows:")
                    logger.info(f"  Trend: {current.signal}")
                    logger.info(f"  Metrics: {current.metrics}")
    
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
        if self.debug_mode:
            logger.debug(f"Getting signal at time: {timestamp}")
            if self.last_signal:
                logger.debug(f"  Returning last signal: {self.last_signal.direction}")
                logger.debug(f"  Metadata: {self.last_signal.metadata}")
            else:
                logger.debug("  No signal available")
                
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