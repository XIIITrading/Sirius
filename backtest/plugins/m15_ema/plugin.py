# backtest/plugins/m15_ema/plugin.py
"""
M15 EMA Crossover Plugin Implementation

This file contains the complete implementation of the M15 EMA Crossover plugin.
It is self-contained and handles:
1. Data fetching (1-minute bars)
2. Aggregation to 15-minute bars
3. EMA crossover analysis
4. Result formatting
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import logging

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backtest.plugins.base_plugin import BacktestPlugin
from modules.calculations.indicators.m15_ema import EMAAnalyzer15M, VolumeSignal as EMASignal
from data.polygon_data_manager import PolygonDataManager

logger = logging.getLogger(__name__)


class M15EMAPlugin(BacktestPlugin):
    """
    Plugin for 15-minute EMA crossover analysis.
    
    This plugin:
    1. Fetches 1-minute bar data
    2. Aggregates to 15-minute bars
    3. Calculates 9 and 21 period EMAs on 15-minute timeframe
    4. Detects crossovers and trend direction
    5. Returns formatted signals for display
    """
    
    def __init__(self):
        """Initialize plugin with configuration"""
        self.data_manager = PolygonDataManager()
        self.config = {
            'ema_short': 9,               # Short-term EMA period
            'ema_long': 21,              # Long-term EMA period
            'buffer_size': 40,           # Number of 15-minute candles to maintain
            'min_candles_required': 21,  # Minimum for valid signal
            'lookback_minutes': 900      # Historical data to fetch (15 hours)
        }
        
    @property
    def name(self) -> str:
        return "15-Min EMA Crossover"
    
    @property
    def version(self) -> str:
        return "2.0.0"
    
    def get_config(self) -> Dict[str, Any]:
        """Get plugin configuration for UI settings"""
        return self.config.copy()
    
    def validate_inputs(self, symbol: str, entry_time: datetime, direction: str) -> bool:
        """
        Validate input parameters.
        
        Args:
            symbol: Must be non-empty string
            entry_time: Must be datetime object
            direction: Must be 'LONG' or 'SHORT'
            
        Returns:
            True if all inputs valid, False otherwise
        """
        if not symbol or not isinstance(symbol, str):
            logger.error(f"Invalid symbol: {symbol}")
            return False
        
        if not isinstance(entry_time, datetime):
            logger.error(f"Invalid entry_time: {entry_time}")
            return False
            
        if direction not in ['LONG', 'SHORT']:
            logger.error(f"Invalid direction: {direction}")
            return False
            
        return True
    
    async def run_analysis(self, symbol: str, entry_time: datetime, direction: str) -> Dict[str, Any]:
        """
        Run the complete M15 EMA crossover analysis.
        
        This is the main method that orchestrates:
        1. Data fetching
        2. Aggregation
        3. Analysis
        4. Result formatting
        """
        try:
            # 1. Fetch required 1-minute data
            logger.info(f"Fetching 1-minute data for {symbol}")
            bars_1min = await self._fetch_data(symbol, entry_time)
            
            if bars_1min.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # 2. Aggregate to 15-minute bars
            logger.info(f"Aggregating to 15-minute bars")
            bars_15min = self._aggregate_to_15min(bars_1min)
            
            if bars_15min.empty:
                raise ValueError(f"No valid 15-minute bars after aggregation")
            
            # 3. Run EMA analysis on 15-minute bars
            logger.info(f"Running EMA crossover analysis on 15-minute data")
            analyzer = EMAAnalyzer15M(
                buffer_size=self.config['buffer_size'],
                ema_short=self.config['ema_short'],
                ema_long=self.config['ema_long'],
                min_candles_required=self.config['min_candles_required']
            )
            
            # Process bars up to entry time
            signal = self._process_bars(analyzer, symbol, bars_15min, entry_time)
            
            # 4. Format and return results
            logger.info(f"Formatting results")
            return self._format_results(signal, entry_time, direction)
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise
    
    async def _fetch_data(self, symbol: str, entry_time: datetime) -> pd.DataFrame:
        """
        Fetch required 1-minute bar data from data source.
        
        This method handles all data fetching logic internally.
        No external data management needed.
        """
        # Calculate time range - need extra time for aggregation alignment
        start_time = entry_time - timedelta(minutes=self.config['lookback_minutes'] + 20)
        
        logger.info(f"Fetching {symbol} 1-minute data from {start_time} to {entry_time}")
        
        # Use PolygonDataManager to fetch 1-minute data
        bars = await self.data_manager.load_bars(
            symbol=symbol,
            start_time=start_time,
            end_time=entry_time,
            timeframe='1min',
            use_cache=True  # Use caching for efficiency
        )
        
        # Validate data
        bars = self._validate_data(bars)
        
        logger.info(f"Fetched {len(bars)} valid 1-minute bars")
        return bars
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean input data"""
        if data.empty:
            return data
            
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return pd.DataFrame()
        
        # Remove rows with NaN in OHLC
        initial_count = len(data)
        data = data.dropna(subset=['open', 'high', 'low', 'close'])
        if len(data) < initial_count:
            logger.warning(f"Dropped {initial_count - len(data)} rows with NaN values")
        
        # Ensure volume is not NaN (fill with 0 if needed)
        data['volume'] = data['volume'].fillna(0)
        
        # Validate OHLC relationships
        invalid = data[
            (data['high'] < data['low']) | 
            (data['high'] < data['open']) | 
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        ]
        
        if not invalid.empty:
            logger.warning(f"Dropping {len(invalid)} bars with invalid OHLC relationships")
            data = data.drop(invalid.index)
        
        # Check for zero or negative prices
        invalid_prices = data[
            (data['open'] <= 0) | 
            (data['high'] <= 0) | 
            (data['low'] <= 0) | 
            (data['close'] <= 0)
        ]
        
        if not invalid_prices.empty:
            logger.warning(f"Dropping {len(invalid_prices)} bars with invalid prices")
            data = data.drop(invalid_prices.index)
        
        # Ensure volume is non-negative
        data.loc[data['volume'] < 0, 'volume'] = 0
        
        # Add vwap if not present
        if 'vwap' not in data.columns:
            data['vwap'] = (data['high'] + data['low'] + data['close']) / 3
        
        # Add transactions if not present
        if 'transactions' not in data.columns:
            data['transactions'] = 0
        
        # Sort by index to ensure chronological order
        data = data.sort_index()
        
        return data
    
    def _aggregate_to_15min(self, bars_1min: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate 1-minute bars to 15-minute bars.
        
        Args:
            bars_1min: Validated 1-minute bars
            
        Returns:
            15-minute aggregated bars
        """
        # Create aggregation rules
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Add transactions if present
        if 'transactions' in bars_1min.columns:
            agg_rules['transactions'] = 'sum'
        
        # Resample to 15-minute bars
        # Use label='right' and closed='right' to match standard behavior
        bars_15min = bars_1min.resample('15T', label='right', closed='right').agg(agg_rules)
        
        # Calculate VWAP for aggregated bars
        # We need to calculate this separately after aggregation
        if 'vwap' in bars_1min.columns and bars_1min['volume'].sum() > 0:
            # Group the 1-minute bars by the 15-minute intervals
            vwap_calc = []
            for idx in bars_15min.index:
                # Get the 1-minute bars that belong to this 15-minute interval
                mask = (bars_1min.index > idx - pd.Timedelta(minutes=15)) & (bars_1min.index <= idx)
                interval_bars = bars_1min.loc[mask]
                
                if not interval_bars.empty and interval_bars['volume'].sum() > 0:
                    # Calculate volume-weighted average price
                    vwap = (interval_bars['vwap'] * interval_bars['volume']).sum() / interval_bars['volume'].sum()
                    vwap_calc.append(vwap)
                else:
                    # Fallback to typical price
                    vwap_calc.append((bars_15min.loc[idx, 'high'] + bars_15min.loc[idx, 'low'] + bars_15min.loc[idx, 'close']) / 3)
            
            bars_15min['vwap'] = vwap_calc
        else:
            # No VWAP data available, use typical price
            bars_15min['vwap'] = (bars_15min['high'] + bars_15min['low'] + bars_15min['close']) / 3
        
        # Remove any rows with NaN in OHLC
        bars_15min = bars_15min.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Ensure volume is not NaN
        bars_15min['volume'] = bars_15min['volume'].fillna(0)
        
        # Ensure transactions column exists
        if 'transactions' not in bars_15min.columns:
            bars_15min['transactions'] = 0
        
        logger.info(f"Aggregated {len(bars_1min)} 1-minute bars to {len(bars_15min)} 15-minute bars")
        
        return bars_15min
    
    def _process_bars(self, analyzer: EMAAnalyzer15M, 
                     symbol: str, bars: pd.DataFrame, 
                     entry_time: datetime) -> Optional[EMASignal]:
        """
        Process 15-minute bars through the EMA analyzer.
        
        Converts DataFrame to format expected by analyzer and
        processes all bars up to entry time.
        """
        candles = []
        
        # Convert bars to candle format expected by analyzer
        for timestamp, row in bars.iterrows():
            # Stop at entry time to prevent look-ahead bias
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
                'vwap': float(row['vwap']),
                'trades': int(row['transactions'])
            }
            candles.append(candle_dict)
        
        logger.info(f"Processing {len(candles)} 15-minute candles")
        
        # Process all historical candles
        signal = analyzer.process_historical_candles(symbol, candles)
        
        # If no signal from historical processing, get current state
        if not signal:
            signal = analyzer.get_current_analysis(symbol)
            
        return signal
    
    def _format_results(self, signal: Optional[EMASignal], 
                       entry_time: datetime, direction: str) -> Dict[str, Any]:
        """
        Format analysis results into standardized output format.
        
        This method transforms the raw calculation output into
        the standardized format expected by the UI.
        """
        if not signal:
            # No signal - return neutral result
            return self._create_neutral_result(entry_time)
        
        # Map signal direction to standard format
        direction_map = {
            'BULL': 'BULLISH',
            'BEAR': 'BEARISH',
            'NEUTRAL': 'NEUTRAL'
        }
        
        signal_direction = direction_map.get(signal.signal, 'NEUTRAL')
        metrics = signal.metrics
        
        # Build display table rows
        display_rows = []
        
        # Timeframe indicator
        display_rows.append(['Timeframe', '15-Minute'])
        
        # EMA values
        ema_9 = metrics.get('ema_9', 0)
        ema_21 = metrics.get('ema_21', 0)
        display_rows.append(['EMA 9', f"${ema_9:.2f}"])
        display_rows.append(['EMA 21', f"${ema_21:.2f}"])
        
        # EMA spread
        ema_spread = metrics.get('ema_spread', 0)
        ema_spread_pct = metrics.get('ema_spread_pct', 0)
        display_rows.append(['EMA Spread', f"${ema_spread:.2f} ({ema_spread_pct:.2f}%)"])
        
        # Price position
        price_vs_ema9 = metrics.get('price_vs_ema9', 'unknown')
        display_rows.append(['Price vs EMA 9', price_vs_ema9.title()])
        
        # Trend information
        trend_strength = metrics.get('trend_strength', 0)
        display_rows.append(['Trend Strength', f"{trend_strength:.0f}%"])
        
        # Crossover information
        last_crossover = metrics.get('last_crossover_type', 'None')
        if last_crossover and last_crossover != 'None':
            display_rows.append(['Last Crossover', last_crossover])
            # Check if crossover is recent (within last 5 bars = 75 minutes)
            crossover_bars_ago = metrics.get('crossover_bars_ago', 999)
            if crossover_bars_ago <= 5:
                display_rows.append(['Crossover Timing', f"{crossover_bars_ago} bars ago"])
        
        # Signal strength
        display_rows.append(['Signal Strength', f"{signal.strength:.0f}%"])
        
        # Create summary text
        summary = f"15-Min EMA {self.config['ema_short']}/{self.config['ema_long']} - {signal_direction}"
        if last_crossover and last_crossover != 'None':
            summary += f" ({last_crossover})"
        
        # Determine if signal aligns with intended direction
        alignment = self._check_direction_alignment(signal_direction, direction)
        if alignment:
            display_rows.append(['Direction Alignment', alignment])
        
        # Add description about timeframe characteristics
        timeframe_note = ("15-minute bars provide a medium-term view suitable for " +
                         "swing trading and position entries with wider stops")
        
        # Return standardized format
        return {
            'plugin_name': self.name,
            'timestamp': entry_time,
            'signal': {
                'direction': signal_direction,
                'strength': float(signal.strength),
                'confidence': float(signal.strength)  # Using strength as confidence
            },
            'details': {
                'timeframe': '15-minute',
                'ema_9': ema_9,
                'ema_21': ema_21,
                'ema_spread': ema_spread,
                'ema_spread_pct': ema_spread_pct,
                'price_vs_ema9': price_vs_ema9,
                'trend_strength': trend_strength,
                'last_crossover_type': last_crossover,
                'reason': signal.reason,
                'timeframe_note': timeframe_note
            },
            'display_data': {
                'summary': summary,
                'description': f"{signal.reason}. {timeframe_note}",
                'table_data': display_rows,
                'chart_markers': self._get_chart_markers(metrics, entry_time)
            }
        }
    
    def _create_neutral_result(self, entry_time: datetime) -> Dict[str, Any]:
        """Create a neutral result when no signal is found"""
        return {
            'plugin_name': self.name,
            'timestamp': entry_time,
            'signal': {
                'direction': 'NEUTRAL',
                'strength': 50.0,
                'confidence': 50.0
            },
            'details': {
                'timeframe': '15-minute',
                'ema_9': 0,
                'ema_21': 0,
                'ema_spread': 0,
                'ema_spread_pct': 0,
                'price_vs_ema9': 'unknown',
                'trend_strength': 0,
                'last_crossover_type': None,
                'reason': 'Insufficient data for 15-minute EMA calculation'
            },
            'display_data': {
                'summary': 'No Signal - Insufficient Data',
                'description': 'Not enough 15-minute candles to calculate EMAs',
                'table_data': [
                    ['Timeframe', '15-Minute'],
                    ['EMA 9', 'N/A'],
                    ['EMA 21', 'N/A'],
                    ['Signal Strength', '50%']
                ]
            }
        }
    
    def _check_direction_alignment(self, signal_direction: str, intended_direction: str) -> str:
        """Check if signal aligns with intended trade direction"""
        if signal_direction == 'NEUTRAL':
            return 'Neutral'
        
        if (intended_direction == 'LONG' and signal_direction == 'BULLISH') or \
           (intended_direction == 'SHORT' and signal_direction == 'BEARISH'):
            return 'Aligned ✓'
        else:
            return 'Opposed ✗'
    
    def _get_chart_markers(self, metrics: Dict, entry_time: datetime) -> List[Dict]:
        """Generate chart markers for visualization"""
        markers = []
        
        # Add EMA line values at entry time
        if metrics.get('ema_9'):
            markers.append({
                'type': 'ema_short',
                'price': metrics['ema_9'],
                'label': '15m EMA 9',
                'color': 'blue'
            })
            
        if metrics.get('ema_21'):
            markers.append({
                'type': 'ema_long',
                'price': metrics['ema_21'],
                'label': '15m EMA 21',
                'color': 'red'
            })
            
        # Add crossover marker if recent
        if metrics.get('last_crossover_type'):
            crossover_bars_ago = metrics.get('crossover_bars_ago', 999)
            if crossover_bars_ago <= 5:  # Within 75 minutes
                markers.append({
                    'type': 'crossover',
                    'label': f"15m {metrics['last_crossover_type']} ({crossover_bars_ago} bars ago)",
                    'color': 'green' if 'Bullish' in metrics['last_crossover_type'] else 'red'
                })
            
        return markers