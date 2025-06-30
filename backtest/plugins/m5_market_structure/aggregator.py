# backtest/plugins/m5_market_structure/aggregator.py
"""Aggregator for M5 Market Structure plugin"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


class M5MarketStructureAggregator:
    """
    Handles data aggregation from 1-minute to 5-minute bars.
    """
    
    def __init__(self):
        self.source_timeframe = '1min'
        self.target_timeframe = '5min'
        
    def aggregate(self, bars_1min: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate 1-minute bars to 5-minute bars.
        
        Args:
            bars_1min: DataFrame with 1-minute OHLCV data
            
        Returns:
            DataFrame with 5-minute OHLCV data
        """
        if bars_1min.empty:
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        data = bars_1min.copy()
        
        # Validate and clean data
        data = self._validate_data(data)
        if data.empty:
            return pd.DataFrame()
        
        # Perform 5-minute aggregation
        bars_5min = data.resample('5T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'vwap': self._aggregate_vwap,
            'transactions': 'sum' if 'transactions' in data.columns else None
        })
        
        # Remove any rows with NaN in OHLC
        bars_5min = bars_5min.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Validate aggregated bars
        bars_5min = self._validate_aggregated_bars(bars_5min)
        
        logger.info(f"Aggregated {len(data)} 1-minute bars into {len(bars_5min)} 5-minute bars")
        
        return bars_5min
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean input data"""
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
        
        # Add VWAP if not present
        if 'vwap' not in data.columns:
            data['vwap'] = (data['high'] + data['low'] + data['close']) / 3
        
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
        
        return data
    
    def _aggregate_vwap(self, series: pd.Series) -> float:
        """
        Aggregate VWAP for the period.
        Simple average of VWAP values (could be improved with volume weighting)
        """
        if series.empty or series.isna().all():
            return float('nan')
        return series.mean()
    
    def _validate_aggregated_bars(self, bars_5min: pd.DataFrame) -> pd.DataFrame:
        """Validate aggregated 5-minute bars"""
        # Ensure OHLC relationships are valid
        invalid = bars_5min[
            (bars_5min['high'] < bars_5min['low']) | 
            (bars_5min['high'] < bars_5min['open']) | 
            (bars_5min['high'] < bars_5min['close']) |
            (bars_5min['low'] > bars_5min['open']) |
            (bars_5min['low'] > bars_5min['close'])
        ]
        
        if not invalid.empty:
            logger.warning(f"Dropping {len(invalid)} aggregated bars with invalid OHLC")
            bars_5min = bars_5min.drop(invalid.index)
        
        # Sort by index to ensure chronological order
        bars_5min = bars_5min.sort_index()
        
        return bars_5min