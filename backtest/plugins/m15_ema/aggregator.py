# backtest/plugins/m15_ema/aggregator.py
"""Aggregator for M15 EMA plugin - converts 1-minute bars to 15-minute bars"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class M15EMAggregator:
    """
    Handles aggregation of 1-minute bars to 15-minute bars.
    Also validates and cleans the data.
    """
    
    def __init__(self):
        self.source_timeframe = '1min'
        self.target_timeframe = '15min'
        
    def aggregate(self, bars_1min: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate 1-minute bars to 15-minute bars.
        
        Args:
            bars_1min: DataFrame with 1-minute OHLCV data
            
        Returns:
            DataFrame with 15-minute OHLCV data
        """
        if bars_1min.empty:
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        data = bars_1min.copy()
        
        # Validate and clean data first
        data = self._validate_data(data)
        if data.empty:
            return pd.DataFrame()
        
        # Sort by index to ensure chronological order
        data = data.sort_index()
        
        # Perform aggregation to 15-minute bars
        bars_15min = self._aggregate_to_15min(data)
        
        logger.info(f"Aggregated {len(data)} 1-minute bars to {len(bars_15min)} 15-minute bars")
        
        return bars_15min
    
    def _aggregate_to_15min(self, bars_1min: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate 1-minute bars to 15-minute bars.
        
        Args:
            bars_1min: Validated 1-minute bars
            
        Returns:
            15-minute aggregated bars
        """
        # Resample to 15-minute bars
        # Use label='right' and closed='right' to match standard behavior
        bars_15min = bars_1min.resample('15T', label='right', closed='right').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'vwap': self._aggregate_vwap,
            'transactions': 'sum' if 'transactions' in bars_1min.columns else None
        })
        
        # Remove any rows with NaN in OHLC
        bars_15min = bars_15min.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Ensure volume is not NaN
        bars_15min['volume'] = bars_15min['volume'].fillna(0)
        
        # Calculate VWAP if not present
        if 'vwap' not in bars_15min.columns or bars_15min['vwap'].isna().all():
            bars_15min['vwap'] = (bars_15min['high'] + bars_15min['low'] + bars_15min['close']) / 3
        
        # Ensure transactions column exists
        if 'transactions' not in bars_15min.columns:
            bars_15min['transactions'] = 0
        
        return bars_15min
    
    def _aggregate_vwap(self, series: pd.Series) -> float:
        """
        Calculate volume-weighted average price for aggregated bars.
        
        Args:
            series: Series of VWAP values with volume in the same DataFrame
            
        Returns:
            Aggregated VWAP
        """
        # Get the corresponding volume data
        if hasattr(series, 'index') and not series.empty:
            # Access the parent DataFrame to get volumes
            idx = series.index
            df = series._parent if hasattr(series, '_parent') else None
            
            if df is not None and 'volume' in df.columns:
                volumes = df.loc[idx, 'volume']
                total_volume = volumes.sum()
                
                if total_volume > 0:
                    # Calculate volume-weighted average
                    return (series * volumes).sum() / total_volume
        
        # Fallback to simple average if volume data not available
        return series.mean() if not series.empty else np.nan
    
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
        
        # Add vwap if not present
        if 'vwap' not in data.columns:
            data['vwap'] = (data['high'] + data['low'] + data['close']) / 3
        
        # Add transactions if not present
        if 'transactions' not in data.columns:
            data['transactions'] = 0
        
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