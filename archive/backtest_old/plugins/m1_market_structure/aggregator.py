# backtest/plugins/m1_market_structure/aggregator.py
"""Aggregator for M1 Market Structure plugin"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


class M1MarketStructureAggregator:
    """
    Handles data validation and preparation for 1-minute bars.
    No aggregation needed since we use 1-minute bars directly.
    """
    
    def __init__(self):
        self.source_timeframe = '1min'
        self.target_timeframe = '1min'
        
    def aggregate(self, bars_1min: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and prepare 1-minute bars.
        
        Args:
            bars_1min: DataFrame with 1-minute OHLCV data
            
        Returns:
            DataFrame with cleaned 1-minute OHLCV data
        """
        if bars_1min.empty:
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        data = bars_1min.copy()
        
        # Validate and clean data
        data = self._validate_data(data)
        if data.empty:
            return pd.DataFrame()
        
        # Sort by index to ensure chronological order
        data = data.sort_index()
        
        logger.info(f"Validated {len(data)} 1-minute bars")
        
        return data
    
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