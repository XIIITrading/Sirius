# New file: backtest/data/trade_preprocessor.py

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

class TradePreprocessor:
    """
    Pre-compute expensive operations on trade/quote data.
    Results are cached for reuse across plugins.
    """
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self._preprocessing_cache = {}
    
    async def get_preprocessed_trades(self, symbol: str, start_time: datetime, 
                                    end_time: datetime) -> Dict[str, pd.DataFrame]:
        """
        Get pre-processed trade data with various computations.
        
        Returns dict with:
        - 'aligned_trades': Trades with synchronized quotes
        - 'volume_bars': 1-minute volume bars with bid/ask breakdown
        - 'imbalance_metrics': Rolling imbalance calculations
        """
        cache_key = f"{symbol}_{start_time}_{end_time}"
        
        if cache_key in self._preprocessing_cache:
            return self._preprocessing_cache[cache_key]
        
        # Get aligned trades
        aligned_trades = await self.data_manager.load_aligned_trades(
            symbol, start_time, end_time
        )
        
        result = {
            'aligned_trades': aligned_trades,
            'volume_bars': self._compute_volume_bars(aligned_trades),
            'imbalance_metrics': self._compute_imbalance_metrics(aligned_trades)
        }
        
        self._preprocessing_cache[cache_key] = result
        return result
    
    def _compute_volume_bars(self, aligned_trades: pd.DataFrame) -> pd.DataFrame:
        """Pre-compute 1-minute volume bars with bid/ask breakdown"""
        if aligned_trades.empty:
            return pd.DataFrame()
        
        # Resample to 1-minute bars
        bars = aligned_trades.resample('1min').agg({
            'price': ['first', 'last', 'min', 'max'],
            'size': 'sum'
        })
        
        # Add bid/ask volume breakdown
        buy_trades = aligned_trades[aligned_trades['trade_side'] == 'buy']
        sell_trades = aligned_trades[aligned_trades['trade_side'] == 'sell']
        
        bars['buy_volume'] = buy_trades.resample('1min')['size'].sum()
        bars['sell_volume'] = sell_trades.resample('1min')['size'].sum()
        bars['net_volume'] = bars['buy_volume'] - bars['sell_volume']
        bars['volume_imbalance'] = bars['net_volume'] / (bars['buy_volume'] + bars['sell_volume'])
        
        return bars.fillna(0)
    
    def _compute_imbalance_metrics(self, aligned_trades: pd.DataFrame) -> pd.DataFrame:
        """Pre-compute rolling imbalance metrics"""
        if aligned_trades.empty:
            return pd.DataFrame()
        
        # Calculate trade-by-trade metrics
        metrics = pd.DataFrame(index=aligned_trades.index)
        
        # Cumulative volumes
        metrics['cum_buy_volume'] = aligned_trades[aligned_trades['trade_side'] == 'buy']['size'].cumsum()
        metrics['cum_sell_volume'] = aligned_trades[aligned_trades['trade_side'] == 'sell']['size'].cumsum()
        metrics.fillna(method='ffill', inplace=True)
        
        # Rolling imbalance (last 100 trades)
        metrics['rolling_buy'] = aligned_trades[aligned_trades['trade_side'] == 'buy']['size'].rolling(100).sum()
        metrics['rolling_sell'] = aligned_trades[aligned_trades['trade_side'] == 'sell']['size'].rolling(100).sum()
        metrics['rolling_imbalance'] = (metrics['rolling_buy'] - metrics['rolling_sell']) / (metrics['rolling_buy'] + metrics['rolling_sell'])
        
        return metrics