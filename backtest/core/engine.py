# backtest/core/engine.py
"""
Main backtest engine that orchestrates the entire backtesting process.
Coordinates data loading, adapter management, and result storage.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json

from ..adapters.base import CalculationAdapter, StandardSignal
from .data_manager import BacktestDataManager
from .result_store import BacktestResultStore, BacktestResult
from .signal_aggregator import SignalAggregator

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest run"""
    symbol: str
    entry_time: datetime  # Must be UTC
    direction: str  # 'LONG' or 'SHORT'
    
    # Time windows
    historical_lookback_hours: int = 2
    forward_bars: int = 60  # 1-minute bars to track
    
    # Data settings
    use_cached_data: bool = True
    fetch_trades: bool = True
    fetch_quotes: bool = True
    
    # Calculation settings
    enabled_calculations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate configuration"""
        if self.entry_time.tzinfo != timezone.utc:
            raise ValueError("Entry time must be UTC")
        if self.direction not in ['LONG', 'SHORT']:
            raise ValueError("Direction must be 'LONG' or 'SHORT'")


class BacktestEngine:
    """
    Main backtesting orchestrator.
    Manages the complete backtest lifecycle from data loading to result storage.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize backtest engine.
        
        Args:
            config_path: Path to JSON configuration file
        """
        self.config_path = config_path
        self.data_manager = BacktestDataManager()
        self.result_store = BacktestResultStore()
        self.signal_aggregator = SignalAggregator()
        
        # Adapter registry - populated by register_adapter()
        self.adapters: Dict[str, CalculationAdapter] = {}
        
        # Performance tracking
        self.backtest_count = 0
        self.total_runtime = 0
        
        if config_path:
            self._load_config(config_path)
            
    def _load_config(self, config_path: str) -> None:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Apply any global settings
                if 'data_cache_dir' in config:
                    self.data_manager.set_cache_dir(config['data_cache_dir'])
                if 'results_dir' in config:
                    self.result_store.set_results_dir(config['results_dir'])
                logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            
    def register_adapter(self, name: str, adapter: CalculationAdapter) -> None:
        """
        Register a calculation adapter.
        
        Args:
            name: Unique name for the adapter
            adapter: Adapter instance
        """
        self.adapters[name] = adapter
        logger.info(f"Registered adapter: {name}")
        
    def register_adapters(self, adapters: Dict[str, CalculationAdapter]) -> None:
        """Register multiple adapters at once"""
        for name, adapter in adapters.items():
            self.register_adapter(name, adapter)
            
    async def run_backtest(self, config: BacktestConfig) -> BacktestResult:
        """
        Run a single backtest iteration.
        
        Args:
            config: Backtest configuration
            
        Returns:
            BacktestResult with analysis and performance metrics
        """
        start_time = datetime.now(timezone.utc)
        logger.info(f"Starting backtest for {config.symbol} at {config.entry_time}")
        
        try:
            # 1. Load historical data
            historical_data = await self._load_historical_data(config)
            
            # 2. Load forward data (60 bars after entry)
            forward_data = await self._load_forward_data(config)
            
            # 3. Initialize adapters with historical data
            await self._initialize_adapters(config.symbol, historical_data, config)
            
            # 4. Get entry signals from all calculations
            entry_signals = await self._get_entry_signals(config)
            
            # 5. Aggregate signals using point & call system
            aggregated_signal = self.signal_aggregator.aggregate_signals(entry_signals)
            
            # 6. Simulate forward price movement
            forward_analysis = await self._analyze_forward_movement(
                config, forward_data, entry_signals
            )
            
            # 7. Create and store result
            result = BacktestResult(
                config=config,
                entry_signals=entry_signals,
                aggregated_signal=aggregated_signal,
                forward_data=forward_data,
                forward_analysis=forward_analysis,
                runtime_seconds=(datetime.now(timezone.utc) - start_time).total_seconds()
            )
            
            # Store result
            self.result_store.store_result(result)
            
            # Update tracking
            self.backtest_count += 1
            self.total_runtime += result.runtime_seconds
            
            logger.info(f"Backtest completed in {result.runtime_seconds:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
            
    async def _load_historical_data(self, config: BacktestConfig) -> Dict[str, pd.DataFrame]:
        """Load historical data for calculations"""
        end_time = config.entry_time
        start_time = end_time - timedelta(hours=config.historical_lookback_hours)
        
        logger.info(f"Loading historical data from {start_time} to {end_time}")
        
        data = {}
        
        # Load 1-minute bars
        data['bars_1min'] = await self.data_manager.load_bars(
            symbol=config.symbol,
            start_time=start_time,
            end_time=end_time,
            timeframe='1min',
            use_cache=config.use_cached_data
        )
        
        # Load trade data if needed
        if config.fetch_trades and self._any_adapter_needs_trades():
            data['trades'] = await self.data_manager.load_trades(
                symbol=config.symbol,
                start_time=start_time,
                end_time=end_time,
                use_cache=config.use_cached_data
            )
            
        # Load quote data if needed
        if config.fetch_quotes and self._any_adapter_needs_quotes():
            data['quotes'] = await self.data_manager.load_quotes(
                symbol=config.symbol,
                start_time=start_time,
                end_time=end_time,
                use_cache=config.use_cached_data
            )
            
        return data
        
    async def _load_forward_data(self, config: BacktestConfig) -> pd.DataFrame:
        """Load forward-looking data for analysis"""
        start_time = config.entry_time
        end_time = start_time + timedelta(minutes=config.forward_bars)
        
        logger.info(f"Loading forward data from {start_time} to {end_time}")
        
        return await self.data_manager.load_bars(
            symbol=config.symbol,
            start_time=start_time,
            end_time=end_time,
            timeframe='1min',
            use_cache=config.use_cached_data
        )
        
    async def _initialize_adapters(self, symbol: str, historical_data: Dict[str, Any], 
                                  config: BacktestConfig) -> None:
        """Initialize all adapters with historical data"""
        # Filter adapters based on enabled calculations
        active_adapters = {
            name: adapter for name, adapter in self.adapters.items()
            if not config.enabled_calculations or name in config.enabled_calculations
        }
        
        logger.info(f"Initializing {len(active_adapters)} adapters")
        
        # Initialize each adapter
        init_tasks = []
        for name, adapter in active_adapters.items():
            init_tasks.append(self._initialize_single_adapter(
                name, adapter, symbol, historical_data
            ))
            
        await asyncio.gather(*init_tasks)
        
    async def _initialize_single_adapter(self, name: str, adapter: CalculationAdapter,
                                       symbol: str, historical_data: Dict[str, Any]) -> None:
        """Initialize a single adapter"""
        try:
            # Initialize calculation
            adapter.initialize(symbol)
            
            # Feed historical bar data
            if 'bars_1min' in historical_data:
                adapter.feed_historical_data(historical_data['bars_1min'], symbol)
                
            # Feed trade data if adapter needs it
            if adapter.requires_trades and 'trades' in historical_data:
                # Process trades in chunks
                trades = historical_data['trades']
                for i in range(0, len(trades), 1000):
                    chunk = trades[i:i+1000]
                    adapter.process_trades(chunk, symbol)
                    
            logger.info(f"Initialized {name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize {name}: {e}")
            # Don't fail entire backtest if one adapter fails
            
    async def _get_entry_signals(self, config: BacktestConfig) -> List[StandardSignal]:
        """Get entry signals from all adapters"""
        signals = []
        
        # Get signal at entry time from each adapter
        for name, adapter in self.adapters.items():
            if (config.enabled_calculations and 
                name not in config.enabled_calculations):
                continue
                
            try:
                signal = adapter.get_signal_at_time(config.entry_time)
                if signal:
                    signals.append(signal)
                else:
                    # Create neutral signal if no signal available
                    signals.append(adapter._create_neutral_signal(config.entry_time))
                    
            except Exception as e:
                logger.error(f"Failed to get signal from {name}: {e}")
                
        logger.info(f"Collected {len(signals)} entry signals")
        return signals
        
    async def _analyze_forward_movement(self, config: BacktestConfig, 
                                      forward_data: pd.DataFrame,
                                      entry_signals: List[StandardSignal]) -> Dict[str, Any]:
        """Analyze price movement for 60 bars after entry"""
        if forward_data.empty:
            return {
                'error': 'No forward data available',
                'max_favorable_move': 0,
                'max_adverse_move': 0
            }
            
        entry_price = forward_data.iloc[0]['close']
        
        # Calculate moves based on direction
        if config.direction == 'LONG':
            favorable_moves = ((forward_data['high'] - entry_price) / entry_price * 100)
            adverse_moves = ((entry_price - forward_data['low']) / entry_price * 100)
        else:  # SHORT
            favorable_moves = ((entry_price - forward_data['low']) / entry_price * 100)
            adverse_moves = ((forward_data['high'] - entry_price) / entry_price * 100)
            
        # Key metrics
        analysis = {
            'entry_price': entry_price,
            'exit_price': forward_data.iloc[-1]['close'],
            'max_favorable_move': favorable_moves.max(),
            'max_adverse_move': adverse_moves.max(),
            'final_pnl': self._calculate_pnl(
                entry_price, 
                forward_data.iloc[-1]['close'],
                config.direction
            ),
            'time_to_max_favorable': favorable_moves.idxmax(),
            'time_to_max_adverse': adverse_moves.idxmax(),
            'bars_analyzed': len(forward_data),
            'volume_profile': {
                'total': forward_data['volume'].sum(),
                'average': forward_data['volume'].mean(),
                'at_extremes': self._analyze_volume_at_extremes(forward_data)
            }
        }
        
        # Add signal accuracy analysis
        consensus_direction = self.signal_aggregator.get_consensus_direction(entry_signals)
        analysis['signal_accuracy'] = {
            'consensus_matched_user': consensus_direction == config.direction.replace('LONG', 'BULLISH').replace('SHORT', 'BEARISH'),
            'profitable': analysis['final_pnl'] > 0,
            'signal_aligned_with_outcome': (
                (consensus_direction == 'BULLISH' and analysis['final_pnl'] > 0) or
                (consensus_direction == 'BEARISH' and analysis['final_pnl'] < 0)
            )
        }
        
        return analysis
        
    def _calculate_pnl(self, entry_price: float, exit_price: float, 
                      direction: str) -> float:
        """Calculate P&L percentage"""
        if direction == 'LONG':
            return ((exit_price - entry_price) / entry_price) * 100
        else:  # SHORT
            return ((entry_price - exit_price) / entry_price) * 100
            
    def _analyze_volume_at_extremes(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze volume at price extremes"""
        high_idx = data['high'].idxmax()
        low_idx = data['low'].idxmin()
        
        return {
            'at_high': data.loc[high_idx, 'volume'],
            'at_low': data.loc[low_idx, 'volume']
        }
        
    def _any_adapter_needs_trades(self) -> bool:
        """Check if any adapter requires trade data"""
        return any(adapter.requires_trades for adapter in self.adapters.values())
        
    def _any_adapter_needs_quotes(self) -> bool:
        """Check if any adapter requires quote data"""
        return any(
            hasattr(adapter, 'requires_quotes') and adapter.requires_quotes 
            for adapter in self.adapters.values()
        )
        
    async def run_bulk_backtest(self, configs: List[BacktestConfig], 
                               max_concurrent: int = 5) -> List[BacktestResult]:
        """
        Run multiple backtests concurrently.
        
        Args:
            configs: List of backtest configurations
            max_concurrent: Maximum concurrent backtests
            
        Returns:
            List of results
        """
        logger.info(f"Starting bulk backtest with {len(configs)} configurations")
        
        results = []
        
        # Process in batches
        for i in range(0, len(configs), max_concurrent):
            batch = configs[i:i + max_concurrent]
            batch_tasks = [self.run_backtest(config) for config in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle results and exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Backtest failed: {result}")
                else:
                    results.append(result)
                    
        logger.info(f"Completed {len(results)} backtests successfully")
        return results
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            'total_backtests': self.backtest_count,
            'total_runtime_seconds': self.total_runtime,
            'average_runtime_seconds': (
                self.total_runtime / self.backtest_count 
                if self.backtest_count > 0 else 0
            ),
            'registered_adapters': list(self.adapters.keys()),
            'data_cache_stats': self.data_manager.get_cache_stats(),
            'result_store_stats': self.result_store.get_statistics()
        }


# Test the engine
if __name__ == "__main__":
    async def test_engine():
        """Test the backtest engine"""
        # Create engine
        engine = BacktestEngine()
        
        # Create test config
        config = BacktestConfig(
            symbol='AAPL',
            entry_time=datetime.now(timezone.utc) - timedelta(hours=1),
            direction='LONG',
            historical_lookback_hours=2,
            forward_bars=60
        )
        
        print(f"Engine initialized: {engine.get_statistics()}")
        
    asyncio.run(test_engine())