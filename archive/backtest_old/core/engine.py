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

from adapters.base import CalculationAdapter, StandardSignal
from core.signal_aggregator import SignalAggregator

# Import the new storage module
try:
    from storage.supabase_storage import BacktestStorage, prepare_bars_for_storage
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Supabase storage module not available")

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
    
    # Storage settings
    store_to_supabase: bool = False  # New field
    
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
    
    def __init__(self, config_path: Optional[str] = None, 
                 data_manager: Optional[Any] = None,
                 enable_supabase_storage: bool = True,
                 plugin_registry: Optional[Any] = None):
        """
        Initialize backtest engine.
        
        Args:
            config_path: Path to JSON configuration file
            data_manager: Optional data manager (uses PolygonDataManager if None)
            enable_supabase_storage: Whether to enable Supabase storage
            plugin_registry: Registry of loaded plugins
        """
        self.config_path = config_path
        
        # Initialize data manager - only use PolygonDataManager
        if data_manager:
            self.data_manager = data_manager
            logger.info("Using provided data manager")
        else:
            # Only use PolygonDataManager - no fallback
            from data.polygon_data_manager import PolygonDataManager
            self.data_manager = PolygonDataManager()
            logger.info("Using PolygonDataManager")
        
        # Initialize local result store
        from core.result_store import BacktestResultStore
        self.result_store = BacktestResultStore()
        
        # Initialize signal aggregator
        self.signal_aggregator = SignalAggregator()
        
        # Store plugin registry reference
        self.plugin_registry = plugin_registry
        
        # Add debug mode flag
        self.debug_mode = False
        
        # Initialize Supabase storage if available and enabled
        self.supabase_storage = None
        self.supabase_enabled = False
        
        if enable_supabase_storage and SUPABASE_AVAILABLE:
            try:
                self.supabase_storage = BacktestStorage(
                    plugin_registry=plugin_registry
                )
                self.supabase_enabled = True
                logger.info("Supabase storage initialized with plugin support")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase storage: {e}")
                logger.info("Continuing with local storage only")
        
        # Adapter registry - populated by register_adapter()
        self.adapters: Dict[str, CalculationAdapter] = {}
        
        # Performance tracking
        self.backtest_count = 0
        self.total_runtime = 0
        
        if config_path:
            self._load_config(config_path)
    
    def enable_debug_mode(self):
        """Enable debug mode for detailed logging"""
        self.debug_mode = True
        logger.setLevel(logging.DEBUG)
        # Also enable debug for all calculation adapters
        for adapter in self.adapters.values():
            if hasattr(adapter, 'debug_mode'):
                adapter.debug_mode = True
            
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
    
    async def _collect_data_requirements(self, config: BacktestConfig) -> Dict[str, Any]:
        """
        Collect data requirements from all active adapters.
        Simplified since plugins handle their own aggregation.
        
        Returns:
            Aggregated requirements for data fetching
        """
        requirements = {
            'max_lookback_minutes': 0,
            'needs_trades': False,
            'needs_quotes': False
        }
        
        # Filter adapters based on enabled calculations
        active_adapters = {
            name: adapter for name, adapter in self.adapters.items()
            if not config.enabled_calculations or name in config.enabled_calculations
        }
        
        # Collect requirements from each adapter
        for name, adapter in active_adapters.items():
            if hasattr(adapter, 'get_data_requirements'):
                adapter_reqs = adapter.get_data_requirements()
                
                # Bar requirements - we only care about lookback now
                if adapter_reqs.get('bars'):
                    bar_reqs = adapter_reqs['bars']
                    requirements['max_lookback_minutes'] = max(
                        requirements['max_lookback_minutes'],
                        bar_reqs.get('lookback_minutes', 0)
                    )
                
                # Trade/quote requirements
                if adapter_reqs.get('trades'):
                    requirements['needs_trades'] = True
                if adapter_reqs.get('quotes'):
                    requirements['needs_quotes'] = True
        
        return requirements
    
    async def _fetch_all_required_data(self, symbol: str, requirements: Dict[str, Any],
                                      start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Fetch all required data based on aggregated requirements.
        Simplified - only fetches 1-minute bars, plugins handle aggregation.
        
        Returns:
            Shared data cache for all adapters
        """
        data_cache = {
            'symbol': symbol,
            'data_start': start_time,
            'data_end': end_time,
            'entry_time': end_time  # data_end is entry_time for historical
        }
        
        # Always fetch 1-minute bars as base - plugins will aggregate as needed
        data_cache['1min_bars'] = await self.data_manager.load_bars(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            timeframe='1min',
            use_cache=True
        )
        
        # Fetch trades if needed
        if requirements['needs_trades']:
            data_cache['trades'] = await self.data_manager.load_trades(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                use_cache=True
            )
        else:
            data_cache['trades'] = []
        
        # Fetch quotes if needed
        if requirements['needs_quotes']:
            data_cache['quotes'] = await self.data_manager.load_quotes(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                use_cache=True
            )
        else:
            data_cache['quotes'] = []
        
        return data_cache
            
    async def run_backtest(self, config: BacktestConfig) -> 'BacktestResult':
        """
        Run a single backtest iteration.
        
        Args:
            config: Backtest configuration
            
        Returns:
            BacktestResult with analysis and performance metrics
        """
        start_time = datetime.now(timezone.utc)
        
        # Debug mode logging
        if self.debug_mode:
            logger.info(f"\n{'='*80}")
            logger.info(f"BACKTEST ENGINE DEBUG MODE")
            logger.info(f"Symbol: {config.symbol}")
            logger.info(f"Entry time: {config.entry_time}")
            logger.info(f"Direction: {config.direction}")
            logger.info(f"Lookback hours: {config.historical_lookback_hours}")
            logger.info(f"Forward bars: {config.forward_bars}")
            logger.info(f"Enabled calculations: {config.enabled_calculations or 'ALL'}")
            logger.info(f"{'='*80}\n")
        
        logger.info(f"Starting backtest for {config.symbol} at {config.entry_time}")
        
        # Log storage configuration
        if config.store_to_supabase and self.supabase_enabled:
            logger.info("Supabase storage is ENABLED for this backtest")
        else:
            logger.info("Supabase storage is DISABLED for this backtest")
        
        try:
            # 1. Collect data requirements from all adapters
            requirements = await self._collect_data_requirements(config)
            
            # 2. Calculate data window based on requirements
            data_start = config.entry_time - timedelta(
                minutes=max(requirements['max_lookback_minutes'], 
                           config.historical_lookback_hours * 60)
            )
            data_end = config.entry_time
            
            # 3. Fetch all required data once
            shared_data_cache = await self._fetch_all_required_data(
                config.symbol, requirements, data_start, data_end
            )
            
            # 4. Load forward data (60 bars after entry)
            forward_data = await self._load_forward_data(config)
            
            # 5. Initialize adapters with shared data cache
            await self._initialize_adapters(config.symbol, shared_data_cache, config)
            
            # 6. Get entry signals from all calculations
            entry_signals = await self._get_entry_signals(config)
            
            # Debug logging for entry signals
            if self.debug_mode:
                logger.info(f"\nEntry signals collected ({len(entry_signals)} total):")
                for sig in entry_signals:
                    logger.info(f"  {sig.name}:")
                    logger.info(f"    Direction: {sig.direction}")
                    logger.info(f"    Strength: {sig.strength}%")
                    logger.info(f"    Confidence: {sig.confidence}%")
                    if 'structure_type' in sig.metadata:
                        logger.info(f"    Structure: {sig.metadata['structure_type']}")
                        logger.info(f"    Trend: {sig.metadata.get('current_trend')}")
                        logger.info(f"    Reason: {sig.metadata.get('reason')}")
            
            # 7. Aggregate signals using point & call system
            aggregated_signal = self.signal_aggregator.aggregate_signals(entry_signals)
            
            # Debug logging for aggregated signal
            if self.debug_mode:
                logger.info(f"\nAggregated Signal:")
                # Safely access keys that may or may not exist
                if isinstance(aggregated_signal, dict):
                    logger.info(f"  Final direction: {aggregated_signal.get('direction', 'N/A')}")
                    logger.info(f"  Total points: {aggregated_signal.get('total_points', 0)}")
                    logger.info(f"  Confidence: {aggregated_signal.get('confidence', 0)}%")
                    if 'point_breakdown' in aggregated_signal:
                        logger.info("  Point breakdown:")
                        for calc, points in aggregated_signal['point_breakdown'].items():
                            logger.info(f"    {calc}: {points} points")
                    # Log all available keys for debugging
                    logger.info(f"  Available keys: {list(aggregated_signal.keys())}")
                else:
                    logger.info(f"  Signal type: {type(aggregated_signal)}")
                    logger.info(f"  Signal value: {aggregated_signal}")
            
            # 8. Simulate forward price movement
            forward_analysis = await self._analyze_forward_movement(
                config, forward_data, entry_signals
            )
            
            # Debug logging for forward analysis
            if self.debug_mode:
                logger.info(f"\nForward Analysis Results:")
                logger.info(f"  Entry price: ${forward_analysis['entry_price']:.2f}")
                logger.info(f"  Exit price: ${forward_analysis['exit_price']:.2f}")
                logger.info(f"  Max favorable move: {forward_analysis['max_favorable_move']:.2f}%")
                logger.info(f"  Max adverse move: {forward_analysis['max_adverse_move']:.2f}%")
                logger.info(f"  Final P&L: {forward_analysis['final_pnl']:.2f}%")
                logger.info(f"  Signal accuracy: {forward_analysis['signal_accuracy']}")
            
            # 9. Prepare bars for storage (even if not auto-storing, prepare for manual push)
            bars_for_storage = None
            if self.supabase_enabled:
                try:
                    historical_bars = shared_data_cache.get('1min_bars', pd.DataFrame())
                    if not historical_bars.empty and not forward_data.empty:
                        bars_for_storage = prepare_bars_for_storage(
                            historical_bars, 
                            forward_data, 
                            config.entry_time
                        )
                        logger.info(f"Prepared {len(bars_for_storage)} bars for storage")
                except Exception as e:
                    logger.warning(f"Could not prepare bars for storage: {e}")
            
            # 10. Create result with all data including bars
            from core.result_store import BacktestResult
            result = BacktestResult(
                config=config,
                entry_signals=entry_signals,
                aggregated_signal=aggregated_signal,
                forward_data=forward_data,
                forward_analysis=forward_analysis,
                runtime_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
                historical_bars=shared_data_cache.get('1min_bars'),  # Store for manual push
                bars_for_storage=bars_for_storage  # Pre-prepared 75 bars
            )
            
            # 11. Store result locally (always)
            self.result_store.store_result(result)
            
            # 12. Store to Supabase if enabled
            if config.store_to_supabase and self.supabase_enabled:
                await self._store_to_supabase(config, shared_data_cache, forward_data, result)
            
            # Update tracking
            self.backtest_count += 1
            self.total_runtime += result.runtime_seconds
            
            logger.info(f"Backtest completed in {result.runtime_seconds:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
    
    async def _store_to_supabase(self, config: BacktestConfig, 
                                historical_data: Dict[str, Any],
                                forward_data: pd.DataFrame,
                                result: 'BacktestResult') -> None:
        """
        Store backtest results to Supabase.
        
        Args:
            config: Backtest configuration
            historical_data: Historical data used
            forward_data: Forward data analyzed
            result: Backtest result
        """
        try:
            # Generate UID
            uid = self.supabase_storage.generate_uid(config)
            logger.info(f"Storing to Supabase with UID: {uid}")
            
            # Check if UID already exists
            exists = await self.supabase_storage.check_uid_exists(uid)
            if exists:
                logger.warning(f"UID {uid} already exists in Supabase, will overwrite")
            
            # Use pre-prepared bars if available
            if result.bars_for_storage is not None:
                bars_to_store = result.bars_for_storage
            else:
                # Prepare bars if not already done
                historical_bars = historical_data.get('1min_bars', pd.DataFrame())
                bars_to_store = prepare_bars_for_storage(
                    historical_bars, 
                    forward_data, 
                    config.entry_time
                )
            
            # Store to Supabase
            storage_result = await self.supabase_storage.store_backtest_data(
                uid=uid,
                config=config,
                bars_df=bars_to_store,
                results=result
            )
            
            if storage_result.success:
                logger.info(f"Successfully stored to Supabase: {storage_result.rows_inserted}")
            else:
                logger.error(f"Failed to store to Supabase: {storage_result.error}")
                
        except Exception as e:
            logger.error(f"Error storing to Supabase: {e}")
            # Don't fail the backtest if storage fails
            # The local storage already succeeded
        
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
        
    async def _initialize_adapters(self, symbol: str, shared_data_cache: Dict[str, Any], 
                                  config: BacktestConfig) -> None:
        """Initialize all adapters with shared data cache"""
        # Filter adapters based on enabled calculations
        active_adapters = {
            name: adapter for name, adapter in self.adapters.items()
            if not config.enabled_calculations or name in config.enabled_calculations
        }
        
        logger.info(f"Initializing {len(active_adapters)} adapters with shared data cache")
        
        # Initialize each adapter
        init_tasks = []
        for name, adapter in active_adapters.items():
            # Set shared data cache
            if hasattr(adapter, 'set_data_cache'):
                adapter.set_data_cache(shared_data_cache)
            
            init_tasks.append(self._initialize_single_adapter(
                name, adapter, symbol, shared_data_cache
            ))
            
        await asyncio.gather(*init_tasks)
        
    async def _initialize_single_adapter(self, name: str, adapter: CalculationAdapter,
                                       symbol: str, historical_data: Dict[str, Any]) -> None:
        """Initialize a single adapter"""
        try:
            # Initialize calculation
            adapter.initialize(symbol)
            
            # Feed historical data
            # Pass empty DataFrame as adapter will use shared cache
            adapter.feed_historical_data(pd.DataFrame(), symbol)
                    
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
                'entry_price': 0,
                'exit_price': 0,
                'max_favorable_move': 0,
                'max_adverse_move': 0,
                'final_pnl': 0
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
                               max_concurrent: int = 5) -> List['BacktestResult']:
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
        stats = {
            'total_backtests': self.backtest_count,
            'total_runtime_seconds': self.total_runtime,
            'average_runtime_seconds': (
                self.total_runtime / self.backtest_count 
                if self.backtest_count > 0 else 0
            ),
            'registered_adapters': list(self.adapters.keys()),
            'supabase_enabled': self.supabase_enabled,
            'result_store_stats': self.result_store.get_statistics()
        }
        
        # Add data manager stats
        if hasattr(self.data_manager, 'get_cache_stats'):
            stats['data_cache_stats'] = self.data_manager.get_cache_stats()
        elif hasattr(self.data_manager, 'get_statistics'):
            stats['data_cache_stats'] = self.data_manager.get_statistics()
            
        # Add Supabase storage stats
        if self.supabase_storage:
            stats['supabase_storage_stats'] = self.supabase_storage.get_statistics()
            
        return stats