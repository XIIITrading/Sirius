"""
Unified Data System - Single entry point for all data operations
"""
import os
from typing import Optional, Dict, Any
from datetime import datetime

from typing import Tuple
import pandas as pd

from .polygon_data_manager import PolygonDataManager
from .protected_data_manager import ProtectedDataManager
from .data_coordinator import DataCoordinator
from .circuit_breaker import CircuitBreaker
from .data_validator import DataValidator
from .trade_quote_aligner import TradeQuoteAligner


class UnifiedDataSystem:
    """
    Unified interface for all data operations in the backtesting system.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 use_circuit_breaker: bool = True,
                 cache_dir: str = './cache',
                 memory_cache_size: int = 100,
                 file_cache_hours: int = 24,
                 circuit_breaker_config: Optional[Dict[str, Any]] = None):
        """
        Initialize unified data system.
        
        Args:
            api_key: Polygon API key (uses env var if not provided)
            use_circuit_breaker: Whether to use circuit breaker protection
            cache_dir: Directory for file cache
            memory_cache_size: Max items in memory cache
            file_cache_hours: Hours to keep file cache valid
            circuit_breaker_config: Optional circuit breaker configuration
        """
        # Get API key
        self.api_key = api_key or os.environ.get('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("Polygon API key not provided")
        
        # Create base data manager
        self.polygon_manager = PolygonDataManager(
            api_key=self.api_key,
            cache_dir=cache_dir,
            memory_cache_size=memory_cache_size,
            file_cache_hours=file_cache_hours,
            extend_window_bars=2000
        )
        
        # Wrap with circuit breaker if requested
        if use_circuit_breaker:
            self.data_manager = ProtectedDataManager(
                polygon_data_manager=self.polygon_manager,
                circuit_breaker_config=circuit_breaker_config
            )
        else:
            self.data_manager = self.polygon_manager
        
        # Create coordinator
        self.coordinator = DataCoordinator(self.data_manager)
        
        # Create validator and aligner
        self.validator = DataValidator()
        self.aligner = TradeQuoteAligner()
        
    def register_module(self, module_name: str, config: Dict[str, Any] = None):
        """Register a calculation module"""
        self.coordinator.register_module(module_name, config or {})
    
    async def fetch_module_data(self, symbol: str, entry_time: datetime, 
                               direction: str) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Fetch all data for registered modules"""
        return await self.coordinator.fetch_all_module_data(symbol, entry_time, direction)
    
    async def validate_and_align_data(self, symbol: str, start_time: datetime, 
                                     end_time: datetime) -> Dict[str, Any]:
        """Fetch, validate, and align all data types"""
        # Fetch data
        bars = await self.data_manager.load_bars(symbol, start_time, end_time)
        trades = await self.data_manager.load_trades(symbol, start_time, end_time)
        quotes = await self.data_manager.load_quotes(symbol, start_time, end_time)
        
        # Validate
        validation_reports = self.validator.validate_data_quality(
            bars_df=bars if not bars.empty else None,
            trades_df=trades if not trades.empty else None,
            quotes_df=quotes if not quotes.empty else None,
            symbol=symbol
        )
        
        # Align trades with quotes
        aligned_trades = None
        alignment_report = None
        order_flow_metrics = None
        
        if not trades.empty and not quotes.empty:
            aligned_trades, alignment_report = self.aligner.align_trades_quotes(trades, quotes)
            order_flow_metrics = self.aligner.calculate_order_flow_metrics(aligned_trades)
        
        return {
            'bars': bars,
            'trades': trades,
            'quotes': quotes,
            'aligned_trades': aligned_trades,
            'validation_reports': validation_reports,
            'alignment_report': alignment_report,
            'order_flow_metrics': order_flow_metrics
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        status = {
            'cache_stats': self.data_manager.get_cache_stats(),
            'aggregator_stats': self.coordinator.aggregator.get_stats(),
            'registered_modules': list(self.coordinator.registered_modules.keys())
        }
        
        # Add circuit breaker status if available
        if hasattr(self.data_manager, 'get_circuit_status'):
            status['circuit_breaker'] = self.data_manager.get_circuit_status()
        
        return status
    
    def generate_reports(self) -> Tuple[str, str]:
        """Generate comprehensive reports"""
        return self.coordinator.generate_data_report()


# Convenience function for quick setup
async def create_data_system(**kwargs) -> UnifiedDataSystem:
    """Create and return a configured UnifiedDataSystem"""
    return UnifiedDataSystem(**kwargs)