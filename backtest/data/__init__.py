# backtest/data/__init__.py
"""
Backtest Data Module - Intelligent data management for backtesting
"""
from .polygon_data_manager import PolygonDataManager
from .request_aggregator import RequestAggregator, DataNeed, DataType
from .data_coordinator import DataCoordinator
from .protected_data_manager import ProtectedDataManager
from .circuit_breaker import CircuitBreaker, CircuitBreakerError, RateLimitError, NoDataAvailableError
from .data_validator import DataValidator, ValidationReport
from .trade_quote_aligner import TradeQuoteAligner, TradeSide
from .unified_data_system import UnifiedDataSystem, create_data_system

__all__ = [
    'PolygonDataManager',
    'RequestAggregator',
    'DataNeed',
    'DataType',
    'DataCoordinator',
    'ProtectedDataManager',
    'CircuitBreaker',
    'CircuitBreakerError',
    'RateLimitError',
    'NoDataAvailableError',
    'DataValidator',
    'ValidationReport',
    'TradeQuoteAligner',
    'TradeSide',
    'UnifiedDataSystem',
    'create_data_system'
]