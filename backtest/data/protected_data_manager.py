# backtest/data/protected_data_manager.py
"""
Module: Protected Data Manager
Purpose: Wrapper for PolygonDataManager with CircuitBreaker protection
Features: Automatic retry, fallback strategies, seamless integration
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd

from .circuit_breaker import CircuitBreaker, CircuitBreakerError, RateLimitError, NoDataAvailableError
from .polygon_data_manager import PolygonDataManager

logger = logging.getLogger(__name__)


class ProtectedDataManager:
    """
    Wrapper for PolygonDataManager that adds circuit breaker protection.
    
    This can be used as a drop-in replacement for PolygonDataManager
    with added reliability features.
    """
    
    def __init__(self, 
                 polygon_data_manager: PolygonDataManager,
                 circuit_breaker_config: Optional[Dict[str, Any]] = None):
        """
        Initialize protected data manager.
        
        Args:
            polygon_data_manager: Existing PolygonDataManager instance
            circuit_breaker_config: Optional circuit breaker configuration
        """
        self.data_manager = polygon_data_manager
        
        # Default configuration
        default_config = {
            'failure_threshold': 0.5,
            'consecutive_failures': 5,
            'recovery_timeout': 60,
            'sliding_window_size': 100,
            'rate_limits': {
                'bars': {'per_minute': 100, 'burst': 10},
                'trades': {'per_minute': 50, 'burst': 5},
                'quotes': {'per_minute': 50, 'burst': 5}
            }
        }
        
        config = {**default_config, **(circuit_breaker_config or {})}
        self.circuit_breaker = CircuitBreaker(**config)
        
        logger.info("ProtectedDataManager initialized with circuit breaker protection")
    
    async def load_bars(self, 
                       symbol: str,
                       start_time: datetime,
                       end_time: datetime,
                       timeframe: str = '1min',
                       use_cache: bool = True) -> pd.DataFrame:
        """
        Load bars with circuit breaker protection.
        
        Handles:
        - Rate limiting
        - Failure detection
        - Data availability checking
        - Automatic fallback to cache on circuit open
        """
        try:
            return await self.circuit_breaker.call(
                self.data_manager.load_bars,
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                timeframe=timeframe,
                operation_type='bars'
            )
        
        except NoDataAvailableError as e:
            # Log and return empty DataFrame with proper structure
            logger.warning(f"No data available: {e}")
            return self._create_empty_bars_df()
        
        except CircuitBreakerError as e:
            # Circuit is open, try to use cached data
            logger.warning(f"Circuit breaker open: {e}")
            return await self._get_cached_bars(symbol, start_time, end_time, timeframe)
        
        except RateLimitError as e:
            # Rate limit hit, could queue for later or return cache
            logger.warning(f"Rate limit hit: {e}")
            return await self._get_cached_bars(symbol, start_time, end_time, timeframe)
    
    async def load_trades(self,
                         symbol: str,
                         start_time: datetime,
                         end_time: datetime,
                         use_cache: bool = True) -> pd.DataFrame:
        """Load trades with circuit breaker protection"""
        try:
            return await self.circuit_breaker.call(
                self.data_manager.load_trades,
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                operation_type='trades'
            )
        
        except NoDataAvailableError as e:
            logger.warning(f"No trade data available: {e}")
            return self._create_empty_trades_df()
        
        except (CircuitBreakerError, RateLimitError) as e:
            logger.warning(f"Circuit breaker/rate limit: {e}")
            return await self._get_cached_trades(symbol, start_time, end_time)
    
    async def load_quotes(self,
                         symbol: str,
                         start_time: datetime,
                         end_time: datetime,
                         use_cache: bool = True) -> pd.DataFrame:
        """Load quotes with circuit breaker protection"""
        try:
            return await self.circuit_breaker.call(
                self.data_manager.load_quotes,
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                operation_type='quotes'
            )
        
        except NoDataAvailableError as e:
            logger.warning(f"No quote data available: {e}")
            return self._create_empty_quotes_df()
        
        except (CircuitBreakerError, RateLimitError) as e:
            logger.warning(f"Circuit breaker/rate limit: {e}")
            return await self._get_cached_quotes(symbol, start_time, end_time)
    
    async def _get_cached_bars(self, symbol: str, start_time: datetime,
                              end_time: datetime, timeframe: str) -> pd.DataFrame:
        """Attempt to get bars from cache when circuit is open"""
        try:
            # With the new modular structure, access cache manager directly
            cached_data = self.data_manager.cache_manager.get(
                symbol, 'bars', timeframe, start_time, end_time
            )
            
            if cached_data is not None:
                logger.info(f"Returning cached bars for {symbol}")
                # Filter to requested range
                return self.data_manager._filter_timerange(cached_data, start_time, end_time)
                
        except Exception as e:
            logger.error(f"Error accessing cache: {e}")
        
        # No cache available
        logger.warning(f"No cached data available for {symbol}")
        return self._create_empty_bars_df()
    
    async def _get_cached_trades(self, symbol: str, start_time: datetime,
                               end_time: datetime) -> pd.DataFrame:
        """Attempt to get trades from cache"""
        try:
            # With the new modular structure, access cache manager directly
            cached_data = self.data_manager.cache_manager.get(
                symbol, 'trades', 'tick', start_time, end_time
            )
            
            if cached_data is not None:
                logger.info(f"Returning cached trades for {symbol}")
                return cached_data
                
        except Exception as e:
            logger.error(f"Error accessing cache: {e}")
        
        return self._create_empty_trades_df()
    
    async def _get_cached_quotes(self, symbol: str, start_time: datetime,
                               end_time: datetime) -> pd.DataFrame:
        """Attempt to get quotes from cache"""
        try:
            # With the new modular structure, access cache manager directly
            cached_data = self.data_manager.cache_manager.get(
                symbol, 'quotes', 'tick', start_time, end_time
            )
            
            if cached_data is not None:
                logger.info(f"Returning cached quotes for {symbol}")
                return cached_data
                
        except Exception as e:
            logger.error(f"Error accessing cache: {e}")
        
        return self._create_empty_quotes_df()
    
    def _create_empty_bars_df(self) -> pd.DataFrame:
        """Create empty bars DataFrame with proper structure"""
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    
    def _create_empty_trades_df(self) -> pd.DataFrame:
        """Create empty trades DataFrame with proper structure"""
        return pd.DataFrame(columns=['price', 'size', 'exchange', 'conditions'])
    
    def _create_empty_quotes_df(self) -> pd.DataFrame:
        """Create empty quotes DataFrame with proper structure"""
        return pd.DataFrame(columns=['bid', 'ask', 'bid_size', 'ask_size'])
    
    def get_circuit_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return self.circuit_breaker.get_status()
    
    def reset_circuit(self):
        """Reset circuit breaker"""
        self.circuit_breaker.reset()
    
    # Delegate other methods to the underlying data manager
    def __getattr__(self, name):
        """Delegate undefined methods to PolygonDataManager"""
        return getattr(self.data_manager, name)


# Example usage showing integration with new structure
async def create_protected_manager(api_key: str, **config) -> ProtectedDataManager:
    """
    Factory function to create a ProtectedDataManager with custom configuration
    """
    # Create base PolygonDataManager
    polygon_manager = PolygonDataManager(
        api_key=api_key,
        cache_dir=config.get('cache_dir', './cache'),
        memory_cache_size=config.get('memory_cache_size', 100),
        file_cache_hours=config.get('file_cache_hours', 24),
        extend_window_bars=config.get('extend_window_bars', 2000),
        report_dir=config.get('report_dir', './temp')
    )
    
    # Wrap with circuit breaker protection
    circuit_config = config.get('circuit_breaker', {})
    protected_manager = ProtectedDataManager(
        polygon_data_manager=polygon_manager,
        circuit_breaker_config=circuit_config
    )
    
    return protected_manager


# Advanced usage example with custom error handling
class EnhancedProtectedDataManager(ProtectedDataManager):
    """
    Enhanced version with additional features like retry logic and notifications
    """
    
    def __init__(self, polygon_data_manager: PolygonDataManager,
                 circuit_breaker_config: Optional[Dict[str, Any]] = None,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        super().__init__(polygon_data_manager, circuit_breaker_config)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
    async def load_bars_with_retry(self, symbol: str, start_time: datetime,
                                   end_time: datetime, timeframe: str = '1min') -> pd.DataFrame:
        """Load bars with automatic retry on transient failures"""
        import asyncio
        
        for attempt in range(self.max_retries):
            try:
                return await self.load_bars(symbol, start_time, end_time, timeframe)
            except (CircuitBreakerError, RateLimitError) as e:
                # Don't retry on circuit open or rate limit
                raise
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"All {self.max_retries} attempts failed")
                    raise
        
        return self._create_empty_bars_df()


if __name__ == "__main__":
    # Example usage
    import asyncio
    from datetime import timezone
    
    async def main():
        # Create protected manager with custom config
        config = {
            'cache_dir': './cache',
            'memory_cache_size': 200,
            'circuit_breaker': {
                'failure_threshold': 0.3,  # More sensitive
                'consecutive_failures': 3,
                'recovery_timeout': 30,
                'rate_limits': {
                    'bars': {'per_minute': 200, 'burst': 20},
                    'trades': {'per_minute': 100, 'burst': 10},
                    'quotes': {'per_minute': 100, 'burst': 10}
                }
            }
        }
        
        manager = await create_protected_manager('your_api_key', **config)
        
        # Test data fetching
        symbol = 'AAPL'
        end_time = datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc)
        start_time = end_time - timedelta(hours=1)
        
        # This will be protected by circuit breaker
        bars = await manager.load_bars(symbol, start_time, end_time)
        print(f"Fetched {len(bars)} bars")
        
        # Check circuit status
        status = manager.get_circuit_status()
        print(f"Circuit state: {status['state']}")
        print(f"Failure rate: {status['failure_rate']:.1%}")
        
    asyncio.run(main())