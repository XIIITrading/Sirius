# backtest/data/debug/test_circuit_breaker.py
"""
Module: Circuit Breaker Tests
Purpose: Test circuit breaker functionality including data availability checks
Features: State transitions, rate limiting, no-data detection, real API tests
"""

import asyncio
import pytest
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, AsyncMock, patch
import pandas as pd
import time

from backtest.data.circuit_breaker import (
    CircuitBreaker, CircuitState, CircuitBreakerError, 
    RateLimitError, NoDataAvailableError, FailureType,
    DataAvailabilityChecker
)
from backtest.data.protected_data_manager import ProtectedDataManager
from backtest.data.polygon_data_manager import PolygonDataManager

logger = logging.getLogger(__name__)


class TestCircuitBreakerCore:
    """Test core circuit breaker functionality"""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create a circuit breaker with test configuration"""
        return CircuitBreaker(
            failure_threshold=0.5,
            consecutive_failures=3,
            recovery_timeout=1,  # 1 second for faster tests
            sliding_window_size=10
        )
    
    @pytest.mark.asyncio
    async def test_initial_state(self, circuit_breaker):
        """Test circuit breaker starts in CLOSED state"""
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.consecutive_failure_count == 0
        assert circuit_breaker.recovery_attempts == 0
    
    @pytest.mark.asyncio
    async def test_successful_calls(self, circuit_breaker):
        """Test successful calls don't open circuit"""
        async def success_func():
            return "success"
        
        # Make several successful calls
        for _ in range(5):
            result = await circuit_breaker.call(success_func)
            assert result == "success"
        
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.consecutive_failure_count == 0
    
    @pytest.mark.asyncio
    async def test_consecutive_failures_open_circuit(self, circuit_breaker):
        """Test consecutive failures open the circuit"""
        async def failing_func():
            raise Exception("API Error")
        
        # Make failures up to threshold
        for i in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_func)
        
        # Circuit should now be open
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Next call should raise CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            await circuit_breaker.call(failing_func)
    
    @pytest.mark.asyncio
    async def test_failure_rate_opens_circuit(self, circuit_breaker):
        """Test high failure rate opens circuit"""
        async def sometimes_failing_func(should_fail):
            if should_fail:
                raise Exception("API Error")
            return "success"
        
        # Mix of success and failures
        pattern = [True, False, True, True, False, True, True]  # 5/7 failures = 71%
        
        for should_fail in pattern:
            try:
                await circuit_breaker.call(sometimes_failing_func, should_fail)
            except Exception:
                pass
        
        # Circuit should be open due to high failure rate
        assert circuit_breaker.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_recovery_to_half_open(self, circuit_breaker):
        """Test circuit transitions to HALF_OPEN after timeout"""
        async def failing_func():
            raise Exception("API Error")
        
        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_func)
        
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Next call should be allowed (HALF_OPEN state)
        async def success_func():
            return "recovered"
        
        result = await circuit_breaker.call(success_func)
        assert result == "recovered"
        assert circuit_breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_half_open_failure_returns_to_open(self, circuit_breaker):
        """Test failure in HALF_OPEN returns to OPEN"""
        async def failing_func():
            raise Exception("Still failing")
        
        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_func)
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # This call will transition to HALF_OPEN and fail
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_func)
        
        # Should be back to OPEN
        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.recovery_attempts == 2  # Incremented
    
    @pytest.mark.asyncio
    async def test_exponential_backoff(self, circuit_breaker):
        """Test exponential backoff for recovery attempts"""
        circuit_breaker.recovery_attempts = 3
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.last_state_change = time.time()
        
        # With 3 attempts, backoff should be 1 * 2^3 = 8 seconds
        wait_time = circuit_breaker._time_until_recovery()
        assert 7.5 < wait_time < 8.5


class TestRateLimiting:
    """Test rate limiting functionality"""
    
    @pytest.fixture
    def rate_limited_breaker(self):
        """Create circuit breaker with rate limits"""
        return CircuitBreaker(
            rate_limits={
                'bars': {'per_minute': 60, 'burst': 5},
                'trades': {'per_minute': 30, 'burst': 3}
            }
        )
    
    @pytest.mark.asyncio
    async def test_rate_limit_allows_burst(self, rate_limited_breaker):
        """Test burst requests are allowed"""
        async def data_func():
            return "data"
        
        # Should allow burst of 5 for bars
        for _ in range(5):
            result = await rate_limited_breaker.call(
                data_func, operation_type='bars'
            )
            assert result == "data"
        
        # 6th request should be rate limited
        with pytest.raises(RateLimitError):
            await rate_limited_breaker.call(
                data_func, operation_type='bars'
            )
    
    @pytest.mark.asyncio
    async def test_rate_limit_refill(self, rate_limited_breaker):
        """Test token refill over time"""
        async def data_func():
            return "data"
        
        # Use all burst tokens
        for _ in range(3):
            await rate_limited_breaker.call(
                data_func, operation_type='trades'
            )
        
        # Should be rate limited
        with pytest.raises(RateLimitError):
            await rate_limited_breaker.call(
                data_func, operation_type='trades'
            )
        
        # Wait for tokens to refill (30/min = 0.5/sec, need 1 token)
        await asyncio.sleep(2.1)
        
        # Should work now
        result = await rate_limited_breaker.call(
            data_func, operation_type='trades'
        )
        assert result == "data"


class TestDataAvailabilityChecker:
    """Test data availability checking"""
    
    @pytest.fixture
    def availability_checker(self):
        """Create availability checker"""
        return DataAvailabilityChecker(cache_duration_minutes=60)
    
    @pytest.mark.asyncio
    async def test_data_exists_check(self, availability_checker):
        """Test checking when data exists"""
        # Mock data fetcher that returns data
        async def mock_fetcher(**kwargs):
            return pd.DataFrame({
                'price': [100, 101, 102],
                'volume': [1000, 1100, 1200]
            }, index=pd.date_range(kwargs['start_time'], periods=3, freq='1min'))
        
        status = await availability_checker.check_data_availability(
            data_fetcher=mock_fetcher,
            symbol='AAPL',
            start_time=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc),
            data_type='bars'
        )
        
        assert status.has_data is True
        assert status.sample_count == 3
    
    @pytest.mark.asyncio
    async def test_no_data_check(self, availability_checker):
        """Test checking when no data exists"""
        # Mock data fetcher that returns empty DataFrame
        async def mock_fetcher(**kwargs):
            return pd.DataFrame()
        
        status = await availability_checker.check_data_availability(
            data_fetcher=mock_fetcher,
            symbol='AAPL',
            start_time=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc),
            data_type='bars'
        )
        
        assert status.has_data is False
        assert status.sample_count == 0
    
    @pytest.mark.asyncio
    async def test_no_data_cache(self, availability_checker):
        """Test no-data ranges are cached"""
        call_count = 0
        
        async def mock_fetcher(**kwargs):
            nonlocal call_count
            call_count += 1
            return pd.DataFrame()
        
        # First check
        status1 = await availability_checker.check_data_availability(
            data_fetcher=mock_fetcher,
            symbol='AAPL',
            start_time=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc),
            data_type='bars'
        )
        
        assert call_count == 1
        
        # Second check for same range should use cache
        status2 = await availability_checker.check_data_availability(
            data_fetcher=mock_fetcher,
            symbol='AAPL',
            start_time=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 15, 10, 45, tzinfo=timezone.utc),
            data_type='bars'
        )
        
        # Should not make another call
        assert call_count == 1
        assert status2.has_data is False
        assert status2.metadata['source'] == 'no_data_cache'


class TestNoDataScenario:
    """Test the specific scenario where backend quotes table hasn't been updated"""
    
    @pytest.mark.asyncio
    async def test_missing_quotes_data(self):
        """Test when quotes data isn't available yet in Polygon"""
        # This simulates the real scenario you mentioned
        circuit_breaker = CircuitBreaker(
            failure_threshold=0.5,
            consecutive_failures=3,
            recovery_timeout=60
        )
        
        # Mock function that simulates Polygon returning empty quotes
        async def fetch_quotes(symbol, start_time, end_time, **kwargs):
            # Simulate the backend returning no data
            return pd.DataFrame()
        
        # First attempt - should detect no data
        with pytest.raises(NoDataAvailableError) as exc_info:
            await circuit_breaker.call(
                fetch_quotes,
                symbol='AAPL',
                start_time=datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc),
                end_time=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
                operation_type='quotes'
            )
        
        assert exc_info.value.symbol == 'AAPL'
        assert circuit_breaker.state == CircuitState.CLOSED  # No data doesn't open circuit
        
        # Check that no-data is cached
        checker = circuit_breaker.availability_checker
        assert 'AAPL' in checker.no_data_cache
    
    @pytest.mark.asyncio
    async def test_quotes_become_available_later(self):
        """Test when quotes data becomes available after initial check"""
        circuit_breaker = CircuitBreaker()
        call_count = 0
        
        async def fetch_quotes(symbol, start_time, end_time, **kwargs):
            nonlocal call_count
            call_count += 1
            
            # First call returns no data, subsequent calls return data
            if call_count == 1:
                return pd.DataFrame()
            else:
                return pd.DataFrame({
                    'bid': [100.00, 100.01],
                    'ask': [100.02, 100.03],
                    'bid_size': [100, 200],
                    'ask_size': [100, 200]
                }, index=pd.date_range(start_time, periods=2, freq='1s'))
        
        # First call - no data
        with pytest.raises(NoDataAvailableError):
            await circuit_breaker.call(
                fetch_quotes,
                symbol='AAPL',
                start_time=datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc),
                end_time=datetime(2024, 1, 15, 9, 35, tzinfo=timezone.utc),
                operation_type='quotes'
            )
        
        # Clear the no-data cache (simulating time passing or manual reset)
        circuit_breaker.availability_checker.no_data_cache.clear()
        
        # Second call - data now available
        result = await circuit_breaker.call(
            fetch_quotes,
            symbol='AAPL',
            start_time=datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 15, 9, 35, tzinfo=timezone.utc),
            operation_type='quotes'
        )
        
        assert len(result) == 2
        assert 'bid' in result.columns


class TestProtectedDataManager:
    """Test the ProtectedDataManager wrapper"""
    
    @pytest.fixture
    def mock_polygon_manager(self):
        """Create mock PolygonDataManager"""
        manager = Mock(spec=PolygonDataManager)
        manager.memory_cache = Mock()
        manager.file_cache = Mock()
        manager._generate_cache_key = Mock(return_value="test_cache_key")
        return manager
    
    @pytest.mark.asyncio
    async def test_successful_data_fetch(self, mock_polygon_manager):
        """Test successful data fetching through protected manager"""
        # Setup mock to return data
        expected_df = pd.DataFrame({'close': [100, 101, 102]})
        mock_polygon_manager.load_bars = AsyncMock(return_value=expected_df)
        
        protected_manager = ProtectedDataManager(mock_polygon_manager)
        
        result = await protected_manager.load_bars(
            symbol='AAPL',
            start_time=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc),
            timeframe='1min'
        )
        
        assert result.equals(expected_df)
        mock_polygon_manager.load_bars.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_no_data_handling(self, mock_polygon_manager):
        """Test handling when no data is available"""
        # Setup mock to return empty DataFrame
        mock_polygon_manager.load_bars = AsyncMock(return_value=pd.DataFrame())
        
        protected_manager = ProtectedDataManager(mock_polygon_manager)
        
        result = await protected_manager.load_bars(
            symbol='AAPL',
            start_time=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc),
            timeframe='1min'
        )
        
        # Should return empty DataFrame with correct structure
        assert result.empty
        assert list(result.columns) == ['open', 'high', 'low', 'close', 'volume']
    
    @pytest.mark.asyncio
    async def test_circuit_open_fallback_to_cache(self, mock_polygon_manager):
        """Test fallback to cache when circuit is open"""
        # Setup mock to fail multiple times
        mock_polygon_manager.load_bars = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        # Setup cache to return data
        cached_df = pd.DataFrame({'close': [100, 101, 102]})
        mock_polygon_manager.memory_cache.get.return_value = cached_df
        
        protected_manager = ProtectedDataManager(mock_polygon_manager)
        
        # Make requests fail to open circuit
        for _ in range(5):
            try:
                await protected_manager.load_bars(
                    symbol='AAPL',
                    start_time=datetime.now(timezone.utc),
                    end_time=datetime.now(timezone.utc),
                    timeframe='1min'
                )
            except:
                pass
        
        # Circuit should be open, next call should return cached data
        result = await protected_manager.load_bars(
            symbol='AAPL',
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            timeframe='1min'
        )
        
        assert result.equals(cached_df)
        mock_polygon_manager.memory_cache.get.assert_called()


class TestRealPolygonIntegration:
    """Integration tests with real Polygon API - use sparingly"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_missing_quotes_scenario(self):
        """Test with real Polygon API for missing quotes scenario"""
        # This test requires actual Polygon credentials
        try:
            from config.polygon import POLYGON_API_KEY
        except ImportError:
            pytest.skip("Polygon API key not available")
        
        # Create real PolygonDataManager
        polygon_manager = PolygonDataManager(
            api_key=POLYGON_API_KEY,
            use_cache=True,
            cache_dir='./test_cache'
        )
        
        # Wrap with protection
        protected_manager = ProtectedDataManager(polygon_manager)
        
        # Test with a time range that might have missing quotes
        # Use a very recent time that might not have data yet
        end_time = datetime.now(timezone.utc) - timedelta(minutes=5)
        start_time = end_time - timedelta(minutes=10)
        
        try:
            result = await protected_manager.load_quotes(
                symbol='AAPL',
                start_time=start_time,
                end_time=end_time
            )
            
            if result.empty:
                logger.info("No quotes data available for recent time period")
            else:
                logger.info(f"Found {len(result)} quotes")
                
        except NoDataAvailableError as e:
            logger.info(f"Correctly detected no data: {e}")
            
        # Check circuit status
        status = protected_manager.get_circuit_status()
        logger.info(f"Circuit status: {status}")


# Test utilities
def create_test_circuit_breaker(**kwargs):
    """Factory function to create test circuit breakers"""
    defaults = {
        'failure_threshold': 0.5,
        'consecutive_failures': 3,
        'recovery_timeout': 1,
        'sliding_window_size': 10
    }
    defaults.update(kwargs)
    return CircuitBreaker(**defaults)


async def simulate_api_failures(circuit_breaker, num_failures):
    """Helper to simulate API failures"""
    async def failing_func():
        raise Exception("Simulated API failure")
    
    for _ in range(num_failures):
        try:
            await circuit_breaker.call(failing_func)
        except Exception:
            pass


if __name__ == "__main__":
    # Run specific test
    pytest.main([__file__, "-v", "-k", "test_missing_quotes_data"])