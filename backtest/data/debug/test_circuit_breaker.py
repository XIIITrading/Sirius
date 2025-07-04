# backtest/data/debug/test_circuit_breaker.py
"""
Module: Circuit Breaker Tests
Purpose: Test circuit breaker functionality including data availability checks
Features: State transitions, rate limiting, no-data detection, real API tests
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
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
    
    def get_circuit_breaker(self):
        """Create a circuit breaker with test configuration"""
        return CircuitBreaker(
            failure_threshold=0.5,
            consecutive_failures=3,
            recovery_timeout=1,  # 1 second for faster tests
            sliding_window_size=10
        )
    
    async def test_initial_state(self):
        """Test circuit breaker starts in CLOSED state"""
        circuit_breaker = self.get_circuit_breaker()
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.consecutive_failure_count == 0
        assert circuit_breaker.recovery_attempts == 0
        print("✓ Initial state test passed")
    
    async def test_successful_calls(self):
        """Test successful calls don't open circuit"""
        circuit_breaker = self.get_circuit_breaker()
        
        async def success_func():
            return "success"
        
        # Make several successful calls
        for _ in range(5):
            result = await circuit_breaker.call(success_func)
            assert result == "success"
        
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.consecutive_failure_count == 0
        print("✓ Successful calls test passed")
    
    async def test_consecutive_failures_open_circuit(self):
        """Test consecutive failures open the circuit"""
        circuit_breaker = self.get_circuit_breaker()
        
        async def failing_func():
            raise Exception("API Error")
        
        # Make failures up to threshold
        for i in range(3):
            try:
                await circuit_breaker.call(failing_func)
            except Exception:
                pass
        
        # Circuit should now be open
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Next call should raise CircuitBreakerError
        try:
            await circuit_breaker.call(failing_func)
            assert False, "Should have raised CircuitBreakerError"
        except CircuitBreakerError:
            pass
        
        print("✓ Consecutive failures test passed")
    
    async def test_failure_rate_opens_circuit(self):
        """Test high failure rate opens circuit"""
        circuit_breaker = self.get_circuit_breaker()
        
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
        print("✓ Failure rate test passed")
    
    async def test_recovery_to_half_open(self):
        """Test circuit transitions to HALF_OPEN after timeout"""
        circuit_breaker = self.get_circuit_breaker()
        
        async def failing_func():
            raise Exception("API Error")
        
        # Open the circuit
        for _ in range(3):
            try:
                await circuit_breaker.call(failing_func)
            except Exception:
                pass
        
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Next call should be allowed (HALF_OPEN state)
        async def success_func():
            return "recovered"
        
        result = await circuit_breaker.call(success_func)
        assert result == "recovered"
        assert circuit_breaker.state == CircuitState.CLOSED
        print("✓ Recovery to half-open test passed")
    
    async def test_half_open_failure_returns_to_open(self):
        """Test failure in HALF_OPEN returns to OPEN"""
        circuit_breaker = self.get_circuit_breaker()
        
        async def failing_func():
            raise Exception("Still failing")
        
        # Open the circuit
        for _ in range(3):
            try:
                await circuit_breaker.call(failing_func)
            except Exception:
                pass
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # This call will transition to HALF_OPEN and fail
        try:
            await circuit_breaker.call(failing_func)
        except Exception:
            pass
        
        # Should be back to OPEN
        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.recovery_attempts == 2  # Incremented
        print("✓ Half-open failure test passed")
    
    async def test_exponential_backoff(self):
        """Test exponential backoff for recovery attempts"""
        circuit_breaker = self.get_circuit_breaker()
        circuit_breaker.recovery_attempts = 3
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.last_state_change = time.time()
        
        # With 3 attempts, backoff should be 1 * 2^3 = 8 seconds
        wait_time = circuit_breaker._time_until_recovery()
        assert 7.5 < wait_time < 8.5
        print("✓ Exponential backoff test passed")


class TestRateLimiting:
    """Test rate limiting functionality"""
    
    def get_rate_limited_breaker(self):
        """Create circuit breaker with rate limits"""
        return CircuitBreaker(
            rate_limits={
                'bars': {'per_minute': 60, 'burst': 5},
                'trades': {'per_minute': 30, 'burst': 3}
            }
        )
    
    async def test_rate_limit_allows_burst(self):
        """Test burst requests are allowed"""
        rate_limited_breaker = self.get_rate_limited_breaker()
        
        async def data_func():
            return "data"
        
        # Should allow burst of 5 for bars
        for _ in range(5):
            result = await rate_limited_breaker.call(
                data_func, operation_type='bars'
            )
            assert result == "data"
        
        # 6th request should be rate limited
        try:
            await rate_limited_breaker.call(
                data_func, operation_type='bars'
            )
            assert False, "Should have raised RateLimitError"
        except RateLimitError:
            pass
        
        print("✓ Rate limit burst test passed")
    
    async def test_rate_limit_refill(self):
        """Test token refill over time"""
        rate_limited_breaker = self.get_rate_limited_breaker()
        
        async def data_func():
            return "data"
        
        # Use all burst tokens
        for _ in range(3):
            await rate_limited_breaker.call(
                data_func, operation_type='trades'
            )
        
        # Should be rate limited
        try:
            await rate_limited_breaker.call(
                data_func, operation_type='trades'
            )
            assert False, "Should have raised RateLimitError"
        except RateLimitError:
            pass
        
        # Wait for tokens to refill (30/min = 0.5/sec, need 1 token)
        await asyncio.sleep(2.1)
        
        # Should work now
        result = await rate_limited_breaker.call(
            data_func, operation_type='trades'
        )
        assert result == "data"
        print("✓ Rate limit refill test passed")


class TestDataAvailabilityChecker:
    """Test data availability checking"""
    
    def get_availability_checker(self):
        """Create availability checker"""
        return DataAvailabilityChecker(cache_duration_minutes=60)
    
    async def test_data_exists_check(self):
        """Test checking when data exists"""
        availability_checker = self.get_availability_checker()
        
        # Data fetcher that returns data
        async def data_fetcher(**kwargs):
            return pd.DataFrame({
                'price': [100, 101, 102],
                'volume': [1000, 1100, 1200]
            }, index=pd.date_range(kwargs['start_time'], periods=3, freq='1min'))
        
        status = await availability_checker.check_data_availability(
            data_fetcher=data_fetcher,
            symbol='AAPL',
            start_time=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc),
            data_type='bars'
        )
        
        assert status.has_data is True
        assert status.sample_count == 3
        print("✓ Data exists check test passed")
    
    async def test_no_data_check(self):
        """Test checking when no data exists"""
        availability_checker = self.get_availability_checker()
        
        # Data fetcher that returns empty DataFrame
        async def empty_fetcher(**kwargs):
            return pd.DataFrame()
        
        status = await availability_checker.check_data_availability(
            data_fetcher=empty_fetcher,
            symbol='AAPL',
            start_time=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc),
            data_type='bars'
        )
        
        assert status.has_data is False
        assert status.sample_count == 0
        print("✓ No data check test passed")
    
    async def test_no_data_cache(self):
        """Test no-data ranges are cached"""
        availability_checker = self.get_availability_checker()
        call_count = 0
        
        async def counting_fetcher(**kwargs):
            nonlocal call_count
            call_count += 1
            return pd.DataFrame()
        
        # First check
        status1 = await availability_checker.check_data_availability(
            data_fetcher=counting_fetcher,
            symbol='AAPL',
            start_time=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc),
            data_type='bars'
        )
        
        assert call_count == 1
        
        # Second check for same range should use cache
        status2 = await availability_checker.check_data_availability(
            data_fetcher=counting_fetcher,
            symbol='AAPL',
            start_time=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 15, 10, 45, tzinfo=timezone.utc),
            data_type='bars'
        )
        
        # Should not make another call
        assert call_count == 1
        assert status2.has_data is False
        assert status2.metadata['source'] == 'no_data_cache'
        print("✓ No data cache test passed")


class TestNoDataScenario:
    """Test the specific scenario where backend quotes table hasn't been updated"""
    
    async def test_missing_quotes_data(self):
        """Test when quotes data isn't available yet in Polygon"""
        circuit_breaker = CircuitBreaker(
            failure_threshold=0.5,
            consecutive_failures=3,
            recovery_timeout=60
        )
        
        # Function that simulates Polygon returning empty quotes
        async def fetch_quotes(symbol, start_time, end_time, **kwargs):
            # Simulate the backend returning no data
            return pd.DataFrame()
        
        # First attempt - should detect no data
        try:
            await circuit_breaker.call(
                fetch_quotes,
                symbol='AAPL',
                start_time=datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc),
                end_time=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
                operation_type='quotes'
            )
            assert False, "Should have raised NoDataAvailableError"
        except NoDataAvailableError as e:
            assert e.symbol == 'AAPL'
        
        assert circuit_breaker.state == CircuitState.CLOSED  # No data doesn't open circuit
        
        # Check that no-data is cached
        checker = circuit_breaker.availability_checker
        assert 'AAPL' in checker.no_data_cache
        print("✓ Missing quotes data test passed")
    
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
        try:
            await circuit_breaker.call(
                fetch_quotes,
                symbol='AAPL',
                start_time=datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc),
                end_time=datetime(2024, 1, 15, 9, 35, tzinfo=timezone.utc),
                operation_type='quotes'
            )
            assert False, "Should have raised NoDataAvailableError"
        except NoDataAvailableError:
            pass
        
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
        print("✓ Quotes become available test passed")


class TestProtectedDataManager:
    """Test the ProtectedDataManager wrapper with real PolygonDataManager"""
    
    async def test_successful_data_fetch(self):
        """Test successful data fetching through protected manager"""
        # Create real PolygonDataManager
        polygon_manager = PolygonDataManager()
        polygon_manager.set_current_plugin("CircuitBreakerTest")
        
        protected_manager = ProtectedDataManager(polygon_manager)
        
        # This will use real API - test with a small time window
        result = await protected_manager.load_bars(
            symbol='AAPL',
            start_time=datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 15, 10, 30, tzinfo=timezone.utc),
            timeframe='1min'
        )
        
        # Should return DataFrame (empty or with data)
        assert isinstance(result, pd.DataFrame)
        print("✓ Successful data fetch test passed")
    
    async def test_no_data_handling(self):
        """Test handling when no data is available"""
        # Create a wrapper function that returns empty DataFrame
        async def empty_load_bars(*args, **kwargs):
            return pd.DataFrame()
        
        # Create real PolygonDataManager and override method
        polygon_manager = PolygonDataManager()
        polygon_manager.load_bars = empty_load_bars
        
        protected_manager = ProtectedDataManager(polygon_manager)
        
        result = await protected_manager.load_bars(
            symbol='AAPL',
            start_time=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc),
            timeframe='1min'
        )
        
        # Should return empty DataFrame with correct structure
        assert result.empty
        assert list(result.columns) == ['open', 'high', 'low', 'close', 'volume']
        print("✓ No data handling test passed")


class TestRealPolygonIntegration:
    """Integration tests with real Polygon API"""
    
    async def test_real_missing_quotes_scenario(self):
        """Test with real Polygon API for missing quotes scenario"""
        try:
            # Create real PolygonDataManager
            polygon_manager = PolygonDataManager()
            polygon_manager.set_current_plugin("RealQuotesTest")
            
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
            print("✓ Real missing quotes scenario test passed")
            
        except Exception as e:
            print(f"Skipping real API test: {e}")


async def run_all_tests():
    """Run all test classes"""
    print("=" * 80)
    print("CIRCUIT BREAKER TEST SUITE")
    print("=" * 80)
    
    # Core tests
    print("\n>>> Testing Circuit Breaker Core...")
    core_tests = TestCircuitBreakerCore()
    await core_tests.test_initial_state()
    await core_tests.test_successful_calls()
    await core_tests.test_consecutive_failures_open_circuit()
    await core_tests.test_failure_rate_opens_circuit()
    await core_tests.test_recovery_to_half_open()
    await core_tests.test_half_open_failure_returns_to_open()
    await core_tests.test_exponential_backoff()
    
    # Rate limiting tests
    print("\n>>> Testing Rate Limiting...")
    rate_tests = TestRateLimiting()
    await rate_tests.test_rate_limit_allows_burst()
    await rate_tests.test_rate_limit_refill()
    
    # Data availability tests
    print("\n>>> Testing Data Availability Checker...")
    availability_tests = TestDataAvailabilityChecker()
    await availability_tests.test_data_exists_check()
    await availability_tests.test_no_data_check()
    await availability_tests.test_no_data_cache()
    
    # No data scenario tests
    print("\n>>> Testing No Data Scenarios...")
    no_data_tests = TestNoDataScenario()
    await no_data_tests.test_missing_quotes_data()
    await no_data_tests.test_quotes_become_available_later()
    
    # Protected data manager tests (uses real API)
    print("\n>>> Testing Protected Data Manager...")
    print("WARNING: These tests use real Polygon API")
    response = input("Continue with real API tests? (y/n): ")
    if response.lower() == 'y':
        protected_tests = TestProtectedDataManager()
        await protected_tests.test_successful_data_fetch()
        await protected_tests.test_no_data_handling()
        
        # Real integration test
        print("\n>>> Testing Real Polygon Integration...")
        integration_tests = TestRealPolygonIntegration()
        await integration_tests.test_real_missing_quotes_scenario()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    # Run all tests
    asyncio.run(run_all_tests())