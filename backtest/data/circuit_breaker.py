# backtest/data/circuit_breaker.py
"""
Module: Circuit Breaker
Purpose: Protect against API failures, rate limits, and missing data scenarios
Features: State management, rate limiting, data availability checking, fallback strategies
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque
import time
import threading

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "CLOSED"          # Normal operation
    OPEN = "OPEN"              # Blocking requests
    HALF_OPEN = "HALF_OPEN"    # Testing recovery


class FailureType(Enum):
    """Types of failures to track differently"""
    NETWORK_ERROR = "NETWORK_ERROR"
    RATE_LIMIT = "RATE_LIMIT"
    SERVER_ERROR = "SERVER_ERROR"
    CLIENT_ERROR = "CLIENT_ERROR"
    TIMEOUT = "TIMEOUT"
    NO_DATA = "NO_DATA"  # New failure type for missing data


@dataclass
class RequestResult:
    """Track result of each request"""
    timestamp: float
    success: bool
    failure_type: Optional[FailureType] = None
    response_time: Optional[float] = None
    error: Optional[Exception] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp,
            'success': self.success,
            'failure_type': self.failure_type.value if self.failure_type else None,
            'response_time': self.response_time,
            'error': str(self.error) if self.error else None
        }


@dataclass
class DataAvailabilityStatus:
    """Status of data availability check"""
    has_data: bool
    last_checked: datetime
    check_duration: float
    sample_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'has_data': self.has_data,
            'last_checked': self.last_checked.isoformat(),
            'check_duration': self.check_duration,
            'sample_count': self.sample_count,
            'metadata': self.metadata
        }


class CircuitBreakerError(Exception):
    """Raised when circuit is open"""
    pass


class RateLimitError(Exception):
    """Raised when rate limit would be exceeded"""
    pass


class NoDataAvailableError(Exception):
    """Raised when no data is available for the requested period"""
    def __init__(self, symbol: str, start_time: datetime, end_time: datetime):
        self.symbol = symbol
        self.start_time = start_time
        self.end_time = end_time
        super().__init__(
            f"No data available for {symbol} between {start_time} and {end_time}"
        )


class DataAvailabilityChecker:
    """
    Checks if data exists before making full requests.
    Uses minimal API calls to probe for data existence.
    """
    
    def __init__(self, cache_duration_minutes: int = 60):
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.availability_cache: Dict[str, DataAvailabilityStatus] = {}
        self.no_data_cache: Dict[str, List[Tuple[datetime, datetime]]] = {}
        self._lock = threading.Lock()  # Thread safety
        
    async def check_data_availability(self, 
                                    data_fetcher: Callable,
                                    symbol: str,
                                    start_time: datetime,
                                    end_time: datetime,
                                    data_type: str = "bars") -> DataAvailabilityStatus:
        """
        Check if data exists for the given symbol and time range.
        Uses a small probe request to minimize API usage.
        """
        cache_key = f"{symbol}_{data_type}_{start_time.date()}_{end_time.date()}"
        
        # Check cache first (thread-safe)
        with self._lock:
            if cache_key in self.availability_cache:
                cached_status = self.availability_cache[cache_key]
                if datetime.now(timezone.utc) - cached_status.last_checked < self.cache_duration:
                    logger.debug(f"Using cached availability status for {cache_key}")
                    return cached_status
        
        # Check if we've marked this range as having no data
        if self._is_in_no_data_cache(symbol, start_time, end_time):
            return DataAvailabilityStatus(
                has_data=False,
                last_checked=datetime.now(timezone.utc),
                check_duration=0,
                sample_count=0,
                metadata={'source': 'no_data_cache'}
            )
        
        # Perform probe request
        logger.info(f"Checking data availability for {symbol} {data_type} "
                   f"[{start_time} to {end_time}]")
        
        start = time.time()
        
        try:
            # Probe with a minimal time window
            if data_type == "bars":
                probe_window = timedelta(hours=1)
            elif data_type in ["trades", "quotes"]:
                probe_window = timedelta(minutes=5)
            else:
                probe_window = timedelta(minutes=15)
            
            probe_end = min(start_time + probe_window, end_time)
            
            # Make probe request
            probe_data = await data_fetcher(
                symbol=symbol,
                start_time=start_time,
                end_time=probe_end
            )
            
            check_duration = time.time() - start
            
            # Handle both DataFrame and empty results
            if hasattr(probe_data, 'empty'):
                has_data = not probe_data.empty
                sample_count = len(probe_data) if not probe_data.empty else 0
            else:
                has_data = probe_data is not None and len(probe_data) > 0
                sample_count = len(probe_data) if probe_data is not None else 0
            
            status = DataAvailabilityStatus(
                has_data=has_data,
                last_checked=datetime.now(timezone.utc),
                check_duration=check_duration,
                sample_count=sample_count,
                metadata={
                    'probe_window': probe_window.total_seconds(),
                    'actual_start': probe_data.index.min().isoformat() if has_data and hasattr(probe_data, 'index') else None,
                    'actual_end': probe_data.index.max().isoformat() if has_data and hasattr(probe_data, 'index') else None
                }
            )
            
            # Cache the result (thread-safe)
            with self._lock:
                self.availability_cache[cache_key] = status
            
            # If no data found, add to no-data cache
            if not has_data:
                self._add_to_no_data_cache(symbol, start_time, end_time)
                logger.warning(f"No data available for {symbol} in range "
                             f"[{start_time} to {end_time}]")
            
            return status
            
        except Exception as e:
            logger.error(f"Error checking data availability: {e}")
            # Assume data might exist on error
            return DataAvailabilityStatus(
                has_data=True,  # Conservative assumption
                last_checked=datetime.now(timezone.utc),
                check_duration=time.time() - start,
                sample_count=0,
                metadata={'error': str(e)}
            )
    
    def _is_in_no_data_cache(self, symbol: str, start_time: datetime, 
                            end_time: datetime) -> bool:
        """Check if time range is marked as having no data"""
        with self._lock:
            if symbol not in self.no_data_cache:
                return False
            
            for cached_start, cached_end in self.no_data_cache[symbol]:
                if cached_start <= start_time and cached_end >= end_time:
                    return True
        return False
    
    def _add_to_no_data_cache(self, symbol: str, start_time: datetime, 
                             end_time: datetime):
        """Add time range to no-data cache"""
        with self._lock:
            if symbol not in self.no_data_cache:
                self.no_data_cache[symbol] = []
            
            # Merge overlapping ranges
            self.no_data_cache[symbol].append((start_time, end_time))
            self._merge_no_data_ranges(symbol)
    
    def _merge_no_data_ranges(self, symbol: str):
        """Merge overlapping no-data ranges for efficiency"""
        if symbol not in self.no_data_cache or len(self.no_data_cache[symbol]) <= 1:
            return
        
        # Sort by start time
        ranges = sorted(self.no_data_cache[symbol])
        merged = [ranges[0]]
        
        for current_start, current_end in ranges[1:]:
            last_start, last_end = merged[-1]
            
            # Check for overlap or adjacent ranges
            if current_start <= last_end + timedelta(hours=1):
                # Merge ranges
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))
        
        self.no_data_cache[symbol] = merged
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear availability cache"""
        with self._lock:
            if symbol:
                # Clear specific symbol
                keys_to_remove = [k for k in self.availability_cache if k.startswith(f"{symbol}_")]
                for key in keys_to_remove:
                    del self.availability_cache[key]
                if symbol in self.no_data_cache:
                    del self.no_data_cache[symbol]
            else:
                # Clear all
                self.availability_cache.clear()
                self.no_data_cache.clear()


class RateLimiter:
    """Token bucket rate limiter with burst support"""
    
    def __init__(self, rate_per_minute: int, burst_size: int = 10):
        self.rate_per_minute = rate_per_minute
        self.burst_size = burst_size
        self.tokens = float(burst_size)
        self.last_refill = time.time()
        self.refill_rate = rate_per_minute / 60.0  # tokens per second
        self._lock = threading.Lock()
        
    def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens, return True if successful"""
        with self._lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on elapsed time
        self.tokens = min(
            self.burst_size,
            self.tokens + (elapsed * self.refill_rate)
        )
        self.last_refill = now
    
    def time_until_available(self, tokens: int = 1) -> float:
        """Return seconds until tokens will be available"""
        with self._lock:
            self._refill()
            
            if self.tokens >= tokens:
                return 0
            
            needed = tokens - self.tokens
            return needed / self.refill_rate
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rate limiter status"""
        with self._lock:
            self._refill()
            return {
                'tokens_available': self.tokens,
                'rate_per_minute': self.rate_per_minute,
                'burst_size': self.burst_size,
                'refill_rate': self.refill_rate
            }


class CircuitBreaker:
    """
    Circuit breaker with rate limiting and data availability checking.
    
    Features:
    - Three states: CLOSED, OPEN, HALF_OPEN
    - Rate limiting per data type
    - Data availability checking
    - Exponential backoff recovery
    - Detailed metrics and monitoring
    """
    
    def __init__(self,
                 failure_threshold: float = 0.5,
                 consecutive_failures: int = 5,
                 recovery_timeout: int = 60,
                 sliding_window_size: int = 100,
                 rate_limits: Optional[Dict[str, Dict[str, int]]] = None):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Failure rate to open circuit (0.5 = 50%)
            consecutive_failures: Alternative trigger for opening
            recovery_timeout: Initial recovery timeout in seconds
            sliding_window_size: Number of requests to track
            rate_limits: Dict of rate limits per operation type
        """
        self.failure_threshold = failure_threshold
        self.consecutive_failures = consecutive_failures
        self.recovery_timeout = recovery_timeout
        self.sliding_window_size = sliding_window_size
        
        # State management
        self.state = CircuitState.CLOSED
        self.last_state_change = time.time()
        self.consecutive_failure_count = 0
        self.recovery_attempts = 0
        self._state_lock = threading.Lock()
        
        # Request tracking
        self.request_history = deque(maxlen=sliding_window_size)
        self._history_lock = threading.Lock()
        
        # Rate limiters
        self.rate_limiters = {}
        if rate_limits:
            for operation, limits in rate_limits.items():
                self.rate_limiters[operation] = RateLimiter(
                    rate_per_minute=limits.get('per_minute', 100),
                    burst_size=limits.get('burst', 10)
                )
        
        # Data availability checker
        self.availability_checker = DataAvailabilityChecker()
        
        # Metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'circuit_opens': 0,
            'rate_limit_hits': 0,
            'no_data_responses': 0,
            'last_failure_time': None,
            'last_failure_type': None
        }
        self._metrics_lock = threading.Lock()
        
        logger.info(f"CircuitBreaker initialized: threshold={failure_threshold}, "
                   f"consecutive={consecutive_failures}, timeout={recovery_timeout}s")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Async function to call
            *args, **kwargs: Arguments for the function
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
            RateLimitError: If rate limit exceeded
            NoDataAvailableError: If no data available
        """
        # Extract operation_type and remove it from kwargs
        operation_type = kwargs.pop('operation_type', 'default')
        
        # Check circuit state
        if not self._should_allow_request():
            with self._metrics_lock:
                self.metrics['circuit_opens'] += 1
            raise CircuitBreakerError(
                f"Circuit breaker is {self.state.value}. "
                f"Recovery attempt in {self._time_until_recovery():.1f}s"
            )
        
        # Check rate limit
        if operation_type in self.rate_limiters:
            if not self.rate_limiters[operation_type].try_acquire():
                with self._metrics_lock:
                    self.metrics['rate_limit_hits'] += 1
                wait_time = self.rate_limiters[operation_type].time_until_available()
                raise RateLimitError(
                    f"Rate limit exceeded for {operation_type}. "
                    f"Retry in {wait_time:.1f}s"
                )
        
        # For data fetching operations, check availability first
        if self._should_check_availability(func, kwargs):
            availability = await self._check_data_availability(func, kwargs)
            if not availability.has_data:
                with self._metrics_lock:
                    self.metrics['no_data_responses'] += 1
                self._record_no_data_result(kwargs)
                raise NoDataAvailableError(
                    symbol=kwargs.get('symbol'),
                    start_time=kwargs.get('start_time'),
                    end_time=kwargs.get('end_time')
                )
        
        # Execute the function
        start_time = time.time()
        with self._metrics_lock:
            self.metrics['total_requests'] += 1
        
        try:
            result = await func(*args, **kwargs)
            response_time = time.time() - start_time
            
            # Record success
            self._record_success(response_time)
            with self._metrics_lock:
                self.metrics['successful_requests'] += 1
            
            return result
            
        except NoDataAvailableError:
            # Re-raise without recording as failure
            raise
            
        except Exception as e:
            response_time = time.time() - start_time
            
            # Classify the error
            failure_type = self._classify_error(e)
            
            # Record failure
            self._record_failure(failure_type, e)
            with self._metrics_lock:
                self.metrics['failed_requests'] += 1
                self.metrics['last_failure_time'] = datetime.now(timezone.utc)
                self.metrics['last_failure_type'] = failure_type.value
            
            # Re-raise the error
            raise
    
    def _should_allow_request(self) -> bool:
        """Check if request should be allowed based on circuit state"""
        with self._state_lock:
            if self.state == CircuitState.CLOSED:
                return True
            
            elif self.state == CircuitState.OPEN:
                # Check if we should transition to HALF_OPEN
                if self._should_attempt_recovery():
                    self._transition_to_half_open()
                    return True
                return False
            
            elif self.state == CircuitState.HALF_OPEN:
                # Allow one request to test recovery
                return True
        
        return False
    
    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery"""
        # Exponential backoff: 60s, 120s, 240s, etc.
        backoff_time = self.recovery_timeout * (2 ** min(self.recovery_attempts, 5))
        return time.time() - self.last_state_change >= backoff_time
    
    def _time_until_recovery(self) -> float:
        """Calculate seconds until next recovery attempt"""
        with self._state_lock:
            backoff_time = self.recovery_timeout * (2 ** min(self.recovery_attempts, 5))
            elapsed = time.time() - self.last_state_change
            return max(0, backoff_time - elapsed)
    
    def _should_check_availability(self, func: Callable, kwargs: Dict) -> bool:
        """Determine if we should check data availability"""
        # Check if this is a data fetching function
        func_name = func.__name__.lower()
        is_data_fetch = any(x in func_name for x in ['load', 'fetch', 'get'])
        
        # Check if we have required parameters
        has_params = all(k in kwargs for k in ['symbol', 'start_time', 'end_time'])
        
        return is_data_fetch and has_params
    
    async def _check_data_availability(self, func: Callable, 
                                     kwargs: Dict) -> DataAvailabilityStatus:
        """Check if data is available before making full request"""
        # Create a minimal probe function
        async def probe_func(**probe_kwargs):
            return await func(**probe_kwargs)
        
        # Determine data type from function name
        func_name = func.__name__.lower()
        if 'trade' in func_name:
            data_type = 'trades'
        elif 'quote' in func_name:
            data_type = 'quotes'
        else:
            data_type = 'bars'
        
        return await self.availability_checker.check_data_availability(
            data_fetcher=probe_func,
            symbol=kwargs['symbol'],
            start_time=kwargs['start_time'],
            end_time=kwargs['end_time'],
            data_type=data_type
        )
    
    def _record_success(self, response_time: float):
        """Record successful request"""
        with self._history_lock:
            self.request_history.append(RequestResult(
                timestamp=time.time(),
                success=True,
                response_time=response_time
            ))
        
        with self._state_lock:
            # Reset consecutive failures
            self.consecutive_failure_count = 0
            
            # Handle state transitions
            if self.state == CircuitState.HALF_OPEN:
                # Success in HALF_OPEN means we can close the circuit
                self._transition_to_closed()
    
    def _record_failure(self, failure_type: FailureType, error: Exception):
        """Record failed request and check if circuit should open"""
        with self._history_lock:
            self.request_history.append(RequestResult(
                timestamp=time.time(),
                success=False,
                failure_type=failure_type,
                error=error
            ))
        
        with self._state_lock:
            # Increment consecutive failures
            self.consecutive_failure_count += 1
            
            # Check if we should open the circuit
            if self.state == CircuitState.CLOSED:
                if (self.consecutive_failure_count >= self.consecutive_failures or
                    self._calculate_failure_rate() >= self.failure_threshold):
                    self._transition_to_open()
            
            elif self.state == CircuitState.HALF_OPEN:
                # Failure in HALF_OPEN means back to OPEN
                self._transition_to_open()
    
    def _record_no_data_result(self, kwargs: Dict):
        """Record when no data is available"""
        # Don't count as failure for circuit breaker purposes
        # But track for metrics
        logger.info(f"No data available for {kwargs.get('symbol')} "
                   f"[{kwargs.get('start_time')} to {kwargs.get('end_time')}]")
    
    def _calculate_failure_rate(self) -> float:
        """Calculate current failure rate"""
        with self._history_lock:
            if not self.request_history:
                return 0.0
            
            # Only consider recent requests based on time window
            recent_cutoff = time.time() - 300  # Last 5 minutes
            recent_requests = [r for r in self.request_history 
                              if r.timestamp > recent_cutoff]
            
            if not recent_requests:
                return 0.0
            
            failures = sum(1 for r in recent_requests if not r.success)
            return failures / len(recent_requests)
    
    def _classify_error(self, error: Exception) -> FailureType:
        """Classify error type for different handling"""
        error_str = str(error).lower()
        
        if '429' in error_str or 'rate limit' in error_str:
            return FailureType.RATE_LIMIT
        elif 'timeout' in error_str:
            return FailureType.TIMEOUT
        elif any(code in error_str for code in ['500', '502', '503', '504']):
            return FailureType.SERVER_ERROR
        elif any(code in error_str for code in ['400', '401', '403', '404']):
            return FailureType.CLIENT_ERROR
        else:
            return FailureType.NETWORK_ERROR
    
    def _transition_to_open(self):
        """Transition to OPEN state"""
        logger.warning(f"Circuit breaker opening. Failure rate: "
                      f"{self._calculate_failure_rate():.1%}, "
                      f"Consecutive failures: {self.consecutive_failure_count}")
        
        self.state = CircuitState.OPEN
        self.last_state_change = time.time()
        self.recovery_attempts += 1
        with self._metrics_lock:
            self.metrics['circuit_opens'] += 1
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""
        logger.info("Circuit breaker transitioning to HALF_OPEN for recovery test")
        
        self.state = CircuitState.HALF_OPEN
        self.last_state_change = time.time()
    
    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        logger.info("Circuit breaker closing. Service recovered.")
        
        self.state = CircuitState.CLOSED
        self.last_state_change = time.time()
        self.recovery_attempts = 0
        self.consecutive_failure_count = 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status"""
        with self._state_lock:
            state = self.state.value
            consecutive_failures = self.consecutive_failure_count
            recovery_attempts = self.recovery_attempts
            time_until_recovery = self._time_until_recovery() if self.state == CircuitState.OPEN else 0
        
        with self._metrics_lock:
            metrics = dict(self.metrics)
        
        return {
            'state': state,
            'failure_rate': self._calculate_failure_rate(),
            'consecutive_failures': consecutive_failures,
            'recovery_attempts': recovery_attempts,
            'time_until_recovery': time_until_recovery,
            'metrics': metrics,
            'rate_limits': {
                name: limiter.get_status()
                for name, limiter in self.rate_limiters.items()
            }
        }
    
    def reset(self):
        """Reset circuit breaker to initial state"""
        logger.info("Circuit breaker reset requested")
        
        with self._state_lock:
            self.state = CircuitState.CLOSED
            self.last_state_change = time.time()
            self.consecutive_failure_count = 0
            self.recovery_attempts = 0
        
        with self._history_lock:
            self.request_history.clear()
        
        # Reset rate limiters
        for limiter in self.rate_limiters.values():
            limiter.tokens = limiter.burst_size
            limiter.last_refill = time.time()
        
        # Clear availability cache
        self.availability_checker.clear_cache()
        
        logger.info("Circuit breaker reset completed")


# Integration helper for the new modular PolygonDataManager
def create_circuit_breaker_for_data_manager(config: Optional[Dict[str, Any]] = None) -> CircuitBreaker:
    """
    Create a circuit breaker configured for use with PolygonDataManager
    
    Args:
        config: Optional configuration overrides
        
    Returns:
        Configured CircuitBreaker instance
    """
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
    
    if config:
        default_config.update(config)
    
    return CircuitBreaker(**default_config)


if __name__ == "__main__":
    # Example usage
    async def example():
        # Create circuit breaker
        breaker = create_circuit_breaker_for_data_manager()
        
        # Example protected function
        async def fetch_data(symbol: str, start_time: datetime, end_time: datetime):
            # Simulate API call
            await asyncio.sleep(0.1)
            return {"data": "example"}
        
        # Use circuit breaker
        try:
            result = await breaker.call(
                fetch_data,
                symbol="AAPL",
                start_time=datetime.now(timezone.utc) - timedelta(hours=1),
                end_time=datetime.now(timezone.utc),
                operation_type='bars'
            )
            print(f"Success: {result}")
        except CircuitBreakerError as e:
            print(f"Circuit open: {e}")
        except RateLimitError as e:
            print(f"Rate limited: {e}")
        except NoDataAvailableError as e:
            print(f"No data: {e}")
        
        # Check status
        status = breaker.get_status()
        print(f"\nCircuit Breaker Status:")
        print(f"  State: {status['state']}")
        print(f"  Failure Rate: {status['failure_rate']:.1%}")
        print(f"  Total Requests: {status['metrics']['total_requests']}")
    
    asyncio.run(example())