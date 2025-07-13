# live_monitor/data/hist_request/base_fetcher.py
"""
Base class for all historical data fetchers
Provides common functionality and interface
"""
import logging
import pandas as pd
from typing import Optional, Dict, List
from datetime import datetime, timezone

from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


class BaseHistoricalFetcher(QObject):
    """
    Base class for all historical data fetchers
    
    Provides:
    - Common signal interface
    - Cache management
    - Error handling
    - Retry logic
    """
    
    # Signals
    fetch_completed = pyqtSignal(dict)  # Emits: {'symbol': str, 'dataframe': pd.DataFrame, 'metadata': dict}
    fetch_failed = pyqtSignal(dict)     # Emits: {'symbol': str, 'error': str, 'fetcher': str}
    fetch_started = pyqtSignal(str)     # Emits: symbol
    
    def __init__(self, rest_client, bars_needed: int, timespan: str, name: str):
        """
        Initialize base fetcher
        
        Args:
            rest_client: PolygonRESTClient instance
            bars_needed: Number of bars to fetch
            timespan: Timespan string ('1min', '5min', '15min')
            name: Fetcher name for logging
        """
        super().__init__()
        self.rest_client = rest_client
        self.bars_needed = bars_needed
        self.timespan = timespan
        self.name = name
        
        # Cache management
        self.cache: Optional[pd.DataFrame] = None
        self.cache_symbol: Optional[str] = None
        self.last_fetch_time: Optional[datetime] = None
        
        # Retry settings
        self.max_retries = 3
        self.retry_count = 0
        
        logger.info(f"Initialized {self.name} fetcher: {bars_needed} {timespan} bars")
    
    def fetch_for_symbol(self, symbol: str, force_refresh: bool = False) -> None:
        """
        Fetch historical data for a symbol
        
        Args:
            symbol: Stock symbol
            force_refresh: Force fetch even if cache exists
        """
        # Check cache validity
        if not force_refresh and self._is_cache_valid(symbol):
            logger.info(f"{self.name}: Using cached data for {symbol}")
            self._emit_cached_data()
            return
        
        # Clear cache if symbol changed
        if symbol != self.cache_symbol:
            self.clear_cache()
        
        # Start fetch
        self.retry_count = 0
        self.fetch_started.emit(symbol)
        self._fetch_data(symbol)
    
    def _fetch_data(self, symbol: str) -> None:
        """Fetch data from REST API"""
        try:
            logger.info(f"{self.name}: Fetching {self.bars_needed} {self.timespan} bars for {symbol}")
            
            # Use REST client to fetch bars
            bars = self.rest_client.fetch_bars(
                symbol=symbol,
                timespan=self.timespan,
                limit=self.bars_needed
            )
            
            if bars:
                # Convert to DataFrame
                df = self._bars_to_dataframe(bars)
                
                # Validate data
                if self._validate_data(df):
                    # Update cache
                    self.cache = df
                    self.cache_symbol = symbol
                    self.last_fetch_time = datetime.now(timezone.utc)
                    
                    # Process and emit
                    self._process_and_emit(symbol, df)
                else:
                    self._handle_fetch_error(symbol, "Data validation failed")
            else:
                self._handle_fetch_error(symbol, "No data returned from API")
                
        except Exception as e:
            self._handle_fetch_error(symbol, str(e))
    
    def _bars_to_dataframe(self, bars: List[Dict]) -> pd.DataFrame:
        """Convert bars list to DataFrame with UTC timestamps as index"""
        df = pd.DataFrame(bars)
        
        # Ensure timestamp is datetime with UTC
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df.set_index('timestamp', inplace=True)
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"{self.name}: Missing required column: {col}")
                
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Validate fetched data"""
        if df.empty:
            logger.warning(f"{self.name}: Empty DataFrame")
            return False
        
        # Check minimum bars
        if len(df) < self.get_minimum_bars():
            logger.warning(f"{self.name}: Insufficient bars: {len(df)} < {self.get_minimum_bars()}")
            return False
        
        # Check for required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            logger.error(f"{self.name}: Missing columns: {missing}")
            return False
        
        # Additional validation in subclasses
        return self._validate_specific_data(df)
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cache is valid for the symbol"""
        if self.cache is None or self.cache_symbol != symbol:
            return False
        
        # Check cache age (5 minutes for intraday data)
        if self.last_fetch_time:
            age = (datetime.now(timezone.utc) - self.last_fetch_time).total_seconds()
            if age > 300:  # 5 minutes
                return False
        
        return True
    
    def _emit_cached_data(self) -> None:
        """Emit cached data"""
        if self.cache is not None and self.cache_symbol:
            self._process_and_emit(self.cache_symbol, self.cache)
    
    def _handle_fetch_error(self, symbol: str, error_msg: str) -> None:
        """Handle fetch errors with retry logic"""
        self.retry_count += 1
        
        if self.retry_count < self.max_retries:
            logger.warning(f"{self.name}: Fetch failed for {symbol}, retry {self.retry_count}/{self.max_retries}: {error_msg}")
            # Retry
            self._fetch_data(symbol)
        else:
            logger.error(f"{self.name}: Fetch failed for {symbol} after {self.max_retries} retries: {error_msg}")
            self.fetch_failed.emit({
                'symbol': symbol,
                'error': error_msg,
                'fetcher': self.name
            })
    
    def _process_and_emit(self, symbol: str, df: pd.DataFrame) -> None:
        """Process data and emit completion signal"""
        # Get any additional metadata
        metadata = self._get_metadata(df)
        
        # Emit completion
        self.fetch_completed.emit({
            'symbol': symbol,
            'dataframe': df,
            'metadata': metadata,
            'fetcher': self.name,
            'timespan': self.timespan,
            'bars': len(df)
        })
        
        logger.info(f"{self.name}: Successfully fetched {len(df)} bars for {symbol}")
    
    def clear_cache(self) -> None:
        """Clear cached data"""
        self.cache = None
        self.cache_symbol = None
        self.last_fetch_time = None
        logger.debug(f"{self.name}: Cache cleared")
    
    # Methods that must be implemented by subclasses
    def get_minimum_bars(self) -> int:
        """Get minimum required bars for this fetcher"""
        raise NotImplementedError("Subclasses must implement get_minimum_bars()")
    
    def _validate_specific_data(self, df: pd.DataFrame) -> bool:
        """Additional validation specific to fetcher type"""
        raise NotImplementedError("Subclasses must implement _validate_specific_data()")
    
    def _get_metadata(self, df: pd.DataFrame) -> Dict:
        """Get additional metadata from the data"""
        raise NotImplementedError("Subclasses must implement _get_metadata()")