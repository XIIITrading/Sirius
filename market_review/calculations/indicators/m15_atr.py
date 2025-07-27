# market_review/calculations/indicators/m15_atr.py
"""
15-Minute Average True Range (ATR) Calculator Module
Purpose: Calculate 15-minute ATR using 14-period lookback
Features: 
  - Optimized for 15-minute timeframe
  - 14-period ATR calculation
  - Integration with polygon_bridge for data fetching
  - Reusable across different tools and dashboards
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple, Union
from dataclasses import dataclass
import logging

# Import your polygon bridge
from market_review.data.polygon_bridge import PolygonHVNBridge

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ATRResult:
    """
    Container for ATR calculation results
    
    Attributes:
        symbol: Stock ticker symbol
        atr_value: The calculated ATR value
        atr_percentage: ATR as percentage of current price
        true_ranges: Series of true range values used in calculation
        current_price: Most recent close price
        calculation_time: When the calculation was performed
        data_end_time: The end time of the data used
        periods_used: Actual number of periods in calculation
    """
    symbol: str
    atr_value: float
    atr_percentage: float
    true_ranges: pd.Series
    current_price: float
    calculation_time: datetime
    data_end_time: datetime
    periods_used: int
    
    def __str__(self):
        """String representation for easy debugging"""
        return (f"ATR({self.symbol}): ${self.atr_value:.2f} "
                f"({self.atr_percentage:.2f}% of ${self.current_price:.2f})")

# THESE FUNCTIONS ARE AT MODULE LEVEL, NOT INSIDE THE DATACLASS
def calculate_true_range(df: pd.DataFrame) -> pd.Series:
    """
    Calculate True Range for each bar in the dataframe.
    
    True Range is the greatest of:
    1. Current High - Current Low
    2. |Current High - Previous Close|
    3. |Current Low - Previous Close|
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        
    Returns:
        pd.Series: True Range values
    """
    # Method 1: High - Low (intraday range)
    high_low = df['high'] - df['low']
    
    # Method 2: |High - Previous Close| (gap up capture)
    # shift(1) gets previous row's value
    high_close = np.abs(df['high'] - df['close'].shift(1))
    
    # Method 3: |Low - Previous Close| (gap down capture)
    low_close = np.abs(df['low'] - df['close'].shift(1))
    
    # True Range = maximum of all three methods
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Name the series for clarity
    true_range.name = 'true_range'
    
    logger.debug(f"Calculated {len(true_range)} true range values")
    
    return true_range

def calculate_atr(true_ranges: pd.Series, periods: int = 14) -> float:
    """
    Calculate Average True Range using simple moving average.
    
    Args:
        true_ranges: Series of true range values
        periods: Number of periods for averaging (default 14)
        
    Returns:
        float: The ATR value (using the most recent 'periods' true ranges)
        
    Note:
        We use simple moving average (SMA) for clarity. Some implementations
        use exponential moving average (EMA) for smoothing.
    """
    # Check if we have enough data
    valid_true_ranges = true_ranges.dropna()
    
    if len(valid_true_ranges) < periods:
        logger.warning(f"Only {len(valid_true_ranges)} periods available, "
                      f"less than requested {periods}")
        # Use what we have
        return valid_true_ranges.mean()
    
    # Calculate ATR as the mean of the last 'periods' true ranges
    atr = valid_true_ranges.tail(periods).mean()
    
    logger.debug(f"ATR calculated using {periods} periods: {atr:.2f}")
    
    return atr

class M15ATRCalculator:
    """
    15-Minute ATR Calculator using Polygon data.
    
    This class provides a simple interface for calculating ATR
    that can be called from any tool or dashboard.
    """
    
    def __init__(self, 
                 periods: int = 14,
                 lookback_bars: int = 100,
                 cache_enabled: bool = True):
        """
        Initialize the ATR calculator.
        
        Args:
            periods: Number of periods for ATR calculation (default 14)
            lookback_bars: How many bars to fetch for calculation buffer
            cache_enabled: Whether to use Polygon's cache
        """
        self.periods = periods
        self.lookback_bars = lookback_bars
        
        # Initialize the polygon bridge with minimal settings
        # We only need basic data fetching capability
        self.bridge = PolygonHVNBridge(
            hvn_levels=10,  # Minimal, we don't need HVN
            hvn_percentile=80.0,  # Default, not used
            lookback_days=10,  # Will calculate based on bars needed
            update_interval_minutes=15,
            cache_enabled=cache_enabled
        )
        
        logger.info(f"M15 ATR Calculator initialized with {periods} periods")

    def calculate(self, 
                  symbol: str, 
                  end_datetime: Optional[datetime] = None) -> ATRResult:
        """
        Calculate 15-minute ATR for a symbol at a specific time.
        
        Args:
            symbol: Stock ticker symbol
            end_datetime: End time for calculation (default: now)
            
        Returns:
            ATRResult: Complete ATR calculation results
            
        Raises:
            ValueError: If insufficient data or calculation fails
        """
        # Use current time if not specified
        if end_datetime is None:
            end_datetime = datetime.now()
            
        calculation_start = datetime.now()
        
        logger.info(f"Calculating 15-min ATR for {symbol} at {end_datetime}")
        
        try:
            # Calculate how many days we need to fetch
            # 15-min bars: 26 bars per day (6.5 hours * 4 bars/hour)
            # We want lookback_bars, so calculate days needed
            days_needed = max(5, (self.lookback_bars // 26) + 2)
            
            # Temporarily set the bridge's lookback days
            original_lookback = self.bridge.lookback_days
            self.bridge.lookback_days = days_needed
            
            # Fetch data using the bridge
            state = self.bridge.calculate_hvn(
                symbol=symbol,
                end_date=end_datetime,
                timeframe='15min'  # Specifically request 15-minute bars
            )
            
            # Restore original lookback
            self.bridge.lookback_days = original_lookback
            
            # Extract the price data
            df = state.recent_bars
            
            if len(df) < 2:
                raise ValueError(f"Insufficient data for {symbol}: only {len(df)} bars")
            
            # Calculate true ranges
            true_ranges = calculate_true_range(df)
            
            # Calculate ATR
            atr_value = calculate_atr(true_ranges, self.periods)
            
            # Get current price and calculate percentage
            current_price = df['close'].iloc[-1]
            atr_percentage = (atr_value / current_price) * 100
            
            # Create and return result
            result = ATRResult(
                symbol=symbol,
                atr_value=atr_value,
                atr_percentage=atr_percentage,
                true_ranges=true_ranges,
                current_price=current_price,
                calculation_time=calculation_start,
                data_end_time=df.index[-1].to_pydatetime(),
                periods_used=min(self.periods, len(true_ranges.dropna()))
            )
            
            logger.info(f"ATR calculation complete: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"ATR calculation failed for {symbol}: {e}")
            raise ValueError(f"Failed to calculate ATR for {symbol}: {str(e)}")
        
    def get_atr_value(self, symbol: str, end_datetime: Optional[datetime] = None) -> float:
        """
        Quick method to get just the ATR value.
        
        Args:
            symbol: Stock ticker
            end_datetime: End time for calculation
            
        Returns:
            float: The ATR value in dollars
        """
        result = self.calculate(symbol, end_datetime)
        return result.atr_value
    
    def get_atr_percentage(self, symbol: str, end_datetime: Optional[datetime] = None) -> float:
        """
        Quick method to get ATR as percentage of price.
        
        Args:
            symbol: Stock ticker
            end_datetime: End time for calculation
            
        Returns:
            float: The ATR as percentage of current price
        """
        result = self.calculate(symbol, end_datetime)
        return result.atr_percentage
    
    def get_volatility_rating(self, symbol: str, end_datetime: Optional[datetime] = None) -> str:
        """
        Get a simple volatility rating based on ATR percentage.
        
        Args:
            symbol: Stock ticker
            end_datetime: End time for calculation
            
        Returns:
            str: 'Low', 'Normal', 'High', or 'Extreme' volatility
        """
        atr_pct = self.get_atr_percentage(symbol, end_datetime)
        
        if atr_pct < 1.0:
            return "Low"
        elif atr_pct < 2.5:
            return "Normal"
        elif atr_pct < 4.0:
            return "High"
        else:
            return "Extreme"

# Module-level instance for easy access
_default_calculator = None

def get_m15_atr(symbol: str, 
                end_datetime: Optional[datetime] = None,
                periods: int = 14) -> ATRResult:
    """
    Module-level convenience function for quick ATR calculations.
    
    This function maintains a singleton calculator instance for efficiency.
    
    Args:
        symbol: Stock ticker
        end_datetime: End time for calculation (default: now)
        periods: ATR periods (only used on first call)
        
    Returns:
        ATRResult: Complete ATR calculation results
    """
    global _default_calculator
    
    if _default_calculator is None:
        _default_calculator = M15ATRCalculator(periods=periods)
    
    return _default_calculator.calculate(symbol, end_datetime)