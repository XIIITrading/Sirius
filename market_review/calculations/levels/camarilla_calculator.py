# market_review/calculations/levels/camarilla_calculator.py
"""
Module: Camarilla Pivots Calculator
Purpose: Calculate Camarilla pivot levels based on prior day's H/L/C
Dependencies: pandas, numpy
Performance: Optimized for live trading calculations
Note: All timestamps in UTC
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class CamarillaLevel:
    """Container for a single Camarilla level"""
    name: str
    price: float
    level_type: str  # 'resistance' or 'support'
    strength: int    # 1-4 (R1/S1 weakest, R4/S4 strongest)


@dataclass
class CamarillaResult:
    """Complete Camarilla pivot analysis result"""
    calculation_time: datetime
    prior_day_high: float
    prior_day_low: float
    prior_day_close: float
    prior_day_range: float
    central_pivot: float
    resistance_levels: Dict[str, float]  # R1-R4
    support_levels: Dict[str, float]     # S1-S4
    all_levels: Dict[str, CamarillaLevel]  # All levels as objects
    ticker: Optional[str] = None
    prior_day_date: Optional[datetime] = None


class CamarillaCalculator:
    """
    Calculate Camarilla pivot levels for intraday trading.
    Uses traditional Camarilla formulas based on prior day's H/L/C.
    """
    
    def __init__(self):
        """Initialize Camarilla calculator."""
        # Traditional Camarilla multipliers
        self.multipliers = {
            'R4': 1.1/2,
            'R3': 1.1/4,
            'R2': 1.1/6,
            'R1': 1.1/12,
            'S1': 1.1/12,
            'S2': 1.1/6,
            'S3': 1.1/4,
            'S4': 1.1/2
        }
        
        logger.info("CamarillaCalculator initialized with traditional multipliers")
    
    def calculate(self, 
                  data: pd.DataFrame,
                  ticker: Optional[str] = None,
                  target_date: Optional[datetime] = None) -> CamarillaResult:
        """
        Calculate Camarilla pivot levels.
        
        Args:
            data: DataFrame with OHLCV data (must include at least 2 days)
            ticker: Optional ticker symbol for reference
            target_date: Date to calculate pivots for (uses prior day's data)
                        If None, uses most recent data
        
        Returns:
            CamarillaResult with all calculated levels
        """
        if data.empty:
            raise ValueError("Empty DataFrame provided")
        
        # Ensure we have required columns
        required_cols = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Get prior day's data
        prior_day_data = self._get_prior_day_data(data, target_date)
        
        if prior_day_data is None:
            raise ValueError("Could not find prior day data for calculation")
        
        # Extract H/L/C
        high = float(prior_day_data['high'])
        low = float(prior_day_data['low'])
        close = float(prior_day_data['close'])
        
        # Calculate range
        daily_range = high - low
        
        if daily_range <= 0:
            raise ValueError(f"Invalid price range: High={high}, Low={low}")
        
        # Calculate pivot levels
        resistance_levels = self._calculate_resistance_levels(close, daily_range)
        support_levels = self._calculate_support_levels(close, daily_range)
        
        # Central pivot (traditional)
        central_pivot = close  # Camarilla uses close as central pivot
        
        # Create level objects
        all_levels = self._create_level_objects(
            resistance_levels, 
            support_levels,
            central_pivot
        )
        
        # Get prior day date if available
        prior_day_date = None
        if hasattr(prior_day_data, 'name') and pd.api.types.is_datetime64_any_dtype(type(prior_day_data.name)):
            prior_day_date = prior_day_data.name
        
        return CamarillaResult(
            calculation_time=datetime.now(timezone.utc),
            prior_day_high=high,
            prior_day_low=low,
            prior_day_close=close,
            prior_day_range=daily_range,
            central_pivot=central_pivot,
            resistance_levels=resistance_levels,
            support_levels=support_levels,
            all_levels=all_levels,
            ticker=ticker,
            prior_day_date=prior_day_date
        )
    
    def _get_prior_day_data(self, 
                           data: pd.DataFrame, 
                           target_date: Optional[datetime] = None) -> Optional[pd.Series]:
        """
        Extract prior day's OHLC data.
        
        Args:
            data: DataFrame with daily or intraday data
            target_date: Date to get prior day for (None = use latest)
            
        Returns:
            Series with prior day's OHLC data
        """
        # Ensure index is datetime
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            else:
                raise ValueError("DataFrame must have datetime index or 'timestamp' column")
        
        # Sort by date
        data = data.sort_index()
        
        # If intraday data, convert to daily
        if len(data) > 0 and self._is_intraday_data(data):
            daily_data = self._convert_to_daily(data)
        else:
            daily_data = data
        
        if daily_data.empty:
            return None
        
        # Determine which day to use
        if target_date is None:
            # Use the last available day
            if len(daily_data) >= 2:
                return daily_data.iloc[-2]  # Prior day
            else:
                logger.warning("Only one day of data available, using it for calculation")
                return daily_data.iloc[-1]
        else:
            # Find the prior trading day before target_date
            if target_date.tzinfo is None:
                target_date = target_date.replace(tzinfo=timezone.utc)
            
            # Get all dates before target
            prior_dates = daily_data[daily_data.index < target_date]
            
            if not prior_dates.empty:
                return prior_dates.iloc[-1]  # Most recent prior day
            else:
                return None
    
    def _is_intraday_data(self, data: pd.DataFrame) -> bool:
        """Check if data is intraday based on frequency."""
        if len(data) < 2:
            return False
        
        # Check time difference between first few rows
        time_diffs = data.index[1:5] - data.index[0:4]
        avg_diff = np.mean([td.total_seconds() for td in time_diffs])
        
        # If average difference is less than 1 day, it's intraday
        return avg_diff < 86400  # seconds in a day
    
    def _convert_to_daily(self, intraday_data: pd.DataFrame) -> pd.DataFrame:
        """Convert intraday data to daily OHLC."""
        # Group by date
        daily_groups = intraday_data.groupby(intraday_data.index.date)
        
        daily_data = pd.DataFrame({
            'open': daily_groups['open'].first(),
            'high': daily_groups['high'].max(),
            'low': daily_groups['low'].min(),
            'close': daily_groups['close'].last(),
            'volume': daily_groups['volume'].sum() if 'volume' in intraday_data.columns else 0
        })
        
        # Convert index back to datetime
        daily_data.index = pd.to_datetime(daily_data.index)
        
        return daily_data
    
    def _calculate_resistance_levels(self, close: float, daily_range: float) -> Dict[str, float]:
        """Calculate resistance levels R1-R4."""
        return {
            'R1': close + (daily_range * self.multipliers['R1']),
            'R2': close + (daily_range * self.multipliers['R2']),
            'R3': close + (daily_range * self.multipliers['R3']),
            'R4': close + (daily_range * self.multipliers['R4'])
        }
    
    def _calculate_support_levels(self, close: float, daily_range: float) -> Dict[str, float]:
        """Calculate support levels S1-S4."""
        return {
            'S1': close - (daily_range * self.multipliers['S1']),
            'S2': close - (daily_range * self.multipliers['S2']),
            'S3': close - (daily_range * self.multipliers['S3']),
            'S4': close - (daily_range * self.multipliers['S4'])
        }
    
    def _create_level_objects(self,
                             resistance_levels: Dict[str, float],
                             support_levels: Dict[str, float],
                             central_pivot: float) -> Dict[str, CamarillaLevel]:
        """Create CamarillaLevel objects for all levels."""
        all_levels = {}
        
        # Add resistance levels
        for name, price in resistance_levels.items():
            strength = int(name[1])  # Extract number from R1, R2, etc.
            all_levels[name] = CamarillaLevel(
                name=name,
                price=price,
                level_type='resistance',
                strength=strength
            )
        
        # Add support levels
        for name, price in support_levels.items():
            strength = int(name[1])  # Extract number from S1, S2, etc.
            all_levels[name] = CamarillaLevel(
                name=name,
                price=price,
                level_type='support',
                strength=strength
            )
        
        # Add central pivot
        all_levels['PP'] = CamarillaLevel(
            name='PP',
            price=central_pivot,
            level_type='pivot',
            strength=0
        )
        
        return all_levels
    
    def get_nearest_levels(self, 
                          result: CamarillaResult, 
                          current_price: float,
                          count: int = 2) -> Tuple[list, list]:
        """
        Get nearest support and resistance levels to current price.
        
        Args:
            result: CamarillaResult object
            current_price: Current market price
            count: Number of levels to return above and below
            
        Returns:
            Tuple of (nearest_resistance_levels, nearest_support_levels)
        """
        resistances = []
        supports = []
        
        # Separate and sort levels
        for level in result.all_levels.values():
            if level.level_type == 'resistance' and level.price > current_price:
                resistances.append(level)
            elif level.level_type == 'support' and level.price < current_price:
                supports.append(level)
        
        # Sort by distance from current price
        resistances.sort(key=lambda x: x.price - current_price)
        supports.sort(key=lambda x: current_price - x.price)
        
        return resistances[:count], supports[:count]


# Example usage and testing
if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    
    print("=== Testing Camarilla Calculator ===\n")
    
    # Test 1: Create sample daily data
    print("Test 1: Sample daily data calculation")
    dates = pd.date_range(end=datetime.now(), periods=5, freq='D')
    sample_data = pd.DataFrame({
        'open': [150.0, 152.0, 151.5, 153.0, 154.0],
        'high': [152.5, 153.5, 154.0, 155.5, 156.0],
        'low': [149.5, 150.5, 151.0, 152.0, 153.5],
        'close': [151.5, 151.0, 153.5, 154.5, 155.0],
        'volume': [1000000, 1100000, 1200000, 1300000, 1400000]
    }, index=dates)
    
    calc = CamarillaCalculator()
    result = calc.calculate(sample_data, ticker='TEST')
    
    print(f"Prior Day: H={result.prior_day_high}, L={result.prior_day_low}, C={result.prior_day_close}")
    print(f"Daily Range: {result.prior_day_range:.2f}")
    print(f"\nCalculated Levels:")
    print(f"Central Pivot: {result.central_pivot:.2f}")
    print("\nResistance Levels:")
    for name, price in sorted(result.resistance_levels.items()):
        print(f"  {name}: {price:.2f}")
    print("\nSupport Levels:")
    for name, price in sorted(result.support_levels.items(), reverse=True):
        print(f"  {name}: {price:.2f}")
    
    # Test 2: Integration with live data
    print("\n\nTest 2: Live data integration test")
    try:
        from market_review.data.polygon_bridge import PolygonHVNBridge
        
        bridge = PolygonHVNBridge()
        print("Fetching TSLA data...")
        
        # Get recent data
        state = bridge.calculate_hvn('TSLA', timeframe='15min')
        
        # Calculate Camarilla levels
        result = calc.calculate(state.recent_bars, ticker='TSLA')
        
        print(f"\nTSLA Camarilla Levels (calculated at {result.calculation_time.strftime('%Y-%m-%d %H:%M:%S UTC')})")
        print(f"Based on: {result.prior_day_date}")
        print(f"Prior Day: H=${result.prior_day_high:.2f}, L=${result.prior_day_low:.2f}, C=${result.prior_day_close:.2f}")
        
        # Get nearest levels to current price
        current_price = state.current_price
        print(f"\nCurrent Price: ${current_price:.2f}")
        
        nearest_r, nearest_s = calc.get_nearest_levels(result, current_price, count=2)
        
        print("\nNearest Resistance Levels:")
        for level in nearest_r:
            distance = level.price - current_price
            print(f"  {level.name}: ${level.price:.2f} (+${distance:.2f})")
        
        print("\nNearest Support Levels:")
        for level in nearest_s:
            distance = current_price - level.price
            print(f"  {level.name}: ${level.price:.2f} (-${distance:.2f})")
        
    except ImportError:
        print("Could not import PolygonHVNBridge - skipping live data test")
    except Exception as e:
        print(f"Error in live data test: {e}")
    
    print("\n=== Camarilla Calculator Test Complete ===")