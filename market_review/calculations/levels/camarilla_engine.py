# market_review/calculations/levels/camarilla_calculator.py
# Updated version with proper UTC handling

"""
Module: Camarilla Pivots Calculator
Purpose: Calculate Camarilla pivot levels based on prior day's H/L/C
Dependencies: pandas, numpy
Performance: Optimized for live trading calculations
Note: All timestamps in UTC, trading day resets at 09:00 UTC
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, time
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)

# Trading session constants (in UTC)
TRADING_DAY_START_UTC = time(9, 0)  # 09:00 UTC = 4:00 AM ET (pre-market start)
TRADING_DAY_END_UTC = time(23, 0)    # 23:00 UTC = 6:00 PM ET (after-hours end)


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
    trading_session_start: Optional[datetime] = None
    trading_session_end: Optional[datetime] = None


class CamarillaCalculator:
    """
    Calculate Camarilla pivot levels for intraday trading.
    Uses traditional Camarilla formulas based on prior day's H/L/C.
    Trading sessions are UTC-based, resetting at 09:00 UTC.
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
        
        logger.info("CamarillaCalculator initialized with UTC trading sessions (09:00 UTC reset)")
    
    def _get_trading_session_bounds(self, dt: datetime) -> Tuple[datetime, datetime]:
        """
        Get the start and end of the trading session for a given datetime.
        Trading session runs from 09:00 UTC to 08:59:59 UTC the next day.
        
        Args:
            dt: Datetime to get session for (must be timezone-aware)
            
        Returns:
            Tuple of (session_start, session_end)
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
            
        # If time is before 09:00 UTC, we're still in the previous trading day
        if dt.time() < TRADING_DAY_START_UTC:
            session_date = dt.date() - timedelta(days=1)
        else:
            session_date = dt.date()
            
        # Session starts at 09:00 UTC
        session_start = datetime.combine(
            session_date, 
            TRADING_DAY_START_UTC,
            tzinfo=timezone.utc
        )
        
        # Session ends at 08:59:59 UTC the next day
        session_end = session_start + timedelta(days=1) - timedelta(seconds=1)
        
        return session_start, session_end
    
    def _convert_to_daily_sessions(self, intraday_data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert intraday data to daily OHLC based on UTC trading sessions.
        Groups data from 09:00 UTC to 08:59:59 UTC as one trading day.
        """
        if intraday_data.empty:
            return pd.DataFrame()
            
        # Ensure timezone awareness
        if intraday_data.index.tz is None:
            # Assume UTC if not specified
            intraday_data.index = intraday_data.index.tz_localize('UTC')
        else:
            # Convert to UTC if in another timezone
            intraday_data.index = intraday_data.index.tz_convert('UTC')
            
        # Create session labels for grouping
        session_labels = []
        session_starts = []
        
        for timestamp in intraday_data.index:
            session_start, _ = self._get_trading_session_bounds(timestamp)
            session_labels.append(session_start.date())
            session_starts.append(session_start)
            
        intraday_data['session_date'] = session_labels
        intraday_data['session_start'] = session_starts
        
        # Group by trading session
        daily_groups = intraday_data.groupby('session_date')
        
        daily_data = pd.DataFrame({
            'open': daily_groups['open'].first(),
            'high': daily_groups['high'].max(),
            'low': daily_groups['low'].min(),
            'close': daily_groups['close'].last(),
            'volume': daily_groups['volume'].sum() if 'volume' in intraday_data.columns else 0,
            'session_start': daily_groups['session_start'].first()
        })
        
        # Use session start as index
        daily_data.index = pd.to_datetime(daily_data['session_start'])
        daily_data = daily_data.drop('session_start', axis=1)
        
        return daily_data
    
    def calculate(self, 
                  data: pd.DataFrame,
                  ticker: Optional[str] = None,
                  target_date: Optional[datetime] = None) -> CamarillaResult:
        """
        Calculate Camarilla pivot levels.
        
        Args:
            data: DataFrame with OHLCV data (must include at least 2 days)
            ticker: Optional ticker symbol for reference
            target_date: Date to calculate pivots for (uses prior session's data)
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
        
        # Get prior trading session's data
        prior_session_data = self._get_prior_session_data(data, target_date)
        
        if prior_session_data is None:
            raise ValueError("Could not find prior session data for calculation")
        
        # Extract H/L/C
        high = float(prior_session_data['high'])
        low = float(prior_session_data['low'])
        close = float(prior_session_data['close'])
        
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
        
        # Get session bounds if available
        session_start = None
        session_end = None
        if hasattr(prior_session_data, 'name') and pd.api.types.is_datetime64_any_dtype(type(prior_session_data.name)):
            session_start = prior_session_data.name
            session_end = session_start + timedelta(days=1) - timedelta(seconds=1)
        
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
            prior_day_date=session_start,
            trading_session_start=session_start,
            trading_session_end=session_end
        )
    
    def _get_prior_session_data(self, 
                               data: pd.DataFrame, 
                               target_date: Optional[datetime] = None) -> Optional[pd.Series]:
        """
        Extract prior trading session's OHLC data based on UTC sessions.
        
        Args:
            data: DataFrame with daily or intraday data
            target_date: Date to get prior session for (None = use latest)
            
        Returns:
            Series with prior session's OHLC data
        """
        # Ensure index is datetime
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            else:
                raise ValueError("DataFrame must have datetime index or 'timestamp' column")
        
        # Sort by date
        data = data.sort_index()
        
        # If intraday data, convert to daily sessions
        if len(data) > 0 and self._is_intraday_data(data):
            daily_data = self._convert_to_daily_sessions(data)
        else:
            daily_data = data
        
        if daily_data.empty:
            return None
        
        # Determine which session to use
        if target_date is None:
            # Use the last available session
            if len(daily_data) >= 2:
                return daily_data.iloc[-2]  # Prior session
            else:
                logger.warning("Only one session of data available, using it for calculation")
                return daily_data.iloc[-1]
        else:
            # Find the prior trading session before target_date
            if target_date.tzinfo is None:
                target_date = target_date.replace(tzinfo=timezone.utc)
            
            # Get session bounds for target date
            target_session_start, _ = self._get_trading_session_bounds(target_date)
            
            # Get all sessions before target
            prior_sessions = daily_data[daily_data.index < target_session_start]
            
            if not prior_sessions.empty:
                return prior_sessions.iloc[-1]  # Most recent prior session
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
    
    def get_current_session_info(self, current_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get information about the current trading session.
        
        Args:
            current_time: Time to check (defaults to now)
            
        Returns:
            Dict with session information
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        elif current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
            
        session_start, session_end = self._get_trading_session_bounds(current_time)
        
        return {
            'current_time_utc': current_time,
            'session_start_utc': session_start,
            'session_end_utc': session_end,
            'session_date': session_start.date(),
            'time_until_new_session': session_end - current_time + timedelta(seconds=1),
            'is_pre_market': current_time.time() < time(13, 30),  # Before 13:30 UTC (9:30 AM ET)
            'is_regular_hours': time(13, 30) <= current_time.time() < time(20, 0),  # 9:30 AM - 4:00 PM ET
            'is_after_hours': current_time.time() >= time(20, 0)  # After 20:00 UTC (4:00 PM ET)
        }