# market_review/calculations/zones/supply_demand.py
"""
Module: Order Blocks & Breaker Blocks Detection
Purpose: Identify order blocks and breaker blocks using swing highs/lows
Features: 
- Swing high/low detection with configurable lookback
- Order block creation on swing breaks
- Breaker block tracking when price reverses through OB
- Maintains last N blocks of each type
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import asyncio

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class Swing:
    """Swing high or low point"""
    price: float
    time: datetime
    bar_index: int
    crossed: bool = False


@dataclass
class OrderBlock:
    """Order Block definition matching PineScript structure"""
    block_type: str  # 'bullish' or 'bearish'
    top: float
    bottom: float
    time: datetime
    bar_index: int
    is_breaker: bool = False
    breaker_time: Optional[datetime] = None
    breaker_bar_index: Optional[int] = None
    
    @property
    def center(self) -> float:
        return (self.top + self.bottom) / 2
        
    @property
    def height(self) -> float:
        return self.top - self.bottom


# For backward compatibility with existing code
@dataclass 
class SupplyDemandZone:
    """Legacy zone definition - maps to OrderBlock"""
    zone_type: str  # 'supply' or 'demand'
    price_low: float
    price_high: float
    center_price: float
    volume_percent: float = 0.0  # Not used in new approach
    fractal_price: float = 0.0
    fractal_time: datetime = None
    strength: float = 50.0
    validated: bool = True
    validation_time: Optional[datetime] = None
    validation_move: Optional[float] = None
    volume_levels: List = field(default_factory=list)
    
    @classmethod
    def from_order_block(cls, ob: OrderBlock) -> 'SupplyDemandZone':
        """Convert OrderBlock to SupplyDemandZone for compatibility"""
        zone_type = 'supply' if ob.block_type == 'bearish' else 'demand'
        return cls(
            zone_type=zone_type,
            price_low=ob.bottom,
            price_high=ob.top,
            center_price=ob.center,
            fractal_price=ob.center,
            fractal_time=ob.time,
            strength=80.0 if not ob.is_breaker else 60.0,
            validated=not ob.is_breaker
        )


class OrderBlockAnalyzer:
    """
    Analyze order blocks and breaker blocks using swing highs/lows
    Matches PineScript logic exactly
    """
    
    def __init__(self,
                 swing_length: int = 7,
                 show_bullish: int = 3,
                 show_bearish: int = 3,
                 use_body: bool = False):
        """
        Initialize Order Block analyzer
        
        Args:
            swing_length: Lookback period for swing detection (default 10)
            show_bullish: Number of bullish OBs to track (default 3)
            show_bearish: Number of bearish OBs to track (default 3)
            use_body: Use candle body instead of wicks (default False)
        """
        self.swing_length = swing_length
        self.show_bullish = show_bullish
        self.show_bearish = show_bearish
        self.use_body = use_body
        
        # Storage
        self.bullish_obs: List[OrderBlock] = []
        self.bearish_obs: List[OrderBlock] = []
        self.swing_highs: List[Swing] = []
        self.swing_lows: List[Swing] = []
        
    def analyze_zones(self, data: pd.DataFrame) -> List[SupplyDemandZone]:
        """
        Main analysis function to identify order blocks
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            List of supply/demand zones (for compatibility)
        """
        logger.info(f"Analyzing order blocks for {len(data)} bars")
        
        # Reset storage
        self.bullish_obs.clear()
        self.bearish_obs.clear()
        self.swing_highs.clear()
        self.swing_lows.clear()
        
        # Ensure we have enough data
        if len(data) < self.swing_length * 2:
            logger.warning("Not enough data for swing analysis")
            return []
            
        # Detect swings
        self._detect_swings(data)
        logger.info(f"Found {len(self.swing_highs)} swing highs and {len(self.swing_lows)} swing lows")
        
        # Process each bar to find order blocks
        self._process_order_blocks(data)
        logger.info(f"Found {len(self.bullish_obs)} bullish and {len(self.bearish_obs)} bearish order blocks")
        
        # Convert to zones for compatibility
        zones = []
        
        # Add most recent bullish blocks
        for ob in self.bullish_obs[-self.show_bullish:]:
            zones.append(SupplyDemandZone.from_order_block(ob))
            
        # Add most recent bearish blocks
        for ob in self.bearish_obs[-self.show_bearish:]:
            zones.append(SupplyDemandZone.from_order_block(ob))
            
        return zones
        
    def _detect_swings(self, data: pd.DataFrame):
        """Detect swing highs and lows"""
        high_values = data['high'].values
        low_values = data['low'].values
        
        # Process each potential swing point
        for i in range(self.swing_length, len(data) - self.swing_length):
            # Check for swing high
            if self._is_swing_high(high_values, i):
                swing = Swing(
                    price=high_values[i],
                    time=data.index[i],
                    bar_index=i
                )
                self.swing_highs.append(swing)
                
            # Check for swing low
            if self._is_swing_low(low_values, i):
                swing = Swing(
                    price=low_values[i],
                    time=data.index[i],
                    bar_index=i
                )
                self.swing_lows.append(swing)
                
    def _is_swing_high(self, highs: np.ndarray, index: int) -> bool:
        """Check if index is a swing high"""
        if index < self.swing_length or index >= len(highs) - self.swing_length:
            return False
            
        high_point = highs[index]
        
        # Check left side
        for i in range(index - self.swing_length, index):
            if highs[i] >= high_point:
                return False
                
        # Check right side
        for i in range(index + 1, index + self.swing_length + 1):
            if i < len(highs) and highs[i] > high_point:
                return False
                
        return True
        
    def _is_swing_low(self, lows: np.ndarray, index: int) -> bool:
        """Check if index is a swing low"""
        if index < self.swing_length or index >= len(lows) - self.swing_length:
            return False
            
        low_point = lows[index]
        
        # Check left side
        for i in range(index - self.swing_length, index):
            if lows[i] <= low_point:
                return False
                
        # Check right side
        for i in range(index + 1, index + self.swing_length + 1):
            if i < len(lows) and lows[i] < low_point:
                return False
                
        return True
        
    def _process_order_blocks(self, data: pd.DataFrame):
        """Process bars to identify order blocks"""
        close_values = data['close'].values
        open_values = data['open'].values
        high_values = data['high'].values
        low_values = data['low'].values
        
        # Get max/min values based on use_body setting
        if self.use_body:
            max_values = np.maximum(close_values, open_values)
            min_values = np.minimum(close_values, open_values)
        else:
            max_values = high_values
            min_values = low_values
            
        # Process each bar
        for i in range(1, len(data)):
            current_close = close_values[i]
            current_time = data.index[i]
            
            # Check for bullish order blocks (close above swing high)
            for swing in self.swing_highs:
                if not swing.crossed and swing.bar_index < i:
                    if current_close > swing.price:
                        swing.crossed = True
                        
                        # Find the order block - looking for last down move before breakout
                        # Start from bar before breakout
                        ob_index = i - 1
                        ob_bottom = low_values[ob_index]
                        ob_top = high_values[ob_index]
                        
                        # Look back to find the lowest low before breakout
                        min_low = low_values[ob_index]
                        min_low_index = ob_index
                        
                        for j in range(i-2, max(swing.bar_index-1, i-10), -1):
                            if j >= 0 and low_values[j] < min_low:
                                min_low = low_values[j]
                                min_low_index = j
                                
                        # The order block is the candle with the lowest low
                        ob_bottom = low_values[min_low_index]
                        ob_top = high_values[min_low_index]
                        ob_time = data.index[min_low_index]
                        
                        ob = OrderBlock(
                            block_type='bullish',
                            top=ob_top,
                            bottom=ob_bottom,
                            time=ob_time,
                            bar_index=min_low_index
                        )
                        self.bullish_obs.append(ob)
                        
            # Check for bearish order blocks (close below swing low)
            for swing in self.swing_lows:
                if not swing.crossed and swing.bar_index < i:
                    if current_close < swing.price:
                        swing.crossed = True
                        
                        # Find the order block - looking for last up move before breakout
                        # Start from bar before breakout
                        ob_index = i - 1
                        ob_bottom = low_values[ob_index]
                        ob_top = high_values[ob_index]
                        
                        # Look back to find the highest high before breakout
                        max_high = high_values[ob_index]
                        max_high_index = ob_index
                        
                        for j in range(i-2, max(swing.bar_index-1, i-10), -1):
                            if j >= 0 and high_values[j] > max_high:
                                max_high = high_values[j]
                                max_high_index = j
                                
                        # The order block is the candle with the highest high
                        ob_bottom = low_values[max_high_index]
                        ob_top = high_values[max_high_index]
                        ob_time = data.index[max_high_index]
                        
                        ob = OrderBlock(
                            block_type='bearish',
                            top=ob_top,
                            bottom=ob_bottom,
                            time=ob_time,
                            bar_index=max_high_index
                        )
                        self.bearish_obs.append(ob)
                        
            # Check for breaker blocks
            self._check_breaker_blocks(i, data)
            
    def _check_breaker_blocks(self, bar_index: int, data: pd.DataFrame):
        """Check if any order blocks have become breaker blocks"""
        if self.use_body:
            current_min = min(data.iloc[bar_index]['close'], data.iloc[bar_index]['open'])
            current_max = max(data.iloc[bar_index]['close'], data.iloc[bar_index]['open'])
        else:
            current_min = data.iloc[bar_index]['low']
            current_max = data.iloc[bar_index]['high']
            
        current_close = data.iloc[bar_index]['close']
        current_time = data.index[bar_index]
        
        # Check bullish OBs for breaks
        for ob in self.bullish_obs:
            if not ob.is_breaker and current_min < ob.bottom:
                # Price went below bottom - OB is broken, becomes resistance
                ob.is_breaker = True
                ob.breaker_time = current_time
                ob.breaker_bar_index = bar_index
                
        # Check bearish OBs for breaks  
        for ob in self.bearish_obs:
            if not ob.is_breaker and current_max > ob.top:
                # Price went above top - OB is broken, becomes support
                ob.is_breaker = True
                ob.breaker_time = current_time
                ob.breaker_bar_index = bar_index
                
        # Remove blocks when price CLOSES outside the zone range
        self.bullish_obs = [
            ob for ob in self.bullish_obs 
            if not (ob.is_breaker and (current_close > ob.top or current_close < ob.bottom))
        ]
        
        self.bearish_obs = [
            ob for ob in self.bearish_obs 
            if not (ob.is_breaker and (current_close > ob.top or current_close < ob.bottom))
        ]
        
    def get_active_blocks(self, current_price: float) -> Dict[str, List[OrderBlock]]:
        """Get currently active order blocks"""
        active_bullish = []
        active_bearish = []
        
        # Get most recent blocks
        for ob in self.bullish_obs[-self.show_bullish:]:
            if not ob.is_breaker or current_price <= ob.top:
                active_bullish.append(ob)
                
        for ob in self.bearish_obs[-self.show_bearish:]:
            if not ob.is_breaker or current_price >= ob.bottom:
                active_bearish.append(ob)
                
        return {
            'bullish': active_bullish,
            'bearish': active_bearish
        }


# Legacy analyzer for compatibility
class SupplyDemandAnalyzer(OrderBlockAnalyzer):
    """Legacy class name for compatibility"""
    pass


# =============== Module-level async functions ===============

# Global data manager instance
_data_manager = None


def set_data_manager(data_manager):
    """Set the global data manager instance"""
    global _data_manager
    _data_manager = data_manager
    logger.info("Data manager set in supply_demand module")
    

async def analyze_supply_demand_zones(ticker: str, 
                                    lookback_days: int = None,  # Keep for compatibility
                                    analysis_lookback_days: int = 30,
                                    display_lookback_days: int = 7,
                                    show_bullish_obs: int = 3,
                                    show_bearish_obs: int = 3,
                                    volume_threshold: float = 1.7) -> Dict:
    """
    Analyze 30 days to find order blocks, but only return 7 days of price data
    """
    # Handle backward compatibility
    if lookback_days is not None:
        display_lookback_days = lookback_days
        analysis_lookback_days = max(30, lookback_days)  # Ensure we analyze enough data
        
    if _data_manager is None:
        raise RuntimeError("Data manager not set. Call set_data_manager() first.")
        
    try:
        # Load MORE data for analysis
        end_date = datetime.now(timezone.utc)
        analysis_start_date = end_date - timedelta(days=analysis_lookback_days)
        display_start_date = end_date - timedelta(days=display_lookback_days)
        
        # Load data using data manager
        data = await _data_manager.load_data_async(
            ticker=ticker,
            timeframe='15min',
            start_date=analysis_start_date,
            end_date=end_date
        )
        
        if data.empty:
            return {
                'error': f'No data available for {ticker}',
                'zones': []
            }
            
        # Get current price FIRST
        current_price = float(data['close'].iloc[-1])
        
        # But only return recent price data for display
        display_data = data[data.index >= display_start_date]
        
        # Calculate ATR for reference
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.mean(tr[-14:])  # Simple 14-period ATR
        
        # Initialize analyzer with consistent swing length
        analyzer = OrderBlockAnalyzer(
            swing_length=7,  # Changed from 10 to match your init
            show_bullish=show_bullish_obs,
            show_bearish=show_bearish_obs,
            use_body=False
        )
        
        # Analyze ALL the data
        zones = analyzer.analyze_zones(data)
        
        # Get ALL order blocks (including broken ones) for the table
        all_order_blocks = analyzer.bullish_obs[-analyzer.show_bullish:] + analyzer.bearish_obs[-analyzer.show_bearish:]
        all_zones = [SupplyDemandZone.from_order_block(ob) for ob in all_order_blocks]
        
        # Get only VALID blocks for chart display
        chart_zones = []
        for ob in all_order_blocks:
            if not ob.is_breaker:
                # Not broken, include it
                chart_zones.append(SupplyDemandZone.from_order_block(ob))
            else:
                # Broken - only include if price is still within the zone
                if ob.bottom <= current_price <= ob.top:
                    chart_zones.append(SupplyDemandZone.from_order_block(ob))
        
        # Get active zones (zones that haven't been invalidated)
        active_blocks = analyzer.get_active_blocks(current_price)
        active_zones = [
            SupplyDemandZone.from_order_block(ob) 
            for ob in active_blocks['bullish'] + active_blocks['bearish']
        ]
        
        # Calculate nearby zones (within 3% of current price)
        price_threshold = current_price * 0.03
        nearby_zones = [
            zone for zone in all_zones
            if abs(zone.center_price - current_price) <= price_threshold
        ]
        
        return {
            'ticker': ticker,
            'current_price': current_price,
            'current_atr': atr,
            'total_zones': len(all_zones),
            'active_zones': len(active_zones),
            'nearby_zones': len(nearby_zones),
            'zones': chart_zones,  # Only valid zones for chart
            'all_zones': all_zones,  # All zones for table
            'active_zone_list': active_zones,
            'nearby_zone_list': nearby_zones,
            'analysis_time': datetime.now(timezone.utc),
            'bullish_obs': len(analyzer.bullish_obs),
            'bearish_obs': len(analyzer.bearish_obs),
            'display_data': display_data
        }
        
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'error': str(e),
            'zones': []
        }


async def get_strongest_zones(ticker: str, 
                            top_n: int = 5,
                            min_strength: float = 70.0) -> List[SupplyDemandZone]:
    """
    Get the strongest order blocks for a ticker
    
    Args:
        ticker: Stock symbol
        top_n: Number of top zones to return
        min_strength: Minimum strength threshold
        
    Returns:
        List of strongest zones
    """
    result = await analyze_supply_demand_zones(ticker)
    
    if 'error' in result:
        return []
        
    # All order blocks have high strength by default
    # Breaker blocks have slightly lower strength
    zones = result['zones']
    
    # Sort by validated (non-breaker) first, then by recency
    zones.sort(key=lambda z: (z.validated, z.strength), reverse=True)
    
    return zones[:top_n]


# For backward compatibility
def get_active_zones(self, current_price: float) -> List[SupplyDemandZone]:
    """Legacy method for compatibility"""
    return self.active_zone_list if hasattr(self, 'active_zone_list') else []
    
    
def get_nearby_zones(self, current_price: float, atr_distance: float = 3.0) -> List[SupplyDemandZone]:
    """Legacy method for compatibility"""
    return self.nearby_zone_list if hasattr(self, 'nearby_zone_list') else []