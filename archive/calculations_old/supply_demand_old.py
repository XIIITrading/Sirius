# market_review/calculations/zones/supply_demand.py
"""
Module: Supply and Demand Zone Detection
Purpose: Identify supply/demand zones using volume profile and fractal pivot points
Features: 
- High volume areas (>70% relative to average)
- Fractal pivot point alignment
- ATR-based zone validation
- Async data loading
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import asyncio

# Import required modules - CORRECTED PATHS
from backtest.calculations.volume.volume_profile import VolumeProfile, PriceLevel
from market_review.calculations.market_structure.m15_market_structure import (
    M15MarketStructureAnalyzer, Fractal
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SupplyDemandZone:
    """Supply or Demand zone definition"""
    zone_type: str  # 'supply' or 'demand'
    price_low: float
    price_high: float
    center_price: float
    volume_percent: float
    fractal_price: float
    fractal_time: datetime
    strength: float  # 0-100 based on volume and ATR validation
    validated: bool
    validation_time: Optional[datetime] = None
    validation_move: Optional[float] = None  # ATR units moved
    volume_levels: List[PriceLevel] = None
    

class ATRCalculator:
    """Calculate Average True Range for 15-minute data"""
    
    @staticmethod
    def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate ATR for given OHLC data
        
        Args:
            data: DataFrame with high, low, close columns
            period: ATR period (default 14)
            
        Returns:
            Series with ATR values
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR using EMA
        atr = tr.ewm(span=period, adjust=False).mean()
        
        return atr


class SupplyDemandAnalyzer:
    """
    Analyze supply and demand zones using volume profile and market structure
    """
    
    def __init__(self,
                 lookback_days: int = 7,  # Changed from 15 to 7
                 volume_threshold_multiplier: float = 1.2,  # >20% above average
                 atr_threshold: float = 2.0,
                 validation_window_minutes: int = 60,
                 min_zone_size_atr: float = 0.5):
        """
        Initialize Supply/Demand analyzer
        
        Args:
            lookback_days: Days of historical data to analyze
            volume_threshold_multiplier: Multiplier above average volume (1.2 = 20% above)
            atr_threshold: ATR units for zone validation
            validation_window_minutes: Time window to check for price movement
            min_zone_size_atr: Minimum zone size in ATR units
        """
        self.lookback_days = lookback_days
        self.volume_threshold_multiplier = volume_threshold_multiplier
        self.atr_threshold = atr_threshold
        self.validation_window_minutes = validation_window_minutes
        self.min_zone_size_atr = min_zone_size_atr
        
        # Initialize components
        self.volume_profile = VolumeProfile(levels=100)
        self.market_structure = M15MarketStructureAnalyzer(
            fractal_length=2,
            buffer_size=100,
            min_candles_required=10
        )
        
        # Storage
        self.zones: List[SupplyDemandZone] = []
        self.current_atr: Optional[float] = None
        
    def analyze_zones(self, data: pd.DataFrame) -> List[SupplyDemandZone]:
        """
        Main analysis function to identify supply/demand zones
        
        Args:
            data: DataFrame with OHLC data (15-minute bars)
            
        Returns:
            List of identified supply/demand zones
        """
        logger.info(f"Analyzing supply/demand zones for {len(data)} bars")
        
        # Ensure we have timestamp column
        if 'timestamp' not in data.columns:
            data['timestamp'] = data.index
            
        # Calculate ATR
        atr_series = ATRCalculator.calculate_atr(data)
        self.current_atr = atr_series.iloc[-1]
        logger.info(f"Current ATR: {self.current_atr:.4f}")
        
        # Build volume profile
        price_levels = self.volume_profile.build_volume_profile(data)
        if not price_levels:
            logger.warning("No price levels found in volume profile")
            return []
            
        # Find high volume areas
        high_volume_zones = self._find_high_volume_zones(price_levels)
        logger.info(f"Found {len(high_volume_zones)} high volume zones")
        
        # Process market structure to find fractals
        fractals = self._extract_fractals_from_data(data)
        logger.info(f"Found {len(fractals)} fractal pivot points")
        
        # Align fractals with high volume zones
        self.zones = self._align_fractals_with_volume(
            fractals, high_volume_zones, data, atr_series
        )
        
        # Validate zones with price movement
        self._validate_zones(data, atr_series)
        
        logger.info(f"Identified {len(self.zones)} supply/demand zones")
        return self.zones
        
    def _find_high_volume_zones(self, price_levels: List[PriceLevel]) -> List[List[PriceLevel]]:
        """
        Find contiguous high volume zones
        
        Args:
            price_levels: List of price levels from volume profile
            
        Returns:
            List of zones (each zone is a list of contiguous price levels)
        """
        if not price_levels:
            return []
            
        # Calculate average volume percentage
        avg_percent = 100.0 / len(price_levels)
        threshold = avg_percent * self.volume_threshold_multiplier
        
        # Get levels above threshold
        high_volume_levels = [
            level for level in price_levels 
            if level.percent_of_total >= threshold
        ]
        
        if not high_volume_levels:
            return []
            
        # Sort by index to ensure contiguity
        high_volume_levels.sort(key=lambda x: x.index)
        
        # Group contiguous levels into zones
        zones = []
        current_zone = [high_volume_levels[0]]
        
        for level in high_volume_levels[1:]:
            if level.index == current_zone[-1].index + 1:
                current_zone.append(level)
            else:
                zones.append(current_zone)
                current_zone = [level]
                
        zones.append(current_zone)
        
        return zones
        
    def _extract_fractals_from_data(self, data: pd.DataFrame) -> List[Dict]:
        """
        Extract fractals from 15-minute data
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            List of fractal dictionaries with price, type, and timestamp
        """
        # Process data through market structure analyzer
        symbol = "ANALYSIS"  # Dummy symbol for internal processing
        
        # Convert DataFrame to candle format expected by analyzer
        candles = []
        for idx, row in data.iterrows():
            candle_dict = {
                'timestamp': idx if isinstance(idx, datetime) else row['timestamp'],
                'o': float(row['open']),
                'h': float(row['high']),
                'l': float(row['low']),
                'c': float(row['close']),
                'v': float(row['volume'])
            }
            candles.append(candle_dict)
            
        # Process historical candles
        self.market_structure.process_historical_candles(symbol, candles)
        
        # Extract fractals
        fractals = []
        
        # Get high fractals
        if symbol in self.market_structure.high_fractals:
            for fractal in self.market_structure.high_fractals[symbol]:
                fractals.append({
                    'price': fractal.price,
                    'type': 'high',
                    'timestamp': fractal.timestamp,
                    'bar_index': fractal.bar_index
                })
                
        # Get low fractals
        if symbol in self.market_structure.low_fractals:
            for fractal in self.market_structure.low_fractals[symbol]:
                fractals.append({
                    'price': fractal.price,
                    'type': 'low',
                    'timestamp': fractal.timestamp,
                    'bar_index': fractal.bar_index
                })
                
        return fractals
        
    def _align_fractals_with_volume(self, 
                                  fractals: List[Dict],
                                  high_volume_zones: List[List[PriceLevel]],
                                  data: pd.DataFrame,
                                  atr_series: pd.Series) -> List[SupplyDemandZone]:
        """
        Align fractal pivot points with high volume zones
        
        Args:
            fractals: List of fractal dictionaries
            high_volume_zones: List of high volume zones
            data: Original OHLC data
            atr_series: ATR values
            
        Returns:
            List of supply/demand zones
        """
        zones = []
        
        for zone_levels in high_volume_zones:
            if not zone_levels:
                continue
                
            # Calculate zone boundaries
            zone_low = min(level.low for level in zone_levels)
            zone_high = max(level.high for level in zone_levels)
            zone_volume = sum(level.percent_of_total for level in zone_levels)
            
            # Calculate weighted center
            total_volume = sum(level.volume for level in zone_levels)
            zone_center = sum(
                level.center * level.volume for level in zone_levels
            ) / total_volume if total_volume > 0 else (zone_low + zone_high) / 2
            
            # Check if zone size meets minimum requirement
            atr_at_zone = atr_series.iloc[-1]  # Use current ATR for simplicity
            zone_size = zone_high - zone_low
            if zone_size < self.min_zone_size_atr * atr_at_zone:
                continue
                
            # Find fractals within or near this zone
            zone_fractals = []
            for fractal in fractals:
                # Check if fractal is within zone or within 1 ATR
                if (zone_low <= fractal['price'] <= zone_high or
                    abs(fractal['price'] - zone_center) <= atr_at_zone):
                    zone_fractals.append(fractal)
                    
            # Create zone if we have fractals
            for fractal in zone_fractals:
                zone_type = 'supply' if fractal['type'] == 'high' else 'demand'
                
                # Calculate initial strength based on volume
                strength = min(100, zone_volume * 10)  # Scale volume percentage
                
                zone = SupplyDemandZone(
                    zone_type=zone_type,
                    price_low=zone_low,
                    price_high=zone_high,
                    center_price=zone_center,
                    volume_percent=zone_volume,
                    fractal_price=fractal['price'],
                    fractal_time=fractal['timestamp'],
                    strength=strength,
                    validated=False,
                    volume_levels=zone_levels
                )
                
                zones.append(zone)
                
        return zones
        
    def _validate_zones(self, data: pd.DataFrame, atr_series: pd.Series):
        """
        Validate zones by checking for 2 ATR move within 60 minutes
        
        Args:
            data: OHLC data
            atr_series: ATR values
        """
        for zone in self.zones:
            # Find the fractal bar in the data
            fractal_time = zone.fractal_time
            
            # Get data after fractal formation
            mask = data.index > fractal_time
            future_data = data[mask].head(
                self.validation_window_minutes // 15  # Convert to 15-min bars
            )
            
            if future_data.empty:
                continue
                
            # Get ATR at fractal time
            atr_at_fractal = atr_series.loc[fractal_time] if fractal_time in atr_series.index else self.current_atr
            
            # Check for price movement
            if zone.zone_type == 'supply':
                # For supply zone, look for move down
                min_low = future_data['low'].min()
                move_from_zone = zone.fractal_price - min_low
                
                if move_from_zone >= self.atr_threshold * atr_at_fractal:
                    zone.validated = True
                    zone.validation_move = move_from_zone / atr_at_fractal
                    zone.validation_time = future_data[
                        future_data['low'] == min_low
                    ].index[0]
                    
            else:  # demand zone
                # For demand zone, look for move up
                max_high = future_data['high'].max()
                move_from_zone = max_high - zone.fractal_price
                
                if move_from_zone >= self.atr_threshold * atr_at_fractal:
                    zone.validated = True
                    zone.validation_move = move_from_zone / atr_at_fractal
                    zone.validation_time = future_data[
                        future_data['high'] == max_high
                    ].index[0]
                    
            # Adjust strength based on validation
            if zone.validated:
                zone.strength = min(100, zone.strength + 20)
                logger.debug(
                    f"Validated {zone.zone_type} zone at {zone.center_price:.2f} "
                    f"with {zone.validation_move:.1f} ATR move"
                )
                
    def get_active_zones(self, current_price: float) -> List[SupplyDemandZone]:
        """
        Get zones that haven't been broken by current price
        
        Args:
            current_price: Current market price
            
        Returns:
            List of active zones
        """
        active_zones = []
        
        for zone in self.zones:
            # Supply zone is active if price is below it
            if zone.zone_type == 'supply' and current_price < zone.price_low:
                active_zones.append(zone)
            # Demand zone is active if price is above it
            elif zone.zone_type == 'demand' and current_price > zone.price_high:
                active_zones.append(zone)
                
        return active_zones
        
    def get_nearby_zones(self, current_price: float, 
                        atr_distance: float = 3.0) -> List[SupplyDemandZone]:
        """
        Get zones within specified ATR distance from current price
        
        Args:
            current_price: Current market price
            atr_distance: Maximum distance in ATR units
            
        Returns:
            List of nearby zones
        """
        if self.current_atr is None:
            return []
            
        max_distance = atr_distance * self.current_atr
        nearby_zones = []
        
        for zone in self.zones:
            distance = min(
                abs(current_price - zone.price_low),
                abs(current_price - zone.price_high),
                abs(current_price - zone.center_price)
            )
            
            if distance <= max_distance:
                nearby_zones.append(zone)
                
        return sorted(nearby_zones, key=lambda z: abs(z.center_price - current_price))


# =============== Module-level async functions for data loading ===============

# Global data manager instance (to be set by the application)
_data_manager = None


def set_data_manager(data_manager):
    """Set the global data manager instance"""
    global _data_manager
    _data_manager = data_manager
    

async def analyze_supply_demand_zones(ticker: str, 
                                    lookback_days: int = 7,  # Changed from 15 to 7
                                    volume_threshold: float = 1.7) -> Dict:
    """
    Async function to analyze supply/demand zones for a ticker
    
    Args:
        ticker: Stock symbol
        lookback_days: Days of historical data
        volume_threshold: Volume threshold multiplier
        
    Returns:
        Dictionary with analysis results
    """
    if _data_manager is None:
        raise RuntimeError("Data manager not set. Call set_data_manager() first.")
        
    try:
        # Load 15-minute data
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=lookback_days)
        
        # Load data using data manager
        data = await _data_manager.load_data_async(
            ticker=ticker,
            timeframe='15min',
            start_date=start_date,
            end_date=end_date
        )
        
        if data.empty:
            return {
                'error': f'No data available for {ticker}',
                'zones': []
            }
            
        # Initialize analyzer
        analyzer = SupplyDemandAnalyzer(
            lookback_days=lookback_days,
            volume_threshold_multiplier=volume_threshold
        )
        
        # Analyze zones
        zones = analyzer.analyze_zones(data)
        
        # Get current price
        current_price = float(data['close'].iloc[-1])
        
        # Categorize zones
        active_zones = analyzer.get_active_zones(current_price)
        nearby_zones = analyzer.get_nearby_zones(current_price)
        
        return {
            'ticker': ticker,
            'current_price': current_price,
            'current_atr': analyzer.current_atr,
            'total_zones': len(zones),
            'active_zones': len(active_zones),
            'nearby_zones': len(nearby_zones),
            'zones': zones,
            'active_zone_list': active_zones,
            'nearby_zone_list': nearby_zones,
            'analysis_time': datetime.now(timezone.utc)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {str(e)}")
        return {
            'error': str(e),
            'zones': []
        }


async def get_strongest_zones(ticker: str, 
                            top_n: int = 5,
                            min_strength: float = 20.0) -> List[SupplyDemandZone]:
    """
    Get the strongest supply/demand zones for a ticker
    
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
        
    # Filter by strength and sort
    strong_zones = [
        zone for zone in result['zones'] 
        if zone.strength >= min_strength
    ]
    
    # Sort by strength and validation
    strong_zones.sort(
        key=lambda z: (z.validated, z.strength), 
        reverse=True
    )
    
    return strong_zones[:top_n]


# =============== Example usage for testing ===============

if __name__ == "__main__":
    # Example synchronous usage for testing
    import asyncio
    
    async def test_analysis():
        # Mock data manager for testing
        class MockDataManager:
            async def load_data_async(self, **kwargs):
                # Generate sample 15-minute data
                dates = pd.date_range(
                    start=kwargs['start_date'],
                    end=kwargs['end_date'],
                    freq='15min'
                )
                
                # Generate random OHLCV data
                np.random.seed(42)
                df = pd.DataFrame({
                    'open': 100 + np.random.randn(len(dates)).cumsum() * 0.5,
                    'high': 0,
                    'low': 0,
                    'close': 0,
                    'volume': np.random.randint(1000, 10000, len(dates))
                }, index=dates)
                
                # Calculate OHLC properly
                df['high'] = df['open'] + np.abs(np.random.randn(len(dates)) * 0.3)
                df['low'] = df['open'] - np.abs(np.random.randn(len(dates)) * 0.3)
                df['close'] = df['low'] + np.random.rand(len(dates)) * (df['high'] - df['low'])
                
                return df
                
        # Set mock data manager
        set_data_manager(MockDataManager())
        
        # Test analysis with 7 days lookback
        result = await analyze_supply_demand_zones('TEST', lookback_days=7)  # Changed to 7
        
        print(f"Analysis Results:")
        print(f"Total zones found: {result['total_zones']}")
        print(f"Active zones: {result['active_zones']}")
        print(f"Current ATR: {result['current_atr']:.4f}")
        
        # Show top zones
        strongest = await get_strongest_zones('TEST', top_n=3)
        print(f"\nTop 3 Strongest Zones:")
        for i, zone in enumerate(strongest, 1):
            print(f"{i}. {zone.zone_type.upper()} Zone:")
            print(f"   Range: ${zone.price_low:.2f} - ${zone.price_high:.2f}")
            print(f"   Volume: {zone.volume_percent:.1f}%")
            print(f"   Strength: {zone.strength:.0f}")
            print(f"   Validated: {zone.validated}")
            
    # Run test
    asyncio.run(test_analysis())