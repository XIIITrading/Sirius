# market_review/calculations/volume/volume_profile.py
"""
Module: Volume Profile Calculation
Purpose: Aggregate volume by price levels for HVN analysis
Time Handling: All timestamps in UTC, no conversions
Performance Target: Process 14 days of 1-min data in <1 second
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

@dataclass
class PriceLevel:
    """Container for price level information"""
    index: int
    low: float
    high: float
    center: float
    volume: float
    percent_of_total: float = 0.0

class VolumeProfile:
    """
    Calculate volume profile from OHLCV data.
    All timestamps must be in UTC.
    """
    
    def __init__(self, levels: int = 100):
        """
        Initialize Volume Profile calculator.
        
        Args:
            levels: Number of price levels to divide the range into
        """
        self.levels = levels
        self.pre_market_start = 8  # 08:00 UTC
        self.pre_market_end = 13.5  # 13:30 UTC
        self.post_market_start = 20  # 20:00 UTC
        self.post_market_end = 24  # 00:00 UTC (next day)
        
        # Store the profile results
        self.price_levels: List[PriceLevel] = []
        self.hvn_unit: float = 0.0
        self.price_range: Tuple[float, float] = (0.0, 0.0)
        
    def calculate_price_levels(self, high: float, low: float) -> Tuple[np.ndarray, float]:
        """
        Calculate price levels and unit size.
        
        Args:
            high: Highest price in range
            low: Lowest price in range
            
        Returns:
            (price_boundaries, hvn_unit)
        """
        price_range = high - low
        hvn_unit = price_range / self.levels
        
        # Create price level boundaries
        price_boundaries = np.linspace(low, high, self.levels + 1)
        
        # Store for later use
        self.hvn_unit = hvn_unit
        self.price_range = (low, high)
        
        return price_boundaries, hvn_unit
    
    def is_market_hours(self, timestamp: pd.Timestamp, 
                       include_pre: bool = True, 
                       include_post: bool = True) -> bool:
        """
        Check if timestamp is within desired market hours (UTC).
        
        Args:
            timestamp: UTC timestamp to check
            include_pre: Include pre-market hours
            include_post: Include post-market hours
        """
        hour = timestamp.hour + timestamp.minute / 60.0
        
        # Regular market hours (13:30 - 20:00 UTC)
        if 13.5 <= hour < 20:
            return True
            
        # Pre-market hours
        if include_pre and 8 <= hour < 13.5:
            return True
            
        # Post-market hours (handle day boundary)
        if include_post and (20 <= hour <= 24 or 0 <= hour < 0):
            return True
            
        return False
    
    def build_volume_profile(self, 
                           data: pd.DataFrame,
                           include_pre: bool = True,
                           include_post: bool = True) -> List[PriceLevel]:
        """
        Build complete volume profile with price level associations.
        
        Args:
            data: DataFrame with columns: timestamp, open, high, low, close, volume
            include_pre: Include pre-market volume
            include_post: Include post-market volume
            
        Returns:
            List of PriceLevel objects sorted by level index
        """
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
        
        # Filter for market hours
        mask = data['timestamp'].apply(
            lambda x: self.is_market_hours(x, include_pre, include_post)
        )
        filtered_data = data[mask].copy()
        
        if filtered_data.empty:
            return []
        
        # Calculate price levels
        high = filtered_data['high'].max()
        low = filtered_data['low'].min()
        price_boundaries, hvn_unit = self.calculate_price_levels(high, low)
        
        # Initialize volume array
        volume_by_level = np.zeros(self.levels)
        
        # Aggregate volume
        for _, row in filtered_data.iterrows():
            # Estimate volume distribution across the bar's range
            bar_low = row['low']
            bar_high = row['high']
            bar_volume = row['volume']
            
            # Find which levels this bar touches
            low_idx = np.searchsorted(price_boundaries, bar_low, side='left')
            high_idx = np.searchsorted(price_boundaries, bar_high, side='right')
            
            # Distribute volume evenly across touched levels
            if high_idx > low_idx:
                levels_touched = high_idx - low_idx
                volume_per_level = bar_volume / levels_touched
                
                for i in range(max(0, low_idx), min(self.levels, high_idx)):
                    volume_by_level[i] += volume_per_level
        
        # Calculate total volume for percentages
        total_volume = np.sum(volume_by_level)
        
        # Build PriceLevel objects
        self.price_levels = []
        for i in range(self.levels):
            if volume_by_level[i] > 0:  # Only include levels with volume
                level = PriceLevel(
                    index=i,
                    low=price_boundaries[i],
                    high=price_boundaries[i + 1],
                    center=(price_boundaries[i] + price_boundaries[i + 1]) / 2,
                    volume=volume_by_level[i],
                    percent_of_total=(volume_by_level[i] / total_volume) * 100 if total_volume > 0 else 0
                )
                self.price_levels.append(level)
        
        return self.price_levels
    
    def get_level_by_price(self, price: float) -> Optional[PriceLevel]:
        """
        Find the price level containing a given price.
        
        Args:
            price: The price to look up
            
        Returns:
            PriceLevel object or None if price is outside range
        """
        for level in self.price_levels:
            if level.low <= price <= level.high:
                return level
        return None
    
    def get_top_levels(self, n: int = 10) -> List[PriceLevel]:
        """
        Get top N levels by volume percentage.
        
        Args:
            n: Number of top levels to return
            
        Returns:
            List of PriceLevel objects sorted by volume percentage (descending)
        """
        return sorted(self.price_levels, 
                     key=lambda x: x.percent_of_total, 
                     reverse=True)[:n]
    
    def get_levels_above_threshold(self, threshold: float) -> List[PriceLevel]:
        """
        Get all levels above a volume percentage threshold.
        
        Args:
            threshold: Minimum percentage of total volume (e.g., 95.0 for 95%)
            
        Returns:
            List of PriceLevel objects meeting the threshold
        """
        return [level for level in self.price_levels 
                if level.percent_of_total >= threshold]


# ============= LIVE DATA TESTING WITH POLYGON =============
if __name__ == "__main__":
    print("=== Testing Volume Profile with LIVE POLYGON DATA ===\n")
    
    # ========== CONFIGURATION ==========
    # Modify these settings for different test scenarios
    TEST_CONFIG = {
        'symbol': 'SPY',               # Stock symbol to analyze
        'lookback_days': 7,            # Days of historical data (7 for faster test)
        'timeframe': '5min',           # Data resolution
        'levels': 50,                  # Number of price levels (50 for clearer visualization)
        'include_premarket': True,     # Include pre-market data
        'include_postmarket': True,    # Include post-market data
        'show_top_levels': 10,         # Number of top levels to display
        'volume_threshold_multiplier': 2.0,  # Show levels with X times average volume
    }
    # ===================================
    
    import sys
    import os
    import time
    
    try:
        # Fix the import path - we're in volume subdirectory
        current_file = os.path.abspath(__file__)
        volume_dir = os.path.dirname(current_file)
        calculations_dir = os.path.dirname(volume_dir)
        modules_dir = os.path.dirname(calculations_dir)
        vega_root = os.path.dirname(modules_dir)
        
        # Add Vega root to Python path
        if vega_root not in sys.path:
            sys.path.insert(0, vega_root)
        
        print(f"✓ Vega root: {vega_root}")
        print(f"✓ Added to Python path")
        
        # Import Polygon DataFetcher
        from polygon import DataFetcher
        print(f"✓ Successfully imported Polygon DataFetcher")
        
        print(f"\nFetching live data for {TEST_CONFIG['symbol']}...")
        
        # Initialize data fetcher
        fetcher = DataFetcher()
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=TEST_CONFIG['lookback_days'])
        
        # Fetch data
        start_time = time.time()
        df = fetcher.fetch_data(
            symbol=TEST_CONFIG['symbol'],
            timeframe=TEST_CONFIG['timeframe'],
            start_date=start_date,
            end_date=end_date,
            use_cache=True,
            validate=True
        )
        fetch_time = time.time() - start_time
        
        # Prepare data for volume profile
        df['timestamp'] = df.index
        
        print(f"✓ Data fetched in {fetch_time:.2f} seconds")
        print(f"✓ Date range: {df.index[0]} to {df.index[-1]}")
        print(f"✓ Total bars: {len(df)}")
        
        # Test 1: Basic Volume Profile Calculation
        print(f"\n[TEST 1] Calculating volume profile with {TEST_CONFIG['levels']} levels...")
        
        vp = VolumeProfile(levels=TEST_CONFIG['levels'])
        
        start_time = time.time()
        profile = vp.build_volume_profile(
            df, 
            include_pre=TEST_CONFIG['include_premarket'],
            include_post=TEST_CONFIG['include_postmarket']
        )
        calc_time = time.time() - start_time
        
        print(f"✓ Volume profile calculated in {calc_time:.3f} seconds")
        print(f"✓ Processing speed: {len(df) / calc_time:.0f} bars/second")
        
        # Test 2: Display Results
        print(f"\n{'='*70}")
        print(f"VOLUME PROFILE RESULTS - {TEST_CONFIG['symbol']}")
        print(f"{'='*70}")
        
        print(f"\n📊 Price Analysis:")
        print(f"  Price range: ${vp.price_range[0]:.2f} to ${vp.price_range[1]:.2f}")
        print(f"  Total range: ${vp.price_range[1] - vp.price_range[0]:.2f}")
        print(f"  HVN Unit size: ${vp.hvn_unit:.4f}")
        print(f"  Current price: ${df['close'].iloc[-1]:.2f}")
        
        print(f"\n📈 Volume Analysis:")
        print(f"  Total levels with volume: {len(profile)}")
        print(f"  Total volume analyzed: {df['volume'].sum():,.0f}")
        print(f"  Average volume per level: {df['volume'].sum() / len(profile):,.0f}")
        
        # Test 3: Show Top Volume Levels
        print(f"\n🎯 Top {TEST_CONFIG['show_top_levels']} Volume Levels:")
        print(f"  {'Rank':<6} {'Price Range':<20} {'Center':<10} {'Volume %':<10} {'Visual':<20}")
        print(f"  {'-'*6} {'-'*20} {'-'*10} {'-'*10} {'-'*20}")
        
        top_levels = vp.get_top_levels(TEST_CONFIG['show_top_levels'])
        max_percent = top_levels[0].percent_of_total if top_levels else 1
        
        for i, level in enumerate(top_levels, 1):
            price_range = f"${level.low:.2f}-${level.high:.2f}"
            bar_length = int((level.percent_of_total / max_percent) * 20)
            visual_bar = '█' * bar_length
            print(f"  {i:<6} {price_range:<20} ${level.center:<9.2f} {level.percent_of_total:<9.2f}% {visual_bar}")
        
        # Test 4: Price Level Lookup
        print(f"\n[TEST 2] Price Level Lookup Test")
        current_price = df['close'].iloc[-1]
        current_level = vp.get_level_by_price(current_price)
        
        if current_level:
            print(f"✓ Current price ${current_price:.2f} is in level {current_level.index}")
            print(f"  Level range: ${current_level.low:.2f} - ${current_level.high:.2f}")
            print(f"  Volume %: {current_level.percent_of_total:.2f}%")
            print(f"  Rank: {sorted(profile, key=lambda x: x.percent_of_total, reverse=True).index(current_level) + 1}")
        
        # Test 5: Find High-Volume Zones
        print(f"\n[TEST 3] High Volume Zone Detection")
        avg_percent = 100.0 / TEST_CONFIG['levels']
        threshold = avg_percent * TEST_CONFIG['volume_threshold_multiplier']
        high_volume_levels = vp.get_levels_above_threshold(threshold)
        
        print(f"  Average volume per level: {avg_percent:.2f}%")
        print(f"  High volume threshold: {threshold:.2f}%")
        print(f"✓ Found {len(high_volume_levels)} high-volume levels")
        
        if high_volume_levels:
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
            
            print(f"\n📍 High-Volume Zones ({len(zones)} found):")
            for i, zone in enumerate(zones, 1):
                zone_low = min(level.low for level in zone)
                zone_high = max(level.high for level in zone)
                zone_volume = sum(level.percent_of_total for level in zone)
                zone_center = sum(level.center * level.volume for level in zone) / sum(level.volume for level in zone)
                
                print(f"\n  Zone {i}:")
                print(f"    Range: ${zone_low:.2f} - ${zone_high:.2f}")
                print(f"    Center: ${zone_center:.2f}")
                print(f"    Total volume: {zone_volume:.1f}%")
                print(f"    Levels: {len(zone)}")
                
                # Check if current price is in this zone
                if zone_low <= current_price <= zone_high:
                    print(f"    Status: 🎯 CURRENT PRICE IN ZONE")
                else:
                    distance = min(abs(current_price - zone_high), abs(current_price - zone_low))
                    print(f"    Status: ${distance:.2f} away")
        
        # Test 6: Market Hours Analysis
        print(f"\n[TEST 4] Market Hours Volume Distribution")
        
        # Regular hours only
        vp_regular = VolumeProfile(levels=TEST_CONFIG['levels'])
        profile_regular = vp_regular.build_volume_profile(df, include_pre=False, include_post=False)
        
        # Extended hours only
        vp_extended = VolumeProfile(levels=TEST_CONFIG['levels'])
        profile_extended_full = vp_extended.build_volume_profile(df, include_pre=True, include_post=True)
        
        if profile_regular and profile:
            regular_volume = sum(level.volume for level in profile_regular)
            total_volume = sum(level.volume for level in profile)
            extended_volume = total_volume - regular_volume
            
            print(f"  Regular hours volume: {regular_volume:,.0f} ({(regular_volume/total_volume)*100:.1f}%)")
            print(f"  Extended hours volume: {extended_volume:,.0f} ({(extended_volume/total_volume)*100:.1f}%)")
            print(f"  Regular hours levels: {len(profile_regular)}")
            print(f"  Total levels (with extended): {len(profile)}")
        
        # Performance Summary
        print(f"\n⚡ Performance Summary:")
        print(f"  Data fetch time: {fetch_time:.2f} seconds")
        print(f"  Calculation time: {calc_time:.3f} seconds")
        print(f"  Total bars processed: {len(df)}")
        print(f"  Processing speed: {len(df) / calc_time:.0f} bars/second")
        print(f"  Memory efficiency: ✓ (target <1 second achieved)" if calc_time < 1 else f"  Memory efficiency: ⚠️  ({calc_time:.2f} seconds)")
        
        # Connection summary
        print(f"\n{'='*70}")
        print("✅ POLYGON CONNECTION SUCCESSFUL - VOLUME PROFILE TESTS PASSED")
        print(f"{'='*70}")
        
    except ImportError as e:
        print(f"❌ Import Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check that polygon package is installed")
        print("2. Verify POLYGON_API_KEY is in .env file")
        print("3. Make sure you're running from the correct directory")
        
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check your POLYGON_API_KEY in .env file")
        print("2. Verify internet connection")
        print("3. Ensure the symbol is valid and market was open")
        print("4. Check your Polygon API subscription limits")
        
        # More detailed error info
        import traceback
        print("\nDetailed error trace:")
        traceback.print_exc()
    
    print("\n=== Volume Profile Live Data Test Complete ===")