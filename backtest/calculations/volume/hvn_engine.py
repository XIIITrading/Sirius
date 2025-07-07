# modules/calculations/volume/hvn_engine.py
"""
Module: HVN (High Volume Node) Calculation Engine
Purpose: Core HVN calculations including ranking, clustering, and proximity detection
Dependencies: volume_profile.py for base calculations
Performance Target: Complete 14-day analysis in <2 seconds
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

# Fix the import path to work from any location
try:
    # Try relative import first (for when running hvn_engine.py directly)
    from .volume_profile import VolumeProfile, PriceLevel
except ImportError:
    # If that fails, try absolute import
    try:
        from modules.calculations.volume.volume_profile import VolumeProfile, PriceLevel
    except ImportError:
        # If still failing, add path and import
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        sys.path.append(parent_dir)
        from volume_profile import VolumeProfile, PriceLevel

@dataclass
class HVNCluster:
    """Container for HVN cluster information"""
    levels: List[PriceLevel]
    cluster_high: float
    cluster_low: float
    center_price: float
    total_volume: float
    total_percent: float
    highest_volume_level: PriceLevel

@dataclass
class HVNResult:
    """Complete HVN analysis result"""
    hvn_unit: float
    price_range: Tuple[float, float]
    clusters: List[HVNCluster]
    ranked_levels: List[PriceLevel]  # All levels with rank
    filtered_levels: List[PriceLevel]  # Levels above threshold

class HVNEngine:
    """
    Main HVN calculation engine.
    Processes volume profile data to identify high volume nodes and clusters.
    """
    
    def __init__(self, 
                 levels: int = 100,
                 percentile_threshold: float = 80.0,
                 proximity_atr_minutes: int = 30):
        """
        Initialize HVN Engine.
        
        Args:
            levels: Number of price levels for volume profile
            percentile_threshold: Percentile threshold (80 = top 20% of levels)
            proximity_atr_minutes: ATR in minutes for proximity alerts
        """
        self.levels = levels
        self.percentile_threshold = percentile_threshold
        self.proximity_atr_minutes = proximity_atr_minutes
        self.volume_profile = VolumeProfile(levels=levels)
        
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range.
        
        Args:
            data: OHLCV DataFrame
            period: ATR period (default 14)
            
        Returns:
            ATR value
        """
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr
    
    def rank_levels(self, levels: List[PriceLevel]) -> List[PriceLevel]:
        """
        Rank levels from 1-100 based on volume percentage.
        100 = highest volume, 1 = lowest volume.
        For ties, closer to current price gets higher rank.
        
        Args:
            levels: List of PriceLevel objects
            
        Returns:
            List of PriceLevel objects with rank attribute added
        """
        if not levels:
            return []
        
        # Get current price (using last level's center as proxy)
        current_price = levels[-1].center
        
        # Sort by volume percentage (descending) and distance to current price (ascending)
        sorted_levels = sorted(
            levels,
            key=lambda x: (-x.percent_of_total, abs(x.center - current_price))
        )
        
        # Assign ranks (1-100 scale)
        total_levels = len(sorted_levels)
        for i, level in enumerate(sorted_levels):
            # Scale rank to 1-100 range
            level.rank = int(100 - (i * 99 / (total_levels - 1))) if total_levels > 1 else 100
            
        return sorted_levels
    
    def filter_by_percentile(self, ranked_levels: List[PriceLevel]) -> List[PriceLevel]:
        """
        Filter levels that are in the top X percentile.
        
        Args:
            ranked_levels: Levels already ranked 1-100
            
        Returns:
            Levels with rank >= percentile_threshold
        """
        return [level for level in ranked_levels 
                if level.rank >= self.percentile_threshold]
    
    def identify_contiguous_clusters(self, 
                                   filtered_levels: List[PriceLevel],
                                   all_levels: List[PriceLevel]) -> List[HVNCluster]:
        """
        Identify clusters of contiguous high-volume levels.
        Checks if adjacent levels are also above threshold.
        
        Args:
            filtered_levels: Levels above percentile threshold
            all_levels: All levels (for checking adjacent)
            
        Returns:
            List of HVNCluster objects
        """
        if not filtered_levels:
            return []
        
        # Create a set of indices for quick lookup
        filtered_indices = {level.index for level in filtered_levels}
        
        # Create index map for all levels
        level_by_index = {level.index: level for level in all_levels}
        
        # Sort filtered levels by index (price order)
        sorted_levels = sorted(filtered_levels, key=lambda x: x.index)
        
        clusters = []
        used_indices = set()
        
        for level in sorted_levels:
            if level.index in used_indices:
                continue
                
            # Start new cluster
            cluster_levels = [level]
            used_indices.add(level.index)
            
            # Check upward (higher prices)
            current_idx = level.index
            while True:
                next_idx = current_idx + 1
                if next_idx in filtered_indices and next_idx not in used_indices:
                    cluster_levels.append(level_by_index[next_idx])
                    used_indices.add(next_idx)
                    current_idx = next_idx
                else:
                    break
            
            # Check downward (lower prices)
            current_idx = level.index
            while True:
                prev_idx = current_idx - 1
                if prev_idx in filtered_indices and prev_idx not in used_indices:
                    cluster_levels.append(level_by_index[prev_idx])
                    used_indices.add(prev_idx)
                    current_idx = prev_idx
                else:
                    break
            
            # Create cluster
            cluster = self._create_cluster(cluster_levels)
            clusters.append(cluster)
        
        return sorted(clusters, key=lambda x: x.total_percent, reverse=True)
    
    def _create_cluster(self, levels: List[PriceLevel]) -> HVNCluster:
        """Helper to create HVNCluster from levels."""
        total_volume = sum(l.volume for l in levels)
        
        return HVNCluster(
            levels=sorted(levels, key=lambda x: x.index),
            cluster_high=max(l.high for l in levels),
            cluster_low=min(l.low for l in levels),
            center_price=sum(l.center * l.volume for l in levels) / total_volume,
            total_volume=total_volume,
            total_percent=sum(l.percent_of_total for l in levels),
            highest_volume_level=max(levels, key=lambda x: x.volume)
        )
    
    def check_proximity(self, current_price: float, clusters: List[HVNCluster], atr: float) -> Dict[int, bool]:
        """
        Check if current price is within proximity of any cluster.
        
        Args:
            current_price: Current market price
            clusters: List of HVN clusters
            atr: Average True Range
            
        Returns:
            Dictionary mapping cluster index to proximity boolean
        """
        proximity_filter = atr * (self.proximity_atr_minutes / 60.0)
        proximity_results = {}
        
        for i, cluster in enumerate(clusters):
            # Check distance to cluster boundaries
            distance_to_high = abs(current_price - cluster.cluster_high)
            distance_to_low = abs(current_price - cluster.cluster_low)
            
            # Within proximity if close to either boundary or inside cluster
            is_proximate = (distance_to_high <= proximity_filter or 
                          distance_to_low <= proximity_filter or
                          (cluster.cluster_low <= current_price <= cluster.cluster_high))
            
            proximity_results[i] = is_proximate
            
        return proximity_results
    
    def calculate_volume_trend(self, data: pd.DataFrame, price_level: float, bars: int = 14) -> str:
        """
        Calculate if volume is increasing or decreasing approaching a level.
        
        Args:
            data: Recent OHLCV data (5-minute bars)
            price_level: The HVN level to check approach
            bars: Number of recent bars to analyze
            
        Returns:
            'increasing', 'decreasing', or 'neutral'
        """
        if len(data) < bars:
            return 'neutral'
            
        recent_data = data.tail(bars).copy()
        
        # Calculate distance to level for each bar
        recent_data['distance'] = abs(recent_data['close'] - price_level)
        
        # Check if we're approaching the level
        distance_trend = recent_data['distance'].diff().mean()
        
        if distance_trend >= 0:  # Moving away
            return 'neutral'
            
        # We're approaching - check volume trend
        volume_first_half = recent_data.head(bars // 2)['volume'].mean()
        volume_second_half = recent_data.tail(bars // 2)['volume'].mean()
        
        if volume_second_half > volume_first_half * 1.1:
            return 'increasing'
        elif volume_second_half < volume_first_half * 0.9:
            return 'decreasing'
        else:
            return 'neutral'
    
    def analyze(self, 
                data: pd.DataFrame,
                include_pre: bool = True,
                include_post: bool = True) -> HVNResult:
        """
        Run complete HVN analysis.
        
        Args:
            data: OHLCV DataFrame with UTC timestamps
            include_pre: Include pre-market data
            include_post: Include post-market data
            
        Returns:
            HVNResult with all calculations
        """
        # Build volume profile
        profile_levels = self.volume_profile.build_volume_profile(
            data, include_pre, include_post
        )
        
        if not profile_levels:
            return HVNResult(
                hvn_unit=0,
                price_range=(0, 0),
                clusters=[],
                ranked_levels=[],
                filtered_levels=[]
            )
        
        # Calculate ATR
        atr = self.calculate_atr(data)
        
        # Rank all levels
        ranked_levels = self.rank_levels(profile_levels)
        
        # Filter levels by percentile (top 20% if threshold=80)
        filtered_levels = self.filter_by_percentile(ranked_levels)
        
        # Identify contiguous clusters
        clusters = self.identify_contiguous_clusters(filtered_levels, profile_levels)
        
        return HVNResult(
            hvn_unit=self.volume_profile.hvn_unit,
            price_range=self.volume_profile.price_range,
            clusters=clusters,
            ranked_levels=ranked_levels,
            filtered_levels=filtered_levels
        )


# ============= LIVE DATA TESTING WITH POLYGON BRIDGE =============
if __name__ == "__main__":
    print("=== Testing HVN Engine with Polygon Bridge Integration ===\n")
    
    # ========== CONFIGURATION ==========
    # Modify these settings for different test scenarios
    TEST_CONFIG = {
        'symbol': 'TSLA',              # Stock symbol to analyze
        'lookback_days': 14,           # Days of historical data
        'timeframe': '15min',           # Data resolution (1min, 5min, 15min, etc.)
        'hvn_levels': 100,             # Number of price levels
        'percentile_threshold': 70.0,  # Top X% of levels (80 = top 20%)
        'proximity_atr_minutes': 30,   # ATR minutes for proximity detection
        'include_premarket': True,     # Include pre-market data
        'include_postmarket': True,    # Include post-market data
    }
    # ===================================
    
    try:
        # Fix the import path - we're in volume subdirectory
        current_file = os.path.abspath(__file__)
        volume_dir = os.path.dirname(current_file)  # .../volume/
        calculations_dir = os.path.dirname(volume_dir)  # .../calculations/
        modules_dir = os.path.dirname(calculations_dir)  # .../modules/
        vega_root = os.path.dirname(modules_dir)  # .../Vega/
        
        # Add Vega root to Python path
        if vega_root not in sys.path:
            sys.path.insert(0, vega_root)
        
        print(f"✓ Current file: {current_file}")
        print(f"✓ Vega root: {vega_root}")
        print(f"✓ Added to Python path")
        
        # Import the bridge - use try/except to handle missing file
        try:
            from modules.data.polygon_bridge import PolygonHVNBridge
            print(f"✓ Successfully imported PolygonHVNBridge")
        except ImportError:
            # If polygon_bridge doesn't exist, let's test with direct Polygon import
            print("⚠️  polygon_bridge.py not found, testing with direct Polygon API...")
            
            # Import Polygon directly
            from polygon import DataFetcher
            
            # Create a simple wrapper class
            class PolygonHVNBridge:
                def __init__(self, hvn_levels, hvn_percentile, lookback_days):
                    self.fetcher = DataFetcher()
                    self.hvn_engine = HVNEngine(hvn_levels, hvn_percentile)
                    self.lookback_days = lookback_days
                
                def calculate_hvn(self, symbol, timeframe='5min'):
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=self.lookback_days)
                    
                    # Fetch data
                    df = self.fetcher.fetch_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date,
                        use_cache=True,
                        validate=True
                    )
                    
                    # Prepare data
                    df['timestamp'] = df.index
                    
                    # Run HVN analysis
                    hvn_result = self.hvn_engine.analyze(df)
                    
                    # Create a simple state object
                    from types import SimpleNamespace
                    state = SimpleNamespace(
                        symbol=symbol,
                        recent_bars=df,
                        current_price=df['close'].iloc[-1],
                        hvn_result=hvn_result,
                        last_calculation=datetime.now()
                    )
                    
                    return state
                
                def get_hvn_levels_near_price(self, symbol, price_range_percent):
                    # Simple implementation
                    return []
            
            print("✓ Created temporary bridge for testing")
        
        print(f"\nTesting Polygon API connection...")
        
        # Test 1: Initialize the bridge
        print("\n[TEST 1] Initializing Polygon HVN Bridge...")
        bridge = PolygonHVNBridge(
            hvn_levels=TEST_CONFIG['hvn_levels'],
            hvn_percentile=TEST_CONFIG['percentile_threshold'],
            lookback_days=TEST_CONFIG['lookback_days']
        )
        print("✓ Bridge initialized successfully")
        
        # Test 2: Fetch live data and calculate HVN
        print(f"\n[TEST 2] Fetching live {TEST_CONFIG['symbol']} data...")
        print(f"  - Timeframe: {TEST_CONFIG['timeframe']}")
        print(f"  - Lookback: {TEST_CONFIG['lookback_days']} days")
        
        state = bridge.calculate_hvn(
            TEST_CONFIG['symbol'],
            timeframe=TEST_CONFIG['timeframe']
        )
        
        print(f"✓ Data fetched and HVN calculated successfully")
        print(f"✓ Received {len(state.recent_bars)} bars of data")
        
        # Test 3: Validate the results
        print(f"\n[TEST 3] Validating HVN Results...")
        result = state.hvn_result
        
        # Basic validation
        assert result.hvn_unit > 0, "HVN unit should be positive"
        assert len(result.ranked_levels) > 0, "Should have ranked levels"
        assert result.price_range[1] > result.price_range[0], "Price range should be valid"
        print("✓ All validations passed")
        
        # Display detailed results
        print(f"\n{'='*70}")
        print(f"POLYGON CONNECTION VALIDATED - HVN ANALYSIS RESULTS")
        print(f"{'='*70}")
        
        print(f"\n📊 Data Summary:")
        print(f"  Symbol: {TEST_CONFIG['symbol']}")
        print(f"  Date Range: {state.recent_bars.index[0]} to {state.recent_bars.index[-1]}")
        print(f"  Total Bars: {len(state.recent_bars)}")
        print(f"  Price Range: ${result.price_range[0]:.2f} - ${result.price_range[1]:.2f}")
        print(f"  Current Price: ${state.current_price:.2f}")
        
        print(f"\n📈 Volume Profile Analysis:")
        print(f"  Total Levels: {len(result.ranked_levels)}")
        print(f"  HVN Levels (top {100-TEST_CONFIG['percentile_threshold']:.0f}%): {len(result.filtered_levels)}")
        print(f"  Contiguous Clusters: {len(result.clusters)}")
        
        # Display top HVN levels
        print(f"\n🎯 Top 5 HVN Levels:")
        print(f"  {'Rank':<6} {'Price':<10} {'Volume %':<10} {'Distance from Current':<20}")
        print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*20}")
        
        for level in result.filtered_levels[:5]:
            distance = abs(level.center - state.current_price)
            print(f"  {level.rank:<6} ${level.center:<9.2f} {level.percent_of_total:<9.2f}% ${distance:<19.2f}")
        
        # Display clusters
        print(f"\n📍 HVN Clusters:")
        for i, cluster in enumerate(result.clusters[:10]):  # Show top 10 clusters
            print(f"\n  Cluster {i+1}:")
            print(f"    Range: ${cluster.cluster_low:.2f} - ${cluster.cluster_high:.2f}")
            print(f"    Center: ${cluster.center_price:.2f}")
            print(f"    Volume: {cluster.total_percent:.2f}% of total")
            print(f"    Levels: {len(cluster.levels)}")
            
            # Proximity check
            if cluster.cluster_low <= state.current_price <= cluster.cluster_high:
                print(f"    Status: 🎯 PRICE INSIDE CLUSTER")
            else:
                distance = min(
                    abs(state.current_price - cluster.cluster_high),
                    abs(state.current_price - cluster.cluster_low)
                )
                print(f"    Status: ${distance:.2f} away")
        
        # Connection summary
        print(f"\n{'='*70}")
        print("✅ POLYGON CONNECTION SUCCESSFUL - ALL TESTS PASSED")
        print(f"{'='*70}")
        
        print("\nNext steps:")
        print("1. Save polygon_bridge.py to modules/data/ for full functionality")
        print("2. Use this validated connection in your trading systems")
        print("3. Set up real-time monitoring with WebSocket feeds")
        
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check your POLYGON_API_KEY in .env file")
        print("2. Verify internet connection")
        print("3. Ensure you're in the Vega project directory")
        print("4. Check that polygon package is installed")
        
        # More detailed error info
        import traceback
        print("\nDetailed error trace:")
        traceback.print_exc()
    
    print("\n=== Polygon Bridge Integration Test Complete ===")