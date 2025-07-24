# market_review/calculations/volume/hvn_engine.py
"""
Module: HVN (High Volume Node) Peak Detection Engine
Purpose: Identify volume peaks in price profiles across multiple timeframes
Performance Target: Complete multi-timeframe analysis in <3 seconds
"""

# Standard library imports
import os
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Third-party imports
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# Local application imports
from market_review.calculations.volume.volume_profile import VolumeProfile, PriceLevel


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


@dataclass
class VolumePeak:
    """Single volume peak information"""
    price: float
    rank: int  # 1 = highest volume peak within timeframe
    volume_percent: float
    level_index: int  # Original level index in volume profile


@dataclass
class TimeframeResult:
    """HVN analysis result for a single timeframe"""
    timeframe_days: int
    price_range: Tuple[float, float]  # High/Low of timeframe
    total_levels: int
    peaks: List[VolumePeak]  # Sorted by rank (volume)
    data_points: int  # Number of bars analyzed


class HVNEngine:
    """
    HVN Peak Detection Engine.
    Identifies absolute peaks in volume profiles across multiple timeframes.
    """
    
    def __init__(self, 
                 levels: int = 100,
                 percentile_threshold: float = 80.0,
                 prominence_threshold: float = 0.5,
                 min_peak_distance: int = 3,
                 proximity_atr_minutes: int = 30):
        """
        Initialize HVN Engine.
        
        Args:
            levels: Number of price levels for volume profile
            percentile_threshold: Percentile threshold for HVN identification
            prominence_threshold: Minimum prominence as % of max volume
            min_peak_distance: Minimum distance between peaks (in levels)
            proximity_atr_minutes: ATR in minutes for proximity alerts
        """
        self.levels = levels
        self.percentile_threshold = percentile_threshold
        self.prominence_threshold = prominence_threshold
        self.min_peak_distance = min_peak_distance
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
    
    def identify_volume_peaks(self, 
                            levels: List[PriceLevel], 
                            percentile_filter: float = 70.0) -> List[PriceLevel]:
        """
        Identify local peaks in the volume profile.
        
        Args:
            levels: All price levels from volume profile
            percentile_filter: Only consider levels above this percentile
            
        Returns:
            List of PriceLevel objects that are peaks
        """
        if not levels:
            return []
        
        # Sort levels by price
        sorted_levels = sorted(levels, key=lambda x: x.center)
        volumes = np.array([level.percent_of_total for level in sorted_levels])
        
        # Calculate thresholds
        max_volume = np.max(volumes)
        min_prominence = max_volume * self.prominence_threshold / 100
        height_threshold = np.percentile(volumes, percentile_filter)
        
        # Find peaks
        peak_indices, properties = find_peaks(
            volumes,
            prominence=min_prominence,
            distance=self.min_peak_distance,
            height=height_threshold
        )
        
        # Extract peak levels
        peak_levels = [sorted_levels[i] for i in peak_indices]
        
        return peak_levels
    
    def analyze_timeframe(self, 
                         data: pd.DataFrame,
                         timeframe_days: int,
                         include_pre: bool = True,
                         include_post: bool = True) -> TimeframeResult:
        """
        Run HVN peak analysis for a single timeframe.
        
        Args:
            data: Complete OHLCV DataFrame
            timeframe_days: Number of days to analyze
            include_pre: Include pre-market data
            include_post: Include post-market data
            
        Returns:
            TimeframeResult with detected peaks
        """
        # Filter data for timeframe
        current_date = data.index[-1]
        start_date = current_date - timedelta(days=timeframe_days)
        timeframe_data = data[data.index >= start_date].copy()
        
        # Build volume profile
        profile_levels = self.volume_profile.build_volume_profile(
            timeframe_data, include_pre, include_post
        )
        
        if not profile_levels:
            return TimeframeResult(
                timeframe_days=timeframe_days,
                price_range=(0, 0),
                total_levels=0,
                peaks=[],
                data_points=len(timeframe_data)
            )
        
        # Identify peaks
        peak_levels = self.identify_volume_peaks(profile_levels)
        
        # Sort peaks by volume and create VolumePeak objects
        sorted_peaks = sorted(peak_levels, key=lambda x: x.percent_of_total, reverse=True)
        volume_peaks = [
            VolumePeak(
                price=peak.center,
                rank=idx + 1,
                volume_percent=peak.percent_of_total,
                level_index=peak.index
            )
            for idx, peak in enumerate(sorted_peaks)
        ]
        
        return TimeframeResult(
            timeframe_days=timeframe_days,
            price_range=self.volume_profile.price_range,
            total_levels=len(profile_levels),
            peaks=volume_peaks,
            data_points=len(timeframe_data)
        )
    
    def analyze_multi_timeframe(self, 
                               data: pd.DataFrame,
                               timeframes: List[int] = [120, 60, 15],
                               include_pre: bool = True,
                               include_post: bool = True) -> Dict[int, TimeframeResult]:
        """
        Run HVN analysis for multiple timeframes.
        
        Args:
            data: Complete OHLCV DataFrame
            timeframes: List of lookback days
            include_pre: Include pre-market data
            include_post: Include post-market data
            
        Returns:
            Dictionary mapping timeframe to TimeframeResult
        """
        results = {}
        
        for days in timeframes:
            results[days] = self.analyze_timeframe(
                data, days, include_pre, include_post
            )
        
        return results
    
    def get_all_peaks_dataframe(self, results: Dict[int, TimeframeResult]) -> pd.DataFrame:
        """
        Convert results to a clean DataFrame for easy access.
        
        Returns DataFrame with columns:
            - timeframe: 120, 60, or 15
            - price: Peak price
            - rank: Rank within timeframe
            - volume_pct: Volume percentage
        """
        rows = []
        
        for days, result in results.items():
            for peak in result.peaks:
                rows.append({
                    'timeframe': days,
                    'price': peak.price,
                    'rank': peak.rank,
                    'volume_pct': peak.volume_percent
                })
        
        return pd.DataFrame(rows)
    
    def get_peaks_summary(self, results: Dict[int, TimeframeResult]) -> Dict:
        """
        Get a clean summary of peaks grouped by timeframe.
        
        Returns:
            {
                120: [{'price': 456.25, 'rank': 1, 'volume_pct': 8.5}, ...],
                60: [{'price': 465.50, 'rank': 1, 'volume_pct': 9.2}, ...],
                15: [{'price': 464.25, 'rank': 1, 'volume_pct': 11.5}, ...]
            }
        """
        summary = {}
        
        for days, result in results.items():
            summary[days] = [
                {
                    'price': peak.price,
                    'rank': peak.rank,
                    'volume_pct': round(peak.volume_percent, 2)
                }
                for peak in result.peaks
            ]
        
        return summary
    
    def print_results(self, results: Dict[int, TimeframeResult], symbol: str = ""):
        """
        Pretty print the results to console.
        """
        print(f"\n{'='*60}")
        print(f"HVN PEAK ANALYSIS{f' - {symbol}' if symbol else ''}")
        print(f"{'='*60}")
        
        for days in sorted(results.keys(), reverse=True):
            result = results[days]
            print(f"\n📊 {days}-Day Timeframe")
            print(f"   Price Range: ${result.price_range[0]:.2f} - ${result.price_range[1]:.2f}")
            print(f"   Data Points: {result.data_points:,} bars")
            print(f"   Total Peaks: {len(result.peaks)}")
            
            if result.peaks:
                print(f"\n   Top Volume Peaks:")
                print(f"   {'Rank':<6} {'Price':<10} {'Volume %'}")
                print(f"   {'-'*6} {'-'*10} {'-'*10}")
                
                for peak in result.peaks[:5]:  # Show top 5
                    print(f"   #{peak.rank:<5} ${peak.price:<9.2f} {peak.volume_percent:.2f}%")