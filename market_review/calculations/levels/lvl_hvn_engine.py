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
                 prominence_threshold: float = 0.5,
                 min_peak_distance: int = 3):
        """
        Initialize HVN Engine.
        
        Args:
            levels: Number of price levels for volume profile
            prominence_threshold: Minimum prominence as % of max volume
            min_peak_distance: Minimum distance between peaks (in levels)
        """
        self.levels = levels
        self.prominence_threshold = prominence_threshold
        self.min_peak_distance = min_peak_distance
        self.volume_profile = VolumeProfile(levels=levels)
        
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