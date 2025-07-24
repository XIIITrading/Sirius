# market_review/calculations/confluence/hvn_confluence.py
"""
Module: HVN Confluence Calculator
Purpose: Identify confluence zones where multiple timeframe peaks align
Features: Zone detection, strength analysis, distance calculation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np

from market_review.calculations.volume.hvn_engine import TimeframeResult


@dataclass
class ConfluenceZone:
    """Represents a confluence zone where multiple timeframe peaks align"""
    zone_id: int
    center_price: float
    zone_high: float
    zone_low: float
    zone_width: float
    timeframes: List[int]
    peaks: List[Tuple[int, float, float]]  # (timeframe, price, volume_pct)
    total_volume_weight: float
    average_volume: float
    distance_from_current: float
    distance_percentage: float
    strength: str  # 'Strong', 'Moderate', 'Weak'
    strength_score: float  # Numerical strength score
    
    def contains_price(self, price: float) -> bool:
        """Check if a price is within this zone"""
        return self.zone_low <= price <= self.zone_high


@dataclass
class ConfluenceAnalysis:
    """Complete confluence analysis results"""
    current_price: float
    analysis_time: datetime
    zones: List[ConfluenceZone]
    total_zones_found: int
    strongest_zone: Optional[ConfluenceZone]
    nearest_zone: Optional[ConfluenceZone]
    price_in_zone: Optional[ConfluenceZone]  # Zone containing current price
    
    def get_zones_by_distance(self, max_distance_pct: float = 5.0) -> List[ConfluenceZone]:
        """Get zones within a certain percentage distance from current price"""
        return [z for z in self.zones if z.distance_percentage <= max_distance_pct]
    
    def get_zones_by_strength(self, min_strength: str = 'Moderate') -> List[ConfluenceZone]:
        """Get zones meeting minimum strength criteria"""
        strength_order = {'Weak': 1, 'Moderate': 2, 'Strong': 3}
        min_level = strength_order.get(min_strength, 2)
        return [z for z in self.zones if strength_order.get(z.strength, 0) >= min_level]


class HVNConfluenceCalculator:
    """
    Calculates confluence zones from multi-timeframe HVN analysis results.
    
    Confluence zones are areas where peaks from multiple timeframes align,
    suggesting stronger support/resistance levels.
    """
    
    def __init__(self,
                 confluence_threshold_percent: float = 0.5,
                 min_peaks_for_zone: int = 2,
                 max_peaks_per_timeframe: int = 10,
                 strength_weights: Optional[Dict[int, float]] = None):
        """
        Initialize confluence calculator.
        
        Args:
            confluence_threshold_percent: Max price difference as % to group peaks
            min_peaks_for_zone: Minimum peaks required to form a zone
            max_peaks_per_timeframe: Max peaks to consider from each timeframe
            strength_weights: Weight multipliers for each timeframe
        """
        self.confluence_threshold_percent = confluence_threshold_percent
        self.min_peaks_for_zone = min_peaks_for_zone
        self.max_peaks_per_timeframe = max_peaks_per_timeframe
        
        # Default weights favor shorter timeframes slightly for intraday
        self.strength_weights = strength_weights or {
            120: 0.8,
            60: 1.0,
            15: 1.2
        }
        
    def calculate(self,
                  results: Dict[int, TimeframeResult],
                  current_price: float,
                  max_zones: int = 10) -> ConfluenceAnalysis:
        """
        Calculate confluence zones from multi-timeframe results.
        
        Args:
            results: Dictionary of timeframe -> TimeframeResult
            current_price: Current market price
            max_zones: Maximum number of zones to return
            
        Returns:
            ConfluenceAnalysis with identified zones
        """
        # Collect all peaks with metadata
        all_peaks = self._collect_peaks(results)
        
        if not all_peaks:
            return self._empty_analysis(current_price)
        
        # Find confluence zones
        zones = self._identify_zones(all_peaks, current_price)
        
        # Calculate zone metrics
        zones = self._calculate_zone_metrics(zones, current_price)
        
        # Sort by distance and limit to max_zones
        zones.sort(key=lambda z: z.distance_from_current)
        zones = zones[:max_zones]
        
        # Assign zone IDs
        for i, zone in enumerate(zones):
            zone.zone_id = i + 1
        
        # Create analysis summary
        return self._create_analysis(zones, current_price)
        
    def _collect_peaks(self, results: Dict[int, TimeframeResult]) -> List[Tuple[int, float, float]]:
        """Collect all peaks from results with timeframe info"""
        all_peaks = []
        
        for timeframe, result in results.items():
            # Take top N peaks from each timeframe
            for peak in result.peaks[:self.max_peaks_per_timeframe]:
                all_peaks.append((timeframe, peak.price, peak.volume_percent))
                
        return all_peaks
    
    def _identify_zones(self, 
                       all_peaks: List[Tuple[int, float, float]], 
                       current_price: float) -> List[ConfluenceZone]:
        """Identify confluence zones from peaks"""
        zones = []
        used_peaks = set()
        confluence_threshold = current_price * (self.confluence_threshold_percent / 100)
        
        # Sort peaks by price for efficient grouping
        sorted_peaks = sorted(enumerate(all_peaks), key=lambda x: x[1][1])
        
        for i, (idx1, (tf1, price1, vol1)) in enumerate(sorted_peaks):
            if idx1 in used_peaks:
                continue
                
            # Start new zone
            zone_peaks = [(tf1, price1, vol1)]
            zone_indices = {idx1}
            
            # Look for nearby peaks
            for j, (idx2, (tf2, price2, vol2)) in enumerate(sorted_peaks[i+1:], i+1):
                if idx2 in used_peaks:
                    continue
                    
                # Check if within threshold of any peak in current zone
                if any(abs(price2 - p[1]) <= confluence_threshold for p in zone_peaks):
                    zone_peaks.append((tf2, price2, vol2))
                    zone_indices.add(idx2)
                else:
                    # Peaks are sorted, so if we're too far, stop looking
                    if price2 - price1 > confluence_threshold:
                        break
            
            # Create zone if it meets criteria
            if self._is_valid_zone(zone_peaks):
                used_peaks.update(zone_indices)
                zone = self._create_zone(zone_peaks, current_price)
                zones.append(zone)
                
        return zones
    
    def _is_valid_zone(self, peaks: List[Tuple[int, float, float]]) -> bool:
        """Check if peaks form a valid confluence zone"""
        if len(peaks) < self.min_peaks_for_zone:
            return False
            
        # Check for multiple timeframes or significant volume
        unique_timeframes = len(set(p[0] for p in peaks))
        total_volume = sum(p[2] for p in peaks)
        
        # Valid if multiple timeframes or high volume concentration
        return unique_timeframes > 1 or (len(peaks) >= 3 and total_volume > 15.0)
    
    def _create_zone(self, 
                    peaks: List[Tuple[int, float, float]], 
                    current_price: float) -> ConfluenceZone:
        """Create a confluence zone from grouped peaks"""
        prices = [p[1] for p in peaks]
        volumes = [p[2] for p in peaks]
        timeframes = list(set(p[0] for p in peaks))
        
        # Calculate weighted center based on volume
        weighted_sum = sum(p * v for p, v in zip(prices, volumes))
        total_volume = sum(volumes)
        center_price = weighted_sum / total_volume if total_volume > 0 else np.mean(prices)
        
        zone = ConfluenceZone(
            zone_id=0,  # Will be assigned later
            center_price=center_price,
            zone_high=max(prices),
            zone_low=min(prices),
            zone_width=max(prices) - min(prices),
            timeframes=sorted(timeframes),
            peaks=sorted(peaks, key=lambda x: x[0]),  # Sort by timeframe
            total_volume_weight=total_volume,
            average_volume=total_volume / len(peaks),
            distance_from_current=abs(center_price - current_price),
            distance_percentage=0,  # Will be calculated
            strength='',  # Will be calculated
            strength_score=0  # Will be calculated
        )
        
        return zone
    
    def _calculate_zone_metrics(self, 
                               zones: List[ConfluenceZone], 
                               current_price: float) -> List[ConfluenceZone]:
        """Calculate additional metrics for each zone"""
        for zone in zones:
            # Distance percentage
            zone.distance_percentage = (zone.distance_from_current / current_price) * 100
            
            # Strength calculation
            zone.strength_score = self._calculate_strength_score(zone)
            zone.strength = self._classify_strength(zone.strength_score)
            
        return zones
    
    def _calculate_strength_score(self, zone: ConfluenceZone) -> float:
        """Calculate numerical strength score for a zone"""
        score = 0.0
        
        # Factor 1: Number of timeframes (0-3 points)
        timeframe_score = len(zone.timeframes) * 1.0
        score += timeframe_score
        
        # Factor 2: Volume weight (0-3 points)
        volume_score = min(zone.total_volume_weight / 20.0, 1.0) * 3.0
        score += volume_score
        
        # Factor 3: Zone tightness (0-2 points)
        tightness_score = max(0, 2.0 - (zone.zone_width / zone.center_price * 100))
        score += tightness_score
        
        # Factor 4: Timeframe weights
        weight_multiplier = sum(self.strength_weights.get(tf, 1.0) for tf in zone.timeframes)
        weight_multiplier /= len(zone.timeframes)  # Average weight
        
        return score * weight_multiplier
    
    def _classify_strength(self, score: float) -> str:
        """Classify zone strength based on score"""
        if score >= 6.0:
            return 'Strong'
        elif score >= 4.0:
            return 'Moderate'
        else:
            return 'Weak'
    
    def _create_analysis(self, 
                        zones: List[ConfluenceZone], 
                        current_price: float) -> ConfluenceAnalysis:
        """Create complete analysis summary"""
        # Find special zones
        strongest_zone = max(zones, key=lambda z: z.strength_score) if zones else None
        nearest_zone = min(zones, key=lambda z: z.distance_from_current) if zones else None
        price_in_zone = next((z for z in zones if z.contains_price(current_price)), None)
        
        return ConfluenceAnalysis(
            current_price=current_price,
            analysis_time=datetime.now(),
            zones=zones,
            total_zones_found=len(zones),
            strongest_zone=strongest_zone,
            nearest_zone=nearest_zone,
            price_in_zone=price_in_zone
        )
    
    def _empty_analysis(self, current_price: float) -> ConfluenceAnalysis:
        """Return empty analysis when no peaks found"""
        return ConfluenceAnalysis(
            current_price=current_price,
            analysis_time=datetime.now(),
            zones=[],
            total_zones_found=0,
            strongest_zone=None,
            nearest_zone=None,
            price_in_zone=None
        )
    
    def format_zone_summary(self, zone: ConfluenceZone, current_price: float) -> str:
        """Format a zone for display"""
        direction = "above" if zone.center_price > current_price else "below"
        
        summary = f"Zone #{zone.zone_id} - {zone.strength}\n"
        summary += f"  Center: ${zone.center_price:.2f} ({zone.distance_percentage:.2f}% {direction})\n"
        summary += f"  Range: ${zone.zone_low:.2f} - ${zone.zone_high:.2f} (width: ${zone.zone_width:.2f})\n"
        summary += f"  Timeframes: {', '.join(f'{tf}d' for tf in zone.timeframes)}\n"
        summary += f"  Combined Volume: {zone.total_volume_weight:.2f}%\n"
        summary += f"  Peaks:\n"
        
        for tf, price, vol in zone.peaks:
            summary += f"    - {tf}d: ${price:.2f} ({vol:.2f}%)\n"
            
        return summary


# Example usage
if __name__ == "__main__":
    # This would normally come from HVN analysis
    from market_review.calculations.volume.hvn_engine import VolumePeak, TimeframeResult
    
    # Mock some results for testing
    mock_results = {
        120: TimeframeResult(
            timeframe_days=120,
            price_range=(200.0, 250.0),
            total_levels=100,
            peaks=[
                VolumePeak(price=220.50, rank=1, volume_percent=3.5, level_index=0),
                VolumePeak(price=235.25, rank=2, volume_percent=2.8, level_index=0),
            ],
            data_points=1000
        ),
        60: TimeframeResult(
            timeframe_days=60,
            price_range=(210.0, 240.0),
            total_levels=100,
            peaks=[
                VolumePeak(price=220.75, rank=1, volume_percent=4.2, level_index=0),
                VolumePeak(price=230.00, rank=2, volume_percent=3.1, level_index=0),
            ],
            data_points=500
        ),
        15: TimeframeResult(
            timeframe_days=15,
            price_range=(215.0, 225.0),
            total_levels=100,
            peaks=[
                VolumePeak(price=220.25, rank=1, volume_percent=5.5, level_index=0),
                VolumePeak(price=223.50, rank=2, volume_percent=4.0, level_index=0),
            ],
            data_points=200
        )
    }
    
    # Calculate confluence
    calculator = HVNConfluenceCalculator()
    analysis = calculator.calculate(mock_results, current_price=222.0)
    
    # Print results
    print(f"Current Price: ${analysis.current_price:.2f}")
    print(f"Total Zones Found: {analysis.total_zones_found}")
    print(f"\nTop 5 Confluence Zones:")
    
    for zone in analysis.zones[:5]:
        print(f"\n{calculator.format_zone_summary(zone, analysis.current_price)}")