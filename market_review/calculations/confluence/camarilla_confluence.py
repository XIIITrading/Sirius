# market_review/calculations/confluence/camarilla_confluence.py
"""
Module: Camarilla Confluence Calculator
Purpose: Identify confluence zones where multiple timeframe Camarilla pivots align
Features: Zone detection, strength analysis, distance calculation, support/resistance classification
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
import numpy as np

from market_review.calculations.pivots.camarilla_engine import CamarillaResult, CamarillaPivot


@dataclass
class CamarillaConfluenceZone:
    """Represents a confluence zone where multiple Camarilla pivots align"""
    zone_id: int
    center_price: float
    zone_high: float
    zone_low: float
    zone_width: float
    zone_type: str  # 'resistance', 'support', or 'mixed'
    timeframes: List[str]  # ['daily', 'weekly', 'monthly']
    pivots: List[Tuple[str, str, float, int]]  # (timeframe, level_name, price, strength)
    total_strength: float
    average_strength: float
    distance_from_current: float
    distance_percentage: float
    strength_classification: str  # 'Strong', 'Moderate', 'Weak'
    strength_score: float  # Numerical strength score
    level_names: Set[str]  # Unique level names in zone (e.g., {'R4', 'R3'})
    
    def contains_price(self, price: float) -> bool:
        """Check if a price is within this zone"""
        return self.zone_low <= price <= self.zone_high
    
    def is_resistance(self) -> bool:
        """Check if this is a resistance zone"""
        return self.zone_type in ['resistance', 'mixed']
    
    def is_support(self) -> bool:
        """Check if this is a support zone"""
        return self.zone_type in ['support', 'mixed']


@dataclass
class CamarillaConfluenceAnalysis:
    """Complete Camarilla confluence analysis results"""
    current_price: float
    analysis_time: datetime
    zones: List[CamarillaConfluenceZone]
    resistance_zones: List[CamarillaConfluenceZone]
    support_zones: List[CamarillaConfluenceZone]
    total_zones_found: int
    strongest_zone: Optional[CamarillaConfluenceZone]
    nearest_resistance: Optional[CamarillaConfluenceZone]
    nearest_support: Optional[CamarillaConfluenceZone]
    price_in_zone: Optional[CamarillaConfluenceZone]
    
    def get_zones_by_distance(self, max_distance_pct: float = 5.0) -> List[CamarillaConfluenceZone]:
        """Get zones within a certain percentage distance from current price"""
        return [z for z in self.zones if z.distance_percentage <= max_distance_pct]
    
    def get_zones_by_strength(self, min_strength: str = 'Moderate') -> List[CamarillaConfluenceZone]:
        """Get zones meeting minimum strength criteria"""
        strength_order = {'Weak': 1, 'Moderate': 2, 'Strong': 3}
        min_level = strength_order.get(min_strength, 2)
        return [z for z in self.zones if strength_order.get(z.strength_classification, 0) >= min_level]
    
    def get_nearest_levels(self) -> Dict[str, Optional[float]]:
        """Get nearest resistance and support levels"""
        return {
            'resistance': self.nearest_resistance.center_price if self.nearest_resistance else None,
            'support': self.nearest_support.center_price if self.nearest_support else None
        }


class CamarillaConfluenceCalculator:
    """
    Calculates confluence zones from multi-timeframe Camarilla analysis results.
    
    Confluence zones are areas where pivots from multiple timeframes align,
    suggesting stronger support/resistance levels.
    """
    
    def __init__(self,
                 confluence_threshold_percent: float = 0.3,
                 min_pivots_for_zone: int = 2,
                 max_pivots_per_timeframe: int = 6,
                 timeframe_weights: Optional[Dict[str, float]] = None):
        """
        Initialize Camarilla confluence calculator.
        
        Args:
            confluence_threshold_percent: Max price difference as % to group pivots
            min_pivots_for_zone: Minimum pivots required to form a zone
            max_pivots_per_timeframe: Max pivots to consider from each timeframe
            timeframe_weights: Weight multipliers for each timeframe
        """
        self.confluence_threshold_percent = confluence_threshold_percent
        self.min_pivots_for_zone = min_pivots_for_zone
        self.max_pivots_per_timeframe = max_pivots_per_timeframe
        
        # Default weights - daily most important for intraday
        self.timeframe_weights = timeframe_weights or {
            'daily': 1.2,
            'weekly': 1.0,
            'monthly': 0.8
        }
        
    def calculate(self,
                  results: Dict[str, CamarillaResult],
                  current_price: float,
                  max_zones: int = 15) -> CamarillaConfluenceAnalysis:
        """
        Calculate confluence zones from multi-timeframe Camarilla results.
        
        Args:
            results: Dictionary of timeframe -> CamarillaResult
            current_price: Current market price
            max_zones: Maximum number of zones to return
            
        Returns:
            CamarillaConfluenceAnalysis with identified zones
        """
        # Collect all pivots with metadata
        all_pivots = self._collect_pivots(results)
        
        if not all_pivots:
            return self._empty_analysis(current_price)
        
        # Find confluence zones
        zones = self._identify_zones(all_pivots, current_price)
        
        # Calculate zone metrics
        zones = self._calculate_zone_metrics(zones, current_price)
        
        # Classify zones by type
        resistance_zones = [z for z in zones if z.zone_type == 'resistance']
        support_zones = [z for z in zones if z.zone_type == 'support']
        
        # Sort by distance
        zones.sort(key=lambda z: z.distance_from_current)
        resistance_zones.sort(key=lambda z: z.center_price)
        support_zones.sort(key=lambda z: z.center_price, reverse=True)
        
        # Limit to max_zones
        zones = zones[:max_zones]
        
        # Assign zone IDs
        for i, zone in enumerate(zones):
            zone.zone_id = i + 1
        
        # Create analysis summary
        return self._create_analysis(zones, resistance_zones, support_zones, current_price)
    
    def _collect_pivots(self, results: Dict[str, CamarillaResult]) -> List[Tuple[str, str, float, int]]:
        """Collect all pivots from results with metadata"""
        all_pivots = []
        
        for timeframe, result in results.items():
            # Take pivots from each timeframe
            for pivot in result.pivots[:self.max_pivots_per_timeframe]:
                all_pivots.append((
                    timeframe,
                    pivot.level_name,
                    pivot.price,
                    pivot.strength
                ))
                
        return all_pivots
    
    def _identify_zones(self, 
                       all_pivots: List[Tuple[str, str, float, int]], 
                       current_price: float) -> List[CamarillaConfluenceZone]:
        """Identify confluence zones from pivots"""
        zones = []
        used_pivots = set()
        confluence_threshold = current_price * (self.confluence_threshold_percent / 100)
        
        # Sort pivots by price for efficient grouping
        sorted_pivots = sorted(enumerate(all_pivots), key=lambda x: x[1][2])
        
        for i, (idx1, (tf1, name1, price1, strength1)) in enumerate(sorted_pivots):
            if idx1 in used_pivots:
                continue
                
            # Start new zone
            zone_pivots = [(tf1, name1, price1, strength1)]
            zone_indices = {idx1}
            
            # Look for nearby pivots
            for j, (idx2, (tf2, name2, price2, strength2)) in enumerate(sorted_pivots[i+1:], i+1):
                if idx2 in used_pivots:
                    continue
                    
                # Check if within threshold of any pivot in current zone
                if any(abs(price2 - p[2]) <= confluence_threshold for p in zone_pivots):
                    zone_pivots.append((tf2, name2, price2, strength2))
                    zone_indices.add(idx2)
                else:
                    # Pivots are sorted, so if we're too far, stop looking
                    if price2 - price1 > confluence_threshold:
                        break
            
            # Create zone if it meets criteria
            if self._is_valid_zone(zone_pivots):
                used_pivots.update(zone_indices)
                zone = self._create_zone(zone_pivots, current_price)
                zones.append(zone)
                
        return zones
    
    def _is_valid_zone(self, pivots: List[Tuple[str, str, float, int]]) -> bool:
        """Check if pivots form a valid confluence zone"""
        if len(pivots) < self.min_pivots_for_zone:
            return False
            
        # Check for multiple timeframes or high strength
        unique_timeframes = len(set(p[0] for p in pivots))
        total_strength = sum(p[3] for p in pivots)
        
        # Valid if multiple timeframes or high combined strength
        return unique_timeframes > 1 or (len(pivots) >= 2 and total_strength >= 10)
    
    def _create_zone(self, 
                    pivots: List[Tuple[str, str, float, int]], 
                    current_price: float) -> CamarillaConfluenceZone:
        """Create a confluence zone from grouped pivots"""
        prices = [p[2] for p in pivots]
        strengths = [p[3] for p in pivots]
        timeframes = list(set(p[0] for p in pivots))
        level_names = set(p[1] for p in pivots)
        
        # Calculate weighted center based on strength
        weighted_sum = sum(price * strength for price, strength in zip(prices, strengths))
        total_strength = sum(strengths)
        center_price = weighted_sum / total_strength if total_strength > 0 else np.mean(prices)
        
        # Determine zone type
        resistance_levels = {'R3', 'R4', 'R6'}
        support_levels = {'S3', 'S4', 'S6'}
        
        has_resistance = bool(level_names & resistance_levels)
        has_support = bool(level_names & support_levels)
        
        if has_resistance and has_support:
            zone_type = 'mixed'
        elif has_resistance:
            zone_type = 'resistance'
        else:
            zone_type = 'support'
        
        zone = CamarillaConfluenceZone(
            zone_id=0,  # Will be assigned later
            center_price=center_price,
            zone_high=max(prices),
            zone_low=min(prices),
            zone_width=max(prices) - min(prices),
            zone_type=zone_type,
            timeframes=sorted(timeframes),
            pivots=sorted(pivots, key=lambda x: (x[0], x[3]), reverse=True),  # Sort by timeframe then strength
            total_strength=float(total_strength),
            average_strength=total_strength / len(pivots),
            distance_from_current=abs(center_price - current_price),
            distance_percentage=0,  # Will be calculated
            strength_classification='',  # Will be calculated
            strength_score=0,  # Will be calculated
            level_names=level_names
        )
        
        return zone
    
    def _calculate_zone_metrics(self, 
                               zones: List[CamarillaConfluenceZone], 
                               current_price: float) -> List[CamarillaConfluenceZone]:
        """Calculate additional metrics for each zone"""
        for zone in zones:
            # Distance percentage
            zone.distance_percentage = (zone.distance_from_current / current_price) * 100
            
            # Strength calculation
            zone.strength_score = self._calculate_strength_score(zone)
            zone.strength_classification = self._classify_strength(zone.strength_score)
            
        return zones
    
    def _calculate_strength_score(self, zone: CamarillaConfluenceZone) -> float:
        """Calculate numerical strength score for a zone"""
        score = 0.0
        
        # Factor 1: Number of timeframes (0-4 points)
        timeframe_score = min(len(zone.timeframes) * 1.5, 4.0)
        score += timeframe_score
        
        # Factor 2: Total strength (0-4 points)
        strength_score = min(zone.total_strength / 5.0, 1.0) * 4.0
        score += strength_score
        
        # Factor 3: Zone tightness (0-2 points)
        tightness_score = max(0, 2.0 - (zone.zone_width / zone.center_price * 200))
        score += tightness_score
        
        # Factor 4: Level type bonus
        if 'R6' in zone.level_names or 'S6' in zone.level_names:
            score += 1.0  # Extreme levels bonus
        if 'R4' in zone.level_names or 'S4' in zone.level_names:
            score += 0.5  # Breakout levels bonus
        
        # Factor 5: Timeframe weights
        weight_multiplier = sum(self.timeframe_weights.get(tf, 1.0) for tf in zone.timeframes)
        weight_multiplier /= len(zone.timeframes)  # Average weight
        
        return score * weight_multiplier
    
    def _classify_strength(self, score: float) -> str:
        """Classify zone strength based on score"""
        if score >= 8.0:
            return 'Strong'
        elif score >= 5.0:
            return 'Moderate'
        else:
            return 'Weak'
    
    def _create_analysis(self, 
                        zones: List[CamarillaConfluenceZone],
                        resistance_zones: List[CamarillaConfluenceZone],
                        support_zones: List[CamarillaConfluenceZone],
                        current_price: float) -> CamarillaConfluenceAnalysis:
        """Create complete analysis summary"""
        # Find special zones
        strongest_zone = max(zones, key=lambda z: z.strength_score) if zones else None
        
        # Nearest resistance (above current price)
        nearest_resistance = next(
            (z for z in resistance_zones if z.center_price > current_price), 
            None
        )
        
        # Nearest support (below current price)
        nearest_support = next(
            (z for z in reversed(support_zones) if z.center_price < current_price), 
            None
        )
        
        # Check if price is in a zone
        price_in_zone = next((z for z in zones if z.contains_price(current_price)), None)
        
        return CamarillaConfluenceAnalysis(
            current_price=current_price,
            analysis_time=datetime.now(),
            zones=zones,
            resistance_zones=[z for z in zones if z.zone_type in ['resistance', 'mixed']],
            support_zones=[z for z in zones if z.zone_type in ['support', 'mixed']],
            total_zones_found=len(zones),
            strongest_zone=strongest_zone,
            nearest_resistance=nearest_resistance,
            nearest_support=nearest_support,
            price_in_zone=price_in_zone
        )
    
    def _empty_analysis(self, current_price: float) -> CamarillaConfluenceAnalysis:
        """Return empty analysis when no pivots found"""
        return CamarillaConfluenceAnalysis(
            current_price=current_price,
            analysis_time=datetime.now(),
            zones=[],
            resistance_zones=[],
            support_zones=[],
            total_zones_found=0,
            strongest_zone=None,
            nearest_resistance=None,
            nearest_support=None,
            price_in_zone=None
        )
    
    def format_zone_summary(self, zone: CamarillaConfluenceZone, current_price: float) -> str:
        """Format a zone for display"""
        direction = "above" if zone.center_price > current_price else "below"
        
        summary = f"Zone #{zone.zone_id} - {zone.zone_type.title()} - {zone.strength_classification}\n"
        summary += f"  Center: ${zone.center_price:.2f} ({zone.distance_percentage:.2f}% {direction})\n"
        summary += f"  Range: ${zone.zone_low:.2f} - ${zone.zone_high:.2f} (width: ${zone.zone_width:.2f})\n"
        summary += f"  Timeframes: {', '.join(zone.timeframes)}\n"
        summary += f"  Combined Strength: {zone.total_strength:.1f}\n"
        summary += f"  Levels: {', '.join(sorted(zone.level_names))}\n"
        summary += f"  Pivots:\n"
        
        for tf, name, price, strength in zone.pivots:
            summary += f"    - {tf} {name}: ${price:.2f} (strength: {strength})\n"
            
        return summary
    
    def get_zone_summary_dict(self, zone: CamarillaConfluenceZone) -> dict:
        """Get zone summary as dictionary for easy access"""
        return {
            'zone_id': zone.zone_id,
            'type': zone.zone_type,
            'strength': zone.strength_classification,
            'center': round(zone.center_price, 2),
            'range': (round(zone.zone_low, 2), round(zone.zone_high, 2)),
            'width': round(zone.zone_width, 2),
            'distance_pct': round(zone.distance_percentage, 2),
            'timeframes': zone.timeframes,
            'levels': sorted(list(zone.level_names)),
            'total_strength': round(zone.total_strength, 1)
        }


# Example usage
if __name__ == "__main__":
    from market_review.calculations.pivots.camarilla_engine import CamarillaResult, CamarillaPivot
    
    # Mock some results for testing
    mock_results = {
        'daily': CamarillaResult(
            timeframe='daily',
            close=220.0,
            high=222.0,
            low=218.0,
            pivots=[
                CamarillaPivot(level_name='R6', price=224.04, strength=6, timeframe='daily'),
                CamarillaPivot(level_name='R4', price=222.20, strength=4, timeframe='daily'),
                CamarillaPivot(level_name='R3', price=221.10, strength=3, timeframe='daily'),
                CamarillaPivot(level_name='S3', price=218.90, strength=3, timeframe='daily'),
                CamarillaPivot(level_name='S4', price=217.80, strength=4, timeframe='daily'),
                CamarillaPivot(level_name='S6', price=215.96, strength=6, timeframe='daily'),
            ],
            range_type='higher',
            central_pivot=220.0
        ),
        'weekly': CamarillaResult(
            timeframe='weekly',
            close=219.0,
            high=225.0,
            low=215.0,
            pivots=[
                CamarillaPivot(level_name='R6', price=225.84, strength=6, timeframe='weekly'),
                CamarillaPivot(level_name='R4', price=224.50, strength=4, timeframe='weekly'),
                CamarillaPivot(level_name='R3', price=221.75, strength=3, timeframe='weekly'),
                CamarillaPivot(level_name='S3', price=216.25, strength=3, timeframe='weekly'),
                CamarillaPivot(level_name='S4', price=213.50, strength=4, timeframe='weekly'),
                CamarillaPivot(level_name='S6', price=212.16, strength=6, timeframe='weekly'),
            ],
            range_type='neutral',
            central_pivot=219.0
        ),
        'monthly': CamarillaResult(
            timeframe='monthly',
            close=215.0,
            high=230.0,
            low=210.0,
            pivots=[
                CamarillaPivot(level_name='R6', price=224.29, strength=6, timeframe='monthly'),
                CamarillaPivot(level_name='R4', price=226.00, strength=4, timeframe='monthly'),
                CamarillaPivot(level_name='R3', price=220.50, strength=3, timeframe='monthly'),
                CamarillaPivot(level_name='S3', price=209.50, strength=3, timeframe='monthly'),
                CamarillaPivot(level_name='S4', price=204.00, strength=4, timeframe='monthly'),
                CamarillaPivot(level_name='S6', price=205.71, strength=6, timeframe='monthly'),
            ],
            range_type='lower',
            central_pivot=215.0
        )
    }
    
    # Calculate confluence
    calculator = CamarillaConfluenceCalculator()
    analysis = calculator.calculate(mock_results, current_price=220.5)
    
    # Print results
    print(f"Current Price: ${analysis.current_price:.2f}")
    print(f"Total Zones Found: {analysis.total_zones_found}")
    print(f"\nNearest Levels:")
    levels = analysis.get_nearest_levels()
    print(f"  Resistance: ${levels['resistance']:.2f}" if levels['resistance'] else "  Resistance: None")
    print(f"  Support: ${levels['support']:.2f}" if levels['support'] else "  Support: None")
    
    print(f"\nTop Confluence Zones:")
    for zone in analysis.zones[:5]:
        print(f"\n{calculator.format_zone_summary(zone, analysis.current_price)}")