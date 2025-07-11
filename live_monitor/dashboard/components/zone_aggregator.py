# market_review/dashboards/components/zone_aggregator.py
"""
Zone Aggregator - Combines overlapping zones from different sources
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class UnifiedZone:
    """Unified zone representation"""
    zone_id: str
    price_low: float
    price_high: float
    center_price: float
    strength: float
    source_type: str  # 'hvn', 'supply_demand', 'combined'
    sources: List[str]  # List of source identifiers
    zone_name: str
    display_color: str
    display_style: str  # 'solid', 'dashed'
    opacity: int  # 0-255


class ZoneAggregator:
    """Aggregates overlapping zones from multiple sources"""
    
    def __init__(self, overlap_threshold: float = 0.1):
        """
        Initialize aggregator
        
        Args:
            overlap_threshold: Minimum overlap percentage to merge zones (0.1 = 10%)
        """
        self.overlap_threshold = overlap_threshold
        
    def aggregate_zones(self, 
                       hvn_result: Optional[Dict] = None,
                       supply_demand_result: Optional[Dict] = None) -> List[UnifiedZone]:
        """
        Aggregate zones from different sources
        
        Returns:
            List of unified zones with overlap detection
        """
        zones = []
        zone_counter = 0
        
        # Process HVN zones
        if hvn_result and hasattr(hvn_result, 'clusters'):
            for i, cluster in enumerate(hvn_result.clusters):
                zone_counter += 1
                zones.append(UnifiedZone(
                    zone_id=f"hvn_{zone_counter}",
                    price_low=cluster.cluster_low,
                    price_high=cluster.cluster_high,
                    center_price=cluster.center_price,
                    strength=cluster.total_percent,
                    source_type='hvn',
                    sources=[f'hvn_cluster_{i}'],
                    zone_name=f"HVN {i+1}",
                    display_color='#10b981',
                    display_style='solid',
                    opacity=80
                ))
        
        # Process Supply/Demand zones
        if supply_demand_result and 'zones' in supply_demand_result:
            for i, zone in enumerate(supply_demand_result['zones']):
                zone_counter += 1
                color = '#ef4444' if zone.zone_type == 'supply' else '#10b981'
                zones.append(UnifiedZone(
                    zone_id=f"sd_{zone_counter}",
                    price_low=zone.price_low,
                    price_high=zone.price_high,
                    center_price=zone.center_price,
                    strength=zone.strength,
                    source_type='supply_demand',
                    sources=[f'{zone.zone_type}_{i}'],
                    zone_name=f"{zone.zone_type.capitalize()} {i+1}",
                    display_color=color,
                    display_style='dashed' if not zone.validated else 'solid',
                    opacity=60
                ))
        
        # Merge overlapping zones
        merged_zones = self._merge_overlapping_zones(zones)
        
        return merged_zones
    
    def _merge_overlapping_zones(self, zones: List[UnifiedZone]) -> List[UnifiedZone]:
        """Merge zones that overlap significantly"""
        if not zones:
            return []
            
        # Sort zones by price_low
        zones.sort(key=lambda z: z.price_low)
        
        merged = []
        current_group = [zones[0]]
        
        for zone in zones[1:]:
            # Check overlap with current group
            group_low = min(z.price_low for z in current_group)
            group_high = max(z.price_high for z in current_group)
            
            overlap = self._calculate_overlap(
                group_low, group_high,
                zone.price_low, zone.price_high
            )
            
            if overlap >= self.overlap_threshold:
                current_group.append(zone)
            else:
                # Finalize current group
                merged.append(self._merge_zone_group(current_group))
                current_group = [zone]
        
        # Don't forget the last group
        if current_group:
            merged.append(self._merge_zone_group(current_group))
            
        return merged
    
    def _calculate_overlap(self, low1: float, high1: float, 
                          low2: float, high2: float) -> float:
        """Calculate overlap percentage between two zones"""
        overlap_low = max(low1, low2)
        overlap_high = min(high1, high2)
        
        if overlap_high <= overlap_low:
            return 0.0
            
        overlap_size = overlap_high - overlap_low
        zone1_size = high1 - low1
        zone2_size = high2 - low2
        min_size = min(zone1_size, zone2_size)
        
        return overlap_size / min_size if min_size > 0 else 0.0
    
    def _merge_zone_group(self, zones: List[UnifiedZone]) -> UnifiedZone:
        """Merge a group of overlapping zones"""
        if len(zones) == 1:
            return zones[0]
            
        # Calculate merged properties
        price_low = min(z.price_low for z in zones)
        price_high = max(z.price_high for z in zones)
        
        # Weighted average for center and strength
        total_strength = sum(z.strength for z in zones)
        if total_strength > 0:
            center_price = sum(z.center_price * z.strength for z in zones) / total_strength
            avg_strength = total_strength / len(zones)
        else:
            center_price = (price_low + price_high) / 2
            avg_strength = sum(z.strength for z in zones) / len(zones)
            
        # Combine sources
        all_sources = []
        source_types = set()
        for z in zones:
            all_sources.extend(z.sources)
            source_types.add(z.source_type)
            
        # Determine display properties
        if len(source_types) > 1:
            display_color = '#f59e0b'  # Orange for combined
            zone_name = f"Combined Zone ({len(zones)} sources)"
            source_type = 'combined'
        else:
            # Use properties from strongest zone
            strongest = max(zones, key=lambda z: z.strength)
            display_color = strongest.display_color
            zone_name = strongest.zone_name + f" (+{len(zones)-1})"
            source_type = strongest.source_type
            
        return UnifiedZone(
            zone_id=f"merged_{id(zones)}",
            price_low=price_low,
            price_high=price_high,
            center_price=center_price,
            strength=avg_strength,
            source_type=source_type,
            sources=all_sources,
            zone_name=zone_name,
            display_color=display_color,
            display_style='solid',
            opacity=100
        )
    
    def get_zones_near_price(self, zones: List[UnifiedZone], 
                            current_price: float,
                            distance_percent: float = 0.03) -> List[UnifiedZone]:
        """Get zones within a certain percentage of current price"""
        nearby = []
        threshold = current_price * distance_percent
        
        for zone in zones:
            # Check if price is inside zone
            if zone.price_low <= current_price <= zone.price_high:
                nearby.append(zone)
                continue
                
            # Check distance to zone boundaries
            distance_to_zone = min(
                abs(current_price - zone.price_low),
                abs(current_price - zone.price_high)
            )
            
            if distance_to_zone <= threshold:
                nearby.append(zone)
                
        return sorted(nearby, key=lambda z: z.strength, reverse=True)