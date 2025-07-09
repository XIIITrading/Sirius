# market_review/dashboards/components/zone_aggregator.py
"""
Module: Zone Aggregator
Purpose: Aggregate zones from multiple sources (HVN, Supply/Demand, etc.) with overlap detection
Features:
- Hierarchical priority system
- Configurable overlap threshold
- Extensible for future zone types
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import IntEnum

import numpy as np
import pandas as pd

# Local imports
from market_review.calculations.volume.hvn_engine import HVNResult, HVNCluster
from market_review.calculations.zones.supply_demand import SupplyDemandZone

logger = logging.getLogger(__name__)


class ZonePriority(IntEnum):
    """Zone source priorities (higher number = higher priority)"""
    HVN = 100
    SUPPLY_DEMAND = 50
    # Future zone types can be added here
    # VWAP_BANDS = 40
    # FIBONACCI = 30
    # CUSTOM = 20


@dataclass
class UnifiedZone:
    """Unified zone structure for all zone types"""
    source_type: str  # 'hvn', 'supply_demand', etc.
    zone_name: str  # Descriptive name
    price_high: float
    price_low: float
    center_price: float
    strength: float  # 0-100 normalized strength
    priority: int
    
    # Visual properties
    display_color: str
    display_style: str  # 'solid', 'dashed', 'dotted'
    opacity: int  # 0-255
    
    # Original data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def height(self) -> float:
        """Zone height in price units"""
        return self.price_high - self.price_low
    
    @property
    def is_valid(self) -> bool:
        """Check if zone is valid"""
        return self.price_high > self.price_low and self.strength > 0


class ZoneAggregator:
    """
    Aggregate zones from multiple sources with overlap detection
    """
    
    # Default colors for different zone types
    DEFAULT_COLORS = {
        'hvn': {
            'high_strength': '#10b981',  # Green
            'med_strength': '#3b82f6',   # Blue
            'low_strength': '#6b7280'    # Gray
        },
        'supply': '#dc2626',  # Red
        'demand': '#16a34a',  # Green
        'breaker_supply': '#f59e0b',  # Orange
        'breaker_demand': '#fbbf24'   # Yellow
    }
    
    def __init__(self, overlap_threshold: float = 0.1):  # 10% default
        """
        Initialize Zone Aggregator
        
        Args:
            overlap_threshold: Minimum overlap percentage to consider zones as duplicates (0.1 = 10%)
        """
        self.overlap_threshold = overlap_threshold
        self.zone_sources = {}
        
        # Register default sources
        self.add_zone_source('hvn', ZonePriority.HVN)
        self.add_zone_source('supply_demand', ZonePriority.SUPPLY_DEMAND)
        
        logger.info(f"ZoneAggregator initialized with {overlap_threshold*100:.0f}% overlap threshold")
        
    def add_zone_source(self, source_name: str, priority: int):
        """Register a new zone source with priority"""
        self.zone_sources[source_name] = priority
        logger.info(f"Added zone source '{source_name}' with priority {priority}")
        
    def aggregate_zones(self,
                       hvn_result: Optional[HVNResult] = None,
                       supply_demand_result: Optional[Dict] = None,
                       future_sources: Optional[Dict[str, Any]] = None) -> List[UnifiedZone]:
        """
        Aggregate zones from all sources, removing duplicates based on priority
        
        Args:
            hvn_result: HVN calculation result
            supply_demand_result: Supply/Demand analysis result
            future_sources: Dictionary of future zone sources
            
        Returns:
            List of unified zones sorted by priority and strength
        """
        all_zones = []
        
        # Convert HVN zones
        if hvn_result and hvn_result.clusters:
            hvn_zones = self._convert_hvn_zones(hvn_result)
            all_zones.extend(hvn_zones)
            logger.info(f"Added {len(hvn_zones)} HVN zones")
            
        # Convert Supply/Demand zones
        if supply_demand_result and 'zones' in supply_demand_result:
            sd_zones = self._convert_supply_demand_zones(supply_demand_result)
            all_zones.extend(sd_zones)
            logger.info(f"Added {len(sd_zones)} Supply/Demand zones")
            
        # Handle future sources
        if future_sources:
            for source_name, source_data in future_sources.items():
                if source_name in self.zone_sources:
                    # Future implementation for other zone types
                    pass
                    
        # Remove duplicates based on overlap and priority
        filtered_zones = self._remove_duplicates(all_zones)
        
        # Sort by priority (descending) then strength (descending)
        filtered_zones.sort(key=lambda z: (z.priority, z.strength), reverse=True)
        
        logger.info(f"Aggregated {len(filtered_zones)} zones from {len(all_zones)} total")
        
        return filtered_zones
        
    def _convert_hvn_zones(self, hvn_result: HVNResult) -> List[UnifiedZone]:
        """Convert HVN clusters to unified zones"""
        zones = []
        
        for i, cluster in enumerate(hvn_result.clusters[:10]):  # Limit to top 10
            # Determine strength and color based on total percent
            if cluster.total_percent >= 10:
                strength = 90
                color = self.DEFAULT_COLORS['hvn']['high_strength']
                opacity = 60
            elif cluster.total_percent >= 5:
                strength = 70
                color = self.DEFAULT_COLORS['hvn']['med_strength']
                opacity = 30
            else:
                strength = 50
                color = self.DEFAULT_COLORS['hvn']['low_strength']
                opacity = 10
                
            zone = UnifiedZone(
                source_type='hvn',
                zone_name=f'HVN Cluster {i+1}',
                price_high=cluster.cluster_high,
                price_low=cluster.cluster_low,
                center_price=cluster.center_price,
                strength=strength,
                priority=ZonePriority.HVN,
                display_color=color,
                display_style='solid',
                opacity=opacity,
                metadata={
                    'cluster': cluster,
                    'total_volume': cluster.total_volume,
                    'total_percent': cluster.total_percent,
                    'num_levels': len(cluster.levels)
                }
            )
            
            if zone.is_valid:
                zones.append(zone)
                
        return zones
        
    def _convert_supply_demand_zones(self, sd_result: Dict) -> List[UnifiedZone]:
        """Convert Supply/Demand zones to unified zones"""
        zones = []
        
        sd_zones = sd_result.get('zones', [])
        
        for zone in sd_zones:
            # Determine color and style based on type and validation
            if zone.zone_type == 'supply':
                if zone.validated:
                    color = self.DEFAULT_COLORS['supply']
                    style = 'solid'
                    opacity = 160
                else:
                    color = self.DEFAULT_COLORS['breaker_supply']
                    style = 'dashed'
                    opacity = 120
            else:  # demand
                if zone.validated:
                    color = self.DEFAULT_COLORS['demand']
                    style = 'solid'
                    opacity = 160
                else:
                    color = self.DEFAULT_COLORS['breaker_demand']
                    style = 'dashed'
                    opacity = 120
                    
            # Normalize strength to 0-100
            normalized_strength = min(100, zone.strength)
            
            unified = UnifiedZone(
                source_type='supply_demand',
                zone_name=f'{zone.zone_type.capitalize()} {"Zone" if zone.validated else "Breaker"}',
                price_high=zone.price_high,
                price_low=zone.price_low,
                center_price=zone.center_price,
                strength=normalized_strength,
                priority=ZonePriority.SUPPLY_DEMAND,
                display_color=color,
                display_style=style,
                opacity=opacity,
                metadata={
                    'original_zone': zone,
                    'zone_type': zone.zone_type,
                    'validated': zone.validated,
                    'fractal_time': zone.fractal_time
                }
            )
            
            if unified.is_valid:
                zones.append(unified)
                
        return zones
        
    def _check_overlap(self, zone1: UnifiedZone, zone2: UnifiedZone) -> float:
        """
        Calculate overlap percentage between two zones
        
        Returns:
            Overlap percentage (0.0 to 1.0)
        """
        # Find overlap range
        overlap_high = min(zone1.price_high, zone2.price_high)
        overlap_low = max(zone1.price_low, zone2.price_low)
        
        # No overlap if ranges don't intersect
        if overlap_low >= overlap_high:
            return 0.0
            
        # Calculate overlap size
        overlap_size = overlap_high - overlap_low
        
        # Use smaller zone as reference for percentage
        smaller_zone_size = min(zone1.height, zone2.height)
        
        # Avoid division by zero
        if smaller_zone_size == 0:
            return 0.0
            
        overlap_percentage = overlap_size / smaller_zone_size
        
        return overlap_percentage
        
    def _remove_duplicates(self, zones: List[UnifiedZone]) -> List[UnifiedZone]:
        """Remove duplicate zones based on overlap and priority"""
        if not zones:
            return []
            
        # Sort by priority (descending) to process higher priority zones first
        sorted_zones = sorted(zones, key=lambda z: z.priority, reverse=True)
        
        filtered_zones = []
        
        for zone in sorted_zones:
            # Check if this zone overlaps with any already accepted zone
            is_duplicate = False
            
            for accepted_zone in filtered_zones:
                overlap = self._check_overlap(zone, accepted_zone)
                
                if overlap >= self.overlap_threshold:
                    # This is a duplicate
                    logger.debug(
                        f"Removing duplicate: {zone.zone_name} "
                        f"({overlap*100:.1f}% overlap with {accepted_zone.zone_name})"
                    )
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                filtered_zones.append(zone)
                
        return filtered_zones
        
    def get_zones_near_price(self, 
                           zones: List[UnifiedZone], 
                           current_price: float,
                           distance_percent: float = 0.02) -> List[UnifiedZone]:
        """
        Get zones within a certain percentage of current price
        
        Args:
            zones: List of unified zones
            current_price: Current market price
            distance_percent: Maximum distance as percentage (0.02 = 2%)
            
        Returns:
            List of nearby zones
        """
        nearby_zones = []
        max_distance = current_price * distance_percent
        
        for zone in zones:
            # Calculate distance to zone
            if zone.price_low <= current_price <= zone.price_high:
                # Price is inside zone
                distance = 0
            else:
                # Distance to nearest edge
                distance = min(
                    abs(current_price - zone.price_high),
                    abs(current_price - zone.price_low)
                )
                
            if distance <= max_distance:
                nearby_zones.append(zone)
                
        return nearby_zones


# Test the aggregator
if __name__ == "__main__":
    print("=== Testing Zone Aggregator ===\n")
    
    # Create test HVN result
    from market_review.calculations.volume.hvn_engine import HVNCluster, PriceLevel
    
    # Mock HVN clusters
    test_clusters = [
        HVNCluster(
            levels=[],
            cluster_high=105.0,
            cluster_low=100.0,
            center_price=102.5,
            total_volume=1000000,
            total_percent=15.0,
            highest_volume_level=None
        ),
        HVNCluster(
            levels=[],
            cluster_high=95.0,
            cluster_low=90.0,
            center_price=92.5,
            total_volume=800000,
            total_percent=8.0,
            highest_volume_level=None
        )
    ]
    
    test_hvn_result = HVNResult(
        hvn_unit=1.0,
        price_range=(85.0, 110.0),
        clusters=test_clusters,
        ranked_levels=[],
        filtered_levels=[]
    )
    
    # Mock Supply/Demand zones
    test_sd_zones = [
        SupplyDemandZone(
            zone_type='supply',
            price_low=104.0,
            price_high=106.0,
            center_price=105.0,
            strength=80.0,
            validated=True,
            fractal_time=datetime.now()
        ),
        SupplyDemandZone(
            zone_type='demand',
            price_low=88.0,
            price_high=91.0,
            center_price=89.5,
            strength=75.0,
            validated=False,
            fractal_time=datetime.now()
        )
    ]
    
    test_sd_result = {
        'zones': test_sd_zones,
        'current_price': 98.0
    }
    
    # Test aggregator
    aggregator = ZoneAggregator(overlap_threshold=0.1)
    
    # Test aggregation
    unified_zones = aggregator.aggregate_zones(
        hvn_result=test_hvn_result,
        supply_demand_result=test_sd_result
    )
    
    print(f"Total zones after aggregation: {len(unified_zones)}\n")
    
    for zone in unified_zones:
        print(f"{zone.zone_name}:")
        print(f"  Range: ${zone.price_low:.2f} - ${zone.price_high:.2f}")
        print(f"  Center: ${zone.center_price:.2f}")
        print(f"  Strength: {zone.strength}")
        print(f"  Priority: {zone.priority}")
        print(f"  Style: {zone.display_style}, Color: {zone.display_color}\n")
    
    # Test overlap detection
    print("\nTesting overlap detection:")
    zone1 = unified_zones[0] if unified_zones else None
    zone2 = unified_zones[1] if len(unified_zones) > 1 else None
    
    if zone1 and zone2:
        overlap = aggregator._check_overlap(zone1, zone2)
        print(f"Overlap between {zone1.zone_name} and {zone2.zone_name}: {overlap*100:.1f}%")