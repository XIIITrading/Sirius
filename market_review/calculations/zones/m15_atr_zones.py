# market_review/calculations/zones/m15_atr_zones.py
"""
Module: 15-Minute ATR Zone Enhancement for Confluence Zones
Purpose: Add ATR-based boundaries to volume confluence zones for volatility context
Features:
  - Single ATR calculation for efficiency
  - Zone relationship analysis
  - Integration with existing confluence zones
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict
from dataclasses import dataclass

# Import required modules
from market_review.calculations.indicators.m15_atr import M15ATRCalculator, ATRResult
from market_review.calculations.confluence.hvn_confluence import ConfluenceAnalysis, ConfluenceZone

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ATRZone:
    """
    ATR-based zone enhancement for a confluence zone.
    
    Provides volatility-based boundaries around volume-based center points.
    """
    confluence_zone_id: int
    center_price: float  # From ConfluenceZone
    atr_value: float
    atr_upper: float  # center + ATR
    atr_lower: float  # center - ATR
    atr_zone_width: float  # 2 * ATR
    
    # Relationship to original volume zone
    volume_zone_upper: float  # Original zone_high
    volume_zone_lower: float  # Original zone_low
    volume_zone_width: float  # Original zone width
    
    # Analysis flags
    atr_extends_above: bool  # ATR upper > volume zone upper
    atr_extends_below: bool  # ATR lower < volume zone lower
    zones_aligned: bool      # ATR zone mostly within volume zone
    
    # Relative measurements
    atr_to_volume_ratio: float  # ATR width / volume zone width
    upper_extension: float  # How much ATR extends above volume zone
    lower_extension: float  # How much ATR extends below volume zone
    
    def contains_price(self, price: float) -> bool:
        """Check if price is within ATR zone"""
        return self.atr_lower <= price <= self.atr_upper
    
    def get_price_position(self, price: float) -> Dict[str, float]:
        """Get price position relative to ATR zone"""
        distance_from_center = price - self.center_price
        atr_percentage = (distance_from_center / self.atr_value) * 100 if self.atr_value > 0 else 0
        
        return {
            'distance_from_center': distance_from_center,
            'atr_percentage': atr_percentage,
            'distance_to_upper': self.atr_upper - price,
            'distance_to_lower': price - self.atr_lower,
            'position_in_zone': (price - self.atr_lower) / self.atr_zone_width if self.atr_zone_width > 0 else 0.5
        }


@dataclass
class ATRZoneAnalysis:
    """Complete ATR zone analysis results"""
    symbol: str
    analysis_time: datetime
    atr_result: ATRResult  # Full ATR calculation details
    atr_zones: List[ATRZone]
    zones_with_price: List[ATRZone]  # Zones containing current price
    
    def get_zone_by_id(self, zone_id: int) -> Optional[ATRZone]:
        """Get ATR zone by confluence zone ID"""
        return next((z for z in self.atr_zones if z.confluence_zone_id == zone_id), None)


class ATRZoneEnhancer:
    """
    Enhances confluence zones with ATR-based volatility boundaries.
    
    Creates volatility-adjusted zones around volume-based confluence centers,
    providing context for potential price movement ranges.
    """
    
    def __init__(self, 
                 periods: int = 14,
                 lookback_bars: int = 100,
                 cache_enabled: bool = True):
        """
        Initialize ATR Zone Enhancer.
        
        Args:
            periods: Number of periods for ATR calculation
            lookback_bars: Number of bars to fetch for ATR calculation
            cache_enabled: Whether to use caching
        """
        self.atr_calculator = M15ATRCalculator(
            periods=periods,
            lookback_bars=lookback_bars,
            cache_enabled=cache_enabled
        )
        logger.info(f"ATR Zone Enhancer initialized with {periods} periods")
    
    def enhance_confluence_zones(self,
                                analysis: ConfluenceAnalysis,
                                symbol: str,
                                end_datetime: Optional[datetime] = None) -> ATRZoneAnalysis:
        """
        Enhance confluence zones with ATR-based boundaries.
        
        Args:
            analysis: Confluence analysis with zones to enhance
            symbol: Stock ticker symbol
            end_datetime: End time for ATR calculation (default: now)
            
        Returns:
            ATRZoneAnalysis with enhanced zone information
        """
        start_time = datetime.now()
        logger.info(f"Enhancing {len(analysis.zones)} confluence zones with ATR data for {symbol}")
        
        try:
            # Calculate ATR once for all zones (efficiency)
            atr_result = self.atr_calculator.calculate(symbol, end_datetime)
            logger.info(f"ATR calculated: ${atr_result.atr_value:.2f} ({atr_result.atr_percentage:.2f}%)")
            
            # Create ATR zones for each confluence zone
            atr_zones = []
            zones_with_price = []
            
            for zone in analysis.zones:
                atr_zone = self._create_atr_zone(zone, atr_result)
                atr_zones.append(atr_zone)
                
                # Check if current price is in this ATR zone
                if atr_zone.contains_price(analysis.current_price):
                    zones_with_price.append(atr_zone)
            
            # Create analysis result
            result = ATRZoneAnalysis(
                symbol=symbol,
                analysis_time=start_time,
                atr_result=atr_result,
                atr_zones=atr_zones,
                zones_with_price=zones_with_price
            )
            
            logger.info(f"ATR zone enhancement complete: {len(atr_zones)} zones created, "
                       f"{len(zones_with_price)} contain current price")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to enhance zones with ATR: {e}")
            # Return analysis with empty ATR zones on error
            return self._create_error_analysis(analysis, symbol, str(e))
    
    def _create_atr_zone(self, confluence_zone: ConfluenceZone, atr_result: ATRResult) -> ATRZone:
        """
        Create an ATR zone from a confluence zone.
        
        Args:
            confluence_zone: Original volume-based confluence zone
            atr_result: ATR calculation result
            
        Returns:
            ATRZone with volatility boundaries
        """
        center_price = confluence_zone.center_price
        atr_value = atr_result.atr_value
        
        # Calculate ATR boundaries
        atr_upper = center_price + atr_value
        atr_lower = center_price - atr_value
        atr_zone_width = atr_value * 2
        
        # Get volume zone boundaries
        volume_upper = confluence_zone.zone_high
        volume_lower = confluence_zone.zone_low
        volume_width = confluence_zone.zone_width
        
        # Calculate extensions
        upper_extension = max(0, atr_upper - volume_upper)
        lower_extension = max(0, volume_lower - atr_lower)
        
        # Determine zone alignment
        # Zones are "aligned" if ATR zone doesn't extend too far beyond volume zone
        max_extension_ratio = 0.5  # ATR can extend up to 50% of volume zone width
        zones_aligned = (upper_extension <= volume_width * max_extension_ratio and
                        lower_extension <= volume_width * max_extension_ratio)
        
        # Create ATR zone
        return ATRZone(
            confluence_zone_id=confluence_zone.zone_id,
            center_price=center_price,
            atr_value=atr_value,
            atr_upper=atr_upper,
            atr_lower=atr_lower,
            atr_zone_width=atr_zone_width,
            volume_zone_upper=volume_upper,
            volume_zone_lower=volume_lower,
            volume_zone_width=volume_width,
            atr_extends_above=atr_upper > volume_upper,
            atr_extends_below=atr_lower < volume_lower,
            zones_aligned=zones_aligned,
            atr_to_volume_ratio=atr_zone_width / volume_width if volume_width > 0 else 0,
            upper_extension=upper_extension,
            lower_extension=lower_extension
        )
    
    def _create_error_analysis(self, 
                              confluence_analysis: ConfluenceAnalysis,
                              symbol: str,
                              error_msg: str) -> ATRZoneAnalysis:
        """Create analysis result when ATR calculation fails"""
        logger.warning(f"Creating error analysis for {symbol}: {error_msg}")
        
        # Create empty ATR zones with None values
        atr_zones = []
        for zone in confluence_analysis.zones:
            # Create a minimal ATR zone indicating calculation failure
            atr_zone = ATRZone(
                confluence_zone_id=zone.zone_id,
                center_price=zone.center_price,
                atr_value=0,
                atr_upper=zone.zone_high,  # Fall back to volume zone
                atr_lower=zone.zone_low,
                atr_zone_width=zone.zone_width,
                volume_zone_upper=zone.zone_high,
                volume_zone_lower=zone.zone_low,
                volume_zone_width=zone.zone_width,
                atr_extends_above=False,
                atr_extends_below=False,
                zones_aligned=True,
                atr_to_volume_ratio=1.0,
                upper_extension=0,
                lower_extension=0
            )
            atr_zones.append(atr_zone)
        
        # Create a dummy ATR result
        dummy_atr_result = ATRResult(
            symbol=symbol,
            atr_value=0,
            atr_percentage=0,
            true_ranges=None,
            current_price=confluence_analysis.current_price,
            calculation_time=datetime.now(),
            data_end_time=datetime.now(),
            periods_used=0
        )
        
        return ATRZoneAnalysis(
            symbol=symbol,
            analysis_time=datetime.now(),
            atr_result=dummy_atr_result,
            atr_zones=atr_zones,
            zones_with_price=[]
        )
    
    def format_atr_zone_summary(self, 
                               atr_zone: ATRZone, 
                               current_price: float,
                               show_analysis: bool = True) -> str:
        """
        Format ATR zone information for display.
        
        Args:
            atr_zone: ATR zone to format
            current_price: Current market price
            show_analysis: Whether to include detailed analysis
            
        Returns:
            Formatted string summary
        """
        summary = f"📈 15-MIN ATR ZONE\n"
        summary += f"   ATR Value: ${atr_zone.atr_value:.2f}\n"
        summary += f"   ATR Upper: ${atr_zone.atr_upper:.2f} (+${atr_zone.atr_value:.2f} from center)\n"
        summary += f"   ATR Lower: ${atr_zone.atr_lower:.2f} (-${atr_zone.atr_value:.2f} from center)\n"
        summary += f"   Zone Width: ${atr_zone.atr_zone_width:.2f}\n"
        
        if show_analysis:
            summary += f"\n⚡ ZONE ANALYSIS\n"
            
            if atr_zone.atr_extends_above:
                summary += f"   ✓ ATR extends above volume zone by ${atr_zone.upper_extension:.2f}\n"
            else:
                summary += f"   ✓ ATR upper within volume zone\n"
                
            if atr_zone.atr_extends_below:
                summary += f"   ✓ ATR extends below volume zone by ${atr_zone.lower_extension:.2f}\n"
            else:
                summary += f"   ✓ ATR lower within volume zone\n"
            
            # Zone relationship interpretation
            if atr_zone.atr_to_volume_ratio > 2.0:
                summary += "   → Suggests high volatility potential vs accumulation\n"
            elif atr_zone.atr_to_volume_ratio < 0.5:
                summary += "   → Suggests tight accumulation vs current volatility\n"
            else:
                summary += "   → Volatility and accumulation are balanced\n"
            
            # Price position
            if atr_zone.contains_price(current_price):
                position = atr_zone.get_price_position(current_price)
                summary += f"\n💹 PRICE POSITION\n"
                summary += f"   ✓ Inside ATR zone\n"
                summary += f"   📍 ${position['distance_from_center']:.2f} from center "
                summary += f"({position['atr_percentage']:.1f}% of ATR)\n"
        
        return summary


# Example usage
if __name__ == "__main__":
    # This would normally come from HVN confluence analysis
    from market_review.calculations.confluence.hvn_confluence import ConfluenceZone, ConfluenceAnalysis
    
    # Create mock confluence zones
    mock_zones = [
        ConfluenceZone(
            zone_id=1,
            center_price=456.25,
            zone_high=457.00,
            zone_low=455.50,
            zone_width=1.50,
            timeframes=[120, 60, 15],
            peaks=[(120, 456.50, 3.5), (60, 456.25, 4.2), (15, 456.00, 5.5)],
            total_volume_weight=13.2,
            average_volume=4.4,
            distance_from_current=0.45,
            distance_percentage=0.1,
            strength='Strong',
            strength_score=7.5
        )
    ]
    
    mock_analysis = ConfluenceAnalysis(
        current_price=456.70,
        analysis_time=datetime.now(),
        zones=mock_zones,
        total_zones_found=1,
        strongest_zone=mock_zones[0],
        nearest_zone=mock_zones[0],
        price_in_zone=mock_zones[0]
    )
    
    # Enhance with ATR zones
    enhancer = ATRZoneEnhancer()
    atr_analysis = enhancer.enhance_confluence_zones(mock_analysis, 'TSLA')
    
    # Print results
    print(f"Symbol: {atr_analysis.symbol}")
    print(f"ATR Value: ${atr_analysis.atr_result.atr_value:.2f}")
    print(f"\nEnhanced Zones:")
    
    for atr_zone in atr_analysis.atr_zones:
        print(f"\n{enhancer.format_atr_zone_summary(atr_zone, mock_analysis.current_price)}")