# live_monitor/dashboard/components/chart/zone_calculator.py
"""
Zone Calculator for Live Monitor
Integrates HVN, Supply/Demand, and Camarilla calculations
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
import pandas as pd

# Import calculation modules
from market_review.calculations.volume.hvn_engine import HVNEngine
from market_review.calculations.levels.camarilla_calculator import CamarillaCalculator
from market_review.calculations.zones.supply_demand import OrderBlockAnalyzer, SupplyDemandZone
from market_review.dashboards.components.zone_aggregator import ZoneAggregator

logger = logging.getLogger(__name__)


class ZoneCalculator(QObject):
    """
    Manages zone calculations for live monitor
    Runs calculations periodically and emits results
    """
    
    # Signals
    zones_calculated = pyqtSignal(dict)  # Emits calculation results
    calculation_error = pyqtSignal(str)
    
    def __init__(self, update_interval_ms: int = 60000):  # Default 1 minute
        super().__init__()
        
        self.update_interval = update_interval_ms
        self.current_symbol = None
        
        # Initialize calculators
        self.hvn_engine = HVNEngine(
            levels=100,
            percentile_threshold=80.0,
            proximity_atr_minutes=30
        )
        
        self.camarilla_calc = CamarillaCalculator()
        
        self.order_block_analyzer = OrderBlockAnalyzer(
            swing_length=7,
            show_bullish=3,
            show_bearish=3
        )
        
        self.zone_aggregator = ZoneAggregator(overlap_threshold=0.1)
        
        # Timer for periodic updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.calculate_zones)
        
        # Data storage
        self.accumulated_data = pd.DataFrame()
        
        logger.info(f"ZoneCalculator initialized with {update_interval_ms}ms interval")
        
    def set_symbol(self, symbol: str):
        """Change the current symbol"""
        if symbol != self.current_symbol:
            self.current_symbol = symbol
            self.accumulated_data = pd.DataFrame()
            logger.info(f"ZoneCalculator: Symbol changed to {symbol}")
            
    def update_data(self, bars: List[Dict]):
        """Update with new bar data"""
        if not bars:
            logger.debug("ZoneCalculator: No bars to update")
            return
            
        logger.info(f"ZoneCalculator: Updating with {len(bars)} bars")
        
        try:
            # Convert bars to DataFrame
            new_data = pd.DataFrame(bars)
            
            # Log the columns we received
            logger.debug(f"ZoneCalculator: Bar columns: {new_data.columns.tolist()}")
            
            if 'timestamp' in new_data.columns:
                # Handle timestamp conversion more carefully
                timestamps = new_data['timestamp']
                
                # Check if timestamps are already datetime objects
                if not pd.api.types.is_datetime64_any_dtype(timestamps):
                    # If they're strings or mixed, convert them
                    try:
                        # First try with UTC assumption
                        new_data['timestamp'] = pd.to_datetime(timestamps, utc=True)
                    except:
                        # If that fails, try without timezone first, then localize
                        new_data['timestamp'] = pd.to_datetime(timestamps)
                        # Make timezone-naive timestamps timezone-aware (UTC)
                        if new_data['timestamp'].dt.tz is None:
                            new_data['timestamp'] = new_data['timestamp'].dt.tz_localize('UTC')
                        else:
                            # Convert to UTC if already has timezone
                            new_data['timestamp'] = new_data['timestamp'].dt.tz_convert('UTC')
                else:
                    # Already datetime, just ensure it's timezone-aware
                    if timestamps.dt.tz is None:
                        new_data['timestamp'] = timestamps.dt.tz_localize('UTC')
                    else:
                        new_data['timestamp'] = timestamps.dt.tz_convert('UTC')
                        
                new_data.set_index('timestamp', inplace=True)
                
            # Append to accumulated data
            if self.accumulated_data.empty:
                self.accumulated_data = new_data
                logger.info(f"ZoneCalculator: Initial data loaded, {len(self.accumulated_data)} bars")
            else:
                # Ensure accumulated data is also timezone-aware
                if self.accumulated_data.index.tz is None:
                    self.accumulated_data.index = self.accumulated_data.index.tz_localize('UTC')
                elif self.accumulated_data.index.tz != timezone.utc:
                    self.accumulated_data.index = self.accumulated_data.index.tz_convert('UTC')
                    
                # Append and remove duplicates
                self.accumulated_data = pd.concat([self.accumulated_data, new_data])
                self.accumulated_data = self.accumulated_data[~self.accumulated_data.index.duplicated(keep='last')]
                
            # Keep only recent data (e.g., 14 days)
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=14)
            old_len = len(self.accumulated_data)
            self.accumulated_data = self.accumulated_data[self.accumulated_data.index >= cutoff_time]
            new_len = len(self.accumulated_data)
            
            if old_len != new_len:
                logger.debug(f"ZoneCalculator: Trimmed data from {old_len} to {new_len} bars")
                
            logger.info(f"ZoneCalculator: Total accumulated bars: {len(self.accumulated_data)}")
            
        except Exception as e:
            logger.error(f"ZoneCalculator: Error updating data: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
    def start_calculations(self):
        """Start periodic zone calculations"""
        if not self.update_timer.isActive():
            self.update_timer.start(self.update_interval)
            logger.info(f"Started zone calculations with {self.update_interval}ms interval")
            
            # Run initial calculation immediately
            QTimer.singleShot(1000, self.calculate_zones)
            
    def stop_calculations(self):
        """Stop periodic zone calculations"""
        if self.update_timer.isActive():
            self.update_timer.stop()
            logger.info("Stopped zone calculations")
            
    def calculate_zones(self):
        """Run all zone calculations"""
        if self.accumulated_data.empty or not self.current_symbol:
            logger.warning(f"ZoneCalculator: Cannot calculate - empty data or no symbol. Symbol: {self.current_symbol}, Data size: {len(self.accumulated_data)}")
            return
            
        try:
            logger.info(f"ZoneCalculator: Starting calculations for {self.current_symbol} with {len(self.accumulated_data)} bars")
            
            # Prepare data
            data = self.accumulated_data.copy()
            
            # Ensure index is timezone-aware
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC')
            else:
                data.index = data.index.tz_convert('UTC')
                
            data['timestamp'] = data.index  # Some calcs need timestamp column
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                logger.error(f"ZoneCalculator: Missing required columns: {missing_cols}")
                logger.error(f"Available columns: {data.columns.tolist()}")
                return
                
            current_price = float(data['close'].iloc[-1])
            logger.info(f"ZoneCalculator: Current price: ${current_price:.2f}")
            
            # 1. Calculate HVN zones
            logger.debug("Calculating HVN zones...")
            hvn_result = self.hvn_engine.analyze(
                data,
                include_pre=True,
                include_post=True
            )
            
            # Convert HVN clusters to zone format
            hvn_zones = []
            for cluster in hvn_result.clusters[:5]:  # Top 5 clusters
                hvn_zones.append({
                    'price_low': cluster.cluster_low,
                    'price_high': cluster.cluster_high,
                    'center_price': cluster.center_price,
                    'strength': cluster.total_percent,
                    'type': 'hvn'
                })
            logger.info(f"ZoneCalculator: Found {len(hvn_zones)} HVN zones")
            
            # 2. Calculate Supply/Demand zones
            logger.debug("Calculating Supply/Demand zones...")
            sd_zones_raw = self.order_block_analyzer.analyze_zones(data)
            
            # Convert to dict format
            sd_zones = []
            for zone in sd_zones_raw:
                sd_zones.append({
                    'price_low': zone.price_low,
                    'price_high': zone.price_high,
                    'center_price': zone.center_price,
                    'strength': zone.strength,
                    'type': zone.zone_type
                })
            logger.info(f"ZoneCalculator: Found {len(sd_zones)} Supply/Demand zones")
            
            # 3. Calculate Camarilla levels
            logger.debug("Calculating Camarilla levels...")
            camarilla_result = self.camarilla_calc.calculate(data, ticker=self.current_symbol)
            
            # Convert to dict format
            camarilla_levels = {
                'R4': camarilla_result.resistance_levels['R4'],
                'R3': camarilla_result.resistance_levels['R3'],
                'R2': camarilla_result.resistance_levels['R2'],
                'R1': camarilla_result.resistance_levels['R1'],
                'PP': camarilla_result.central_pivot,
                'S1': camarilla_result.support_levels['S1'],
                'S2': camarilla_result.support_levels['S2'],
                'S3': camarilla_result.support_levels['S3'],
                'S4': camarilla_result.support_levels['S4']
            }
            logger.info(f"ZoneCalculator: Calculated Camarilla levels - Pivot: ${camarilla_levels['PP']:.2f}")
            
            # 4. Aggregate zones
            logger.debug("Aggregating zones...")
            aggregated_zones = self.zone_aggregator.aggregate_zones(
                hvn_result=hvn_result,
                supply_demand_result={'zones': sd_zones_raw}
            )
            
            # Get zones near current price
            nearby_zones = self.zone_aggregator.get_zones_near_price(
                aggregated_zones,
                current_price,
                distance_percent=0.03
            )
            logger.info(f"ZoneCalculator: {len(aggregated_zones)} aggregated zones, {len(nearby_zones)} nearby")
            
            # Emit results
            result = {
                'symbol': self.current_symbol,
                'current_price': current_price,
                'hvn_zones': hvn_zones,
                'supply_demand_zones': sd_zones,
                'camarilla_levels': camarilla_levels,
                'aggregated_zones': aggregated_zones,
                'nearby_zones': nearby_zones,
                'calculation_time': datetime.now(timezone.utc)
            }
            
            self.zones_calculated.emit(result)
            logger.info(f"ZoneCalculator: Calculation complete and emitted!")
            
        except Exception as e:
            logger.error(f"ZoneCalculator: Error calculating zones: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.calculation_error.emit(str(e))