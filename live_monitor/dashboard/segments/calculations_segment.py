# live_monitor/dashboard/segments/calculations_segment.py
"""
Calculations Segment - All calculation-related methods for the dashboard
"""

import logging
import pandas as pd
import numpy as np
from PyQt6.QtCore import QTimer

logger = logging.getLogger(__name__)


class CalculationsSegment:
    """Dashboard segment containing all calculation methods"""
    
    def run_calculations(self):
        """Run HVN and Order Block calculations with M1, M5, and M15 EMA signals"""
        if not self.current_symbol or len(self.accumulated_data) < 100:
            logger.warning(f"Not enough data for calculations: {len(self.accumulated_data)} bars")
            return
            
        try:
            # Convert accumulated data to DataFrame
            df = pd.DataFrame(self.accumulated_data)
            
            # Ensure timestamp column exists and is timezone-aware
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                if df['timestamp'].dt.tz is None:
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
                # Keep timestamp as column for HVN engine
            
            current_price = float(df['close'].iloc[-1])
            
            # Update signal interpreter context
            self.signal_interpreter.set_symbol_context(
                self.current_symbol, 
                current_price
            )
            
            # Calculate M15 ATR for the tables
            m15_atr = self.calculate_m15_atr(df)
            
            # Create a copy with timestamp as index for EMA and Statistical calculations
            df_with_index = df.copy()
            df_with_index.set_index('timestamp', inplace=True)
            
            # Process EMA calculations
            self._process_ema_calculations(df_with_index)
            
            # Process Statistical Trend calculations
            self._process_statistical_trend(df_with_index)
            
            # Process zone calculations - pass df with timestamp as column
            self._process_zone_calculations(df, current_price, m15_atr)
            
            self.status_bar.showMessage(
                f"Calculations updated | M15 ATR: ${m15_atr:.2f}", 
                3000
            )
            
        except Exception as e:
            logger.error(f"Error in calculations: {e}", exc_info=True)
            self.status_bar.showMessage(f"Calculation error: {str(e)}", 5000)
    
    def _process_ema_calculations(self, df: pd.DataFrame):
        """Process all EMA calculations - respects active_entry_sources"""
        # M1 EMA - Always calculate for display, but check active status for entries
        if self.m1_ema_calculator:
            m1_ema_result = self.m1_ema_calculator.calculate(df)
            if m1_ema_result:
                standard_signal = self.signal_interpreter.process_m1_ema(m1_ema_result)
                self.update_signal_display(
                    standard_signal.value,
                    standard_signal.category.value,
                    'M1'
                )
                active_status = "ACTIVE" if self.active_entry_sources.get('M1_EMA', True) else "DISPLAY ONLY"
                logger.info(
                    f"M1 EMA Signal [{active_status}]: {standard_signal.value:+.1f} "
                    f"({standard_signal.category.value}) "
                    f"Confidence: {standard_signal.confidence:.0%}"
                )
        
        # M5 EMA - Always calculate for display, but check active status for entries
        if self.m5_ema_calculator:
            m5_ema_result = self.m5_ema_calculator.calculate(df)
            if m5_ema_result:
                standard_signal = self.signal_interpreter.process_m5_ema(m5_ema_result)
                self.update_signal_display(
                    standard_signal.value,
                    standard_signal.category.value,
                    'M5'
                )
                active_status = "ACTIVE" if self.active_entry_sources.get('M5_EMA', True) else "DISPLAY ONLY"
                logger.info(
                    f"M5 EMA Signal [{active_status}]: {standard_signal.value:+.1f} "
                    f"({standard_signal.category.value}) "
                    f"Confidence: {standard_signal.confidence:.0%}"
                )
        
        # M15 EMA - Always calculate for display, but check active status for entries
        if self.m15_ema_calculator:
            m15_ema_result = self.m15_ema_calculator.calculate(df, timeframe='1min')
            if m15_ema_result:
                standard_signal = self.signal_interpreter.process_m15_ema(m15_ema_result)
                self.update_signal_display(
                    standard_signal.value,
                    standard_signal.category.value,
                    'M15'
                )
                active_status = "ACTIVE" if self.active_entry_sources.get('M15_EMA', True) else "DISPLAY ONLY"
                logger.info(
                    f"M15 EMA Signal [{active_status}]: {standard_signal.value:+.1f} "
                    f"({standard_signal.category.value}) "
                    f"Original: {m15_ema_result.signal} "
                    f"Confidence: {standard_signal.confidence:.0%}"
                )

    def _process_statistical_trend(self, df: pd.DataFrame):
        """Process Statistical Trend calculations - respects active_entry_sources"""
        # Statistical Trend (1-minute based with 10-bar lookback)
        if len(df) >= 10:  # Need minimum bars for calculation
            try:
                stat_result = self.statistical_trend_calculator.analyze(
                    self.current_symbol, 
                    df, 
                    df.index[-1]  # Use last bar timestamp as entry_time
                )
                if stat_result:
                    standard_signal = self.signal_interpreter.process_statistical_trend(stat_result)
                    self.update_signal_display(
                        standard_signal.value,
                        standard_signal.category.value,
                        'STAT'
                    )
                    active_status = "ACTIVE" if self.active_entry_sources.get('STATISTICAL_TREND', True) else "DISPLAY ONLY"
                    logger.info(
                        f"Statistical Trend Signal [{active_status}]: {standard_signal.value:+.1f} "
                        f"({standard_signal.category.value}) "
                        f"Vol-Adj Strength: {stat_result.volatility_adjusted_strength:.2f} "
                        f"Confidence: {standard_signal.confidence:.0%}"
                    )
            except Exception as e:
                logger.error(f"Error in statistical trend calculation: {e}")
        else:
            logger.debug(f"Insufficient data for statistical trend: {len(df)} bars < 10 required") 
    
    def _process_zone_calculations(self, df: pd.DataFrame, current_price: float, m15_atr: float):
        """Process HVN and Order Block calculations"""
        # Run HVN calculation
        logger.info("Running HVN calculation...")
        hvn_result = self.hvn_engine.analyze(df, include_pre=True, include_post=True)
        
        # Convert HVN clusters to display format
        hvn_zones = []
        for cluster in hvn_result.clusters[:5]:  # Top 5 clusters
            hvn_zones.append({
                'price_low': cluster.cluster_low,
                'price_high': cluster.cluster_high,
                'center_price': cluster.center_price,
                'strength': cluster.total_percent,
                'type': 'hvn'
            })
        
        # Update HVN table
        self.hvn_table.update_hvn_zones(hvn_zones, current_price, m15_atr)
        
        # Run Order Block calculation
        logger.info("Running Order Block calculation...")
        order_blocks_raw = self.order_block_analyzer.analyze_zones(df)
        
        # Convert to display format
        order_blocks = []
        for ob in self.order_block_analyzer.bullish_obs[-3:] + self.order_block_analyzer.bearish_obs[-3:]:
            order_blocks.append({
                'block_type': ob.block_type,
                'top': ob.top,
                'bottom': ob.bottom,
                'center': ob.center,
                'is_breaker': ob.is_breaker,
                'time': ob.time
            })
        
        # Update Order Blocks table
        self.order_blocks_table.update_order_blocks(order_blocks, current_price, m15_atr)
        
        # For Supply/Demand, we'll use placeholder data for now
        supply_zones = [
            {'price_low': current_price * 1.01, 'price_high': current_price * 1.02, 
             'center_price': current_price * 1.015, 'strength': 75},
        ]
        demand_zones = [
            {'price_low': current_price * 0.98, 'price_high': current_price * 0.99, 
             'center_price': current_price * 0.985, 'strength': 80},
        ]
        
        # Update Supply/Demand table
        self.supply_demand_table.update_zones(supply_zones, demand_zones, current_price, m15_atr)
    
    def calculate_m15_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate M15 ATR from 1-minute data"""
        try:
            # Create a copy to avoid modifying original
            df_copy = df.copy()
            
            # If timestamp is not the index, set it
            if 'timestamp' in df_copy.columns:
                df_copy.set_index('timestamp', inplace=True)
            
            # Resample to 15-minute bars
            df_15m = df_copy.resample('15T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            if len(df_15m) < period:
                return 0.0
                
            # Calculate ATR
            high = df_15m['high'].values
            low = df_15m['low'].values
            close = df_15m['close'].values
            
            # True Range calculation
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = np.mean(tr[-period:])
            
            return float(atr)
        except Exception as e:
            logger.error(f"Error calculating M15 ATR: {e}")
            return 0.0