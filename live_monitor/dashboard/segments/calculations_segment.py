# live_monitor/dashboard/segments/calculations_segment.py
"""
Calculations Segment - All calculation-related methods for the dashboard
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timezone
from PyQt6.QtCore import QTimer

logger = logging.getLogger(__name__)


class CalculationsSegment:
    """Dashboard segment containing all calculation methods"""
    
    def setup_historical_data_connections(self):
        """Connect to historical data ready signals from PolygonDataManager"""
        if hasattr(self, 'data_manager') and self.data_manager:
            logger.info("Setting up historical data connections")
            
            # Connect EMA data
            self.data_manager.ema_data_ready.connect(self._on_ema_historical_data)
            
            # Connect Market Structure data
            self.data_manager.structure_data_ready.connect(self._on_structure_historical_data)
            
            # Connect Statistical Trend data
            self.data_manager.trend_data_ready.connect(self._on_trend_historical_data)
            
            # Connect Zone data
            self.data_manager.zone_data_ready.connect(self._on_zone_historical_data)
            
            logger.info("Historical data connections established")
    
    def _on_ema_historical_data(self, data: dict):
        """Process historical EMA data when it becomes available"""
        logger.info(f"Received historical EMA data: {list(data.keys())}")
        
        try:
            # Get current price for signal interpreter
            current_price = self._get_current_price_from_data(data)
            if current_price and self.current_symbol:
                self.signal_interpreter.set_symbol_context(
                    self.current_symbol, 
                    current_price
                )
            
            # M1 EMA
            if 'M1' in data and self.m1_ema_calculator:
                df = data['M1']
                logger.info(f"Processing M1 EMA with {len(df)} bars")
                result = self.m1_ema_calculator.calculate(df)
                if result and self.signal_interpreter:
                    signal = self.signal_interpreter.process_m1_ema(result)
                    self.update_signal_display(signal.value, signal.category.value, 'M1')
                    logger.info(f"M1 EMA Historical: {signal.value:+.1f} ({signal.category.value})")
            
            # M5 EMA
            if 'M5' in data and self.m5_ema_calculator:
                df = data['M5']
                logger.info(f"Processing M5 EMA with {len(df)} bars")
                result = self.m5_ema_calculator.calculate(df)
                if result and self.signal_interpreter:
                    signal = self.signal_interpreter.process_m5_ema(result)
                    self.update_signal_display(signal.value, signal.category.value, 'M5')
                    logger.info(f"M5 EMA Historical: {signal.value:+.1f} ({signal.category.value})")
            
            # M15 EMA
            if 'M15' in data and self.m15_ema_calculator:
                df = data['M15']
                logger.info(f"Processing M15 EMA with {len(df)} bars")
                result = self.m15_ema_calculator.calculate(df, timeframe='15min')  # Native 15min bars
                if result and self.signal_interpreter:
                    signal = self.signal_interpreter.process_m15_ema(result)
                    self.update_signal_display(signal.value, signal.category.value, 'M15')
                    logger.info(f"M15 EMA Historical: {signal.value:+.1f} ({signal.category.value})")
                    
        except Exception as e:
            logger.error(f"Error processing historical EMA data: {e}", exc_info=True)
    
    def _on_structure_historical_data(self, data: dict):
        """Process historical market structure data"""
        logger.info(f"Received historical structure data: {list(data.keys())}")
        
        try:
            # M1 Market Structure
            if 'M1' in data and hasattr(self, 'm1_market_structure'):
                df = data['M1']
                logger.info(f"Processing M1 Market Structure with {len(df)} bars")
                # Process with market structure analyzer if available
                # Note: You'll need to add market structure analyzers to your dashboard
                
            # M5 Market Structure
            if 'M5' in data and hasattr(self, 'm5_market_structure'):
                df = data['M5']
                logger.info(f"Processing M5 Market Structure with {len(df)} bars")
                
            # M15 Market Structure
            if 'M15' in data and hasattr(self, 'm15_market_structure'):
                df = data['M15']
                logger.info(f"Processing M15 Market Structure with {len(df)} bars")
                
        except Exception as e:
            logger.error(f"Error processing historical structure data: {e}", exc_info=True)
    
    def _on_trend_historical_data(self, data: dict):
        """Process historical statistical trend data"""
        logger.info(f"Received historical trend data: {list(data.keys())}")
        
        try:
            # Get current price for signal interpreter
            current_price = self._get_current_price_from_data(data)
            if current_price and self.current_symbol:
                self.signal_interpreter.set_symbol_context(
                    self.current_symbol, 
                    current_price
                )
            
            # M1 Statistical Trend (we primarily use this one)
            if 'M1' in data and self.statistical_trend_calculator:
                df = data['M1']
                logger.info(f"Processing Statistical Trend with {len(df)} bars")
                
                if len(df) >= 10:  # Need minimum bars
                    result = self.statistical_trend_calculator.analyze(
                        self.current_symbol, 
                        df, 
                        df.index[-1]  # Use last bar timestamp
                    )
                    if result and self.signal_interpreter:
                        signal = self.signal_interpreter.process_statistical_trend(result)
                        self.update_signal_display(signal.value, signal.category.value, 'STAT')
                        logger.info(
                            f"Statistical Trend Historical: {signal.value:+.1f} "
                            f"({signal.category.value}) Vol-Adj: {result.volatility_adjusted_strength:.2f}"
                        )
            # M5 Statistical Trend
            if 'M5' in data and self.statistical_trend_5min:
                df = data['M5']
                logger.info(f"Processing M5 Statistical Trend (Historical) with {len(df)} bars")
                
                if len(df) >= 10:  # Need minimum bars
                    result = self.statistical_trend_5min.analyze(
                        self.current_symbol, 
                        df, 
                        df.index[-1]
                    )
                    if result and self.signal_interpreter:
                        signal = self.signal_interpreter.process_m5_statistical_trend(result)
                        self.update_signal_display(signal.value, signal.category.value, 'M5 TREND')
                        logger.info(
                            f"M5 Statistical Trend Historical: {result.signal} "
                            f"Value: {signal.value:+.1f} ({signal.category.value})"
                        )
                        
        except Exception as e:
            logger.error(f"Error processing historical trend data: {e}", exc_info=True)
    
    def _on_zone_historical_data(self, data: dict):
        """Process historical zone data (HVN and Order Blocks)"""
        logger.info(f"Received historical zone data: {list(data.keys())}")
        
        try:
            current_price = self._get_current_price_from_data(data)
            if not current_price:
                logger.warning("No current price available for zone calculations")
                return
            
            # HVN Data
            if 'HVN' in data and self.hvn_engine:
                df_hvn = data['HVN'].copy()
                logger.info(f"Processing HVN with {len(df_hvn)} bars")
                
                # Ensure timestamp is a column for HVN engine
                if 'timestamp' not in df_hvn.columns:
                    df_hvn.reset_index(inplace=True)
                
                # Calculate M15 ATR
                m15_atr = self.calculate_m15_atr(df_hvn)
                
                # Run HVN analysis
                hvn_result = self.hvn_engine.analyze(df_hvn, include_pre=True, include_post=True)
                
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
                logger.info(f"Updated HVN table with {len(hvn_zones)} zones")
            
            # Order Blocks Data
            if 'OrderBlocks' in data and self.order_block_analyzer:
                df_ob = data['OrderBlocks'].copy()
                logger.info(f"Processing Order Blocks with {len(df_ob)} bars")
                
                # Ensure timestamp is a column for Order Block analyzer
                if 'timestamp' not in df_ob.columns:
                    df_ob.reset_index(inplace=True)
                
                # Calculate M15 ATR if not already done
                if 'HVN' not in data:
                    m15_atr = self.calculate_m15_atr(df_ob)
                
                # Run Order Block analysis
                order_blocks_raw = self.order_block_analyzer.analyze_zones(df_ob)
                
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
                logger.info(f"Updated Order Blocks table with {len(order_blocks)} blocks")
                
                # Update Supply/Demand with placeholder data for now
                supply_zones = [
                    {'price_low': current_price * 1.01, 'price_high': current_price * 1.02, 
                     'center_price': current_price * 1.015, 'strength': 75},
                ]
                demand_zones = [
                    {'price_low': current_price * 0.98, 'price_high': current_price * 0.99, 
                     'center_price': current_price * 0.985, 'strength': 80},
                ]
                
                self.supply_demand_table.update_zones(supply_zones, demand_zones, current_price, m15_atr)
                
        except Exception as e:
            logger.error(f"Error processing historical zone data: {e}", exc_info=True)
    
    def _get_current_price_from_data(self, data: dict) -> Optional[float]:
        """Extract current price from any available dataframe in the data dict"""
        for key, df in data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                if 'close' in df.columns:
                    return float(df['close'].iloc[-1])
        return None
    
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
            self._process_m5_statistical_trend(df_with_index)
            
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
        """Process 1-minute Statistical Trend calculations"""
        if len(df) >= 10:
            try:
                stat_result = self.statistical_trend_calculator_1min.analyze(  # <- Fix name
                    self.current_symbol, 
                    df, 
                    df.index[-1]
                )
                if stat_result:
                    standard_signal = self.signal_interpreter.process_statistical_trend(stat_result)
                    self.update_signal_display(
                        standard_signal.value,
                        standard_signal.category.value,
                        'STAT'
                    )
                    active_status = "ACTIVE" if self.active_entry_sources.get('STATISTICAL_TREND_1M', True) else "DISPLAY ONLY"
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
    
    def _process_m5_statistical_trend(self, df: pd.DataFrame):
        """Process M5 Statistical Trend analysis"""
        try:
            # Resample to 5-minute bars
            df_5min = df.resample('5T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            # Need at least 10 5-minute bars
            if len(df_5min) < 10:
                self.logger.warning("Insufficient 5-minute bars for M5 statistical trend")
                return
            
            # Run analysis
            result = self.statistical_trend_5min.analyze(
                symbol=self.current_symbol,
                bars_df=df_5min,
                entry_time=df_5min.index[-1]
            )
            
            # Process through signal interpreter
            standard_signal = self.signal_interpreter.process_m5_statistical_trend(result)
            
            # Update signal display
            self.update_signal_display(
                standard_signal.value,
                standard_signal.category.value,
                'M5 TREND'
            )
            
            # Log the signal
            active_status = "ACTIVE" if self.active_entry_sources.get('STATISTICAL_TREND_5M', True) else "DISPLAY ONLY"
            self.logger.info(
                f"M5 Statistical Trend [{active_status}]: {result.signal} "
                f"Signal Value: {standard_signal.value:+.1f} ({standard_signal.category.value}) "
                f"Vol-Adj: {result.volatility_adjusted_strength:.2f} "
                f"Confidence: {standard_signal.confidence:.0%}"
            )
            
        except Exception as e:
            self.logger.error(f"Error in M5 statistical trend calculation: {e}", exc_info=True)
    
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