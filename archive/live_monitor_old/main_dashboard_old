# live_monitor/dashboard/main_dashboard.py
"""
Main Live Monitor Dashboard with Polygon Data Integration
Updated to use three table widgets instead of charts
Enhanced with Signal Interpreter for M1, M5, and M15 EMA signals
"""

import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QSplitter, QStatusBar, QLabel)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent, QPainter, QBrush, QPen, QColor

# Import styles and components
from ..styles import BaseStyles
from .components import (TickerEntry, TickerCalculations, EntryCalculations,
                        PointCallEntry, PointCallExit, HVNTableWidget,
                        SupplyDemandTableWidget, OrderBlocksTableWidget)

# Import data manager
from live_monitor.data import PolygonDataManager

# Import calculation modules
from live_monitor.calculations.volume.hvn_engine import HVNEngine
from live_monitor.calculations.zones.supply_demand import OrderBlockAnalyzer
from live_monitor.calculations.indicators.m1_ema import M1EMACalculator
from live_monitor.calculations.indicators.m5_ema import M5EMACalculator
from live_monitor.calculations.indicators.m15_ema import M15EMACalculator

# Import signal interpreter
from live_monitor.signals.signal_interpreter import SignalInterpreter

# Configure logging
logger = logging.getLogger(__name__)


class ServerStatusWidget(QWidget):
    """Custom widget for server connection status indicator"""
    
    def __init__(self):
        super().__init__()
        self.is_connected = False
        self.setFixedSize(100, 30)
        
    def set_connected(self, connected: bool):
        """Update connection status"""
        self.is_connected = connected
        self.update()  # Trigger repaint
        
    def paintEvent(self, event):
        """Custom paint for the indicator"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw background
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        
        # Draw border
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        
        # Draw circle
        circle_color = QColor(68, 255, 68) if self.is_connected else QColor(255, 68, 68)
        painter.setBrush(QBrush(circle_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(10, 8, 14, 14)
        
        # Draw text
        painter.setPen(QPen(Qt.GlobalColor.white))
        painter.drawText(30, 20, "Server")


class LiveMonitorDashboard(QMainWindow):
    """Main dashboard window for live monitoring with real-time data"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize data manager
        self.data_manager = PolygonDataManager()
        
        # Initialize signal interpreter
        self.signal_interpreter = SignalInterpreter()
        
        # Initialize calculation engines
        self.hvn_engine = HVNEngine(
            levels=100,
            percentile_threshold=80.0,
            proximity_atr_minutes=30
        )
        
        self.order_block_analyzer = OrderBlockAnalyzer(
            swing_length=7,
            show_bullish=3,
            show_bearish=3
        )
        
        # Initialize M1 EMA calculator
        self.m1_ema_calculator = M1EMACalculator()
        
        # Initialize M5 EMA calculator
        self.m5_ema_calculator = M5EMACalculator()
        
        # Initialize M15 EMA calculator
        self.m15_ema_calculator = M15EMACalculator()
        
        # Data storage
        self.accumulated_data = []
        self.current_symbol = None
        
        # Initialize UI
        self.init_ui()
        self.apply_styles()
        self.connect_signals()
        self.setup_data_connections()
        self.setup_signal_connections()
        
        # Connect to Polygon server after UI is ready
        QTimer.singleShot(100, self.connect_to_polygon)
        
        # Start calculation timer
        self.calculation_timer = QTimer()
        self.calculation_timer.timeout.connect(self.run_calculations)
        self.calculation_timer.start(30000)  # Run every 30 seconds
        
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("Live Monitor - Trading Dashboard")
        self.setGeometry(100, 100, 1600, 900)
        
        # Create main widget that will hold everything
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Main vertical layout to hold header and content
        main_vertical_layout = QVBoxLayout(main_widget)
        main_vertical_layout.setContentsMargins(0, 0, 0, 0)
        main_vertical_layout.setSpacing(0)
        
        # Create header widget
        header_widget = QWidget()
        header_widget.setFixedHeight(40)
        header_widget.setStyleSheet("QWidget { background-color: #2a2a2a; border-bottom: 1px solid #444; }")
        
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(10, 5, 10, 5)
        
        # Add title label to header
        title_label = QLabel("Live Monitor Dashboard")
        title_label.setStyleSheet("QLabel { color: white; font-size: 16px; font-weight: bold; }")
        header_layout.addWidget(title_label)
        
        # Add stretch to push indicator to the right
        header_layout.addStretch()
        
        # Add server status indicator
        self.server_status_widget = ServerStatusWidget()
        header_layout.addWidget(self.server_status_widget)
        
        # Add header to main layout
        main_vertical_layout.addWidget(header_widget)
        
        # Create content widget (original dashboard content)
        content_widget = QWidget()
        main_vertical_layout.addWidget(content_widget, 1)
        
        # Original main horizontal layout now goes in content widget
        main_layout = QHBoxLayout(content_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Create main horizontal splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # LEFT COLUMN
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(5)
        
        # Ticker Entry
        self.ticker_entry = TickerEntry()
        left_layout.addWidget(self.ticker_entry)
        
        # Ticker Calculations
        self.ticker_calculations = TickerCalculations()
        left_layout.addWidget(self.ticker_calculations, 1)
        
        # Entry/Size Calculations
        self.entry_calculations = EntryCalculations()
        left_layout.addWidget(self.entry_calculations, 1)
        
        # MIDDLE AND RIGHT SECTION
        middle_right_widget = QWidget()
        middle_right_layout = QVBoxLayout(middle_right_widget)
        middle_right_layout.setContentsMargins(0, 0, 0, 0)
        middle_right_layout.setSpacing(5)
        
        # Create vertical splitter for top/bottom sections
        vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # TOP SECTION - Point & Call Entry and Exit side by side
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(5)
        
        # Point & Call Entry (left side of top)
        self.point_call_entry = PointCallEntry()
        top_layout.addWidget(self.point_call_entry, 1)
        
        # Point & Call Exit (right side of top)
        self.point_call_exit = PointCallExit()
        top_layout.addWidget(self.point_call_exit, 1)
        
        # BOTTOM SECTION - Three table widgets side by side
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(5)
        
        # HVN Table
        self.hvn_table = HVNTableWidget()
        bottom_layout.addWidget(self.hvn_table, 1)
        
        # Supply/Demand Table
        self.supply_demand_table = SupplyDemandTableWidget()
        bottom_layout.addWidget(self.supply_demand_table, 1)
        
        # Order Blocks Table
        self.order_blocks_table = OrderBlocksTableWidget()
        bottom_layout.addWidget(self.order_blocks_table, 1)
        
        # Add top and bottom to vertical splitter
        vertical_splitter.addWidget(top_widget)
        vertical_splitter.addWidget(bottom_widget)
        
        # Set vertical splitter proportions (40% top, 60% bottom)
        vertical_splitter.setSizes([360, 540])
        
        # Add vertical splitter to middle_right layout
        middle_right_layout.addWidget(vertical_splitter)
        
        # Add widgets to main horizontal splitter
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(middle_right_widget)
        
        # Set horizontal splitter proportions (25% left, 75% middle/right)
        main_splitter.setSizes([400, 1200])
        
        # Add splitter to main layout
        main_layout.addWidget(main_splitter)
        
        # Add status bar
        self.setup_status_bar()
        
    def setup_status_bar(self):
        """Setup status bar with connection indicator and separate signal displays"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Connection status label
        self.connection_label = QLabel("● Disconnected")
        self.connection_label.setStyleSheet("QLabel { color: #ff4444; font-weight: bold; }")
        self.status_bar.addWidget(self.connection_label)
        
        # Current symbol label
        self.symbol_label = QLabel("Symbol: None")
        self.status_bar.addWidget(self.symbol_label)
        
        # Signal status labels for different timeframes
        self.m1_signal_label = QLabel("M1: --")
        self.m1_signal_label.setStyleSheet("QLabel { font-weight: bold; margin-left: 10px; }")
        self.status_bar.addWidget(self.m1_signal_label)
        
        self.m5_signal_label = QLabel("M5: --")
        self.m5_signal_label.setStyleSheet("QLabel { font-weight: bold; margin-left: 10px; }")
        self.status_bar.addWidget(self.m5_signal_label)
        
        self.m15_signal_label = QLabel("M15: --")
        self.m15_signal_label.setStyleSheet("QLabel { font-weight: bold; margin-left: 10px; }")
        self.status_bar.addWidget(self.m15_signal_label)
        
        # Last update time
        self.update_time_label = QLabel("Last Update: Never")
        self.status_bar.addPermanentWidget(self.update_time_label)
        
    def apply_styles(self):
        """Apply dark theme styles"""
        self.setStyleSheet(BaseStyles.get_base_stylesheet())
        
    def connect_signals(self):
        """Connect component signals"""
        # Connect ticker changes
        self.ticker_entry.ticker_changed.connect(self._on_ticker_changed)
        
        # Connect entry calculations
        self.entry_calculations.calculation_complete.connect(self._on_calculation_complete)
        
        # Connect point & call selections
        self.point_call_entry.entry_selected.connect(self._on_entry_selected)
        self.point_call_exit.exit_selected.connect(self._on_exit_selected)
        
        # Connect supply/demand table signals
        self.supply_demand_table.add_zone_requested.connect(self._on_add_zone_requested)
        self.supply_demand_table.refresh_requested.connect(self._on_refresh_zones_requested)
        
    def setup_data_connections(self):
        """Setup connections to data manager"""
        # Connect data signals
        self.data_manager.market_data_updated.connect(self._on_market_data_updated)
        self.data_manager.chart_data_updated.connect(self._on_chart_data_updated)
        self.data_manager.entry_signal_generated.connect(self._on_entry_signal)
        self.data_manager.exit_signal_generated.connect(self._on_exit_signal)
        
        # Connect status signals
        self.data_manager.connection_status_changed.connect(self._on_connection_status_changed)
        self.data_manager.error_occurred.connect(self._on_data_error)
    
    def setup_signal_connections(self):
        """Setup connections for signal interpreter"""
        # Connect interpreter signals to data manager signals
        # This creates the signal flow: Interpreter → Data Manager → Dashboard
        self.signal_interpreter.entry_signal_generated.connect(
            self.data_manager.entry_signal_generated.emit
        )
        self.signal_interpreter.exit_signal_generated.connect(
            self.data_manager.exit_signal_generated.emit
        )
        
    def connect_to_polygon(self):
        """Connect to Polygon data server"""
        logger.info("Connecting to Polygon data server...")
        self.status_bar.showMessage("Connecting to Polygon server...", 2000)
        self.data_manager.connect()
        
    def _on_ticker_changed(self, ticker):
        """Handle ticker symbol changes"""
        if ticker:
            logger.info(f"Ticker changed to: {ticker}")
            self.current_symbol = ticker
            self.symbol_label.setText(f"Symbol: {ticker}")
            self.status_bar.showMessage(f"Subscribing to {ticker}...", 2000)
            
            # Clear existing data
            self.ticker_calculations.clear_calculations()
            self.point_call_entry.clear_signals()
            self.point_call_exit.clear_signals()
            self.accumulated_data.clear()
            
            # Clear tables
            self.hvn_table.clear_zones()
            self.supply_demand_table.clear_zones()
            self.order_blocks_table.clear_blocks()
            
            # Reset all signal labels
            self.m1_signal_label.setText("M1: --")
            self.m1_signal_label.setStyleSheet("QLabel { font-weight: bold; margin-left: 10px; }")
            self.m5_signal_label.setText("M5: --")
            self.m5_signal_label.setStyleSheet("QLabel { font-weight: bold; margin-left: 10px; }")
            self.m15_signal_label.setText("M15: --")
            self.m15_signal_label.setStyleSheet("QLabel { font-weight: bold; margin-left: 10px; }")
            
            # Change symbol in data manager
            self.data_manager.change_symbol(ticker)
            
            # Run calculations after a delay to allow data to load
            QTimer.singleShot(5000, self.run_calculations)
        
    def _on_market_data_updated(self, data):
        """Handle market data updates"""
        # Update ticker calculations
        self.ticker_calculations.update_calculations(data)
        
        # Update last update time
        if 'last_update' in data:
            self.update_time_label.setText(f"Last Update: {data['last_update']}")
        
        # Show in status bar
        if 'last_price' in data:
            self.status_bar.showMessage(
                f"Last: ${data['last_price']:.2f} | "
                f"Bid: ${data.get('bid', 0):.2f} | "
                f"Ask: ${data.get('ask', 0):.2f}", 
                2000
            )
    
    def _on_chart_data_updated(self, data: dict):
        """Handle chart data updates - accumulate for calculations"""
        if data.get('bars'):
            # Add bars to accumulated data
            self.accumulated_data.extend(data['bars'])
            
            # Keep only recent data (e.g., last 2000 bars)
            if len(self.accumulated_data) > 2000:
                self.accumulated_data = self.accumulated_data[-2000:]
            
            logger.info(f"Accumulated {len(self.accumulated_data)} bars for {data['symbol']}")
    
    def calculate_m15_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate M15 ATR from 1-minute data"""
        try:
            # Resample to 15-minute bars
            df_15m = df.resample('15T').agg({
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
    
    def update_signal_display(self, signal_value: float, category: str, timeframe: str):
        """Update the signal label with color coding for specific timeframe"""
        # Determine which label to update
        if timeframe == 'M1':
            label = self.m1_signal_label
        elif timeframe == 'M5':
            label = self.m5_signal_label
        elif timeframe == 'M15':
            label = self.m15_signal_label
        else:
            return
        
        # Format the signal display
        label.setText(f"{timeframe}: {category} ({signal_value:+.1f})")
        
        # Apply color based on category
        if signal_value >= 25:
            # Bullish - Green
            label.setStyleSheet("QLabel { color: #26a69a; font-weight: bold; margin-left: 10px; }")
        elif signal_value > 0:
            # Weak Bullish - Light Green
            label.setStyleSheet("QLabel { color: #66bb6a; font-weight: bold; margin-left: 10px; }")
        elif signal_value > -25:
            # Weak Bearish - Light Red
            label.setStyleSheet("QLabel { color: #ef5350; font-weight: bold; margin-left: 10px; }")
        else:
            # Bearish - Red
            label.setStyleSheet("QLabel { color: #d32f2f; font-weight: bold; margin-left: 10px; }")
    
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
                df.set_index('timestamp', inplace=True)
            
            current_price = float(df['close'].iloc[-1])
            
            # Update signal interpreter context
            self.signal_interpreter.set_symbol_context(
                self.current_symbol, 
                current_price
            )
            
            # Calculate M15 ATR for the tables
            m15_atr = self.calculate_m15_atr(df)
            
            # Run M1 EMA calculation and signal generation
            m1_ema_result = self.m1_ema_calculator.calculate(df)
            if m1_ema_result:
                # Process through signal interpreter
                standard_signal = self.signal_interpreter.process_m1_ema(m1_ema_result)
                
                # Update signal display for M1
                self.update_signal_display(
                    standard_signal.value,
                    standard_signal.category.value,
                    'M1'
                )
                
                # Log the signal
                logger.info(
                    f"M1 EMA Signal: {standard_signal.value:+.1f} "
                    f"({standard_signal.category.value}) "
                    f"Confidence: {standard_signal.confidence:.0%}"
                )
            else:
                logger.warning("M1 EMA calculation returned None")
            
            # Run M5 EMA calculation and signal generation
            m5_ema_result = self.m5_ema_calculator.calculate(df)
            if m5_ema_result:
                # Process through signal interpreter
                standard_signal = self.signal_interpreter.process_m5_ema(m5_ema_result)
                
                # Update signal display for M5
                self.update_signal_display(
                    standard_signal.value,
                    standard_signal.category.value,
                    'M5'
                )
                
                # Log the signal
                logger.info(
                    f"M5 EMA Signal: {standard_signal.value:+.1f} "
                    f"({standard_signal.category.value}) "
                    f"Confidence: {standard_signal.confidence:.0%}"
                )
            else:
                logger.warning("M5 EMA calculation returned None")
            
            # Run M15 EMA calculation and signal generation
            m15_ema_result = self.m15_ema_calculator.calculate(df, timeframe='1min')
            if m15_ema_result:
                # Process through signal interpreter
                standard_signal = self.signal_interpreter.process_m15_ema(m15_ema_result)
                
                # Update signal display for M15
                self.update_signal_display(
                    standard_signal.value,
                    standard_signal.category.value,
                    'M15'
                )
                
                # Log the signal
                logger.info(
                    f"M15 EMA Signal: {standard_signal.value:+.1f} "
                    f"({standard_signal.category.value}) "
                    f"Original: {m15_ema_result.signal} "
                    f"Confidence: {standard_signal.confidence:.0%}"
                )
            else:
                logger.warning("M15 EMA calculation returned None")
            
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
            # In production, this would come from Supabase
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
            
            self.status_bar.showMessage(
                f"Calculations updated: {len(hvn_zones)} HVN zones, "
                f"{len(order_blocks)} order blocks | "
                f"M15 ATR: ${m15_atr:.2f}", 
                3000
            )
            
        except Exception as e:
            logger.error(f"Error in calculations: {e}", exc_info=True)
            self.status_bar.showMessage(f"Calculation error: {str(e)}", 5000)
    
    def _on_add_zone_requested(self):
        """Handle request to add new supply/demand zone"""
        # In production, this would open a dialog to add zone to Supabase
        logger.info("Add zone requested - would open Supabase dialog")
        self.status_bar.showMessage("Add zone feature coming soon...", 2000)
    
    def _on_refresh_zones_requested(self):
        """Handle request to refresh supply/demand zones"""
        # In production, this would fetch latest zones from Supabase
        logger.info("Refresh zones requested - would query Supabase")
        self.status_bar.showMessage("Refreshing zones from database...", 2000)
        # Run calculations to update with latest data
        self.run_calculations()
    
    def _on_entry_signal(self, signal_data):
        """Handle new entry signal"""
        logger.info(f"Entry signal received: {signal_data}")
        self.point_call_entry.add_entry_signal(
            time=signal_data.get('time', ''),
            signal_type=signal_data.get('signal_type', 'LONG'),
            price=signal_data.get('price', ''),
            signal=signal_data.get('signal', ''),
            strength=signal_data.get('strength', 'Medium'),
            notes=signal_data.get('notes', '')
        )
    
    def _on_exit_signal(self, signal_data):
        """Handle new exit signal"""
        logger.info(f"Exit signal received: {signal_data}")
        self.point_call_exit.add_exit_signal(
            time=signal_data.get('time', ''),
            exit_type=signal_data.get('exit_type', 'TARGET'),
            price=signal_data.get('price', ''),
            pnl=signal_data.get('pnl', ''),
            signal=signal_data.get('signal', ''),
            urgency=signal_data.get('urgency', 'Normal')
        )
    
    def _on_connection_status_changed(self, is_connected):
        """Handle connection status changes"""
        # Update top-right indicator
        self.server_status_widget.set_connected(is_connected)
        
        # Update status bar indicator
        if is_connected:
            self.connection_label.setText("● Connected")
            self.connection_label.setStyleSheet("QLabel { color: #44ff44; font-weight: bold; }")
            self.setWindowTitle("Live Monitor - Trading Dashboard [Connected]")
            self.status_bar.showMessage("Connected to Polygon server", 2000)
        else:
            self.connection_label.setText("● Disconnected")
            self.connection_label.setStyleSheet("QLabel { color: #ff4444; font-weight: bold; }")
            self.setWindowTitle("Live Monitor - Trading Dashboard [Disconnected]")
            self.status_bar.showMessage("Disconnected from server", 2000)
    
    def _on_data_error(self, error_msg):
        """Handle data errors"""
        logger.error(f"Data error: {error_msg}")
        self.status_bar.showMessage(f"Error: {error_msg}", 5000)
        
    def _on_calculation_complete(self, results):
        """Handle calculation completion"""
        logger.info(f"Calculation complete: {results}")
        
    def _on_entry_selected(self, entry_data):
        """Handle entry signal selection"""
        logger.info(f"Entry selected: {entry_data}")
        if 'price' in entry_data:
            try:
                price_str = entry_data['price'].replace('$', '').replace(',', '')
                price = float(price_str)
                self.entry_calculations.update_entry_price(price)
            except ValueError:
                logger.error(f"Could not parse price: {entry_data['price']}")
            
    def _on_exit_selected(self, exit_data):
        """Handle exit signal selection"""
        logger.info(f"Exit selected: {exit_data}")
    
    def closeEvent(self, event: QCloseEvent):
        """Handle window close event"""
        logger.info("Closing dashboard...")
        
        # Stop calculation timer
        self.calculation_timer.stop()
        
        # Disconnect from data server
        self.data_manager.disconnect()
        
        # Accept the close event
        event.accept()


def main():
    """Main entry point"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create application
    app = QApplication(sys.argv)
    
    # Create and show dashboard
    dashboard = LiveMonitorDashboard()
    dashboard.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()