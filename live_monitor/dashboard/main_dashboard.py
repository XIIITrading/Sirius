# live_monitor/dashboard/main_dashboard.py
"""
Main Live Monitor Dashboard with Polygon Data Integration
Refactored to use segments for better organization
"""

import sys
import logging
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QCloseEvent

# Import segments
from .segments import (CalculationsSegment, DataHandlerSegment, 
                      UIBuilderSegment, SignalDisplaySegment)

# Import data manager
from live_monitor.data import PolygonDataManager

# Import calculation modules
from live_monitor.calculations.volume.hvn_engine import HVNEngine
from live_monitor.calculations.zones.supply_demand import OrderBlockAnalyzer
from live_monitor.calculations.indicators.m1_ema import M1EMACalculator
from live_monitor.calculations.indicators.m5_ema import M5EMACalculator
from live_monitor.calculations.indicators.m15_ema import M15EMACalculator
from live_monitor.calculations.trend.statistical_trend_1min import StatisticalTrend1MinSimplified

# Import signal interpreter
from live_monitor.signals.signal_interpreter import SignalInterpreter

# Configure logging
logger = logging.getLogger(__name__)


class LiveMonitorDashboard(QMainWindow, UIBuilderSegment, DataHandlerSegment, 
                          CalculationsSegment, SignalDisplaySegment):
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
        
        # Initialize EMA calculators
        self.m1_ema_calculator = M1EMACalculator()
        self.m5_ema_calculator = M5EMACalculator()
        self.m15_ema_calculator = M15EMACalculator()

        # Initialize Statistical Trend
        self.statistical_trend_calculator = StatisticalTrend1MinSimplified(lookback_periods=10)
        
        # Data storage
        self.accumulated_data = []
        self.current_symbol = None
        
        # Initialize UI (from UIBuilderSegment)
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