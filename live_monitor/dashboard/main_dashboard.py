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

# Import historical fetch coordinator
from live_monitor.data.hist_request.fetch_coordinator import HistoricalFetchCoordinator

# Import calculation modules
from live_monitor.calculations.volume.hvn_engine import HVNEngine
from live_monitor.calculations.zones.supply_demand import OrderBlockAnalyzer
from live_monitor.calculations.indicators.m1_ema import M1EMACalculator
from live_monitor.calculations.indicators.m5_ema import M5EMACalculator
from live_monitor.calculations.indicators.m15_ema import M15EMACalculator
from live_monitor.calculations.trend.statistical_trend_1min import StatisticalTrend1MinSimplified
from live_monitor.calculations.trend.statistical_trend_5min import StatisticalTrend5Min
from live_monitor.calculations.trend.statistical_trend_15min import StatisticalTrend15Min
from live_monitor.calculations.market_structure.m1_market_structure import MarketStructureAnalyzer
from live_monitor.calculations.market_structure.m5_market_structure import M5MarketStructureAnalyzer
from live_monitor.calculations.market_structure.m15_market_structure import M15MarketStructureAnalyzer


# Import signal interpreter
from live_monitor.signals.signal_interpreter import SignalInterpreter

# Configure logging
logger = logging.getLogger(__name__)


class LiveMonitorDashboard(QMainWindow, UIBuilderSegment, DataHandlerSegment, 
                          CalculationsSegment, SignalDisplaySegment):
    """Main dashboard window for live monitoring with real-time data"""
    
    def __init__(self):
        super().__init__()
        
        # Configuration for active entry signal sources
        # TOGGLE THESE TO ENABLE/DISABLE ENTRY SIGNAL GENERATION
        self.active_entry_sources = {
            'M1_EMA': True,          # Set to False to disable M1 EMA entry signals
            'M5_EMA': True,          # Set to False to disable M5 EMA entry signals
            'M15_EMA': True,         # Set to False to disable M15 EMA entry signals
            'STATISTICAL_TREND_1M': True, # Set to False to disable Statistical Trend entry signals
            'STATISTICAL_TREND_5M': True, # Set to False to disable Statistical Trend entry signals
            'STATISTICAL_TREND_15M': True, # Set to False to disable Statistical Trend entry signals
            'M1_MARKET_STRUCTURE': True,  # Set to False to disable M1 Market Structure entry signals
            'M5_MARKET_STRUCTURE': True,    
            'M15_MARKET_STRUCTURE': True,
        }
        
        # Initialize data manager
        self.data_manager = PolygonDataManager()
        
        # Initialize signal interpreter
        self.signal_interpreter = SignalInterpreter()
        # Pass the active sources configuration
        self.signal_interpreter.set_active_entry_sources(self.active_entry_sources)
        
        # Initialize historical fetch coordinator
        self.fetch_coordinator = HistoricalFetchCoordinator(self.data_manager.rest_client)
        
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
        self.statistical_trend_calculator_1min = StatisticalTrend1MinSimplified(lookback_periods=10)
        self.statistical_trend_5min = StatisticalTrend5Min(lookback_periods=10)
        self.statistical_trend_15min = StatisticalTrend15Min(lookback_periods=10)

        # Initialize Market Structure Analyzer
        self.m1_market_structure_analyzer = MarketStructureAnalyzer(
            fractal_length=5,
            buffer_size=200,
            min_candles_required=21,
            bars_needed=200
        )

        self.m5_market_structure_analyzer = M5MarketStructureAnalyzer(
            fractal_length=3,
            buffer_size=100,
            min_candles_required=15,
            bars_needed=100
        )

        self.m15_market_structure_analyzer = M15MarketStructureAnalyzer(
            fractal_length=2,
            buffer_size=60,
            min_candles_required=10,
            bars_needed=60
        )

        # Data storage
        self.accumulated_data = []
        self.current_symbol = None
        
        # Historical data storage
        self.historical_data = {
            'M1': None,
            'M5': None,
            'M15': None
        }
        
        # Initialize UI (from UIBuilderSegment)
        self.init_ui()
        self.apply_styles()
        self.connect_signals()
        self.setup_data_connections()
        self.setup_signal_connections()
        self.setup_fetch_coordinator_connections()
        
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
    
    def setup_fetch_coordinator_connections(self):
        """Setup connections for historical fetch coordinator"""
        # Connect fetch coordinator signals
        self.fetch_coordinator.fetch_started.connect(self._on_fetch_started)
        self.fetch_coordinator.fetch_progress.connect(self._on_fetch_progress)
        self.fetch_coordinator.all_fetches_completed.connect(self._on_all_fetches_completed)
        self.fetch_coordinator.fetch_error.connect(self._on_fetch_error)
        
        # Connect data ready signals
        self.fetch_coordinator.ema_data_ready.connect(self._on_historical_ema_data)
        self.fetch_coordinator.structure_data_ready.connect(self._on_historical_structure_data)
        self.fetch_coordinator.trend_data_ready.connect(self._on_historical_trend_data)
        self.fetch_coordinator.zone_data_ready.connect(self._on_historical_zone_data)
    
    def connect_to_polygon(self):
        """Connect to Polygon data server"""
        logger.info("Connecting to Polygon data server...")
        self.status_bar.showMessage("Connecting to Polygon server...", 2000)
        self.data_manager.connect()
    
    def toggle_entry_source(self, source: str, enabled: bool):
        """Toggle whether a calculation source generates entry signals"""
        if source in self.active_entry_sources:
            self.active_entry_sources[source] = enabled
            # Update signal interpreter
            self.signal_interpreter.set_active_entry_sources(self.active_entry_sources)
            logger.info(f"Entry source {source} set to {enabled}")
    
    def _on_fetch_started(self, symbol: str):
        """Handle start of historical data fetch"""
        self.status_bar.showMessage(f"Fetching historical data for {symbol}...", 2000)
    
    def _on_fetch_progress(self, progress: dict):
        """Handle fetch progress updates"""
        completed = progress.get('completed', 0)
        total = progress.get('total', 0)
        percentage = progress.get('percentage', 0)
        self.status_bar.showMessage(
            f"Loading historical data: {completed}/{total} ({percentage:.0f}%)", 
            1000
        )
    
    def _on_all_fetches_completed(self, symbol: str):
        """Handle completion of all historical fetches"""
        logger.info(f"All historical data fetched for {symbol}")
        self.status_bar.showMessage(f"Historical data loaded for {symbol}", 2000)
        
        # Run calculations with full historical data
        QTimer.singleShot(1000, self.run_calculations)
    
    def _on_fetch_error(self, error_data: dict):
        """Handle fetch errors"""
        errors = error_data.get('errors', [])
        if errors:
            logger.error(f"Historical fetch errors: {errors}")
            self.status_bar.showMessage("Some historical data failed to load", 3000)
    
    def _on_historical_ema_data(self, data: dict):
        """Process historical EMA data"""
        logger.info(f"Received historical EMA data - M1: {len(data.get('M1', [])) if data.get('M1') is not None else 0} bars, "
                    f"M5: {len(data.get('M5', [])) if data.get('M5') is not None else 0} bars, "
                    f"M15: {len(data.get('M15', [])) if data.get('M15') is not None else 0} bars")
        
        # Store historical data
        for timeframe in ['M1', 'M5', 'M15']:
            if timeframe in data and data[timeframe] is not None:
                self.historical_data[timeframe] = data[timeframe]
        
        # Run EMA calculations with historical data
        self._run_ema_calculations_with_historical()
    
    def _on_historical_structure_data(self, data: dict):
        """Process historical market structure data"""
        logger.info("Received historical market structure data")
        # Future: Use for market structure analysis
    
    def _on_historical_trend_data(self, data: dict):
        """Process historical trend data"""
        logger.info("Received historical statistical trend data")
        # Future: Use for trend analysis
    
    def _on_historical_zone_data(self, data: dict):
        """Process historical zone data"""
        logger.info("Received historical zone data")
        # Future: Use for zone analysis
    
    def _run_ema_calculations_with_historical(self):
        """Run EMA calculations using historical data"""
        current_price = None
        
        # Get current price from accumulated data or historical
        if self.accumulated_data:
            current_price = float(self.accumulated_data[-1]['close'])
        elif self.historical_data.get('M1') is not None and not self.historical_data['M1'].empty:
            current_price = float(self.historical_data['M1']['close'].iloc[-1])
        
        if current_price:
            # Update signal interpreter context
            self.signal_interpreter.set_symbol_context(
                self.current_symbol, 
                current_price
            )
        
        # Process M1 EMA with historical data
        if self.historical_data.get('M1') is not None and self.m1_ema_calculator:
            m1_ema_result = self.m1_ema_calculator.calculate(self.historical_data['M1'])
            if m1_ema_result:
                standard_signal = self.signal_interpreter.process_m1_ema(m1_ema_result)
                self.update_signal_display(
                    standard_signal.value,
                    standard_signal.category.value,
                    'M1'
                )
                logger.info(f"M1 EMA (Historical): {standard_signal.value:+.1f} ({standard_signal.category.value})")
        
        # Process M5 EMA with historical data
        if self.historical_data.get('M5') is not None and self.m5_ema_calculator:
            m5_ema_result = self.m5_ema_calculator.calculate(self.historical_data['M5'])
            if m5_ema_result:
                standard_signal = self.signal_interpreter.process_m5_ema(m5_ema_result)
                self.update_signal_display(
                    standard_signal.value,
                    standard_signal.category.value,
                    'M5'
                )
                logger.info(f"M5 EMA (Historical): {standard_signal.value:+.1f} ({standard_signal.category.value})")
        
        # Process M15 EMA with historical data
        if self.historical_data.get('M15') is not None and self.m15_ema_calculator:
            m15_ema_result = self.m15_ema_calculator.calculate(self.historical_data['M15'], timeframe='15min')
            if m15_ema_result:
                standard_signal = self.signal_interpreter.process_m15_ema(m15_ema_result)
                self.update_signal_display(
                    standard_signal.value,
                    standard_signal.category.value,
                    'M15'
                )
                logger.info(f"M15 EMA (Historical): {standard_signal.value:+.1f} ({standard_signal.category.value})")
    
    def _on_ticker_changed(self, ticker):
        """Handle ticker symbol changes - OVERRIDE from DataHandlerSegment"""
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
            
            # Clear historical data
            self.historical_data = {
                'M1': None,
                'M5': None,
                'M15': None
            }
            
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
            self.stat_signal_label.setText("STAT: --")
            self.stat_signal_label.setStyleSheet("QLabel { font-weight: bold; margin-left: 10px; }")
            self.m1_mstruct_label.setText("M1 MSTRUCT: --")
            self.m1_mstruct_label.setStyleSheet("QLabel { font-weight: bold; margin-left: 10px; }")
            self.m5_mstruct_label.setText("M5 MSTRUCT: --")  # Fixed indentation
            self.m5_mstruct_label.setStyleSheet("QLabel { font-weight: bold; margin-left: 10px; }")
            self.m15_mstruct_label.setText("M15 MSTRUCT: --")
            self.m15_mstruct_label.setStyleSheet("QLabel { font-weight: bold; margin-left: 10px; }")
            
            # Change symbol in data manager
            self.data_manager.change_symbol(ticker)
            
            # Fetch historical data for all indicators
            self.fetch_coordinator.fetch_all_for_symbol(ticker)
            
            # Don't run calculations immediately - wait for historical data
    
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