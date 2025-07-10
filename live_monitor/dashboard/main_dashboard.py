"""
Main Live Monitor Dashboard with Polygon Data Integration
"""

import sys
import logging
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QSplitter, QStatusBar, QLabel)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent, QPainter, QBrush, QPen, QColor

# Import styles and components
from ..styles import BaseStyles
from .components import (TickerEntry, TickerCalculations, EntryCalculations,
                        PointCallEntry, PointCallExit, ChartWidget)

# Import data manager
from live_monitor.data import PolygonDataManager

# Polygon Aggregate Data Handler
from live_monitor.dashboard.components.chart.data.aggregate_data_handler import AggregateDataHandler

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
        
        # Create and connect aggregate handler for chart data
        self.aggregate_handler = AggregateDataHandler()
        self.aggregate_handler.chart_data_updated.connect(
            lambda data: logger.info(f"Chart update: {data['symbol']} {data['timeframe']} - {len(data['bars'])} bars")
        )
        self.data_manager.set_aggregate_handler(self.aggregate_handler)
        logger.info("Aggregate handler connected to data manager")

        
        # Initialize UI
        self.init_ui()
        self.apply_styles()
        self.connect_signals()
        self.setup_data_connections()
        
        # Connect to Polygon server after UI is ready
        QTimer.singleShot(100, self.connect_to_polygon)
        
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
        
        # BOTTOM SECTION - Chart spanning full width
        self.chart_widget = ChartWidget()
        
        # Add top and bottom to vertical splitter
        vertical_splitter.addWidget(top_widget)
        vertical_splitter.addWidget(self.chart_widget)
        
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
        """Setup status bar with connection indicator"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Connection status label
        self.connection_label = QLabel("● Disconnected")
        self.connection_label.setStyleSheet("QLabel { color: #ff4444; font-weight: bold; }")
        self.status_bar.addWidget(self.connection_label)
        
        # Current symbol label
        self.symbol_label = QLabel("Symbol: None")
        self.status_bar.addWidget(self.symbol_label)
        
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
        
        # Connect chart controls
        self.chart_widget.timeframe_changed.connect(self._on_timeframe_changed)
        self.chart_widget.indicator_toggled.connect(self._on_indicator_toggled)
        
    def setup_data_connections(self):
        """Setup connections to data manager"""
        # Connect data signals
        self.data_manager.market_data_updated.connect(self._on_market_data_updated)
        
        # IMPORTANT: Connect chart data updates to chart widget
        self.data_manager.chart_data_updated.connect(self.chart_widget.update_chart_data)
        logger.info("Chart widget connected to data manager chart updates")
        
        self.data_manager.entry_signal_generated.connect(self._on_entry_signal)
        self.data_manager.exit_signal_generated.connect(self._on_exit_signal)
        
        # Connect status signals
        self.data_manager.connection_status_changed.connect(self._on_connection_status_changed)
        self.data_manager.error_occurred.connect(self._on_data_error)
        
    def connect_to_polygon(self):
        """Connect to Polygon data server"""
        logger.info("Connecting to Polygon data server...")
        self.status_bar.showMessage("Connecting to Polygon server...", 2000)
        self.data_manager.connect()
        
    def _on_ticker_changed(self, ticker):
        """Handle ticker symbol changes"""
        if ticker:
            logger.info(f"Ticker changed to: {ticker}")
            self.symbol_label.setText(f"Symbol: {ticker}")
            self.status_bar.showMessage(f"Subscribing to {ticker}...", 2000)
            
            # Clear existing data
            self.ticker_calculations.clear_calculations()
            self.point_call_entry.clear_signals()
            self.point_call_exit.clear_signals()
            
            # Change symbol in data manager
            self.data_manager.change_symbol(ticker)
        
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
        
    def _on_timeframe_changed(self, timeframe):
        """Handle timeframe changes"""
        logger.info(f"Timeframe changed to: {timeframe}")
        
    def _on_indicator_toggled(self, indicator, enabled):
        """Handle indicator toggle"""
        logger.info(f"Indicator {indicator} toggled: {enabled}")
    
    def closeEvent(self, event: QCloseEvent):
        """Handle window close event"""
        logger.info("Closing dashboard...")
        
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