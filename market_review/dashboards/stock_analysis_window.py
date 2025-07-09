# market_review/dashboards/stock_analysis_window.py
"""
Module: Stock Analysis Window
Purpose: Detailed analysis window for individual stocks with multiple chart views
UI Framework: PyQt6
"""

import logging
import re
from typing import Optional

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QTabWidget, QPushButton, QLabel, QGroupBox,
                            QSplitter, QTextEdit, QLineEdit)
from PyQt6.QtCore import Qt, pyqtSignal

# Local imports
from market_review.dashboards.components.dual_hvn_chart import DualHVNChart
from market_review.dashboards.components.camarilla_pivot_chart import CamarillaPivotChart
from market_review.dashboards.components.supply_demand_chart import SupplyDemandChart
from market_review.dashboards.components.summary_chart import SummaryChart

# Configure logging
logger = logging.getLogger(__name__)


class StockAnalysisWindow(QMainWindow):
    """Dedicated window for detailed stock analysis."""
    
    # Signals
    window_closed = pyqtSignal(str)  # Emits ticker when closed
    
    def __init__(self, ticker: str, parent=None):
        super().__init__(parent)
        self.ticker = ticker.upper()
        
        self.setWindowTitle(f"Stock Analysis - {self.ticker}")
        self.setGeometry(150, 150, 1400, 900)
        self.setMinimumSize(1200, 700)
        
        # Apply dark theme
        self.apply_dark_theme()
        
        # Initialize UI
        self.init_ui()
        
        # Load data for all tabs
        self.load_ticker_data()
        
    def apply_dark_theme(self):
        """Apply dark theme styling."""
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1a1a1a;
                color: #ffffff;
                font-family: 'Arial', sans-serif;
                font-size: 12px;
            }
            QTabWidget::pane {
                border: 1px solid #333333;
                background-color: #1a1a1a;
            }
            QTabBar::tab {
                background-color: #2a2a2a;
                color: #ffffff;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #374151;
                border-bottom: 2px solid #10b981;
            }
            QTabBar::tab:hover {
                background-color: #374151;
            }
            QGroupBox {
                border: 1px solid #333333;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #10b981;
            }
            QPushButton {
                background-color: #10b981;
                border: none;
                border-radius: 3px;
                padding: 6px 12px;
                font-weight: bold;
                color: #000000;
            }
            QPushButton:hover {
                background-color: #059669;
            }
            QLabel {
                color: #e5e7eb;
            }
            QTextEdit {
                background-color: #2a2a2a;
                border: 1px solid #444444;
                border-radius: 3px;
                padding: 5px;
            }
            QTableWidget {
                background-color: #2a2a2a;
                border: 1px solid #444444;
                gridline-color: #444444;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #374151;
            }
            QHeaderView::section {
                background-color: #333333;
                color: #ffffff;
                padding: 5px;
                border: 1px solid #444444;
                font-weight: bold;
            }
            QLineEdit {
                background-color: #2a2a2a;
                border: 1px solid #444444;
                border-radius: 3px;
                padding: 5px;
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
            }
            QLineEdit:focus {
                border: 1px solid #10b981;
            }
        """)
        
    def init_ui(self):
        """Initialize the UI components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Header section
        header_widget = self.create_header_section()
        main_layout.addWidget(header_widget)
        
        # Tab widget for different views
        self.tab_widget = QTabWidget()
        
        # HVN Analysis Tab
        self.hvn_tab = self.create_hvn_tab()
        self.tab_widget.addTab(self.hvn_tab, "HVN Analysis")
        
        # Supply/Demand Tab
        self.supply_demand_tab = self.create_supply_demand_tab()
        self.tab_widget.addTab(self.supply_demand_tab, "Supply/Demand")
        
        # Camarilla Pivots Tab
        self.camarilla_tab = self.create_camarilla_tab()
        self.tab_widget.addTab(self.camarilla_tab, "Camarilla Pivots")

        # Summary Chart Tab
        self.summary_tab = self.create_summary_tab()
        self.tab_widget.addTab(self.summary_tab, "Summary")
        
        # Trading Plan Tab (placeholder for now)
        self.trading_plan_tab = self.create_trading_plan_tab()
        self.tab_widget.addTab(self.trading_plan_tab, "Trading Plan")
                
        # Add tabs to main layout
        main_layout.addWidget(self.tab_widget, 1)
        
    def create_header_section(self):
        """Create header with ticker info and controls."""
        group = QGroupBox(f"Analysis: {self.ticker}")
        layout = QHBoxLayout()
        
        # Ticker input section
        ticker_input_layout = QHBoxLayout()
        ticker_input_layout.setSpacing(5)
        
        # Ticker label
        ticker_label = QLabel("Ticker:")
        ticker_label.setStyleSheet("font-weight: bold; color: #e5e7eb;")
        ticker_input_layout.addWidget(ticker_label)
        
        # Ticker input field
        self.ticker_input = QLineEdit(self.ticker)
        self.ticker_input.setMaximumWidth(100)
        self.ticker_input.setStyleSheet("""
            QLineEdit {
                background-color: #2a2a2a;
                border: 1px solid #444444;
                border-radius: 3px;
                padding: 5px;
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
            }
            QLineEdit:focus {
                border: 1px solid #10b981;
            }
        """)
        # Allow pressing Enter to load ticker
        self.ticker_input.returnPressed.connect(self.on_ticker_change)
        ticker_input_layout.addWidget(self.ticker_input)
        
        # Load button
        self.load_btn = QPushButton("Load")
        self.load_btn.setMaximumWidth(60)
        self.load_btn.clicked.connect(self.on_ticker_change)
        ticker_input_layout.addWidget(self.load_btn)
        
        layout.addLayout(ticker_input_layout)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #10b981; margin-left: 20px;")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # Refresh button
        self.refresh_btn = QPushButton("Refresh Data")
        self.refresh_btn.clicked.connect(self.load_ticker_data)
        layout.addWidget(self.refresh_btn)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("background-color: #ef4444;")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        
        group.setLayout(layout)
        
        # Store the group box to update title later
        self.header_group = group
        return group
        
    def create_hvn_tab(self):
        """Create HVN analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Use existing DualHVNChart
        self.hvn_chart = DualHVNChart(
            lookback_periods=[7, 14],
            display_bars=390  # 5 days
        )
        
        # Connect signals
        self.hvn_chart.loading_started.connect(
            lambda: self.update_status("Loading HVN data...")
        )
        self.hvn_chart.loading_finished.connect(
            lambda: self.update_status("HVN analysis complete", success=True)
        )
        self.hvn_chart.error_occurred.connect(
            lambda err: self.update_status(f"HVN Error: {err}", error=True)
        )
        
        layout.addWidget(self.hvn_chart)
        return widget
        
    def create_supply_demand_tab(self):
        """Create Supply/Demand analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Create Supply/Demand chart
        self.supply_demand_chart = SupplyDemandChart(
            lookback_days=7,
            display_bars=182  # 7 days of 15-min bars
        )
        
        # Connect signals
        self.supply_demand_chart.loading_started.connect(
            lambda: self.update_status("Analyzing supply/demand zones...")
        )
        self.supply_demand_chart.loading_finished.connect(
            lambda: self.update_status("Supply/Demand analysis complete", success=True)
        )
        self.supply_demand_chart.error_occurred.connect(
            lambda err: self.update_status(f"S/D Error: {err}", error=True)
        )
        self.supply_demand_chart.zone_selected.connect(
            self.on_zone_selected
        )
        
        layout.addWidget(self.supply_demand_chart)
        return widget
        
    def create_camarilla_tab(self):
        """Create Camarilla pivots tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Create Camarilla chart
        self.camarilla_chart = CamarillaPivotChart()
        
        # Connect signals
        self.camarilla_chart.loading_started.connect(
            lambda: self.update_status("Calculating Camarilla levels...")
        )
        self.camarilla_chart.loading_finished.connect(
            lambda: self.update_status("Camarilla analysis complete", success=True)
        )
        self.camarilla_chart.error_occurred.connect(
            lambda err: self.update_status(f"Camarilla Error: {err}", error=True)
        )
        
        layout.addWidget(self.camarilla_chart)
        return widget
        
    def create_trading_plan_tab(self):
        """Create trading plan tab (placeholder)."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Placeholder content
        info_label = QLabel("Trading Plan functionality coming soon...")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("color: #9ca3af; font-size: 16px; padding: 20px;")
        
        layout.addWidget(info_label)
        return widget
    
    def create_summary_tab(self):
        """Create summary analysis tab with all indicators."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Create Summary chart
        self.summary_chart = SummaryChart(
            lookback_days=7,
            display_bars=182  # 7 days of 15-min bars
        )
        
        # Connect signals
        self.summary_chart.loading_started.connect(
            lambda: self.update_status("Loading comprehensive analysis...")
        )
        self.summary_chart.loading_finished.connect(
            lambda: self.update_status("Summary analysis complete", success=True)
        )
        self.summary_chart.error_occurred.connect(
            lambda err: self.update_status(f"Summary Error: {err}", error=True)
        )
        
        layout.addWidget(self.summary_chart)
        return widget
        
    def validate_ticker(self, ticker: str) -> bool:
        """Basic ticker validation."""
        # Check length (most tickers are 1-5 characters)
        if len(ticker) < 1 or len(ticker) > 10:
            return False
            
        # Check for valid characters (letters and possibly dots/dashes for some tickers)
        if not re.match(r'^[A-Z][A-Z0-9\-\.]*$', ticker):
            return False
            
        return True
        
    def on_ticker_change(self):
        """Handle ticker change from input field."""
        new_ticker = self.ticker_input.text().strip().upper()
        
        # Validate ticker input
        if not new_ticker:
            self.update_status("Please enter a ticker symbol", error=True)
            return
            
        if not self.validate_ticker(new_ticker):
            self.update_status("Invalid ticker format", error=True)
            return
            
        if new_ticker == self.ticker:
            self.update_status("Already showing " + new_ticker)
            return
            
        # Update ticker
        self.ticker = new_ticker
        
        # Update window title
        self.setWindowTitle(f"Stock Analysis - {self.ticker}")
        
        # Update header group title
        if hasattr(self, 'header_group'):
            self.header_group.setTitle(f"Analysis: {self.ticker}")
        
        # Clear status
        self.update_status(f"Loading {self.ticker}...")
        
        # Reload all data
        self.load_ticker_data()
        
        logger.info(f"Changed ticker to {self.ticker}")
        
    def load_ticker_data(self):
        """Load data for all tabs."""
        logger.info(f"Loading analysis data for {self.ticker}")
        
        # Load HVN data
        self.hvn_chart.load_ticker(self.ticker)
        
        # Load Supply/Demand data
        self.supply_demand_chart.load_ticker(self.ticker)
        
        # Load Camarilla data
        self.camarilla_chart.load_ticker(self.ticker)
        
        # Load Summary data
        self.summary_chart.load_ticker(self.ticker)
        
    def update_status(self, message: str, success: bool = False, error: bool = False):
        """Update status label."""
        self.status_label.setText(message)
        
        if success:
            self.status_label.setStyleSheet("color: #10b981; margin-left: 20px;")
        elif error:
            self.status_label.setStyleSheet("color: #ef4444; margin-left: 20px;")
        else:
            self.status_label.setStyleSheet("color: #f59e0b; margin-left: 20px;")
            
    def on_zone_selected(self, zone_data: dict):
        """Handle supply/demand zone selection."""
        zone = zone_data['zone']
        logger.info(
            f"Zone selected: {zone.zone_type} at ${zone.center_price:.2f} "
            f"(Strength: {zone.strength:.0f}%)"
        )
        
        # You can add additional handling here, such as:
        # - Highlighting the zone on the chart
        # - Showing additional zone details
        # - Creating alerts for the zone
        
    def closeEvent(self, event):
        """Handle window close event."""
        self.window_closed.emit(self.ticker)
        event.accept()


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    
    # Create application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Configure PyQtGraph
    import pyqtgraph as pg
    pg.setConfigOptions(antialias=True)
    pg.setConfigOption('background', '#1a1a1a')
    pg.setConfigOption('foreground', '#ffffff')
    
    # Initialize data manager and set it for supply/demand module
    try:
        from market_review.dashboards.data_manager import DataManager
        from market_review.calculations.zones.supply_demand import set_data_manager
        
        # Get data manager instance
        data_manager = DataManager.get_instance()
        
        # Set data manager for supply/demand module
        set_data_manager(data_manager)
        
        print("Data manager initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize data manager: {e}")
        print("Supply/Demand analysis may not work properly")
    
    # Create and show window with a test ticker
    ticker = "TSLA"  # Change to any ticker you want
    if len(sys.argv) > 1:
        ticker = sys.argv[1]  # Allow passing ticker as command line argument
    
    window = StockAnalysisWindow(ticker)
    window.show()
    
    print(f"Stock Analysis Window opened for {ticker}")
    print("Available tabs:")
    print("  - HVN Analysis: Volume profile analysis")
    print("  - Supply/Demand: Order blocks and breaker blocks detection")
    print("  - Camarilla Pivots: Support/resistance levels")
    print("  - Trading Plan: (Coming soon)")
    print("\nYou can change tickers using the input field in the header")
    print("You can also pass a ticker as argument: python stock_analysis_window.py AAPL")
    
    sys.exit(app.exec())