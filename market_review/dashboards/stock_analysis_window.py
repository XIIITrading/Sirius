# market_review/dashboards/stock_analysis_window.py
"""
Module: Stock Analysis Window
Purpose: Detailed analysis window for individual stocks with multiple chart views
UI Framework: PyQt6
"""

import logging
from typing import Optional

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QTabWidget, QPushButton, QLabel, QGroupBox,
                            QSplitter, QTextEdit)
from PyQt6.QtCore import Qt, pyqtSignal

# Local imports
from market_review.dashboards.components.dual_hvn_chart import DualHVNChart
from market_review.dashboards.components.camarilla_pivot_chart import CamarillaPivotChart

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
        
        # Camarilla Pivots Tab
        self.camarilla_tab = self.create_camarilla_tab()
        self.tab_widget.addTab(self.camarilla_tab, "Camarilla Pivots")
        
        # Trading Plan Tab (placeholder for now)
        self.trading_plan_tab = self.create_trading_plan_tab()
        self.tab_widget.addTab(self.trading_plan_tab, "Trading Plan")
        
        # Add tabs to main layout
        main_layout.addWidget(self.tab_widget, 1)
        
    def create_header_section(self):
        """Create header with ticker info and controls."""
        group = QGroupBox(f"Analysis: {self.ticker}")
        layout = QHBoxLayout()
        
        # Ticker info
        self.ticker_label = QLabel(f"Ticker: {self.ticker}")
        self.ticker_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #10b981;")
        layout.addWidget(self.ticker_label)
        
        # Status label
        self.status_label = QLabel("Loading data...")
        self.status_label.setStyleSheet("color: #f59e0b;")
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
        
    def load_ticker_data(self):
        """Load data for all tabs."""
        logger.info(f"Loading analysis data for {self.ticker}")
        
        # Load HVN data
        self.hvn_chart.load_ticker(self.ticker)
        
        # Load Camarilla data
        self.camarilla_chart.load_ticker(self.ticker)
        
    def update_status(self, message: str, success: bool = False, error: bool = False):
        """Update status label."""
        self.status_label.setText(message)
        
        if success:
            self.status_label.setStyleSheet("color: #10b981;")
        elif error:
            self.status_label.setStyleSheet("color: #ef4444;")
        else:
            self.status_label.setStyleSheet("color: #f59e0b;")
            
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
    
    # Create and show window with a test ticker
    ticker = "TSLA"  # Change to any ticker you want
    if len(sys.argv) > 1:
        ticker = sys.argv[1]  # Allow passing ticker as command line argument
    
    window = StockAnalysisWindow(ticker)
    window.show()
    
    print(f"Stock Analysis Window opened for {ticker}")
    print("You can also pass a ticker as argument: python stock_analysis_window.py AAPL")
    
    sys.exit(app.exec())