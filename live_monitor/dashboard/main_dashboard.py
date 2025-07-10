"""
Main Live Monitor Dashboard
"""

import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter
from PyQt6.QtCore import Qt
from ..styles import BaseStyles
from .components import (TickerEntry, TickerCalculations, EntryCalculations,
                        PointCallEntry, PointCallExit, ChartWidget)


class LiveMonitorDashboard(QMainWindow):
    """Main dashboard window for live monitoring"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.apply_styles()
        self.connect_signals()
        
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("Live Monitor - Trading Dashboard")
        self.setGeometry(100, 100, 1600, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main horizontal layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Create main horizontal splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # LEFT COLUMN (unchanged)
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
        
        # MIDDLE AND RIGHT SECTION (new layout)
        # Create a widget to hold the middle/right content
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
        
    def _on_ticker_changed(self, ticker):
        """Handle ticker symbol changes"""
        print(f"Ticker changed to: {ticker}")
        # Future: Trigger data fetch and updates
        
    def _on_calculation_complete(self, results):
        """Handle calculation completion"""
        print(f"Calculation complete: {results}")
        # Future: Update relevant displays
        
    def _on_entry_selected(self, entry_data):
        """Handle entry signal selection"""
        print(f"Entry selected: {entry_data}")
        # Future: Update entry calculations with selected price
        if 'price' in entry_data:
            self.entry_calculations.update_entry_price(float(entry_data['price']))
            
    def _on_exit_selected(self, exit_data):
        """Handle exit signal selection"""
        print(f"Exit selected: {exit_data}")
        # Future: Process exit signal
        
    def _on_timeframe_changed(self, timeframe):
        """Handle timeframe changes"""
        print(f"Timeframe changed to: {timeframe}")
        # Future: Update chart data
        
    def _on_indicator_toggled(self, indicator, enabled):
        """Handle indicator toggle"""
        print(f"Indicator {indicator} toggled: {enabled}")
        # Future: Show/hide indicator on chart


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    dashboard = LiveMonitorDashboard()
    dashboard.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()