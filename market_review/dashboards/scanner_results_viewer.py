# market_review/dashboards/scanner_results_viewer.py
"""
Module: Scanner Results Viewer Dashboard
Purpose: Display premarket scanner results from Supabase with integrated HVN charts
UI Framework: PyQt6 with PyQtGraph
Note: All times are in UTC
"""

# Standard library imports
import logging
import os
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

# Third-party imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QTableWidget, 
                            QTableWidgetItem, QHeaderView, QGroupBox, QSplitter,
                            QComboBox, QSpinBox, QDateEdit, QCheckBox,
                            QAbstractItemView, QSizePolicy)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDate, QTimer
from PyQt6.QtGui import QFont, QColor, QBrush

# Local application imports
from market_review.data.supabase_client import SupabaseClient
from market_review.dashboards.components.dual_hvn_chart import DualHVNChart

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScannerDataWorker(QThread):
    """Background thread for fetching scanner results from Supabase."""
    data_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, fetch_method: str, **kwargs):
        super().__init__()
        self.fetch_method = fetch_method
        self.kwargs = kwargs
        
    def run(self):
        try:
            client = SupabaseClient()
            
            if self.fetch_method == 'today':
                data = client.get_today_scans(**self.kwargs)
            elif self.fetch_method == 'by_date':
                data = client.get_scans_by_date(**self.kwargs)
            elif self.fetch_method == 'recent':
                data = client.get_recent_scans(**self.kwargs)
            else:
                data = client.get_today_scans()
                
            self.data_ready.emit(data)
            
        except Exception as e:
            logger.error(f"Error fetching scanner data: {e}")
            self.error_occurred.emit(str(e))


class ScannerResultsViewer(QMainWindow):
    """Main scanner results viewer dashboard."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Premarket Scanner Results - HVN Analysis")
        self.setGeometry(100, 100, 1600, 900)
        self.setMinimumSize(1400, 700)
        
        # State
        self.scanner_data = []
        self.data_worker = None
        self.current_ticker = None
        
        # Apply dark theme
        self.apply_dark_theme()
        
        # Initialize UI
        self.init_ui()
        
        # Load initial data
        QTimer.singleShot(100, self.load_today_scans)
        
    def apply_dark_theme(self):
        """Apply dark theme styling."""
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1a1a1a;
                color: #ffffff;
                font-family: 'Arial', sans-serif;
                font-size: 12px;
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
            QLabel {
                color: #e5e7eb;
            }
            QComboBox, QSpinBox, QDateEdit {
                background-color: #2a2a2a;
                border: 1px solid #444444;
                border-radius: 3px;
                padding: 5px;
                color: #ffffff;
            }
            QComboBox:hover, QSpinBox:hover, QDateEdit:hover {
                border: 1px solid #10b981;
            }
            QComboBox::drop-down {
                border: none;
            }
            QCheckBox {
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #444444;
                border-radius: 3px;
                background-color: #2a2a2a;
            }
            QCheckBox::indicator:checked {
                background-color: #10b981;
                border-color: #10b981;
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
            QPushButton:pressed {
                background-color: #047857;
            }
            QPushButton:disabled {
                background-color: #4b5563;
                color: #9ca3af;
            }
            QTableWidget {
                background-color: #2a2a2a;
                border: 1px solid #444444;
                gridline-color: #333333;
                selection-background-color: #374151;
            }
            QTableWidget::item {
                padding: 5px;
                border: none;
            }
            QTableWidget::item:selected {
                background-color: #374151;
            }
            QHeaderView::section {
                background-color: #1f2937;
                border: none;
                padding: 8px;
                font-weight: bold;
                border-right: 1px solid #333333;
            }
            QScrollBar:vertical {
                background: #2a2a2a;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #4b5563;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #6b7280;
            }
            QSplitter::handle {
                background-color: #333333;
            }
            QSplitter::handle:hover {
                background-color: #10b981;
            }
        """)
        
    def init_ui(self):
        """Initialize the main UI components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Top section: Controls
        controls_widget = self.create_controls_section()
        controls_widget.setMaximumHeight(100)
        main_layout.addWidget(controls_widget)
        
        # Main content: Splitter
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: Scanner results table
        table_widget = self.create_table_section()
        self.main_splitter.addWidget(table_widget)
        
        # Right side: HVN Charts
        chart_widget = self.create_chart_section()
        self.main_splitter.addWidget(chart_widget)
        
        # Set initial splitter sizes (40% table, 60% charts)
        self.main_splitter.setSizes([640, 960])
        self.main_splitter.setStretchFactor(0, 40)
        self.main_splitter.setStretchFactor(1, 60)
        
        # Add splitter to main layout
        main_layout.addWidget(self.main_splitter, 1)
        
    def create_controls_section(self):
        """Create the controls section."""
        group = QGroupBox("Scanner Controls")
        layout = QHBoxLayout()
        layout.setSpacing(15)
        
        # Date controls
        self.date_combo = QComboBox()
        self.date_combo.addItems(["Today", "Yesterday", "Custom Date", "Last N Days"])
        self.date_combo.currentTextChanged.connect(self.on_date_mode_changed)
        layout.addWidget(QLabel("Date:"))
        layout.addWidget(self.date_combo)
        
        # Custom date picker (hidden by default)
        self.custom_date = QDateEdit()
        self.custom_date.setCalendarPopup(True)
        self.custom_date.setDate(QDate.currentDate())
        self.custom_date.setDisplayFormat("yyyy-MM-dd")
        self.custom_date.hide()
        layout.addWidget(self.custom_date)
        
        # Days spinner (hidden by default)
        self.days_spin = QSpinBox()
        self.days_spin.setRange(1, 30)
        self.days_spin.setValue(5)
        self.days_spin.setSuffix(" days")
        self.days_spin.hide()
        layout.addWidget(self.days_spin)
        
        # Filters
        layout.addWidget(QLabel("Filters:"))
        
        self.passed_only_check = QCheckBox("Passed Only")
        self.passed_only_check.setChecked(True)
        layout.addWidget(self.passed_only_check)
        
        # Sort options
        layout.addWidget(QLabel("Sort by:"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Rank", "Interest Score", "PM Volume", "ATR %"])
        layout.addWidget(self.sort_combo)
        
        # Refresh button
        self.refresh_btn = QPushButton("Refresh Data")
        self.refresh_btn.clicked.connect(self.refresh_data)
        layout.addWidget(self.refresh_btn)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #10b981; font-weight: bold;")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        group.setLayout(layout)
        return group
        
    def create_table_section(self):
        """Create the scanner results table section."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Table info
        self.table_info = QLabel("No data loaded")
        self.table_info.setStyleSheet("color: #9ca3af; padding: 5px;")
        layout.addWidget(self.table_info)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        
        # Define columns
        self.table_columns = [
            "Rank", "Ticker", "Price", "Interest", "PM Volume", 
            "Avg Volume", "PM/Avg %", "ATR", "ATR %", "$ Volume", "Chart"
        ]
        
        self.results_table.setColumnCount(len(self.table_columns))
        self.results_table.setHorizontalHeaderLabels(self.table_columns)
        
        # Set column widths
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Rank
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)  # Ticker
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Price
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)  # Interest
        header.setSectionResizeMode(10, QHeaderView.ResizeMode.Fixed)  # Chart button
        self.results_table.setColumnWidth(10, 80)
        
        # Hide vertical header
        self.results_table.verticalHeader().setVisible(False)
        
        layout.addWidget(self.results_table)
        
        # Set minimum width
        widget.setMinimumWidth(600)
        
        return widget
        
    def create_chart_section(self):
        """Create the HVN chart section."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Chart info
        self.chart_info = QLabel("Select a ticker to view HVN analysis")
        self.chart_info.setStyleSheet("color: #9ca3af; padding: 5px; font-size: 14px;")
        layout.addWidget(self.chart_info)
        
        # Dual HVN Chart with custom parameters
        # 1092 bars = approximately 14 days of 15-minute bars
        self.dual_chart = DualHVNChart(
            lookback_periods=[14, 28],
            display_bars=1092  # 14 days * 6.5 hours * 4 bars/hour ≈ 1092 bars
        )
        self.dual_chart.loading_started.connect(self.on_chart_loading_started)
        self.dual_chart.loading_finished.connect(self.on_chart_loading_finished)
        self.dual_chart.error_occurred.connect(self.on_chart_error)
        
        layout.addWidget(self.dual_chart)
        
        widget.setMinimumWidth(800)
        return widget
        
    def on_date_mode_changed(self, mode):
        """Handle date mode change."""
        self.custom_date.hide()
        self.days_spin.hide()
        
        if mode == "Custom Date":
            self.custom_date.show()
        elif mode == "Last N Days":
            self.days_spin.show()
            
    def refresh_data(self):
        """Refresh data based on current settings."""
        mode = self.date_combo.currentText()
        
        if mode == "Today":
            self.load_today_scans()
        elif mode == "Yesterday":
            self.load_yesterday_scans()
        elif mode == "Custom Date":
            self.load_custom_date_scans()
        elif mode == "Last N Days":
            self.load_recent_scans()
            
    def load_today_scans(self):
        """Load today's scanner results."""
        self.status_label.setText("Loading today's scans...")
        self.status_label.setStyleSheet("color: #f59e0b;")
        
        self.data_worker = ScannerDataWorker(
            'today',
            passed_filters_only=self.passed_only_check.isChecked(),
            sort_by=self.get_sort_column(),
            ascending=self.get_sort_order()
        )
        
        self.data_worker.data_ready.connect(self.on_data_ready)
        self.data_worker.error_occurred.connect(self.on_data_error)
        self.data_worker.start()
        
    def load_yesterday_scans(self):
        """Load yesterday's scanner results."""
        from datetime import timedelta
        yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
        
        self.status_label.setText("Loading yesterday's scans...")
        self.status_label.setStyleSheet("color: #f59e0b;")
        
        self.data_worker = ScannerDataWorker(
            'by_date',
            scan_date=yesterday,
            passed_filters_only=self.passed_only_check.isChecked(),
            sort_by=self.get_sort_column(),
            ascending=self.get_sort_order()
        )
        
        self.data_worker.data_ready.connect(self.on_data_ready)
        self.data_worker.error_occurred.connect(self.on_data_error)
        self.data_worker.start()
        
    def load_custom_date_scans(self):
        """Load scanner results for custom date."""
        selected_date = self.custom_date.date().toPyDate()
        
        self.status_label.setText(f"Loading scans for {selected_date}...")
        self.status_label.setStyleSheet("color: #f59e0b;")
        
        self.data_worker = ScannerDataWorker(
            'by_date',
            scan_date=selected_date,
            passed_filters_only=self.passed_only_check.isChecked(),
            sort_by=self.get_sort_column(),
            ascending=self.get_sort_order()
        )
        
        self.data_worker.data_ready.connect(self.on_data_ready)
        self.data_worker.error_occurred.connect(self.on_data_error)
        self.data_worker.start()
        
    def load_recent_scans(self):
        """Load recent scanner results."""
        days = self.days_spin.value()
        
        self.status_label.setText(f"Loading last {days} days...")
        self.status_label.setStyleSheet("color: #f59e0b;")
        
        self.data_worker = ScannerDataWorker(
            'recent',
            days=days,
            passed_filters_only=self.passed_only_check.isChecked(),
            sort_by='scan_date',
            include_today=True
        )
        
        self.data_worker.data_ready.connect(self.on_data_ready)
        self.data_worker.error_occurred.connect(self.on_data_error)
        self.data_worker.start()
        
    def get_sort_column(self):
        """Get database column name for sorting."""
        sort_map = {
            "Rank": "rank",
            "Interest Score": "interest_score",
            "PM Volume": "premarket_volume",
            "ATR %": "atr_percent"
        }
        return sort_map.get(self.sort_combo.currentText(), "rank")
        
    def get_sort_order(self):
        """Get sort order based on column."""
        # Rank should be ascending, others descending
        return self.sort_combo.currentText() == "Rank"
        
    def on_data_ready(self, data: List[Dict[str, Any]]):
        """Handle data ready from worker."""
        self.scanner_data = data
        self.update_table()
        
        if data:
            self.status_label.setText(f"Loaded {len(data)} results")
            self.status_label.setStyleSheet("color: #10b981;")
        else:
            self.status_label.setText("No results found")
            self.status_label.setStyleSheet("color: #f59e0b;")
            
    def on_data_error(self, error_msg: str):
        """Handle data error from worker."""
        self.status_label.setText("Error loading data")
        self.status_label.setToolTip(error_msg)
        self.status_label.setStyleSheet("color: #ef4444;")
        logger.error(f"Data error: {error_msg}")
        
    def update_table(self):
        """Update the results table with scanner data."""
        self.results_table.setRowCount(len(self.scanner_data))
        
        if not self.scanner_data:
            self.table_info.setText("No data to display")
            return
            
        # Update table info
        date_range = set()
        for item in self.scanner_data:
            date_range.add(item['scan_date'])
        
        if len(date_range) == 1:
            self.table_info.setText(f"Showing {len(self.scanner_data)} results for {list(date_range)[0]}")
        else:
            self.table_info.setText(f"Showing {len(self.scanner_data)} results from {len(date_range)} days")
        
        # Populate table
        for row, scan in enumerate(self.scanner_data):
            # Rank
            rank_item = QTableWidgetItem(str(scan['rank']))
            rank_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_table.setItem(row, 0, rank_item)
            
            # Ticker
            ticker_item = QTableWidgetItem(scan['ticker'])
            ticker_item.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            ticker_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_table.setItem(row, 1, ticker_item)
            
            # Price
            price_item = QTableWidgetItem(f"${scan['price']:.2f}")
            price_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.results_table.setItem(row, 2, price_item)
            
            # Interest Score
            interest_item = QTableWidgetItem(f"{scan['interest_score']:.1f}")
            interest_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            # Color code interest score
            if scan['interest_score'] >= 80:
                interest_item.setForeground(QBrush(QColor("#10b981")))
            elif scan['interest_score'] >= 60:
                interest_item.setForeground(QBrush(QColor("#f59e0b")))
            self.results_table.setItem(row, 3, interest_item)
            
            # PM Volume
            pm_vol_item = QTableWidgetItem(f"{scan['premarket_volume']:,}")
            pm_vol_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.results_table.setItem(row, 4, pm_vol_item)
            
            # Avg Volume
            avg_vol_item = QTableWidgetItem(f"{scan['avg_daily_volume']:,}")
            avg_vol_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.results_table.setItem(row, 5, avg_vol_item)
            
            # PM/Avg Ratio
            if scan['avg_daily_volume'] > 0:
                ratio = (scan['premarket_volume'] / scan['avg_daily_volume']) * 100
                ratio_item = QTableWidgetItem(f"{ratio:.1f}%")
                if ratio >= 20:
                    ratio_item.setForeground(QBrush(QColor("#10b981")))
            else:
                ratio_item = QTableWidgetItem("N/A")
            ratio_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_table.setItem(row, 6, ratio_item)
            
            # ATR
            atr_item = QTableWidgetItem(f"${scan['atr']:.2f}")
            atr_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.results_table.setItem(row, 7, atr_item)
            
            # ATR %
            atr_pct_item = QTableWidgetItem(f"{scan['atr_percent']:.2f}%")
            atr_pct_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if scan['atr_percent'] >= 3.0:
                atr_pct_item.setForeground(QBrush(QColor("#10b981")))
            self.results_table.setItem(row, 8, atr_pct_item)
            
            # Dollar Volume
            dollar_vol = scan['dollar_volume'] / 1_000_000  # Convert to millions
            dollar_vol_item = QTableWidgetItem(f"${dollar_vol:.1f}M")
            dollar_vol_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.results_table.setItem(row, 9, dollar_vol_item)
            
            # Chart button
            chart_btn = QPushButton("View")
            chart_btn.setMaximumWidth(70)
            chart_btn.clicked.connect(lambda checked, t=scan['ticker']: self.load_ticker_chart(t))
            self.results_table.setCellWidget(row, 10, chart_btn)
            
    def load_ticker_chart(self, ticker: str):
        """Load HVN charts for selected ticker."""
        self.current_ticker = ticker
        self.chart_info.setText(f"Loading HVN analysis for {ticker}...")
        self.chart_info.setStyleSheet("color: #f59e0b; padding: 5px; font-size: 14px;")
        
        # Load in dual chart
        self.dual_chart.load_ticker(ticker)
        
    def on_chart_loading_started(self):
        """Handle chart loading started."""
        logger.info(f"Started loading charts for {self.current_ticker}")
        
    def on_chart_loading_finished(self):
        """Handle chart loading finished."""
        self.chart_info.setText(f"HVN Analysis: {self.current_ticker}")
        self.chart_info.setStyleSheet("color: #10b981; padding: 5px; font-size: 14px; font-weight: bold;")
        logger.info(f"Finished loading charts for {self.current_ticker}")
        
    def on_chart_error(self, error_msg: str):
        """Handle chart error."""
        self.chart_info.setText(f"Error loading {self.current_ticker}: {error_msg}")
        self.chart_info.setStyleSheet("color: #ef4444; padding: 5px; font-size: 14px;")
        logger.error(f"Chart error: {error_msg}")
        
    def resizeEvent(self, event):
        """Handle window resize."""
        super().resizeEvent(event)
        
        # Adjust table column widths on resize
        if hasattr(self, 'results_table'):
            # Force table to recalculate column widths
            self.results_table.resizeColumnsToContents()


# ============= MAIN =============
if __name__ == "__main__":
    print("=== Scanner Results Viewer Test ===\n")
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Create and show viewer
    viewer = ScannerResultsViewer()
    viewer.show()
    
    print("Scanner Results Viewer launched.")
    print("Features:")
    print("- Loads today's premarket scan results on startup")
    print("- Click 'View' button to load HVN charts for any ticker")
    print("- Charts show 7-day and 28-day HVN lookback periods")
    print("- Use controls to filter and sort results")
    print("- Window and charts resize dynamically")
    
    sys.exit(app.exec())