# modules/ui/dashboards/hvn_dashboard.py
"""
Module: HVN Visualization Dashboard
Purpose: Clean visualization of HVN analysis with live data integration
UI Framework: PyQt6 with PyQtGraph for charts
Note: All times are in UTC for consistency with trading data
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time, timezone
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, 
                           QDoubleSpinBox, QTableWidget, QTableWidgetItem, 
                           QHeaderView, QGroupBox, QSplitter, QComboBox,
                           QCheckBox, QDateTimeEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDateTime, QTimer
from PyQt6.QtGui import QFont, QColor
import pyqtgraph as pg
import os

# Get the Vega root directory - MUST BE BEFORE ANY LOCAL IMPORTS
current_dir = os.path.dirname(os.path.abspath(__file__))  # dashboards
ui_dir = os.path.dirname(current_dir)  # ui
modules_dir = os.path.dirname(ui_dir)  # modules
vega_root = os.path.dirname(modules_dir)  # Vega

# Remove any existing 'polygon' from sys.modules to avoid conflicts
if 'polygon' in sys.modules:
    del sys.modules['polygon']

# Add vega_root to the FRONT of sys.path - matching polygon_bridge.py exactly
sys.path.insert(0, vega_root)

# Now import YOUR modules
from modules.calculations.volume.hvn_engine import HVNEngine
from modules.data.polygon_bridge import PolygonHVNBridge

# Remove vega_root from path after imports to keep things clean
sys.path.remove(vega_root)

# Configure PyQtGraph
pg.setConfigOptions(antialias=True, useOpenGL=True)
pg.setConfigOption('background', '#1a1a1a')
pg.setConfigOption('foreground', '#ffffff')


class DataWorker(QThread):
    """Background thread for data fetching and HVN calculation."""
    data_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, symbol, lookback_days, end_datetime, timeframe, hvn_levels, percentile):
        super().__init__()
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.end_datetime = end_datetime
        self.timeframe = timeframe
        self.hvn_levels = hvn_levels
        self.percentile = percentile
        
    def run(self):
        try:
            # Initialize bridge
            bridge = PolygonHVNBridge(
                hvn_levels=self.hvn_levels,
                hvn_percentile=self.percentile,
                lookback_days=self.lookback_days
            )
            
            # Calculate HVN with custom end date
            # This WILL fetch data up to the specified end_datetime
            state = bridge.calculate_hvn(
                self.symbol, 
                end_date=self.end_datetime,
                timeframe=self.timeframe
            )
            
            # Add end_datetime to state for display
            state.end_datetime = self.end_datetime
            
            self.data_ready.emit(state)
        except Exception as e:
            self.error_occurred.emit(str(e))


class CandlestickItem(pg.GraphicsObject):
    """Custom candlestick item for cleaner display."""
    
    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.data = data
        self.generatePicture()
        
    def generatePicture(self):
        self.picture = pg.QtGui.QPicture()
        p = pg.QtGui.QPainter(self.picture)
        
        for i, (_, row) in enumerate(self.data.iterrows()):
            # Determine color
            if row['close'] >= row['open']:
                p.setPen(pg.mkPen('#10b981', width=1))
                p.setBrush(pg.mkBrush('#10b981'))
            else:
                p.setPen(pg.mkPen('#ef4444', width=1))
                p.setBrush(pg.mkBrush('#ef4444'))
            
            # Draw high-low line
            p.drawLine(pg.QtCore.QPointF(i, row['low']), 
                      pg.QtCore.QPointF(i, row['high']))
            
            # Draw open-close rectangle
            height = abs(row['close'] - row['open'])
            if height > 0:
                p.drawRect(pg.QtCore.QRectF(i - 0.3, 
                                           min(row['open'], row['close']), 
                                           0.6, 
                                           height))
        
        p.end()
    
    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)
    
    def boundingRect(self):
        return pg.QtCore.QRectF(self.picture.boundingRect())


class HVNDashboard(QMainWindow):
    """Main HVN Visualization Dashboard."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HVN Analysis Dashboard - UTC Times")
        self.setGeometry(100, 100, 1500, 800)
        
        # Set minimum window size
        self.setMinimumSize(1200, 600)
        
        # Apply dark theme
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
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QDateTimeEdit {
                background-color: #2a2a2a;
                border: 1px solid #444444;
                border-radius: 3px;
                padding: 5px;
                color: #ffffff;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QDateTimeEdit:focus {
                border: 1px solid #10b981;
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
                padding: 8px 16px;
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
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #333333;
                border: none;
                padding: 8px;
                font-weight: bold;
            }
            QLabel {
                color: #e5e7eb;
            }
        """)
        
        self.data_worker = None
        self.current_state = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the main UI components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Remove margins for better space usage
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)  # Reduce spacing between sections
        
        # Top section: Configuration controls with FIXED HEIGHT
        config_widget = self.create_config_section()
        config_widget.setMaximumHeight(120)  # Fix the height of config section
        main_layout.addWidget(config_widget)
        
        # Main content: Splitter (this will expand to fill remaining space)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: Tables
        tables_widget = self.create_tables_section()
        self.main_splitter.addWidget(tables_widget)
        
        # Right side: Chart
        chart_widget = self.create_chart_section()
        self.main_splitter.addWidget(chart_widget)
        
        # Set splitter proportions (35% tables, 65% chart)
        self.main_splitter.setSizes([525, 975])
        
        # Make splitter responsive
        self.main_splitter.setStretchFactor(0, 35)  # Tables
        self.main_splitter.setStretchFactor(1, 65)  # Chart
        
        # Add splitter with stretch factor so it takes all remaining space
        main_layout.addWidget(self.main_splitter, 1)  # The '1' gives it all extra space
        
    def create_config_section(self):
        """Create configuration controls."""
        group = QGroupBox("HVN Configuration")
        group.setMaximumHeight(120)  # Ensure the group box respects height limit
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)  # Reduce internal margins
        main_layout.setSpacing(5)  # Reduce spacing
        
        # Row 1: Basic config
        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(10)  # Reduce horizontal spacing
        
        # Symbol
        symbol_label = QLabel("Symbol:")
        self.symbol_input = QLineEdit("TSLA")
        self.symbol_input.setMaximumWidth(80)
        row1_layout.addWidget(symbol_label)
        row1_layout.addWidget(self.symbol_input)
        
        # Timeframe
        timeframe_label = QLabel("Timeframe:")
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(["1min", "5min", "15min", "30min", "1h"])
        self.timeframe_combo.setCurrentText("15min")
        self.timeframe_combo.setMaximumWidth(80)
        row1_layout.addWidget(timeframe_label)
        row1_layout.addWidget(self.timeframe_combo)
        
        # Lookback days
        lookback_label = QLabel("Days:")
        self.lookback_spin = QSpinBox()
        self.lookback_spin.setRange(1, 180)
        self.lookback_spin.setValue(5)
        self.lookback_spin.setMaximumWidth(60)
        row1_layout.addWidget(lookback_label)
        row1_layout.addWidget(self.lookback_spin)
        
        # HVN Levels
        levels_label = QLabel("Levels:")
        self.levels_spin = QSpinBox()
        self.levels_spin.setRange(50, 200)
        self.levels_spin.setValue(100)
        self.levels_spin.setMaximumWidth(60)
        row1_layout.addWidget(levels_label)
        row1_layout.addWidget(self.levels_spin)
        
        # Percentile
        percentile_label = QLabel("Percentile:")
        self.percentile_spin = QDoubleSpinBox()
        self.percentile_spin.setRange(50.0, 95.0)
        self.percentile_spin.setValue(80.0)
        self.percentile_spin.setSuffix("%")
        self.percentile_spin.setMaximumWidth(80)
        row1_layout.addWidget(percentile_label)
        row1_layout.addWidget(self.percentile_spin)
        
        row1_layout.addStretch()
        main_layout.addLayout(row1_layout)
        
        # Row 2: Time controls
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(10)  # Reduce horizontal spacing
        
        # Live data checkbox
        self.live_data_check = QCheckBox("Live Data")
        self.live_data_check.setChecked(True)
        self.live_data_check.toggled.connect(self.toggle_time_controls)
        row2_layout.addWidget(self.live_data_check)
        
        # Custom end time controls
        row2_layout.addWidget(QLabel("End Time (UTC):"))
        
        # Date/Time picker
        self.end_datetime_edit = QDateTimeEdit()
        self.end_datetime_edit.setTimeSpec(Qt.TimeSpec.UTC)
        self.end_datetime_edit.setDateTime(QDateTime.currentDateTimeUtc())
        self.end_datetime_edit.setCalendarPopup(True)
        self.end_datetime_edit.setDisplayFormat("yyyy-MM-dd HH:mm UTC")
        self.end_datetime_edit.setEnabled(False)
        self.end_datetime_edit.setMinimumWidth(160)
        row2_layout.addWidget(self.end_datetime_edit)
        
        # Quick presets
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Custom",
            "Today NY Open (14:30 UTC)",
            "Yesterday NY Close (21:00 UTC)",
            "Yesterday NY Open (14:30 UTC)",
            "Last Friday NY Close",
            "1 Hour Ago"
        ])
        self.preset_combo.setEnabled(False)
        self.preset_combo.currentTextChanged.connect(self.apply_time_preset)
        self.preset_combo.setMinimumWidth(160)
        row2_layout.addWidget(self.preset_combo)
        
        # Calculate button
        self.calculate_btn = QPushButton("Calculate HVN")
        self.calculate_btn.clicked.connect(self.fetch_and_calculate)
        row2_layout.addWidget(self.calculate_btn)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #10b981;")
        row2_layout.addWidget(self.status_label)
        
        row2_layout.addStretch()
        main_layout.addLayout(row2_layout)
        
        group.setLayout(main_layout)
        return group
    
    def toggle_time_controls(self, checked):
        """Enable/disable time controls based on live data checkbox."""
        self.end_datetime_edit.setEnabled(not checked)
        self.preset_combo.setEnabled(not checked)
        
        if not checked:
            # Set to current UTC time when switching to custom
            self.end_datetime_edit.setDateTime(QDateTime.currentDateTimeUtc())
            self.preset_combo.setCurrentText("Custom")
    
    def apply_time_preset(self, preset_text):
        """Apply time preset to datetime picker (all in UTC)."""
        if preset_text == "Custom":
            return
            
        # Work in UTC
        now_utc = datetime.now(timezone.utc)
        
        if preset_text == "Today NY Open (14:30 UTC)":
            # NY market opens at 9:30 AM ET = 14:30 UTC (or 13:30 during DST)
            target = now_utc.replace(hour=14, minute=30, second=0, microsecond=0)
        elif preset_text == "Yesterday NY Close (21:00 UTC)":
            # NY market closes at 4:00 PM ET = 21:00 UTC (or 20:00 during DST)
            yesterday = now_utc - timedelta(days=1)
            target = yesterday.replace(hour=21, minute=0, second=0, microsecond=0)
        elif preset_text == "Yesterday NY Open (14:30 UTC)":
            yesterday = now_utc - timedelta(days=1)
            target = yesterday.replace(hour=14, minute=30, second=0, microsecond=0)
        elif preset_text == "Last Friday NY Close":
            days_since_friday = (now_utc.weekday() - 4) % 7
            if days_since_friday == 0:
                days_since_friday = 7
            last_friday = now_utc - timedelta(days=days_since_friday)
            target = last_friday.replace(hour=21, minute=0, second=0, microsecond=0)
        elif preset_text == "1 Hour Ago":
            target = now_utc - timedelta(hours=1)
        else:
            return
        
        # Convert to QDateTime in UTC
        qt_datetime = QDateTime(
            target.year, target.month, target.day,
            target.hour, target.minute, target.second
        )
        qt_datetime.setTimeSpec(Qt.TimeSpec.UTC)
        self.end_datetime_edit.setDateTime(qt_datetime)
        
    def create_tables_section(self):
        """Create tables section."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Time info label
        self.time_info_label = QLabel("Data Period: --")
        self.time_info_label.setStyleSheet("color: #9ca3af; padding: 5px;")
        layout.addWidget(self.time_info_label)
        
        # Create a splitter for the two tables
        tables_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # HVN Levels Table
        levels_group = QGroupBox("HVN Levels (5 Above / 5 Below)")
        levels_layout = QVBoxLayout()
        levels_layout.setContentsMargins(5, 5, 5, 5)
        
        self.levels_table = QTableWidget(10, 4)
        self.levels_table.setHorizontalHeaderLabels(["Price", "Volume %", "Distance", "Position"])
        
        # Make table responsive
        header = self.levels_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        
        self.levels_table.verticalHeader().setVisible(False)
        self.levels_table.setAlternatingRowColors(True)
        levels_layout.addWidget(self.levels_table)
        
        levels_group.setLayout(levels_layout)
        tables_splitter.addWidget(levels_group)
        
        # HVN Clusters Table
        clusters_group = QGroupBox("Top 5 HVN Clusters")
        clusters_layout = QVBoxLayout()
        clusters_layout.setContentsMargins(5, 5, 5, 5)
        
        self.clusters_table = QTableWidget(5, 5)
        self.clusters_table.setHorizontalHeaderLabels(["Range", "Center", "Volume %", "Levels", "Status"])
        
        # Make table responsive
        header = self.clusters_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        
        self.clusters_table.verticalHeader().setVisible(False)
        self.clusters_table.setAlternatingRowColors(True)
        clusters_layout.addWidget(self.clusters_table)
        
        clusters_group.setLayout(clusters_layout)
        tables_splitter.addWidget(clusters_group)
        
        # Set equal stretch factors for both tables
        tables_splitter.setStretchFactor(0, 1)
        tables_splitter.setStretchFactor(1, 1)
        
        layout.addWidget(tables_splitter)
        
        # Set minimum width for tables section
        widget.setMinimumWidth(400)
        
        return widget
        
    def create_chart_section(self):
        """Create chart section."""
        self.chart_widget = pg.GraphicsLayoutWidget()
        
        # Set minimum size for chart
        self.chart_widget.setMinimumSize(600, 400)
        
        # Main candlestick plot
        self.price_plot = self.chart_widget.addPlot(row=0, col=0)
        self.price_plot.setLabel('left', 'Price', units='$')
        self.price_plot.setLabel('bottom', 'Time (UTC)')
        self.price_plot.showGrid(x=False, y=False)
        
        # Volume profile plot
        self.volume_plot = self.chart_widget.addPlot(row=0, col=1)
        self.volume_plot.setLabel('bottom', 'Volume %')
        self.volume_plot.hideAxis('left')
        self.volume_plot.setMaximumWidth(150)
        self.volume_plot.showGrid(x=False, y=False)
        
        # Link Y axes
        self.volume_plot.setYLink(self.price_plot)
        
        return self.chart_widget
    
    def fetch_and_calculate(self):
        """Fetch data and calculate HVN in background."""
        # Check if worker is already running
        if self.data_worker and self.data_worker.isRunning():
            self.status_label.setText("Calculation in progress...")
            self.status_label.setStyleSheet("color: #f59e0b;")
            return
            
        self.status_label.setText("Fetching data...")
        self.status_label.setStyleSheet("color: #f59e0b;")
        self.calculate_btn.setEnabled(False)
        
        # Determine end datetime
        if self.live_data_check.isChecked():
            # Use current UTC time for live data
            end_datetime = datetime.now(timezone.utc).replace(tzinfo=None)
        else:
            # Convert QDateTime to Python datetime (already in UTC)
            qt_dt = self.end_datetime_edit.dateTime()
            end_datetime = datetime(
                qt_dt.date().year(), qt_dt.date().month(), qt_dt.date().day(),
                qt_dt.time().hour(), qt_dt.time().minute(), qt_dt.time().second(),
                tzinfo=timezone.utc
            ).replace(tzinfo=None)  # Remove timezone for compatibility
        
        # Start background worker
        self.data_worker = DataWorker(
            self.symbol_input.text().upper(),
            self.lookback_spin.value(),
            end_datetime,  # This will fetch data up to this exact time
            self.timeframe_combo.currentText(),
            self.levels_spin.value(),
            self.percentile_spin.value()
        )
        
        self.data_worker.data_ready.connect(self.on_data_ready)
        self.data_worker.error_occurred.connect(self.on_error)
        self.data_worker.start()
        
    def on_data_ready(self, state):
        """Handle data ready from worker thread."""
        self.current_state = state
        self.update_display()
        
        # Update status
        self.status_label.setText(f"Updated: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
        self.status_label.setStyleSheet("color: #10b981;")
        self.calculate_btn.setEnabled(True)
        
    def on_error(self, error_msg):
        """Handle error from worker thread."""
        # Show truncated error in status
        self.status_label.setText("Error: See details")
        self.status_label.setToolTip(error_msg)  # Full error in tooltip
        self.status_label.setStyleSheet("color: #ef4444;")
        self.calculate_btn.setEnabled(True)
        
        # Log full error
        print(f"HVN Calculation Error: {error_msg}")
        
    def update_display(self):
        """Update all displays with new data."""
        if not self.current_state:
            return
            
        # Validate data
        if self.current_state.recent_bars.empty:
            self.status_label.setText("No data to display")
            return
            
        if not self.current_state.hvn_result.filtered_levels:
            self.status_label.setText("No HVN levels found")
            return
        
        # Update time info (timestamps should already be in UTC from Polygon)
        data = self.current_state.recent_bars
        start_time = data.index[0].strftime('%Y-%m-%d %H:%M UTC')
        end_time = data.index[-1].strftime('%Y-%m-%d %H:%M UTC')
        self.time_info_label.setText(f"Data Period: {start_time} to {end_time} ({len(data)} bars)")
        
        # Update levels table
        self.update_levels_table()
        
        # Update clusters table
        self.update_clusters_table()
        
        # Update chart
        self.update_chart()
        
    def update_levels_table(self):
        """Update HVN levels table (5 above, 5 below current price)."""
        current_price = self.current_state.current_price
        levels = self.current_state.hvn_result.filtered_levels
        
        # Separate levels above and below current price
        above_levels = [l for l in levels if l.center > current_price]
        below_levels = [l for l in levels if l.center <= current_price]
        
        # Sort by distance to current price
        above_levels.sort(key=lambda x: x.center - current_price)
        below_levels.sort(key=lambda x: current_price - x.center)
        
        # Take 5 closest from each
        display_levels = above_levels[:5] + below_levels[:5]
        
        self.levels_table.setRowCount(len(display_levels))
        
        for i, level in enumerate(display_levels):
            # Price
            self.levels_table.setItem(i, 0, QTableWidgetItem(f"${level.center:.2f}"))
            
            # Volume %
            self.levels_table.setItem(i, 1, QTableWidgetItem(f"{level.percent_of_total:.2f}%"))
            
            # Distance
            distance = abs(level.center - current_price)
            self.levels_table.setItem(i, 2, QTableWidgetItem(f"${distance:.2f}"))
            
            # Position
            position = "Above" if level.center > current_price else "Below"
            position_item = QTableWidgetItem(position)
            if position == "Above":
                position_item.setForeground(QColor("#10b981"))
            else:
                position_item.setForeground(QColor("#ef4444"))
            self.levels_table.setItem(i, 3, position_item)
            
    def update_clusters_table(self):
        """Update HVN clusters table."""
        clusters = self.current_state.hvn_result.clusters[:5]  # Top 5 clusters
        current_price = self.current_state.current_price
        
        self.clusters_table.setRowCount(len(clusters))
        
        for i, cluster in enumerate(clusters):
            # Range
            range_text = f"${cluster.cluster_low:.2f} - ${cluster.cluster_high:.2f}"
            self.clusters_table.setItem(i, 0, QTableWidgetItem(range_text))
            
            # Center
            self.clusters_table.setItem(i, 1, QTableWidgetItem(f"${cluster.center_price:.2f}"))
            
            # Volume %
            self.clusters_table.setItem(i, 2, QTableWidgetItem(f"{cluster.total_percent:.2f}%"))
            
            # Levels
            self.clusters_table.setItem(i, 3, QTableWidgetItem(str(len(cluster.levels))))
            
            # Status
            if cluster.cluster_low <= current_price <= cluster.cluster_high:
                status = "Inside"
                status_item = QTableWidgetItem(status)
                status_item.setForeground(QColor("#f59e0b"))
            else:
                distance = min(
                    abs(current_price - cluster.cluster_high),
                    abs(current_price - cluster.cluster_low)
                )
                status = f"${distance:.2f} away"
                status_item = QTableWidgetItem(status)
                status_item.setForeground(QColor("#6b7280"))
            
            self.clusters_table.setItem(i, 4, status_item)
            
    def update_chart(self):
        """Update the chart with candlesticks and volume profile."""
        # Clear previous items
        self.price_plot.clear()
        self.volume_plot.clear()
        
        # Get data
        data = self.current_state.recent_bars
        
        # Limit to last N candles for performance if needed
        max_candles = 500
        if len(data) > max_candles:
            data = data.tail(max_candles)
        
        # Reset index for plotting
        data_reset = data.reset_index()
        
        # Create custom x-axis with time labels (in UTC)
        time_strings = [t.strftime('%m/%d %H:%M') for t in data.index]
        x_dict = dict(enumerate(time_strings))
        
        # Only show every Nth label to avoid crowding
        step = max(1, len(time_strings) // 10)
        x_dict_sparse = {k: v for k, v in x_dict.items() if k % step == 0}
        
        stringaxis = pg.AxisItem(orientation='bottom')
        stringaxis.setTicks([list(x_dict_sparse.items())])
        self.price_plot.setAxisItems(axisItems={'bottom': stringaxis})
        
        # Draw candlesticks
        candlestick = CandlestickItem(data_reset)
        self.price_plot.addItem(candlestick)
        
        # Add HVN cluster zones as horizontal bars
        for i, cluster in enumerate(self.current_state.hvn_result.clusters[:5]):
            # Use different colors for different clusters
            colors = [
                (16, 185, 129, 40),   # Green
                (59, 130, 246, 40),   # Blue
                (168, 85, 247, 40),   # Purple
                (251, 146, 60, 40),   # Orange
                (236, 72, 153, 40),   # Pink
            ]
            
            zone = pg.LinearRegionItem(
                values=(cluster.cluster_low, cluster.cluster_high),
                orientation='horizontal',
                brush=pg.mkBrush(*colors[i % len(colors)]),
                movable=False
            )
            self.price_plot.addItem(zone)
        
        # Draw volume profile
        price_levels = self.current_state.hvn_result.ranked_levels
        max_volume_pct = max([level.percent_of_total for level in price_levels]) if price_levels else 1
        
        for level in price_levels:
            # Create horizontal bar
            bar_width = (level.percent_of_total / max_volume_pct) * 40  # Scale for visibility
            rect = pg.QtWidgets.QGraphicsRectItem(0, level.low, bar_width, level.high - level.low)
            rect.setPen(pg.mkPen(None))
            
            # Color based on rank
            if level.rank >= 80:
                rect.setBrush(pg.mkBrush(16, 185, 129, 180))  # High volume - green
            elif level.rank >= 60:
                rect.setBrush(pg.mkBrush(59, 130, 246, 150))  # Medium - blue
            else:
                rect.setBrush(pg.mkBrush(107, 114, 128, 80))  # Low - gray
                
            self.volume_plot.addItem(rect)
        
        # Add current price line
        current_price = self.current_state.current_price
        price_line = pg.InfiniteLine(
            pos=current_price, 
            angle=0, 
            pen=pg.mkPen('#f59e0b', width=2, style=Qt.PenStyle.DashLine),
            label=f'Current: ${current_price:.2f}',
            labelOpts={'position': 0.95, 'color': '#f59e0b'}
        )
        self.price_plot.addItem(price_line)
        
        # Set axis ranges
        self.price_plot.setXRange(0, len(data))
        y_min = data['low'].min() * 0.998
        y_max = data['high'].max() * 1.002
        self.price_plot.setYRange(y_min, y_max)
        self.volume_plot.setXRange(0, 50)
    
    def resizeEvent(self, event):
        """Handle window resize events."""
        super().resizeEvent(event)
        
        # Adjust volume plot width based on window size
        window_width = self.width()
        if window_width > 1600:
            self.volume_plot.setMaximumWidth(200)
        elif window_width > 1200:
            self.volume_plot.setMaximumWidth(150)
        else:
            self.volume_plot.setMaximumWidth(120)
        
        # Force chart redraw if data exists
        if hasattr(self, 'current_state') and self.current_state:
            # Schedule a chart update after resize is complete
            QTimer.singleShot(100, self.update_chart)


# ============= MAIN =============
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show dashboard
    dashboard = HVNDashboard()
    dashboard.show()
    
    sys.exit(app.exec())