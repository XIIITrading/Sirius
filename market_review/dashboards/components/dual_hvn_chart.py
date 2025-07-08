# market_review/dashboards/components/dual_hvn_chart.py
"""
Module: Dual HVN Chart Component
Purpose: Reusable component showing two vertically stacked HVN charts
         with configurable lookback periods and display bars
UI Framework: PyQt6 with PyQtGraph
Note: All times are in UTC
"""

# Standard library imports
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

# Third-party imports
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QSplitter, QGroupBox, QSizePolicy, QLineEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor

# Local application imports
from market_review.calculations.volume.hvn_engine import HVNEngine
from market_review.data.polygon_bridge import PolygonHVNBridge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure PyQtGraph
pg.setConfigOptions(antialias=True)


class ChartDataWorker(QThread):
    """Background thread for fetching data and calculating HVN."""
    data_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(str)
    
    def __init__(self, ticker: str, lookback_days_hvn: int, display_bars: int = 390):
        """
        Args:
            ticker: Stock symbol
            lookback_days_hvn: Days for HVN calculation
            display_bars: Number of bars to display (390 = ~5 days of 15min bars)
        """
        super().__init__()
        self.ticker = ticker
        self.lookback_days_hvn = lookback_days_hvn
        self.display_bars = display_bars
        
    def run(self):
        try:
            self.progress_update.emit(f"Fetching data for {self.ticker}...")
            
            # Initialize bridge with HVN lookback period
            bridge = PolygonHVNBridge(
                hvn_levels=100,
                hvn_percentile=80.0,
                lookback_days=self.lookback_days_hvn
            )
            
            # Calculate HVN
            state = bridge.calculate_hvn(
                self.ticker,
                timeframe='15min'
            )
            
            # Package results
            result = {
                'ticker': self.ticker,
                'lookback_days': self.lookback_days_hvn,
                'state': state,
                'display_bars': self.display_bars
            }
            
            self.data_ready.emit(result)
            
        except Exception as e:
            logger.error(f"Error in ChartDataWorker: {e}")
            self.error_occurred.emit(str(e))


class CandlestickItem(pg.GraphicsObject):
    """Custom candlestick item for charts."""
    
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


class DualHVNChart(QWidget):
    """
    Reusable dual HVN chart component.
    Shows two vertically stacked charts with configurable lookback periods.
    """
    
    # Signals
    loading_started = pyqtSignal()
    loading_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self, parent=None, lookback_periods=None, display_bars=390):
        """
        Initialize the dual HVN chart.
        
        Args:
            parent: Parent widget
            lookback_periods: List of two integers for lookback periods [period1, period2]
                            Default is [7, 28]
            display_bars: Number of bars to display (default: 390 = ~5 days of 15min bars)
                         - 78 bars = ~1 day (6.5 hours)
                         - 390 bars = ~5 days
                         - 1092 bars = ~14 days
        """
        super().__init__(parent)
        
        # Set configurable parameters
        self.lookback_periods = lookback_periods or [7, 28]
        self.display_bars = display_bars
        
        # Validate lookback periods
        if len(self.lookback_periods) != 2:
            raise ValueError("lookback_periods must contain exactly 2 values")
        
        self.ticker = None
        self.chart_data = {
            f'{self.lookback_periods[0]}_day': None,
            f'{self.lookback_periods[1]}_day': None
        }
        self.workers = []
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Create splitter for vertical stacking
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setChildrenCollapsible(False)
        
        # Create chart widgets with dynamic titles
        self.chart_1 = self.create_chart_widget(f"{self.lookback_periods[0]}-Day HVN Lookback")
        self.chart_2 = self.create_chart_widget(f"{self.lookback_periods[1]}-Day HVN Lookback")
        
        # Add to splitter
        self.splitter.addWidget(self.chart_1['container'])
        self.splitter.addWidget(self.chart_2['container'])
        
        # Set equal sizes initially
        self.splitter.setSizes([400, 400])
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)
        
        # Add to main layout
        layout.addWidget(self.splitter)
        
        # Set size policy for dynamic resizing
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
    def create_chart_widget(self, title: str) -> Dict[str, Any]:
        """Create a single chart widget with HVN visualization."""
        container = QGroupBox(title)
        container.setStyleSheet("""
            QGroupBox {
                border: 1px solid #333333;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                background-color: #1a1a1a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #10b981;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Info label
        info_label = QLabel("No data loaded")
        info_label.setStyleSheet("color: #9ca3af; padding: 2px;")
        layout.addWidget(info_label)
        
        # Chart widget
        chart_widget = pg.GraphicsLayoutWidget()
        chart_widget.setBackground('#1a1a1a')
        
        # Create plots
        price_plot = chart_widget.addPlot(row=0, col=0)
        price_plot.setLabel('left', 'Price', units='$')
        price_plot.setLabel('bottom', 'Time (UTC)')
        price_plot.showGrid(x=True, y=True, alpha=0.3)
        
        volume_plot = chart_widget.addPlot(row=0, col=1)
        volume_plot.setLabel('bottom', 'Volume %')
        volume_plot.hideAxis('left')
        volume_plot.setMaximumWidth(150)
        volume_plot.showGrid(x=True, y=False, alpha=0.3)
        
        # Link Y axes
        volume_plot.setYLink(price_plot)
        
        layout.addWidget(chart_widget)
        container.setLayout(layout)
        
        return {
            'container': container,
            'info_label': info_label,
            'chart_widget': chart_widget,
            'price_plot': price_plot,
            'volume_plot': volume_plot
        }
    
    def update_parameters(self, lookback_periods=None, display_bars=None):
        """
        Update chart parameters. Requires reloading the ticker.
        
        Args:
            lookback_periods: New lookback periods [period1, period2]
            display_bars: New number of bars to display
        """
        if lookback_periods is not None:
            if len(lookback_periods) != 2:
                raise ValueError("lookback_periods must contain exactly 2 values")
            self.lookback_periods = lookback_periods
            
            # Update chart titles
            self.chart_1['container'].setTitle(f"{self.lookback_periods[0]}-Day HVN Lookback")
            self.chart_2['container'].setTitle(f"{self.lookback_periods[1]}-Day HVN Lookback")
            
            # Update chart data keys
            self.chart_data = {
                f'{self.lookback_periods[0]}_day': None,
                f'{self.lookback_periods[1]}_day': None
            }
        
        if display_bars is not None:
            self.display_bars = display_bars
        
        # Reload current ticker if one is loaded
        if self.ticker:
            self.load_ticker(self.ticker)
    
    def load_ticker(self, ticker: str):
        """
        Load data for a specific ticker.
        
        Args:
            ticker: Stock symbol to load
        """
        if not ticker:
            return
            
        self.ticker = ticker.upper()
        logger.info(f"Loading ticker: {self.ticker}")
        
        # Clear any existing workers
        self.clear_workers()
        
        # Clear existing charts
        self.clear_charts()
        
        # Update labels
        self.chart_1['info_label'].setText(f"Loading {self.ticker}...")
        self.chart_2['info_label'].setText(f"Loading {self.ticker}...")
        
        # Emit loading signal
        self.loading_started.emit()
        
        # Start workers for both lookback periods
        worker_1 = ChartDataWorker(
            self.ticker, 
            lookback_days_hvn=self.lookback_periods[0],
            display_bars=self.display_bars
        )
        worker_1.data_ready.connect(
            lambda data: self.on_data_ready(data, f'{self.lookback_periods[0]}_day')
        )
        worker_1.error_occurred.connect(
            lambda err: self.on_error(err, f'{self.lookback_periods[0]}_day')
        )
        worker_1.progress_update.connect(
            lambda msg: self.chart_1['info_label'].setText(msg)
        )
        
        worker_2 = ChartDataWorker(
            self.ticker, 
            lookback_days_hvn=self.lookback_periods[1],
            display_bars=self.display_bars
        )
        worker_2.data_ready.connect(
            lambda data: self.on_data_ready(data, f'{self.lookback_periods[1]}_day')
        )
        worker_2.error_occurred.connect(
            lambda err: self.on_error(err, f'{self.lookback_periods[1]}_day')
        )
        worker_2.progress_update.connect(
            lambda msg: self.chart_2['info_label'].setText(msg)
        )
        
        self.workers = [worker_1, worker_2]
        
        # Start workers
        worker_1.start()
        worker_2.start()
    
    def clear_workers(self):
        """Stop and clear any running workers."""
        for worker in self.workers:
            if worker.isRunning():
                worker.terminate()
                worker.wait()
        self.workers.clear()
    
    def clear_charts(self):
        """Clear all chart data."""
        self.chart_1['price_plot'].clear()
        self.chart_1['volume_plot'].clear()
        self.chart_2['price_plot'].clear()
        self.chart_2['volume_plot'].clear()
    
    def on_data_ready(self, data: dict, chart_key: str):
        """Handle data ready from worker."""
        self.chart_data[chart_key] = data
        
        # Determine which chart to update
        if chart_key == f'{self.lookback_periods[0]}_day':
            chart = self.chart_1
        else:
            chart = self.chart_2
            
        self.update_chart(chart, data)
        
        # Check if both charts are loaded
        if all(self.chart_data.values()):
            self.loading_finished.emit()
    
    def on_error(self, error_msg: str, chart_key: str):
        """Handle error from worker."""
        if chart_key == f'{self.lookback_periods[0]}_day':
            chart = self.chart_1
        else:
            chart = self.chart_2
            
        chart['info_label'].setText(f"Error: {error_msg}")
        chart['info_label'].setStyleSheet("color: #ef4444; padding: 2px;")
        
        self.error_occurred.emit(f"{chart_key} error: {error_msg}")
    
    def update_chart(self, chart_dict: dict, data: dict):
        """Update a single chart with data."""
        state = data['state']
        lookback_days = data['lookback_days']
        display_bars = data['display_bars']
        
        # Clear previous items
        chart_dict['price_plot'].clear()
        chart_dict['volume_plot'].clear()
        
        # Get ALL available data
        all_data = state.recent_bars
        if all_data.empty:
            chart_dict['info_label'].setText("No data available")
            return
        
        # Use ALL data for the chart (don't limit it)
        display_data = all_data.copy()
        
        # Update info label
        start_time = display_data.index[0].strftime('%Y-%m-%d %H:%M UTC')
        end_time = display_data.index[-1].strftime('%Y-%m-%d %H:%M UTC')
        
        # Calculate time span
        time_span = display_data.index[-1] - display_data.index[0]
        approx_days = time_span.total_seconds() / 86400
        
        chart_dict['info_label'].setText(
            f"{state.symbol} | {start_time} to {end_time} | "
            f"{len(display_data)} bars (~{approx_days:.1f} days) | Current: ${state.current_price:.2f}"
        )
        chart_dict['info_label'].setStyleSheet("color: #10b981; padding: 2px;")
        
        # Reset index for plotting
        display_data_reset = display_data.reset_index()
        
        # Create time axis
        time_strings = [t.strftime('%m/%d %H:%M') for t in display_data.index]
        x_dict = dict(enumerate(time_strings))
        
        # Show subset of labels
        step = max(1, len(time_strings) // 8)
        x_dict_sparse = {k: v for k, v in x_dict.items() if k % step == 0}
        
        stringaxis = pg.AxisItem(orientation='bottom')
        stringaxis.setTicks([list(x_dict_sparse.items())])
        chart_dict['price_plot'].setAxisItems(axisItems={'bottom': stringaxis})
        
        # Draw candlesticks
        candlestick = CandlestickItem(display_data_reset)
        chart_dict['price_plot'].addItem(candlestick)
        
        # Add HVN cluster zones
        colors = [
            (16, 185, 129, 30),   # Green
            (59, 130, 246, 30),   # Blue
            (168, 85, 247, 30),   # Purple
            (251, 146, 60, 30),   # Orange
            (236, 72, 153, 30),   # Pink
        ]
        
        for i, cluster in enumerate(state.hvn_result.clusters[:5]):
            zone = pg.LinearRegionItem(
                values=(cluster.cluster_low, cluster.cluster_high),
                orientation='horizontal',
                brush=pg.mkBrush(*colors[i % len(colors)]),
                movable=False
            )
            chart_dict['price_plot'].addItem(zone)
        
        # Draw volume profile
        price_levels = state.hvn_result.ranked_levels
        max_volume_pct = max([level.percent_of_total for level in price_levels]) if price_levels else 1
        
        for level in price_levels:
            bar_width = (level.percent_of_total / max_volume_pct) * 40
            rect = pg.QtWidgets.QGraphicsRectItem(0, level.low, bar_width, level.high - level.low)
            rect.setPen(pg.mkPen(None))
            
            if level.rank >= 80:
                rect.setBrush(pg.mkBrush(16, 185, 129, 180))
            elif level.rank >= 60:
                rect.setBrush(pg.mkBrush(59, 130, 246, 150))
            else:
                rect.setBrush(pg.mkBrush(107, 114, 128, 80))
                
            chart_dict['volume_plot'].addItem(rect)
        
        # Add current price line
        current_price = state.current_price
        price_line = pg.InfiniteLine(
            pos=current_price, 
            angle=0, 
            pen=pg.mkPen('#f59e0b', width=2, style=Qt.PenStyle.DashLine),
            label=f'${current_price:.2f}',
            labelOpts={'position': 0.95, 'color': '#f59e0b'}
        )
        chart_dict['price_plot'].addItem(price_line)
        
        # Set initial view range to show only the requested bars
        # But all data is still there for scrolling
        total_bars = len(display_data)
        if total_bars > display_bars:
            # Show the most recent 'display_bars' bars
            start_x = total_bars - display_bars
            end_x = total_bars
        else:
            # Show all available bars
            start_x = 0
            end_x = total_bars
        
        # Set the initial view
        chart_dict['price_plot'].setXRange(start_x, end_x)
        
        # Set Y range based on visible data
        visible_data = display_data.iloc[start_x:end_x] if total_bars > display_bars else display_data
        y_min = visible_data['low'].min() * 0.998
        y_max = visible_data['high'].max() * 1.002
        chart_dict['price_plot'].setYRange(y_min, y_max)
        
        # Enable mouse interaction for scrolling/zooming
        chart_dict['price_plot'].enableAutoRange()
        chart_dict['price_plot'].setMouseEnabled(x=True, y=True)
        
        chart_dict['volume_plot'].setXRange(0, 50)
    
    def resizeEvent(self, event):
        """Handle widget resize events."""
        super().resizeEvent(event)
        
        # Adjust volume plot widths based on widget width
        widget_width = self.width()
        if widget_width > 1200:
            volume_width = 150
        elif widget_width > 800:
            volume_width = 120
        else:
            volume_width = 100
            
        self.chart_1['volume_plot'].setMaximumWidth(volume_width)
        self.chart_2['volume_plot'].setMaximumWidth(volume_width)


# ============= STANDALONE TEST SCRIPT =============
if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QSpinBox, QLineEdit
    
    class TestWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Dual HVN Chart Test")
            self.setGeometry(100, 100, 1200, 900)
            
            # Set dark theme
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #1a1a1a;
                }
                QLabel {
                    color: #ffffff;
                }
            """)
            
            # Create central widget
            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)
            
            # Test controls
            controls = QHBoxLayout()
            
            # Ticker input
            controls.addWidget(QLabel("Ticker:"))
            self.ticker_input = QLineEdit("TSLA")
            self.ticker_input.setMaximumWidth(100)
            self.ticker_input.setStyleSheet("""
                QLineEdit {
                    background-color: #2a2a2a;
                    border: 1px solid #444444;
                    border-radius: 3px;
                    padding: 5px;
                    color: #ffffff;
                }
            """)
            controls.addWidget(self.ticker_input)
            
            # Lookback period controls
            controls.addWidget(QLabel("Lookback 1:"))
            self.lookback1_spin = QSpinBox()
            self.lookback1_spin.setRange(3, 60)
            self.lookback1_spin.setValue(14)
            self.lookback1_spin.setStyleSheet("""
                QSpinBox {
                    background-color: #2a2a2a;
                    border: 1px solid #444444;
                    border-radius: 3px;
                    padding: 5px;
                    color: #ffffff;
                }
            """)
            controls.addWidget(self.lookback1_spin)
            
            controls.addWidget(QLabel("Lookback 2:"))
            self.lookback2_spin = QSpinBox()
            self.lookback2_spin.setRange(5, 90)
            self.lookback2_spin.setValue(28)
            self.lookback2_spin.setStyleSheet("""
                QSpinBox {
                    background-color: #2a2a2a;
                    border: 1px solid #444444;
                    border-radius: 3px;
                    padding: 5px;
                    color: #ffffff;
                }
            """)
            controls.addWidget(self.lookback2_spin)
            
            # Display bars control
            controls.addWidget(QLabel("Display Bars:"))
            self.bars_spin = QSpinBox()
            self.bars_spin.setRange(78, 2340)  # 1 day to 30 days
            self.bars_spin.setValue(1092)  # Default 14 days
            self.bars_spin.setSingleStep(78)  # Increment by 1 day
            self.bars_spin.setStyleSheet("""
                QSpinBox {
                    background-color: #2a2a2a;
                    border: 1px solid #444444;
                    border-radius: 3px;
                    padding: 5px;
                    color: #ffffff;
                }
            """)
            controls.addWidget(self.bars_spin)
            
            # Info label for bars
            self.bars_info = QLabel("(~14 days)")
            self.bars_info.setStyleSheet("color: #9ca3af;")
            controls.addWidget(self.bars_info)
            self.bars_spin.valueChanged.connect(self.update_bars_info)
            
            # Load button
            load_btn = QPushButton("Load Ticker")
            load_btn.setStyleSheet("""
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
            """)
            load_btn.clicked.connect(self.load_ticker)
            controls.addWidget(load_btn)
            
            # Update parameters button
            update_btn = QPushButton("Update Parameters")
            update_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f59e0b;
                    border: none;
                    border-radius: 3px;
                    padding: 8px 16px;
                    font-weight: bold;
                    color: #000000;
                }
                QPushButton:hover {
                    background-color: #d97706;
                }
            """)
            update_btn.clicked.connect(self.update_parameters)
            controls.addWidget(update_btn)
            
            controls.addStretch()
            
            layout.addLayout(controls)
            
            # Add dual chart with custom initial parameters
            self.dual_chart = DualHVNChart(
                lookback_periods=[14, 28],
                display_bars=1092  # 14 days worth of bars
            )
            self.dual_chart.loading_started.connect(
                lambda: print("Loading started...")
            )
            self.dual_chart.loading_finished.connect(
                lambda: print("Loading finished!")
            )
            self.dual_chart.error_occurred.connect(
                lambda err: print(f"Error: {err}")
            )
            
            layout.addWidget(self.dual_chart)
        
        def update_bars_info(self, value):
            """Update the bars info label."""
            days = value / 78  # 78 bars per day
            self.bars_info.setText(f"(~{days:.1f} days)")
        
        def load_ticker(self):
            ticker = self.ticker_input.text().strip()
            if ticker:
                print(f"Loading ticker: {ticker}")
                self.dual_chart.load_ticker(ticker)
        
        def update_parameters(self):
            lookback1 = self.lookback1_spin.value()
            lookback2 = self.lookback2_spin.value()
            display_bars = self.bars_spin.value()
            
            print(f"Updating parameters: Lookback periods=[{lookback1}, {lookback2}], Display bars={display_bars}")
            self.dual_chart.update_parameters(
                lookback_periods=[lookback1, lookback2],
                display_bars=display_bars
            )
    
    print("=== Testing Dual HVN Chart Component ===\n")
    print("Bar count reference (15-minute bars):")
    print("- 78 bars = ~1 trading day")
    print("- 390 bars = ~5 trading days")
    print("- 1092 bars = ~14 trading days")
    print("- 1560 bars = ~20 trading days\n")
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Configure PyQtGraph
    pg.setConfigOptions(antialias=True)
    pg.setConfigOption('background', '#1a1a1a')
    pg.setConfigOption('foreground', '#ffffff')
    
    window = TestWindow()
    window.show()
    
    print("Test window opened with configurable parameters.")
    print("- Adjust lookback periods and display bars")
    print("- Click 'Update Parameters' to apply changes")
    print("- Both charts will reload with new settings")
    
    sys.exit(app.exec())