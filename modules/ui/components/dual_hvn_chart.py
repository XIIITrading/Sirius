# modules/ui/components/dual_hvn_chart.py
"""
Module: Dual HVN Chart Component
Purpose: Reusable component showing two vertically stacked HVN charts
         with different lookback periods (7 and 28 days)
UI Framework: PyQt6 with PyQtGraph
Note: All times are in UTC
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import logging

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QSplitter, QGroupBox, QSizePolicy)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor
import pyqtgraph as pg

# Setup path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
components_dir = current_dir
ui_dir = os.path.dirname(components_dir)
modules_dir = os.path.dirname(ui_dir)
vega_root = os.path.dirname(modules_dir)

if 'polygon' in sys.modules:
    del sys.modules['polygon']

sys.path.insert(0, vega_root)

from modules.calculations.volume.hvn_engine import HVNEngine
from modules.data.polygon_bridge import PolygonHVNBridge

sys.path.remove(vega_root)

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
    
    def __init__(self, ticker: str, lookback_days_hvn: int, data_days: int = 5):
        super().__init__()
        self.ticker = ticker
        self.lookback_days_hvn = lookback_days_hvn
        self.data_days = data_days
        
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
                'data_days': self.data_days
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
    Shows two vertically stacked charts with different lookback periods.
    """
    
    # Signals
    loading_started = pyqtSignal()
    loading_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.ticker = None
        self.chart_data = {
            '7_day': None,
            '28_day': None
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
        
        # Create chart widgets
        self.chart_7day = self.create_chart_widget("7-Day HVN Lookback")
        self.chart_28day = self.create_chart_widget("28-Day HVN Lookback")
        
        # Add to splitter
        self.splitter.addWidget(self.chart_7day['container'])
        self.splitter.addWidget(self.chart_28day['container'])
        
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
        self.chart_7day['info_label'].setText(f"Loading {self.ticker}...")
        self.chart_28day['info_label'].setText(f"Loading {self.ticker}...")
        
        # Emit loading signal
        self.loading_started.emit()
        
        # Start workers for both lookback periods
        worker_7day = ChartDataWorker(self.ticker, lookback_days_hvn=7)
        worker_7day.data_ready.connect(lambda data: self.on_data_ready(data, '7_day'))
        worker_7day.error_occurred.connect(lambda err: self.on_error(err, '7_day'))
        worker_7day.progress_update.connect(
            lambda msg: self.chart_7day['info_label'].setText(msg)
        )
        
        worker_28day = ChartDataWorker(self.ticker, lookback_days_hvn=28)
        worker_28day.data_ready.connect(lambda data: self.on_data_ready(data, '28_day'))
        worker_28day.error_occurred.connect(lambda err: self.on_error(err, '28_day'))
        worker_28day.progress_update.connect(
            lambda msg: self.chart_28day['info_label'].setText(msg)
        )
        
        self.workers = [worker_7day, worker_28day]
        
        # Start workers
        worker_7day.start()
        worker_28day.start()
    
    def clear_workers(self):
        """Stop and clear any running workers."""
        for worker in self.workers:
            if worker.isRunning():
                worker.terminate()
                worker.wait()
        self.workers.clear()
    
    def clear_charts(self):
        """Clear all chart data."""
        for chart_key in ['7_day', '28_day']:
            chart = self.chart_7day if chart_key == '7_day' else self.chart_28day
            chart['price_plot'].clear()
            chart['volume_plot'].clear()
    
    def on_data_ready(self, data: dict, chart_key: str):
        """Handle data ready from worker."""
        self.chart_data[chart_key] = data
        
        # Update appropriate chart
        chart = self.chart_7day if chart_key == '7_day' else self.chart_28day
        self.update_chart(chart, data)
        
        # Check if both charts are loaded
        if all(self.chart_data.values()):
            self.loading_finished.emit()
    
    def on_error(self, error_msg: str, chart_key: str):
        """Handle error from worker."""
        chart = self.chart_7day if chart_key == '7_day' else self.chart_28day
        chart['info_label'].setText(f"Error: {error_msg}")
        chart['info_label'].setStyleSheet("color: #ef4444; padding: 2px;")
        
        self.error_occurred.emit(f"{chart_key} error: {error_msg}")
    
    def update_chart(self, chart_dict: dict, data: dict):
        """Update a single chart with data."""
        state = data['state']
        lookback_days = data['lookback_days']
        
        # Clear previous items
        chart_dict['price_plot'].clear()
        chart_dict['volume_plot'].clear()
        
        # Get recent data (5 days for display)
        all_data = state.recent_bars
        if all_data.empty:
            chart_dict['info_label'].setText("No data available")
            return
            
        # Filter to last 5 days of data
        cutoff_date = all_data.index[-1] - timedelta(days=data['data_days'])
        display_data = all_data[all_data.index > cutoff_date].copy()
        
        # Update info label
        start_time = display_data.index[0].strftime('%Y-%m-%d %H:%M UTC')
        end_time = display_data.index[-1].strftime('%Y-%m-%d %H:%M UTC')
        chart_dict['info_label'].setText(
            f"{state.symbol} | {start_time} to {end_time} | "
            f"{len(display_data)} bars | Current: ${state.current_price:.2f}"
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
        
        # Set axis ranges
        chart_dict['price_plot'].setXRange(0, len(display_data))
        y_min = display_data['low'].min() * 0.998
        y_max = display_data['high'].max() * 1.002
        chart_dict['price_plot'].setYRange(y_min, y_max)
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
            
        self.chart_7day['volume_plot'].setMaximumWidth(volume_width)
        self.chart_28day['volume_plot'].setMaximumWidth(volume_width)


# ============= STANDALONE TEST SCRIPT =============
if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton
    
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
            """)
            
            # Create central widget
            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)
            
            # Test controls
            controls = QHBoxLayout()
            
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
            
            controls.addWidget(QLabel("Ticker:"))
            controls.addWidget(self.ticker_input)
            controls.addWidget(load_btn)
            controls.addStretch()
            
            layout.addLayout(controls)
            
            # Add dual chart
            self.dual_chart = DualHVNChart()
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
        
        def load_ticker(self):
            ticker = self.ticker_input.text().strip()
            if ticker:
                print(f"Loading ticker: {ticker}")
                self.dual_chart.load_ticker(ticker)
    
    print("=== Testing Dual HVN Chart Component ===\n")
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Configure PyQtGraph
    pg.setConfigOptions(antialias=True)
    pg.setConfigOption('background', '#1a1a1a')
    pg.setConfigOption('foreground', '#ffffff')
    
    window = TestWindow()
    window.show()
    
    print("Test window opened. Try loading different tickers.")
    print("Both charts should display with different HVN lookback periods.")
    
    sys.exit(app.exec())