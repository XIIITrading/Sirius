# market_review/dashboards/components/camarilla_pivot_chart.py
"""
Module: Camarilla Pivot Chart Component
Purpose: Display Camarilla pivot levels on a price chart
UI Framework: PyQt6 with PyQtGraph
Note: All times are in UTC
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import pandas as pd
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGroupBox
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

# Local imports
from market_review.calculations.pivots.camarilla_engine import CamarillaCalculator, CamarillaResult
from market_review.data.polygon_bridge import PolygonHVNBridge

# Configure logging
logger = logging.getLogger(__name__)

# Configure PyQtGraph
pg.setConfigOptions(antialias=True)


class CamarillaDataWorker(QThread):
    """Background thread for fetching data and calculating Camarilla levels."""
    data_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(str)
    
    def __init__(self, ticker: str, lookback_days: int = 14):
        super().__init__()
        self.ticker = ticker
        self.lookback_days = lookback_days
        
    def run(self):
        try:
            self.progress_update.emit(f"Fetching data for {self.ticker}...")
            
            # Initialize bridge
            bridge = PolygonHVNBridge(lookback_days=self.lookback_days)
            
            # Fetch data
            state = bridge.calculate_hvn(self.ticker, timeframe='15min')
            
            # Calculate Camarilla levels
            self.progress_update.emit("Calculating Camarilla pivot levels...")
            calc = CamarillaCalculator()
            camarilla_result = calc.calculate(state.recent_bars, ticker=self.ticker)
            
            # Get nearest levels
            current_price = state.current_price
            nearest_r, nearest_s = calc.get_nearest_levels(camarilla_result, current_price, count=4)
            
            # Package results
            result = {
                'ticker': self.ticker,
                'price_data': state.recent_bars,
                'current_price': current_price,
                'camarilla_result': camarilla_result,
                'nearest_resistance': nearest_r,
                'nearest_support': nearest_s
            }
            
            self.data_ready.emit(result)
            
        except Exception as e:
            logger.error(f"Error in CamarillaDataWorker: {e}")
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


class CamarillaPivotChart(QWidget):
    """
    Camarilla Pivot chart component.
    Shows price chart with Camarilla pivot levels overlaid.
    """
    
    # Signals
    loading_started = pyqtSignal()
    loading_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.ticker = None
        self.chart_data = None
        self.worker = None
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Container
        self.container = QGroupBox("Camarilla Pivot Levels")
        self.container.setStyleSheet("""
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
                color: #f59e0b;
            }
        """)
        
        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(5, 5, 5, 5)
        
        # Info label
        self.info_label = QLabel("No data loaded")
        self.info_label.setStyleSheet("color: #9ca3af; padding: 2px;")
        container_layout.addWidget(self.info_label)
        
        # Levels info
        self.levels_label = QLabel("")
        self.levels_label.setStyleSheet("color: #10b981; padding: 2px; font-size: 11px;")
        container_layout.addWidget(self.levels_label)
        
        # Chart widget
        self.chart_widget = pg.GraphicsLayoutWidget()
        self.chart_widget.setBackground('#1a1a1a')
        
        # Create plot
        self.plot = self.chart_widget.addPlot(row=0, col=0)
        self.plot.setLabel('left', 'Price', units='$')
        self.plot.setLabel('bottom', 'Time (UTC)')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        
        container_layout.addWidget(self.chart_widget)
        self.container.setLayout(container_layout)
        
        layout.addWidget(self.container)
    
    def load_ticker(self, ticker: str):
        """Load data and calculate Camarilla levels for ticker."""
        if not ticker:
            return
            
        self.ticker = ticker.upper()
        logger.info(f"Loading Camarilla levels for: {self.ticker}")
        
        # Clear any existing worker
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        
        # Clear chart
        self.plot.clear()
        
        # Update labels
        self.info_label.setText(f"Loading {self.ticker}...")
        self.levels_label.setText("")
        
        # Emit loading signal
        self.loading_started.emit()
        
        # Start worker
        self.worker = CamarillaDataWorker(self.ticker)
        self.worker.data_ready.connect(self.on_data_ready)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.progress_update.connect(
            lambda msg: self.info_label.setText(msg)
        )
        
        self.worker.start()
    
    def on_data_ready(self, data: dict):
        """Handle data ready from worker."""
        self.chart_data = data
        self.update_chart()
        self.loading_finished.emit()
    
    def on_error(self, error_msg: str):
        """Handle error from worker."""
        self.info_label.setText(f"Error: {error_msg}")
        self.info_label.setStyleSheet("color: #ef4444; padding: 2px;")
        self.error_occurred.emit(error_msg)
    
    def update_chart(self):
        """Update chart with data and Camarilla levels."""
        if not self.chart_data:
            return
            
        # Clear previous items
        self.plot.clear()
        
        # Get data
        price_data = self.chart_data['price_data']
        camarilla_result = self.chart_data['camarilla_result']
        current_price = self.chart_data['current_price']
        
        # Limit display to recent data (5 days)
        display_bars = 390  # ~5 days of 15-min bars
        if len(price_data) > display_bars:
            display_data = price_data.tail(display_bars).copy()
        else:
            display_data = price_data.copy()
        
        # Update info label
        self.info_label.setText(
            f"{self.ticker} | Current: ${current_price:.2f} | "
            f"Prior Day: H=${camarilla_result.prior_day_high:.2f} "
            f"L=${camarilla_result.prior_day_low:.2f} "
            f"C=${camarilla_result.prior_day_close:.2f}"
        )
        self.info_label.setStyleSheet("color: #10b981; padding: 2px;")
        
        # Update levels label
        levels_text = "Levels: "
        for name in ['R4', 'R3', 'R2', 'R1']:
            price = camarilla_result.resistance_levels[name]
            levels_text += f"{name}=${price:.2f} "
        levels_text += "| "
        for name in ['S1', 'S2', 'S3', 'S4']:
            price = camarilla_result.support_levels[name]
            levels_text += f"{name}=${price:.2f} "
        
        self.levels_label.setText(levels_text)
        
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
        self.plot.setAxisItems(axisItems={'bottom': stringaxis})
        
        # Draw candlesticks
        candlestick = CandlestickItem(display_data_reset)
        self.plot.addItem(candlestick)
        
        # Draw Camarilla levels
        level_colors = {
            'R4': '#ef4444',  # Red
            'R3': '#f59e0b',  # Orange
            'R2': '#fbbf24',  # Yellow
            'R1': '#fde047',  # Light yellow
            'S1': '#84cc16',  # Light green
            'S2': '#22c55e',  # Green
            'S3': '#10b981',  # Emerald
            'S4': '#06b6d4',  # Cyan
        }
        
        # Draw resistance levels
        for name, price in camarilla_result.resistance_levels.items():
            line = pg.InfiniteLine(
                pos=price, 
                angle=0, 
                pen=pg.mkPen(level_colors[name], width=2, style=Qt.PenStyle.DashLine),
                label=f'{name}: ${price:.2f}',
                labelOpts={'position': 0.98, 'color': level_colors[name]}
            )
            self.plot.addItem(line)
        
        # Draw support levels
        for name, price in camarilla_result.support_levels.items():
            line = pg.InfiniteLine(
                pos=price, 
                angle=0, 
                pen=pg.mkPen(level_colors[name], width=2, style=Qt.PenStyle.DashLine),
                label=f'{name}: ${price:.2f}',
                labelOpts={'position': 0.98, 'color': level_colors[name]}
            )
            self.plot.addItem(line)
        
        # Draw central pivot
        pivot_line = pg.InfiniteLine(
            pos=camarilla_result.central_pivot, 
            angle=0, 
            pen=pg.mkPen('#a855f7', width=3, style=Qt.PenStyle.DotLine),
            label=f'Pivot: ${camarilla_result.central_pivot:.2f}',
            labelOpts={'position': 0.02, 'color': '#a855f7'}
        )
        self.plot.addItem(pivot_line)
        
        # Draw current price line
        current_line = pg.InfiniteLine(
            pos=current_price, 
            angle=0, 
            pen=pg.mkPen('#3b82f6', width=2),
            label=f'Current: ${current_price:.2f}',
            labelOpts={'position': 0.95, 'color': '#3b82f6'}
        )
        self.plot.addItem(current_line)
        
        # Set view range
        self.plot.setXRange(0, len(display_data))
        
        # Calculate Y range to include all levels
        all_prices = [
            display_data['high'].max(),
            display_data['low'].min(),
            camarilla_result.resistance_levels['R4'],
            camarilla_result.support_levels['S4']
        ]
        y_min = min(all_prices) * 0.995
        y_max = max(all_prices) * 1.005
        self.plot.setYRange(y_min, y_max)
        
        # Enable mouse interaction
        self.plot.enableAutoRange()
        self.plot.setMouseEnabled(x=True, y=True)