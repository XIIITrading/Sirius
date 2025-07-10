"""
Chart Widget Module - Real-time chart with HVN, Supply/Demand, and Camarilla overlays
"""

import logging
from typing import Optional, Dict, List, Deque
from datetime import datetime, timedelta
from collections import deque

import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QComboBox, QCheckBox)
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QTimer
from PyQt6.QtGui import QFont, QColor

import pandas as pd
import numpy as np

from ...styles import ChartStyles, BaseStyles

# Try to import chart data models
try:
    from .chart.data.models import Bar, TimeframeType
except ImportError:
    # Fallback definitions if models not yet created
    from typing import TypedDict, Literal
    
    TimeframeType = Literal['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    
    class Bar:
        def __init__(self, timestamp, open, high, low, close, volume, trades=0, vwap=None):
            self.timestamp = timestamp
            self.open = open
            self.high = high
            self.low = low
            self.close = close
            self.volume = volume
            self.trades = trades
            self.vwap = vwap
            
        def to_dict(self):
            return {
                'timestamp': self.timestamp,
                'open': self.open,
                'high': self.high,
                'low': self.low,
                'close': self.close,
                'volume': self.volume,
                'trades': self.trades,
                'vwap': self.vwap
            }

# Configure logging
logger = logging.getLogger(__name__)

# Configure PyQtGraph
pg.setConfigOptions(antialias=True)


class CandlestickItem(pg.GraphicsObject):
    """Custom candlestick item for real-time updates"""
    
    def __init__(self):
        pg.GraphicsObject.__init__(self)
        self.bars: List[Bar] = []
        self.picture = None
        
    def set_data(self, bars: List[Bar]):
        """Update with new bar data"""
        self.bars = bars
        self.generatePicture()
        self.update()
        
    def append_bar(self, bar: Bar):
        """Append a single bar"""
        self.bars.append(bar)
        self.generatePicture()
        self.update()
        
    def update_last_bar(self, bar: Bar):
        """Update the last bar (for incomplete bars)"""
        if self.bars:
            self.bars[-1] = bar
            self.generatePicture()
            self.update()
            
    def generatePicture(self):
        """Generate the picture for painting"""
        self.picture = pg.QtGui.QPicture()
        p = pg.QtGui.QPainter(self.picture)
        
        for i, bar in enumerate(self.bars):
            # Determine color
            if bar.close >= bar.open:
                p.setPen(pg.mkPen('#10b981', width=1))
                p.setBrush(pg.mkBrush('#10b981'))
            else:
                p.setPen(pg.mkPen('#ef4444', width=1))
                p.setBrush(pg.mkBrush('#ef4444'))
            
            # Draw high-low line
            p.drawLine(pg.QtCore.QPointF(i, bar.low), 
                      pg.QtCore.QPointF(i, bar.high))
            
            # Draw open-close rectangle
            height = abs(bar.close - bar.open)
            if height > 0:
                p.drawRect(pg.QtCore.QRectF(i - 0.3, 
                                           min(bar.open, bar.close), 
                                           0.6, 
                                           height))
        
        p.end()
    
    def paint(self, p, *args):
        if self.picture:
            p.drawPicture(0, 0, self.picture)
    
    def boundingRect(self):
        if self.picture:
            return pg.QtCore.QRectF(self.picture.boundingRect())
        return pg.QtCore.QRectF()


class ChartWidget(QWidget):
    """Widget for displaying real-time charts with indicators"""
    
    # Signals
    timeframe_changed = pyqtSignal(str)
    indicator_toggled = pyqtSignal(str, bool)
    lookback_changed = pyqtSignal(int)
    
    def __init__(self, parent=None, max_bars: int = 500):
        super().__init__(parent)
        
        self.max_bars = max_bars
        self.current_timeframe: TimeframeType = '5m'
        self.current_symbol: Optional[str] = None
        self.current_lookback = 390  # Default lookback
        
        # Data storage
        self.bar_data: Dict[TimeframeType, Deque[Bar]] = {
            tf: deque(maxlen=max_bars) 
            for tf in ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        }
        
        # Chart items
        self.candlestick_item = None
        self.overlay_items = []
        self.current_price_line = None
        
        # Overlay data (to be populated by calculations)
        self.hvn_zones = []
        self.supply_demand_zones = []
        self.camarilla_levels = None
        
        # Overlay visibility
        self.show_hvn = True
        self.show_order_blocks = True
        self.show_camarilla = True
        
        self.init_ui()
        self.apply_styles()
        
        # Setup update timer for smooth animations
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.refresh_display)
        self.update_timer.start(1000)  # Update every second
        
    def init_ui(self):
        """Initialize the UI"""
        self.setObjectName("chart_container")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header = QLabel("Chart Analysis")
        header.setObjectName("chart_header")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # Info bar
        self.info_label = QLabel("No data")
        self.info_label.setStyleSheet(f"color: {BaseStyles.TEXT_SECONDARY}; padding: 5px;")
        layout.addWidget(self.info_label)
        
        # Controls bar
        controls_widget = self.create_controls()
        layout.addWidget(controls_widget)
        
        # Chart widget using pyqtgraph
        self.chart_widget = pg.GraphicsLayoutWidget()
        self.chart_widget.setBackground(ChartStyles.CHART_BACKGROUND)
        
        # Create plot
        self.plot = self.chart_widget.addPlot(row=0, col=0)
        self.plot.setLabel('left', 'Price', units='$', color=BaseStyles.TEXT_PRIMARY)
        self.plot.setLabel('bottom', 'Time', color=BaseStyles.TEXT_PRIMARY)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Customize plot appearance
        self.plot.getAxis('left').setPen(BaseStyles.TEXT_SECONDARY)
        self.plot.getAxis('left').setTextPen(BaseStyles.TEXT_SECONDARY)
        self.plot.getAxis('bottom').setPen(BaseStyles.TEXT_SECONDARY)
        self.plot.getAxis('bottom').setTextPen(BaseStyles.TEXT_SECONDARY)
        
        # Add legend
        self.legend = self.plot.addLegend(offset=(10, 10))
        self.legend.setBrush(pg.mkBrush(20, 20, 20, 200))
        self.legend.setParentItem(self.plot.vb)
        self.legend.anchor((0, 0), (0, 0))
        
        layout.addWidget(self.chart_widget, 1)  # Give it stretch factor
        
        # Initialize candlestick item
        self.candlestick_item = CandlestickItem()
        self.plot.addItem(self.candlestick_item)
        
    def create_controls(self):
        """Create control widgets"""
        controls_widget = QWidget()
        controls_widget.setObjectName("chart_controls")
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(10, 5, 10, 5)
        
        # Lookback selector (NEW)
        self.lookback_combo = QComboBox()
        self.lookback_combo.setObjectName("lookback_selector")
        self.lookback_combo.addItems(["50", "100", "200", "390", "500", "1000"])
        self.lookback_combo.setCurrentText("390")
        self.lookback_combo.currentTextChanged.connect(self.on_lookback_changed)
        controls_layout.addWidget(QLabel("Bars:"))
        controls_layout.addWidget(self.lookback_combo)
        
        controls_layout.addSpacing(10)
        
        # Timeframe selector
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.setObjectName("timeframe_selector")
        self.timeframe_combo.addItems(["1m", "5m", "15m", "30m", "1h", "4h", "1d"])
        self.timeframe_combo.setCurrentText("5m")
        self.timeframe_combo.currentTextChanged.connect(self.on_timeframe_changed)
        controls_layout.addWidget(QLabel("Timeframe:"))
        controls_layout.addWidget(self.timeframe_combo)
        
        controls_layout.addSpacing(20)
        
        # Indicator toggles
        self.indicator_checkboxes = {}
        
        # HVN toggle
        hvn_check = QCheckBox("HVN")
        hvn_check.setObjectName("indicator_toggle")
        hvn_check.setChecked(True)
        hvn_check.setStyleSheet("color: #10b981;")
        hvn_check.toggled.connect(lambda checked: self.toggle_overlay('HVN', checked))
        self.indicator_checkboxes['HVN'] = hvn_check
        controls_layout.addWidget(hvn_check)
        
        # Order Blocks toggle
        ob_check = QCheckBox("Order Blocks")
        ob_check.setObjectName("indicator_toggle")
        ob_check.setChecked(True)
        ob_check.setStyleSheet("color: #f59e0b;")
        ob_check.toggled.connect(lambda checked: self.toggle_overlay('Order Blocks', checked))
        self.indicator_checkboxes['Order Blocks'] = ob_check
        controls_layout.addWidget(ob_check)
        
        # Camarilla toggle
        cam_check = QCheckBox("Camarilla")
        cam_check.setObjectName("indicator_toggle")
        cam_check.setChecked(True)
        cam_check.setStyleSheet("color: #a855f7;")
        cam_check.toggled.connect(lambda checked: self.toggle_overlay('Camarilla', checked))
        self.indicator_checkboxes['Camarilla'] = cam_check
        controls_layout.addWidget(cam_check)
        
        controls_layout.addStretch()
        
        # Auto-scale button
        auto_scale_btn = QPushButton("Auto Scale")
        auto_scale_btn.setObjectName("chart_button")
        auto_scale_btn.clicked.connect(self.auto_scale)
        controls_layout.addWidget(auto_scale_btn)
        
        # Zoom controls
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setObjectName("chart_button")
        zoom_in_btn.setFixedSize(30, 30)
        zoom_in_btn.clicked.connect(self.zoom_in)
        
        zoom_out_btn = QPushButton("-")
        zoom_out_btn.setObjectName("chart_button")
        zoom_out_btn.setFixedSize(30, 30)
        zoom_out_btn.clicked.connect(self.zoom_out)
        
        controls_layout.addWidget(zoom_in_btn)
        controls_layout.addWidget(zoom_out_btn)
        
        return controls_widget
        
    def apply_styles(self):
        """Apply styles to the widget"""
        self.setStyleSheet(ChartStyles.get_stylesheet())
        
    def on_lookback_changed(self, lookback: str):
        """Handle lookback period change"""
        bars = int(lookback)
        self.current_lookback = bars
        logger.info(f"Lookback changed to {bars} bars")
        
        # Emit signal for data loading
        self.lookback_changed.emit(bars)
        
        # Update display
        self.refresh_display()
        
    @pyqtSlot(dict)
    def update_chart_data(self, data: dict):
        """
        Update chart with new data from AggregateDataHandler
        
        Expected data format:
        {
            'symbol': str,
            'timeframe': str,
            'bars': List[dict],  # Bar dictionaries
            'is_update': bool,
            'latest_bar_complete': bool
        }
        """
        try:
            symbol = data['symbol']
            timeframe = data['timeframe']
            bars_data = data['bars']
            is_update = data['is_update']
            latest_bar_complete = data['latest_bar_complete']
            
            # Update current symbol
            if symbol != self.current_symbol:
                self.current_symbol = symbol
                self.clear_chart()
            
            # Convert dict bars to Bar objects
            bars = []
            for bar_dict in bars_data:
                bar = Bar(
                    timestamp=bar_dict['timestamp'],
                    open=bar_dict['open'],
                    high=bar_dict['high'],
                    low=bar_dict['low'],
                    close=bar_dict['close'],
                    volume=bar_dict['volume'],
                    trades=bar_dict.get('trades', 0),
                    vwap=bar_dict.get('vwap')
                )
                bars.append(bar)
            
            # Update bar storage
            if is_update and latest_bar_complete:
                # New complete bar
                for bar in bars:
                    self.bar_data[timeframe].append(bar)
            elif is_update and not latest_bar_complete:
                # Update incomplete bar
                if bars and self.bar_data[timeframe]:
                    # Check if we're updating the last bar
                    last_bar = self.bar_data[timeframe][-1]
                    new_bar = bars[0]
                    
                    if last_bar.timestamp == new_bar.timestamp:
                        # Update existing bar
                        self.bar_data[timeframe][-1] = new_bar
                    else:
                        # New bar
                        self.bar_data[timeframe].append(new_bar)
            else:
                # Full refresh
                self.bar_data[timeframe].clear()
                self.bar_data[timeframe].extend(bars)
            
            # Update display if this is the current timeframe
            if timeframe == self.current_timeframe:
                self.refresh_display()
                
            # Update info label
            if bars:
                latest_bar = bars[-1]
                self.info_label.setText(
                    f"{symbol} | {timeframe} | "
                    f"O: ${latest_bar.open:.2f} H: ${latest_bar.high:.2f} "
                    f"L: ${latest_bar.low:.2f} C: ${latest_bar.close:.2f} | "
                    f"Vol: {latest_bar.volume:,}"
                )
                
        except Exception as e:
            logger.error(f"Error updating chart data: {e}", exc_info=True)
    
    def refresh_display(self):
        """Refresh the chart display"""
        if not self.current_symbol:
            return
            
        bars = list(self.bar_data[self.current_timeframe])
        if not bars:
            return
            
        # Apply lookback limit
        display_bars = bars
        if len(bars) > self.current_lookback:
            display_bars = bars[-self.current_lookback:]
            
        # Update candlesticks
        self.candlestick_item.set_data(display_bars)
        
        # Update time axis
        self.update_time_axis(display_bars)
        
        # Update current price line
        self.update_current_price(display_bars[-1].close if display_bars else 0)
        
        # Update overlays
        self.update_overlays()
        
        # Set X range
        self.plot.setXRange(0, len(display_bars))
    
    def update_time_axis(self, bars: List[Bar]):
        """Update time axis labels"""
        if not bars:
            return
            
        # Create time strings
        time_strings = []
        for bar in bars:
            if self.current_timeframe in ['1d']:
                time_strings.append(bar.timestamp.strftime('%m/%d'))
            elif self.current_timeframe in ['1h', '4h']:
                time_strings.append(bar.timestamp.strftime('%m/%d %H:%M'))
            else:
                time_strings.append(bar.timestamp.strftime('%H:%M'))
        
        # Create tick dictionary
        x_dict = dict(enumerate(time_strings))
        
        # Show subset of labels to avoid crowding
        num_bars = len(bars)
        if num_bars > 50:
            step = num_bars // 10
        elif num_bars > 20:
            step = 5
        else:
            step = 1
            
        x_dict_sparse = {k: v for k, v in x_dict.items() if k % step == 0}
        
        # Update axis
        stringaxis = pg.AxisItem(orientation='bottom')
        stringaxis.setTicks([list(x_dict_sparse.items())])
        stringaxis.setTextPen(BaseStyles.TEXT_SECONDARY)
        self.plot.setAxisItems(axisItems={'bottom': stringaxis})
    
    def update_current_price(self, price: float):
        """Update current price line"""
        if self.current_price_line:
            self.plot.removeItem(self.current_price_line)
            
        self.current_price_line = pg.InfiniteLine(
            pos=price,
            angle=0,
            pen=pg.mkPen('#3b82f6', width=2),
            label=f'${price:.2f}',
            labelOpts={'position': 0.95, 'color': '#3b82f6'}
        )
        self.plot.addItem(self.current_price_line)
    
    def on_timeframe_changed(self, timeframe: str):
        """Handle timeframe change"""
        self.current_timeframe = timeframe
        self.timeframe_changed.emit(timeframe)
        self.refresh_display()
    
    def toggle_overlay(self, overlay_name: str, enabled: bool):
        """Toggle overlay visibility"""
        if overlay_name == 'HVN':
            self.show_hvn = enabled
        elif overlay_name == 'Order Blocks':
            self.show_order_blocks = enabled
        elif overlay_name == 'Camarilla':
            self.show_camarilla = enabled
            
        self.indicator_toggled.emit(overlay_name, enabled)
        self.update_overlays()
    
    def update_overlays(self):
        """Update all overlays based on visibility settings"""
        # Clear existing overlay items
        for item in self.overlay_items:
            self.plot.removeItem(item)
        self.overlay_items.clear()
        
        # Add overlays based on visibility
        if self.show_hvn and self.hvn_zones:
            self._draw_hvn_zones()
            
        if self.show_order_blocks and self.supply_demand_zones:
            self._draw_supply_demand_zones()
            
        if self.show_camarilla and self.camarilla_levels:
            self._draw_camarilla_levels()
    
    def _draw_hvn_zones(self):
        """Draw HVN zones"""
        for zone in self.hvn_zones:
            color = QColor('#10b981')
            color.setAlpha(80)
            
            zone_rect = pg.LinearRegionItem(
                values=(zone['price_low'], zone['price_high']),
                orientation='horizontal',
                brush=pg.mkBrush(color),
                pen=pg.mkPen('#10b981', width=1),
                movable=False
            )
            
            self.plot.addItem(zone_rect)
            self.overlay_items.append(zone_rect)
    
    def _draw_supply_demand_zones(self):
        """Draw supply/demand zones"""
        for zone in self.supply_demand_zones:
            if zone['type'] == 'supply':
                color = QColor('#ef4444')
            else:
                color = QColor('#10b981')
            color.setAlpha(60)
            
            zone_rect = pg.LinearRegionItem(
                values=(zone['price_low'], zone['price_high']),
                orientation='horizontal',
                brush=pg.mkBrush(color),
                pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine),
                movable=False
            )
            
            self.plot.addItem(zone_rect)
            self.overlay_items.append(zone_rect)
    
    def _draw_camarilla_levels(self):
        """Draw Camarilla pivot levels"""
        if not self.camarilla_levels:
            return
            
        level_styles = {
            'R4': {'color': '#ef4444', 'width': 2, 'style': Qt.PenStyle.DashLine},
            'R3': {'color': '#f59e0b', 'width': 1.5, 'style': Qt.PenStyle.DashLine},
            'R2': {'color': '#fbbf24', 'width': 1, 'style': Qt.PenStyle.DotLine},
            'R1': {'color': '#fde047', 'width': 1, 'style': Qt.PenStyle.DotLine},
            'S1': {'color': '#84cc16', 'width': 1, 'style': Qt.PenStyle.DotLine},
            'S2': {'color': '#22c55e', 'width': 1, 'style': Qt.PenStyle.DotLine},
            'S3': {'color': '#10b981', 'width': 1.5, 'style': Qt.PenStyle.DashLine},
            'S4': {'color': '#06b6d4', 'width': 2, 'style': Qt.PenStyle.DashLine},
        }
        
        # Draw levels
        for level_name, level_price in self.camarilla_levels.items():
            if level_name in level_styles:
                style = level_styles[level_name]
                line = pg.InfiniteLine(
                    pos=level_price,
                    angle=0,
                    pen=pg.mkPen(
                        style['color'],
                        width=style['width'],
                        style=style['style']
                    ),
                    label=f'{level_name}: ${level_price:.2f}',
                    labelOpts={'position': 0.02, 'color': style['color']}
                )
                self.plot.addItem(line)
                self.overlay_items.append(line)
    
    def add_hvn_zones(self, zones: List[dict]):
        """Add HVN zones to chart"""
        self.hvn_zones = zones
        self.update_overlays()
        
    def add_supply_demand_zones(self, zones: List[dict]):
        """Add supply/demand zones to chart"""
        self.supply_demand_zones = zones
        self.update_overlays()
        
    def add_camarilla_levels(self, levels: dict):
        """Add Camarilla pivot levels to chart"""
        self.camarilla_levels = levels
        self.update_overlays()
        
    def clear_chart(self):
        """Clear all chart data"""
        self.plot.clear()
        self.candlestick_item = CandlestickItem()
        self.plot.addItem(self.candlestick_item)
        self.overlay_items.clear()
        self.current_price_line = None
        
        # Clear data
        for tf in self.bar_data:
            self.bar_data[tf].clear()
            
        # Clear overlays
        self.hvn_zones = []
        self.supply_demand_zones = []
        self.camarilla_levels = None
        
        # Update display
        self.info_label.setText("Chart cleared")
        
    def auto_scale(self):
        """Auto-scale the chart"""
        self.plot.enableAutoRange()
        
    def zoom_in(self):
        """Zoom in on chart"""
        self.plot.scaleBy((0.8, 1))
        
    def zoom_out(self):
        """Zoom out on chart"""
        self.plot.scaleBy((1.25, 1))
        
    def add_entry_marker(self, price: float, time: datetime):
        """Add entry marker to chart"""
        # Find the bar index for this time
        bars = list(self.bar_data[self.current_timeframe])
        bar_index = None
        
        for i, bar in enumerate(bars):
            if bar.timestamp <= time < bar.timestamp + timedelta(minutes=1):
                bar_index = i
                break
                
        if bar_index is not None:
            # Add arrow marker
            arrow = pg.ArrowItem(
                pos=(bar_index, price),
                angle=-90,
                tipAngle=30,
                headLen=10,
                tailLen=0,
                brush='#10b981'
            )
            self.plot.addItem(arrow)
            self.overlay_items.append(arrow)
        
    def add_exit_marker(self, price: float, time: datetime):
        """Add exit marker to chart"""
        # Find the bar index for this time
        bars = list(self.bar_data[self.current_timeframe])
        bar_index = None
        
        for i, bar in enumerate(bars):
            if bar.timestamp <= time < bar.timestamp + timedelta(minutes=1):
                bar_index = i
                break
                
        if bar_index is not None:
            # Add arrow marker
            arrow = pg.ArrowItem(
                pos=(bar_index, price),
                angle=90,
                tipAngle=30,
                headLen=10,
                tailLen=0,
                brush='#ef4444'
            )
            self.plot.addItem(arrow)
            self.overlay_items.append(arrow)