# market_review/dashboards/components/supply_demand_chart.py
"""
Module: Supply/Demand Zone Chart Component
Purpose: Visualize supply and demand zones with price action
Updated: Aligned candlestick rendering with dual_hvn_chart style
         7-day lookback with 15-minute bars
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import asyncio

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                            QTableWidget, QTableWidgetItem, QHeaderView,
                            QLabel, QGroupBox, QPushButton)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QColor, QPainter, QPicture
import pyqtgraph as pg
import pandas as pd
import numpy as np

# Import the supply/demand analyzer
from market_review.calculations.zones.supply_demand import (
    analyze_supply_demand_zones, get_strongest_zones, 
    SupplyDemandZone, set_data_manager
)

logger = logging.getLogger(__name__)

# Module-level data manager instance
_data_manager = None
_current_plugin = 'supply_demand'


def initialize_data_manager(data_manager):
    """Initialize the module-level data manager"""
    global _data_manager
    _data_manager = data_manager
    
    # Set data manager for the supply_demand module
    set_data_manager(data_manager)
    
    # Tell data manager this plugin is active
    if hasattr(data_manager, 'set_current_plugin'):
        data_manager.set_current_plugin(_current_plugin)
    
    logger.info("Supply/Demand data manager initialized")


class PolygonDataManager:
    """Data manager wrapper for Polygon data fetching"""
    
    def __init__(self):
        from market_review.data.polygon_bridge import PolygonHVNBridge
        self.bridge = PolygonHVNBridge(
            hvn_levels=100,
            hvn_percentile=80.0,
            lookback_days=14  # Max lookback for data fetching
        )
        
    async def load_data_async(self, **kwargs):
        """Load data asynchronously using Polygon bridge"""
        ticker = kwargs.get('ticker')
        timeframe = kwargs.get('timeframe', '15min')
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        
        # Use the bridge to calculate HVN (which fetches the data)
        state = self.bridge.calculate_hvn(
            symbol=ticker,
            end_date=end_date,
            timeframe=timeframe
        )
        
        # Extract the price data from the state
        if state and state.recent_bars is not None:
            # Filter data to match requested date range
            df = state.recent_bars
            if start_date and end_date:
                # Ensure timezone awareness
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                if start_date.tzinfo is None:
                    start_date = pd.Timestamp(start_date).tz_localize('UTC')
                if end_date.tzinfo is None:
                    end_date = pd.Timestamp(end_date).tz_localize('UTC')
                    
                # Filter to requested range
                df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            return df
        else:
            return pd.DataFrame()


class CandlestickItem(pg.GraphicsObject):
    """Custom candlestick item matching dual_hvn_chart style"""
    
    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.data = data
        self.generatePicture()
        
    def generatePicture(self):
        self.picture = QPicture()
        p = QPainter(self.picture)
        
        for i, (_, row) in enumerate(self.data.iterrows()):
            # Determine color - matching dual_hvn_chart colors
            if row['close'] >= row['open']:
                p.setPen(pg.mkPen('#10b981', width=1))  # Green
                p.setBrush(pg.mkBrush('#10b981'))
            else:
                p.setPen(pg.mkPen('#ef4444', width=1))  # Red
                p.setBrush(pg.mkBrush('#ef4444'))
            
            # Draw high-low line (wick)
            p.drawLine(pg.QtCore.QPointF(i, row['low']), 
                      pg.QtCore.QPointF(i, row['high']))
            
            # Draw open-close rectangle (body)
            height = abs(row['close'] - row['open'])
            if height > 0:
                p.drawRect(pg.QtCore.QRectF(
                    i - 0.3,  # x position - width/2
                    min(row['open'], row['close']),  # y position (bottom of body)
                    0.6,  # width
                    height  # height
                ))
        
        p.end()
    
    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)
    
    def boundingRect(self):
        return pg.QtCore.QRectF(self.picture.boundingRect())


class SupplyDemandWorker(QThread):
    """Worker thread for async supply/demand analysis"""
    
    # Signals
    analysis_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(str)
    
    def __init__(self, ticker: str, lookback_days: int = 7):
        super().__init__()
        self.ticker = ticker
        self.lookback_days = lookback_days
        
    def run(self):
        """Run the analysis in separate thread"""
        try:
            import asyncio
            
            self.progress_update.emit("Analyzing supply/demand zones...")
            
            # Create new event loop for thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Ensure data manager is set
            self._ensure_data_manager()
            
            # Run async analysis
            result = loop.run_until_complete(
                analyze_supply_demand_zones(
                    self.ticker,
                    lookback_days=self.lookback_days
                )
            )
            
            if 'error' in result:
                self.error_occurred.emit(result['error'])
            else:
                self.analysis_complete.emit(result)
                
        except Exception as e:
            self.error_occurred.emit(str(e))
            
    def _ensure_data_manager(self):
        """Ensure data manager is set for supply_demand module"""
        logger.info("Starting _ensure_data_manager")
        
        # Check if supply_demand module already has data manager
        try:
            from market_review.calculations.zones.supply_demand import _data_manager as sd_data_manager
            
            if sd_data_manager is not None:
                logger.info("Data manager already set in supply_demand module")
                return
        except Exception as e:
            logger.error(f"Error checking supply_demand data manager: {e}")
        
        # Declare global BEFORE using it
        global _data_manager
        
        # Try to use the module-level data manager from our module
        if _data_manager is not None:
            try:
                from market_review.calculations.zones.supply_demand import set_data_manager
                set_data_manager(_data_manager)
                logger.info("Successfully set data manager from supply_demand_chart module")
                return
            except Exception as e:
                logger.error(f"Error setting data manager from module: {e}")
        
        # Create a PolygonDataManager instance
        logger.info("Creating PolygonDataManager instance...")
        try:
            polygon_dm = PolygonDataManager()
            from market_review.calculations.zones.supply_demand import set_data_manager
            set_data_manager(polygon_dm)
            
            # Also set our module-level instance
            _data_manager = polygon_dm
            logger.info("Successfully created and set PolygonDataManager")
            
        except Exception as e:
            logger.error(f"Failed to create PolygonDataManager: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Data manager could not be initialized: {str(e)}")


class SupplyDemandChart(QWidget):
    """Chart widget for displaying supply/demand zones with proper candlesticks"""
    
    # Signals
    loading_started = pyqtSignal()
    loading_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    zone_selected = pyqtSignal(dict)
    
    def __init__(self, data_manager=None, lookback_days: int = 7, display_bars: int = 182, parent=None):
        """
        Initialize chart with 7 days lookback and proper display bars.
        
        Args:
            data_manager: Data manager instance (optional, can use module-level)
            lookback_days: Days to look back (default: 7)
            display_bars: Number of 15-min bars to display (default: 182 = 7 days)
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Initialize data manager if provided
        if data_manager:
            initialize_data_manager(data_manager)
        else:
            # Create PolygonDataManager if no data manager provided
            try:
                logger.info("No data manager provided, creating PolygonDataManager")
                polygon_dm = PolygonDataManager()
                initialize_data_manager(polygon_dm)
            except Exception as e:
                logger.warning(f"Could not initialize data manager at startup: {e}")
                # Don't fail here - worker thread will try again
        
        self.ticker = None
        self.lookback_days = lookback_days
        self.display_bars = display_bars  # 7 * 6.5 * 4 = 182 bars
        self.analysis_result = None
        self.price_data = None
        
        # Zone colors with transparency
        self.supply_color = QColor(220, 38, 127, 80)  # Pink/Red
        self.demand_color = QColor(34, 197, 94, 80)   # Green
        self.validated_alpha = 120
        self.unvalidated_alpha = 60
        
        # Style settings matching dual_hvn_chart
        self.setStyleSheet("""
            QWidget {
                background-color: #1a1a1a;
                color: #ffffff;
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
                color: #ffffff;
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
            QTableWidget {
                background-color: #2a2a2a;
                alternate-background-color: #333333;
                gridline-color: #444444;
            }
            QHeaderView::section {
                background-color: #1a1a1a;
                color: #10b981;
                font-weight: bold;
                border: 1px solid #444444;
                padding: 4px;
            }
        """)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - Chart
        chart_widget = self.create_chart_widget()
        splitter.addWidget(chart_widget)
        
        # Right side - Zone details
        details_widget = self.create_details_widget()
        splitter.addWidget(details_widget)
        
        # Set splitter sizes (70% chart, 30% details)
        splitter.setSizes([700, 300])
        
        layout.addWidget(splitter)
        
    def create_chart_widget(self):
        """Create the chart widget with proper styling"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create plot widget with dark background
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1a1a1a')
        self.plot_widget.setLabel('left', 'Price', units='$')
        self.plot_widget.setLabel('bottom', 'Time (UTC)')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Add crosshair
        self.crosshair_v = pg.InfiniteLine(
            angle=90, movable=False, 
            pen=pg.mkPen('w', width=0.5, style=Qt.PenStyle.DashLine)
        )
        self.crosshair_h = pg.InfiniteLine(
            angle=0, movable=False, 
            pen=pg.mkPen('w', width=0.5, style=Qt.PenStyle.DashLine)
        )
        self.plot_widget.addItem(self.crosshair_v, ignoreBounds=True)
        self.plot_widget.addItem(self.crosshair_h, ignoreBounds=True)
        
        # Price label
        self.price_label = pg.TextItem(
            anchor=(0, 1), 
            color='w', 
            fill=pg.mkBrush(30, 30, 30, 180)
        )
        self.plot_widget.addItem(self.price_label, ignoreBounds=True)
        
        # Connect mouse move
        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_moved)
        
        layout.addWidget(self.plot_widget)
        return widget
        
    def create_details_widget(self):
        """Create the zone details widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Summary group
        summary_group = QGroupBox("Zone Summary")
        summary_layout = QVBoxLayout()
        
        self.summary_label = QLabel("No data loaded")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("color: #9ca3af; padding: 2px;")
        summary_layout.addWidget(self.summary_label)
        
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        # Zone table
        zones_group = QGroupBox("Supply/Demand Zones")
        zones_layout = QVBoxLayout()
        
        # Refresh button
        self.refresh_btn = QPushButton("Refresh Analysis")
        self.refresh_btn.clicked.connect(self.refresh_analysis)
        zones_layout.addWidget(self.refresh_btn)
        
        # Zone table
        self.zone_table = QTableWidget()
        self.zone_table.setColumnCount(6)
        self.zone_table.setHorizontalHeaderLabels([
            "Type", "Range", "Volume%", "Strength", "Valid", "Distance"
        ])
        self.zone_table.setAlternatingRowColors(True)
        
        # Set column widths
        header = self.zone_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        
        self.zone_table.setColumnWidth(0, 60)
        self.zone_table.setColumnWidth(2, 70)
        self.zone_table.setColumnWidth(3, 70)
        self.zone_table.setColumnWidth(4, 50)
        self.zone_table.setColumnWidth(5, 70)
        
        # Enable row selection
        self.zone_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.zone_table.itemSelectionChanged.connect(self.on_zone_selected)
        
        zones_layout.addWidget(self.zone_table)
        zones_group.setLayout(zones_layout)
        layout.addWidget(zones_group)
        
        return widget
    
    def set_data_manager(self, data_manager):
        """Set or update the data manager"""
        initialize_data_manager(data_manager)
        
    def load_ticker(self, ticker: str):
        """Load supply/demand analysis for ticker"""
        self.ticker = ticker
        self.loading_started.emit()
        
        # Clear existing data
        self.plot_widget.clear()
        # Re-add crosshair items after clearing
        self.plot_widget.addItem(self.crosshair_v, ignoreBounds=True)
        self.plot_widget.addItem(self.crosshair_h, ignoreBounds=True)
        self.plot_widget.addItem(self.price_label, ignoreBounds=True)
        
        self.zone_table.setRowCount(0)
        self.summary_label.setText(f"Loading {ticker}...")
        self.summary_label.setStyleSheet("color: #f59e0b; padding: 2px;")
        
        # Start worker thread with 7 days lookback
        self.worker = SupplyDemandWorker(ticker, self.lookback_days)
        self.worker.analysis_complete.connect(self.on_analysis_complete)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.progress_update.connect(self.update_status)
        self.worker.start()
        
    @pyqtSlot(dict)
    def on_analysis_complete(self, result: dict):
        """Handle analysis completion"""
        self.analysis_result = result
        self.loading_finished.emit()
        
        # Update summary with green success color
        summary_text = f"""
        <b>Current Price:</b> ${result['current_price']:.2f}<br>
        <b>Current ATR:</b> ${result['current_atr']:.2f}<br>
        <b>Total Zones:</b> {result['total_zones']}<br>
        <b>Active Zones:</b> {result['active_zones']}<br>
        <b>Nearby Zones:</b> {result['nearby_zones']}
        """
        self.summary_label.setText(summary_text)
        self.summary_label.setStyleSheet("color: #10b981; padding: 2px;")
        
        # Load price data and plot
        self.load_price_data()
        
        # Populate zone table
        if 'all_zones' in result and result['all_zones']:
            self.populate_zone_table(result['all_zones'])
        elif result['zones']:
            self.populate_zone_table(result['zones'])  # Fallbackif 'all_zones' in result and result['all_zones']:

    
    def load_price_data(self):
        """Load price data for charting"""
        try:
            # Check if we have a data manager
            if _data_manager is None:
                logger.error("No data manager available")
                self.error_occurred.emit("Data manager not initialized")
                return
            
            # Create event loop for async operation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Load 15-minute data for 7 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)
            
            # Run async load
            self.price_data = loop.run_until_complete(
                _data_manager.load_data_async(
                    ticker=self.ticker,
                    timeframe='15min',
                    start_date=start_date,
                    end_date=end_date
                )
            )
            
            if self.price_data is not None and not self.price_data.empty:
                self.plot_price_data()
                self.plot_zones()
            else:
                logger.error("No price data returned")
                self.error_occurred.emit("No price data available")
                
        except Exception as e:
            logger.error(f"Error loading price data: {e}")
            self.error_occurred.emit(f"Failed to load price data: {str(e)}")
            
    def plot_price_data(self):
        """Plot candlestick price data using CandlestickItem"""
        if self.price_data is None or self.price_data.empty:
            return
        
        # Reset index for plotting
        df = self.price_data.reset_index()
        
        # Create and add candlestick item
        candlesticks = CandlestickItem(df)
        self.plot_widget.addItem(candlesticks)
        
        # Create time axis labels
        time_strings = [t.strftime('%m/%d %H:%M') for t in self.price_data.index]
        x_dict = dict(enumerate(time_strings))
        
        # Show subset of labels (every 6.5 hours = ~26 bars)
        step = max(1, len(time_strings) // 8)
        x_dict_sparse = {k: v for k, v in x_dict.items() if k % step == 0}
        
        # Create custom axis
        stringaxis = pg.AxisItem(orientation='bottom')
        stringaxis.setTicks([list(x_dict_sparse.items())])
        self.plot_widget.setAxisItems(axisItems={'bottom': stringaxis})
        
        # Add current price line
        if self.analysis_result:
            current_price = self.analysis_result['current_price']
            price_line = pg.InfiniteLine(
                pos=current_price, 
                angle=0, 
                pen=pg.mkPen('#f59e0b', width=2, style=Qt.PenStyle.DashLine),
                label=f'${current_price:.2f}',
                labelOpts={'position': 0.95, 'color': '#f59e0b'}
            )
            self.plot_widget.addItem(price_line)
        
        # Set Y-axis range
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_range = price_max - price_min
        padding = price_range * 0.1
        
        self.plot_widget.setYRange(price_min - padding, price_max + padding)
        
        # Set X-axis to show recent data (display_bars)
        total_bars = len(df)
        if total_bars > self.display_bars:
            start_x = total_bars - self.display_bars
            end_x = total_bars
        else:
            start_x = 0
            end_x = total_bars
            
        self.plot_widget.setXRange(start_x, end_x)
        
        # Enable mouse interaction
        self.plot_widget.setMouseEnabled(x=True, y=True)
        
    def plot_zones(self):
        """Plot supply/demand zones"""
        if self.analysis_result is None or 'zones' not in self.analysis_result:
            return
            
        if self.price_data is None or self.price_data.empty:
            return
            
        # Get price bounds
        price_min = self.price_data['low'].min()
        price_max = self.price_data['high'].max()
        price_range = price_max - price_min
        
        # Add padding
        padding = price_range * 0.05
        display_min = price_min - padding
        display_max = price_max + padding
        
        zones_plotted = 0
        
        for zone in self.analysis_result['zones']:
            # Only plot zones within range
            if zone.price_high < display_min or zone.price_low > display_max:
                continue
                
            # Clip zone to display range
            zone_bottom = max(zone.price_low, display_min)
            zone_top = min(zone.price_high, display_max)
            
            # Set color based on type
            if zone.zone_type == 'supply':
                color = QColor(220, 38, 127)  # Pink/Red
            else:
                color = QColor(34, 197, 94)   # Green
                
            # Adjust for breaker blocks
            if not zone.validated:  # This is a breaker block
                # Use dashed line for breakers
                pen_style = Qt.PenStyle.DashLine
                color.setAlpha(self.unvalidated_alpha)
                label_suffix = " (B)"  # Add (B) for breaker
            else:
                pen_style = Qt.PenStyle.SolidLine
                color.setAlpha(self.validated_alpha)
                label_suffix = ""
                
            # Create zone rectangle with appropriate pen style
            zone_rect = pg.LinearRegionItem(
                values=(zone_bottom, zone_top),
                orientation='horizontal',
                brush=pg.mkBrush(color),
                pen=pg.mkPen(color, width=2, style=pen_style),
                movable=False
            )
            
            self.plot_widget.addItem(zone_rect)
            zones_plotted += 1
            
            # Add zone label
            label_text = f"{zone.zone_type.upper()}{label_suffix}\n{zone.strength:.0f}%"
            zone_label = pg.TextItem(
                text=label_text,
                color='w',
                anchor=(1, 0.5)
            )
            # Position label
            label_x = len(self.price_data) - 10
            zone_label.setPos(label_x, zone.center_price)
            self.plot_widget.addItem(zone_label)
            
        logger.info(f"Plotted {zones_plotted} zones")
            
    def populate_zone_table(self, zones: List[SupplyDemandZone]):
        """Populate the zone details table"""
        # Filter and sort zones
        if self.price_data is not None and not self.price_data.empty:
            price_min = self.price_data['low'].min()
            price_max = self.price_data['high'].max()
            price_range = price_max - price_min
            
            display_min = price_min - price_range * 0.2
            display_max = price_max + price_range * 0.2
            
            visible_zones = [
                zone for zone in zones 
                if not (zone.price_high < display_min or zone.price_low > display_max)
            ]
        else:
            visible_zones = zones
            
        self.zone_table.setRowCount(len(visible_zones))
        
        current_price = self.analysis_result.get('current_price', 0)
        current_atr = self.analysis_result.get('current_atr', 1)
        
        # Sort by distance
        sorted_zones = sorted(visible_zones, key=lambda z: abs(z.center_price - current_price))
        
        for i, zone in enumerate(sorted_zones):
            # Type
            type_item = QTableWidgetItem(zone.zone_type.upper())
            type_item.setForeground(
                QColor(220, 38, 127) if zone.zone_type == 'supply' 
                else QColor(34, 197, 94)
            )
            self.zone_table.setItem(i, 0, type_item)
            
            # Range
            range_text = f"${zone.price_low:.2f}-${zone.price_high:.2f}"
            self.zone_table.setItem(i, 1, QTableWidgetItem(range_text))
            
            # Volume %
            vol_item = QTableWidgetItem(f"{zone.volume_percent:.1f}%")
            self.zone_table.setItem(i, 2, vol_item)
            
            # Strength
            strength_item = QTableWidgetItem(f"{zone.strength:.0f}")
            if zone.strength >= 80:
                strength_item.setForeground(QColor(34, 197, 94))
            elif zone.strength >= 60:
                strength_item.setForeground(QColor(251, 191, 36))
            self.zone_table.setItem(i, 3, strength_item)
            
            # Validated column - show if it's broken
            if zone.validated:
                val_text = "✓"
                val_color = QColor(34, 197, 94)  # Green for valid
            else:
                val_text = "Breaker"
                val_color = QColor(245, 158, 11)  # Orange for breaker blocks
                
            val_item = QTableWidgetItem(val_text)
            val_item.setForeground(val_color)
            self.zone_table.setItem(i, 4, val_item)
            
            # Distance
            distance = abs(zone.center_price - current_price)
            distance_atr = distance / current_atr if current_atr > 0 else 0
            dist_item = QTableWidgetItem(f"{distance_atr:.1f} ATR")
            self.zone_table.setItem(i, 5, dist_item)
            
    def on_zone_selected(self):
        """Handle zone selection in table"""
        selected_rows = self.zone_table.selectionModel().selectedRows()
        if selected_rows:
            row = selected_rows[0].row()
            # Emit signal with zone info
            if self.analysis_result and 'zones' in self.analysis_result:
                zones = self.analysis_result['zones']
                if row < len(zones):
                    self.zone_selected.emit({
                        'zone': zones[row],
                        'row': row
                    })
                
    def on_mouse_moved(self, pos):
        """Handle mouse movement for crosshair"""
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            
            # Update crosshair
            self.crosshair_v.setPos(mouse_point.x())
            self.crosshair_h.setPos(mouse_point.y())
            
            # Update price label
            self.price_label.setText(f"${mouse_point.y():.2f}")
            self.price_label.setPos(mouse_point.x(), mouse_point.y())
            
    @pyqtSlot(str)
    def on_error(self, error_msg: str):
        """Handle errors"""
        self.error_occurred.emit(error_msg)
        self.summary_label.setText(f"Error: {error_msg}")
        self.summary_label.setStyleSheet("color: #ef4444; padding: 2px;")
        self.loading_finished.emit()
        
    @pyqtSlot(str)
    def update_status(self, message: str):
        """Update status message"""
        self.summary_label.setText(message)
        
    def refresh_analysis(self):
        """Refresh the analysis"""
        if self.ticker:
            self.load_ticker(self.ticker)

# Export the chart class
__all__ = ['SupplyDemandChart', 'CandlestickItem', 'initialize_data_manager']