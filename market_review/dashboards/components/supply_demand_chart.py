# market_review/dashboards/components/supply_demand_chart.py
"""
Module: Supply/Demand Zone Chart Component
Purpose: Visualize supply and demand zones with price action
UI Framework: PyQt6 with PyQtGraph
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                            QTableWidget, QTableWidgetItem, QHeaderView,
                            QLabel, QGroupBox, QPushButton)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QColor
import pyqtgraph as pg
import pandas as pd
import numpy as np

# Import the supply/demand analyzer
from market_review.calculations.zones.supply_demand import (
    analyze_supply_demand_zones, get_strongest_zones, 
    SupplyDemandZone, set_data_manager
)

logger = logging.getLogger(__name__)


class MockDataManager:
    """Mock data manager for testing when real DataManager is not available"""
    
    async def load_data_async(self, **kwargs):
        """Generate mock data for testing"""
        ticker = kwargs.get('ticker', 'TEST')
        timeframe = kwargs.get('timeframe', '15min')
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        
        # Generate time series
        if timeframe == '15min':
            freq = '15min'
        elif timeframe == '5min':
            freq = '5min'
        else:
            freq = '1h'
            
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Generate realistic OHLCV data
        np.random.seed(42)
        base_price = 100
        returns = np.random.randn(len(dates)) * 0.02  # 2% volatility
        close_prices = base_price * np.exp(returns.cumsum())
        
        df = pd.DataFrame(index=dates)
        df['close'] = close_prices
        df['open'] = df['close'].shift(1).fillna(base_price)
        df['high'] = df[['open', 'close']].max(axis=1) + np.abs(np.random.randn(len(dates)) * 0.5)
        df['low'] = df[['open', 'close']].min(axis=1) - np.abs(np.random.randn(len(dates)) * 0.5)
        df['volume'] = np.random.randint(100000, 1000000, len(dates))
        
        return df


class SupplyDemandWorker(QThread):
    """Worker thread for async supply/demand analysis"""
    
    # Signals
    analysis_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(str)
    
    def __init__(self, ticker: str, lookback_days: int = 15):
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
        """Ensure data manager is set for supply/demand module"""
        from market_review.calculations.zones import supply_demand
        
        # Check if data manager is already set
        if supply_demand._data_manager is None:
            logger.warning("Data manager not set, attempting to initialize...")
            
            try:
                # Try to get the real DataManager
                from market_review.dashboards.data_manager import DataManager
                data_manager = DataManager.get_instance()
                set_data_manager(data_manager)
                logger.info("Successfully set DataManager")
            except ImportError:
                # Use mock data manager as fallback
                logger.warning("DataManager not available, using mock data")
                mock_manager = MockDataManager()
                set_data_manager(mock_manager)


class SupplyDemandChart(QWidget):
    """Chart widget for displaying supply/demand zones"""
    
    # Signals
    loading_started = pyqtSignal()
    loading_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    zone_selected = pyqtSignal(dict)
    
    def __init__(self, lookback_days: int = 15, display_bars: int = None):
        super().__init__()
        self.ticker = None
        self.lookback_days = lookback_days
        # Calculate display bars: 15 days * 6.5 hours * 4 (15-min bars per hour)
        self.display_bars = display_bars or (15 * 6.5 * 4)  # ~390 15-min bars for 15 days
        self.analysis_result = None
        self.price_data = None
        self.price_data_15min = None  # Store 15-minute data
        
        # Zone colors
        self.supply_color = QColor(220, 38, 127, 80)  # Pink/Red with transparency
        self.demand_color = QColor(34, 197, 94, 80)   # Green with transparency
        self.validated_alpha = 120  # Higher alpha for validated zones
        self.unvalidated_alpha = 60  # Lower alpha for unvalidated zones
        
        # Initialize data manager on creation
        self._initialize_data_manager()
        
        self.init_ui()
        
    def _initialize_data_manager(self):
        """Initialize data manager for supply/demand module"""
        try:
            from market_review.dashboards.data_manager import DataManager
            data_manager = DataManager.get_instance()
            set_data_manager(data_manager)
            logger.info("DataManager initialized for Supply/Demand")
        except ImportError:
            logger.warning("DataManager not available, will use mock data")
            # Don't set mock here, let the worker handle it
        
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
        """Create the chart widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Price ($)')
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Add crosshair
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('w', width=0.5, style=Qt.PenStyle.DashLine))
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('w', width=0.5, style=Qt.PenStyle.DashLine))
        self.plot_widget.addItem(self.crosshair_v, ignoreBounds=True)
        self.plot_widget.addItem(self.crosshair_h, ignoreBounds=True)
        
        # Price label
        self.price_label = pg.TextItem(anchor=(0, 1), color='w', fill=pg.mkBrush(30, 30, 30, 180))
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
        
        # Start worker thread
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
        
        # Update summary
        summary_text = f"""
        <b>Current Price:</b> ${result['current_price']:.2f}<br>
        <b>Current ATR:</b> ${result['current_atr']:.2f}<br>
        <b>Total Zones:</b> {result['total_zones']}<br>
        <b>Active Zones:</b> {result['active_zones']}<br>
        <b>Nearby Zones:</b> {result['nearby_zones']}
        """
        self.summary_label.setText(summary_text)
        
        # Load price data and plot
        self.load_price_data()
        
        # Populate zone table
        if result['zones']:
            self.populate_zone_table(result['zones'])
        
    def load_price_data(self):
        """Load 15-minute price data for charting"""
        try:
            # Try to get data manager
            try:
                from market_review.dashboards.data_manager import DataManager
                data_manager = DataManager.get_instance()
                
                # Load 15-minute data for the same period as analysis
                end_date = datetime.now()
                start_date = end_date - timedelta(days=self.lookback_days)
                
                self.price_data_15min = data_manager.load_data(
                    ticker=self.ticker,
                    timeframe='15min',  # Changed to 15-minute
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Use the 15-minute data for display
                self.price_data = self.price_data_15min
                
            except (ImportError, Exception) as e:
                # Fallback: use mock 15-minute data
                logger.warning(f"DataManager not available for charting: {e}")
                
                # Generate 15-minute data for visualization
                if self.analysis_result and 'current_price' in self.analysis_result:
                    current_price = self.analysis_result['current_price']
                    
                    # Generate 15 days of 15-minute data
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=self.lookback_days)
                    dates = pd.date_range(start=start_date, end=end_date, freq='15min')
                    
                    # Filter for market hours only (optional)
                    dates = [d for d in dates if 13 <= d.hour < 20 or (d.hour == 20 and d.minute == 0)]
                    
                    # Generate price data around current price
                    np.random.seed(hash(self.ticker) % 2**32)
                    returns = np.random.randn(len(dates)) * 0.002  # Slightly higher volatility for 15-min
                    close_prices = current_price * np.exp(returns.cumsum())
                    
                    self.price_data = pd.DataFrame(index=dates)
                    self.price_data['close'] = close_prices
                    self.price_data['open'] = self.price_data['close'].shift(1).fillna(current_price)
                    self.price_data['high'] = self.price_data[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.randn(len(dates)) * 0.002))
                    self.price_data['low'] = self.price_data[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.randn(len(dates)) * 0.002))
                    self.price_data['volume'] = np.random.randint(50000, 500000, len(dates))
            
            if self.price_data is not None and not self.price_data.empty:
                self.plot_price_data()
                self.plot_zones()
                
        except Exception as e:
            logger.error(f"Error loading price data: {e}")
            self.error_occurred.emit(f"Failed to load price data: {str(e)}")
            
    def plot_price_data(self):
        """Plot candlestick price data"""
        if self.price_data is None or self.price_data.empty:
            return
            
        # Use all available data (15 days of 15-min bars)
        df = self.price_data
        x = np.arange(len(df))
        
        # Plot candlesticks
        for i, (idx, row) in enumerate(df.iterrows()):
            color = 'g' if row['close'] >= row['open'] else 'r'
            
            # Body
            body = pg.PlotDataItem(
                [i, i], [row['open'], row['close']],
                pen=pg.mkPen(color, width=3)  # Thinner for 15-min bars
            )
            self.plot_widget.addItem(body)
            
            # Wicks
            wick = pg.PlotDataItem(
                [i, i], [row['low'], row['high']],
                pen=pg.mkPen(color, width=1)
            )
            self.plot_widget.addItem(wick)
            
        # Set x-axis with better labels for 15-minute data
        tick_labels = []
        # Show labels every day
        for i in range(0, len(df), 26):  # ~26 15-min bars per day (market hours)
            if i < len(df):
                tick_labels.append((i, df.index[i].strftime('%m/%d')))
        
        self.plot_widget.getAxis('bottom').setTicks([tick_labels])
        
        # Set Y-axis range based on data
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_range = price_max - price_min
        padding = price_range * 0.1  # 10% padding
        
        self.plot_widget.setYRange(price_min - padding, price_max + padding)
        self.plot_widget.setXRange(0, len(df))
        
    def plot_zones(self):
        """Plot supply/demand zones - only show zones within price bounds"""
        if self.analysis_result is None or 'zones' not in self.analysis_result:
            return
            
        if self.price_data is None or self.price_data.empty:
            return
            
        # Get price bounds from the displayed data
        price_min = self.price_data['low'].min()
        price_max = self.price_data['high'].max()
        price_range = price_max - price_min
        
        # Add some padding to include zones slightly outside the range
        padding = price_range * 0.05  # 5% padding
        display_min = price_min - padding
        display_max = price_max + padding
        
        zones_plotted = 0
        
        for zone in self.analysis_result['zones']:
            # Only plot zones that are within or near the price range
            if zone.price_high < display_min or zone.price_low > display_max:
                continue  # Skip zones completely outside the range
                
            # Clip zone to display range
            zone_bottom = max(zone.price_low, display_min)
            zone_top = min(zone.price_high, display_max)
            
            # Set color based on type
            if zone.zone_type == 'supply':
                color = QColor(220, 38, 127)  # Pink/Red
            else:
                color = QColor(34, 197, 94)   # Green
                
            # Adjust alpha for validation
            if zone.validated:
                color.setAlpha(self.validated_alpha)
            else:
                color.setAlpha(self.unvalidated_alpha)
                
            # Create zone rectangle
            zone_rect = pg.LinearRegionItem(
                values=(zone_bottom, zone_top),
                orientation='horizontal',
                brush=pg.mkBrush(color),
                pen=pg.mkPen(color, width=2),
                movable=False
            )
            
            self.plot_widget.addItem(zone_rect)
            zones_plotted += 1
            
            # Add zone label
            label_text = f"{zone.zone_type.upper()}\n{zone.strength:.0f}%"
            zone_label = pg.TextItem(
                text=label_text,
                color='w',
                anchor=(1, 0.5)  # Anchor to right side
            )
            # Position label at the right edge of the chart
            label_x = len(self.price_data) - 10
            zone_label.setPos(label_x, zone.center_price)
            self.plot_widget.addItem(zone_label)
            
        logger.info(f"Plotted {zones_plotted} zones within price bounds")
            
    def populate_zone_table(self, zones: List[SupplyDemandZone]):
        """Populate the zone details table"""
        # Filter zones to show only those within reasonable range
        if self.price_data is not None and not self.price_data.empty:
            price_min = self.price_data['low'].min()
            price_max = self.price_data['high'].max()
            price_range = price_max - price_min
            
            # Show zones within 20% of the price range
            display_min = price_min - price_range * 0.2
            display_max = price_max + price_range * 0.2
            
            # Filter zones
            visible_zones = [
                zone for zone in zones 
                if not (zone.price_high < display_min or zone.price_low > display_max)
            ]
        else:
            visible_zones = zones
            
        self.zone_table.setRowCount(len(visible_zones))
        
        current_price = self.analysis_result.get('current_price', 0)
        current_atr = self.analysis_result.get('current_atr', 1)
        
        # Sort zones by distance from current price
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
            
            # Validated
            val_item = QTableWidgetItem("✓" if zone.validated else "✗")
            val_item.setForeground(
                QColor(34, 197, 94) if zone.validated 
                else QColor(156, 163, 175)
            )
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
            # Need to account for filtered zones
            if self.analysis_result and 'zones' in self.analysis_result:
                # Get the visible zones in the same order as the table
                zones = self.analysis_result['zones']
                
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
                    
                current_price = self.analysis_result.get('current_price', 0)
                sorted_zones = sorted(visible_zones, key=lambda z: abs(z.center_price - current_price))
                
                if row < len(sorted_zones):
                    zone = sorted_zones[row]
                    self.zone_selected.emit({
                        'zone': zone,
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
        self.loading_finished.emit()
        
    @pyqtSlot(str)
    def update_status(self, message: str):
        """Update status message"""
        self.summary_label.setText(message)
        
    def refresh_analysis(self):
        """Refresh the analysis"""
        if self.ticker:
            self.load_ticker(self.ticker)