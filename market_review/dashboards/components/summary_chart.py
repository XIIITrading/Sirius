# market_review/dashboards/components/summary_chart.py
"""
Module: Summary Chart Component
Purpose: Integrated chart showing HVN, Supply/Demand zones, and Camarilla pivots
Features:
- 7-day lookback with 15-minute bars
- Aggregated zones with overlap detection
- Camarilla pivot levels
- Unified visualization
"""

import logging
from typing import Optional, Dict, Any, List

import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QGroupBox, QPushButton, QCheckBox, QSlider)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QColor

import pandas as pd
import numpy as np

# Local imports
from market_review.calculations.volume.hvn_engine import HVNEngine
from market_review.calculations.pivots.camarilla_engine import CamarillaCalculator
from market_review.calculations.zones.supply_demand import analyze_supply_demand_zones
from market_review.data.polygon_bridge import PolygonHVNBridge
from market_review.dashboards.components.zone_aggregator import ZoneAggregator, UnifiedZone

# Configure logging
logger = logging.getLogger(__name__)

# Configure PyQtGraph
pg.setConfigOptions(antialias=True)


class SummaryDataWorker(QThread):
    """Background worker for fetching and processing all data"""
    
    data_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(str)
    
    def __init__(self, ticker: str, lookback_days: int = 7):
        super().__init__()
        self.ticker = ticker
        self.lookback_days = lookback_days
        
    def run(self):
        """Run comprehensive analysis"""
        try:
            import asyncio
            from datetime import datetime, timezone
            
            self.progress_update.emit(f"Fetching data for {self.ticker}...")
            
            # Initialize bridge
            bridge = PolygonHVNBridge(
                hvn_levels=100,
                hvn_percentile=80.0,
                lookback_days=14  # Fetch more for calculations
            )
            
            # Fetch price data and calculate HVN
            state = bridge.calculate_hvn(self.ticker, timeframe='15min')
            
            if not state or state.recent_bars.empty:
                self.error_occurred.emit("No data available")
                return
                
            # Get price data
            price_data = state.recent_bars
            current_price = state.current_price
            
            # 1. HVN Results (already calculated)
            self.progress_update.emit("Processing HVN zones...")
            hvn_result = state.hvn_result
            
            # 2. Camarilla Pivots
            self.progress_update.emit("Calculating Camarilla levels...")
            camarilla_calc = CamarillaCalculator()
            camarilla_result = camarilla_calc.calculate(price_data, ticker=self.ticker)
            
            # 3. Supply/Demand Zones
            self.progress_update.emit("Analyzing supply/demand zones...")
            
            # Create event loop for async analysis
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Ensure data manager is set for supply_demand
            try:
                from market_review.dashboards.components.supply_demand_chart import PolygonDataManager, initialize_data_manager
                polygon_dm = PolygonDataManager()
                initialize_data_manager(polygon_dm)
            except Exception as e:
                logger.warning(f"Could not initialize data manager: {e}")
            
            # Run supply/demand analysis
            sd_result = loop.run_until_complete(
                analyze_supply_demand_zones(
                    self.ticker,
                    lookback_days=self.lookback_days,
                    analysis_lookback_days=30,  # Analyze more days
                    show_bullish_obs=5,
                    show_bearish_obs=5
                )
            )
            
            # 4. Aggregate zones
            self.progress_update.emit("Aggregating zones...")
            aggregator = ZoneAggregator(overlap_threshold=0.1)  # 10% overlap
            
            aggregated_zones = aggregator.aggregate_zones(
                hvn_result=hvn_result,
                supply_demand_result=sd_result
            )
            
            # Get zones near current price
            nearby_zones = aggregator.get_zones_near_price(
                aggregated_zones, 
                current_price,
                distance_percent=0.03  # 3% of price
            )
            
            # Package results
            result = {
                'ticker': self.ticker,
                'price_data': price_data,
                'current_price': current_price,
                'hvn_result': hvn_result,
                'camarilla_result': camarilla_result,
                'supply_demand_result': sd_result,
                'aggregated_zones': aggregated_zones,
                'nearby_zones': nearby_zones,
                'timestamp': datetime.now(timezone.utc)
            }
            
            self.data_ready.emit(result)
            
        except Exception as e:
            logger.error(f"Error in SummaryDataWorker: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.error_occurred.emit(str(e))


class CandlestickItem(pg.GraphicsObject):
    """Custom candlestick item for charts"""
    
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


class SummaryChart(QWidget):
    """
    Summary chart showing all analysis types in one view
    """
    
    # Signals
    loading_started = pyqtSignal()
    loading_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    zone_selected = pyqtSignal(dict)
    
    def __init__(self, parent=None, lookback_days: int = 7, display_bars: int = 182):
        """
        Initialize Summary Chart
        
        Args:
            parent: Parent widget
            lookback_days: Days to look back (default: 7)
            display_bars: Number of 15-min bars to display (default: 182 = 7 days)
        """
        super().__init__(parent)
        
        self.lookback_days = lookback_days
        self.display_bars = display_bars
        self.ticker = None
        self.chart_data = None
        self.worker = None
        
        # Zone display toggles
        self.show_hvn_zones = True
        self.show_sd_zones = True
        self.show_camarilla = True
        
        # Zone items for toggling
        self.zone_items = []
        self.camarilla_items = []
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Container
        self.container = QGroupBox("Summary Analysis")
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
                color: #a855f7;
            }
        """)
        
        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(5, 5, 5, 5)
        
        # Controls section
        controls_widget = self.create_controls()
        container_layout.addWidget(controls_widget)
        
        # Info label
        self.info_label = QLabel("No data loaded")
        self.info_label.setStyleSheet("color: #9ca3af; padding: 2px;")
        container_layout.addWidget(self.info_label)
        
        # Summary stats
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("color: #10b981; padding: 2px; font-size: 11px;")
        container_layout.addWidget(self.stats_label)
        
        # Chart widget
        self.chart_widget = pg.GraphicsLayoutWidget()
        self.chart_widget.setBackground('#1a1a1a')
        
        # Create plot
        self.plot = self.chart_widget.addPlot(row=0, col=0)
        self.plot.setLabel('left', 'Price', units='$')
        self.plot.setLabel('bottom', 'Time (UTC)')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Add legend
        self.legend = self.plot.addLegend(offset=(10, 10))
        self.legend.setParentItem(self.plot.vb)
        self.legend.anchor((0, 0), (0, 0))
        
        container_layout.addWidget(self.chart_widget)
        self.container.setLayout(container_layout)
        
        layout.addWidget(self.container)
        
    def create_controls(self):
        """Create control widgets"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Zone toggles
        self.hvn_check = QCheckBox("HVN Zones")
        self.hvn_check.setChecked(True)
        self.hvn_check.setStyleSheet("color: #10b981;")
        self.hvn_check.stateChanged.connect(self.toggle_hvn_zones)
        layout.addWidget(self.hvn_check)
        
        self.sd_check = QCheckBox("Supply/Demand")
        self.sd_check.setChecked(True)
        self.sd_check.setStyleSheet("color: #f59e0b;")
        self.sd_check.stateChanged.connect(self.toggle_sd_zones)
        layout.addWidget(self.sd_check)
        
        self.camarilla_check = QCheckBox("Camarilla Pivots")
        self.camarilla_check.setChecked(True)
        self.camarilla_check.setStyleSheet("color: #a855f7;")
        self.camarilla_check.stateChanged.connect(self.toggle_camarilla)
        layout.addWidget(self.camarilla_check)
        
        layout.addStretch()
        
        # Refresh button
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setStyleSheet("""
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
        """)
        self.refresh_btn.clicked.connect(self.refresh_analysis)
        layout.addWidget(self.refresh_btn)
        
        return widget
        
    def load_ticker(self, ticker: str):
        """Load data and perform analysis for ticker"""
        if not ticker:
            return
            
        self.ticker = ticker.upper()
        logger.info(f"Loading summary analysis for: {self.ticker}")
        
        # Clear any existing worker
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            
        # Clear chart
        self.clear_chart()
        
        # Update labels
        self.info_label.setText(f"Loading {self.ticker}...")
        self.stats_label.setText("")
        
        # Emit loading signal
        self.loading_started.emit()
        
        # Start worker
        self.worker = SummaryDataWorker(self.ticker, self.lookback_days)
        self.worker.data_ready.connect(self.on_data_ready)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.progress_update.connect(
            lambda msg: self.info_label.setText(msg)
        )
        
        self.worker.start()
        
    def clear_chart(self):
        """Clear all chart items"""
        self.plot.clear()
        self.zone_items.clear()
        self.camarilla_items.clear()
        self.legend.clear()
        
    @pyqtSlot(dict)
    def on_data_ready(self, data: dict):
        """Handle data ready from worker"""
        self.chart_data = data
        self.update_chart()
        self.loading_finished.emit()
        
    @pyqtSlot(str)
    def on_error(self, error_msg: str):
        """Handle error from worker"""
        self.info_label.setText(f"Error: {error_msg}")
        self.info_label.setStyleSheet("color: #ef4444; padding: 2px;")
        self.error_occurred.emit(error_msg)
        
    def update_chart(self):
        """Update chart with all data"""
        if not self.chart_data:
            return
            
        # Clear previous items
        self.clear_chart()
        
        # Get data
        price_data = self.chart_data['price_data']
        current_price = self.chart_data['current_price']
        aggregated_zones = self.chart_data['aggregated_zones']
        camarilla_result = self.chart_data.get('camarilla_result')
        
        # Limit display to recent data
        if len(price_data) > self.display_bars:
            display_data = price_data.tail(self.display_bars).copy()
        else:
            display_data = price_data.copy()
            
        # Update info label
        self.info_label.setText(
            f"{self.ticker} | Current: ${current_price:.2f} | "
            f"Zones: {len(aggregated_zones)} | "
            f"Nearby: {len(self.chart_data['nearby_zones'])}"
        )
        self.info_label.setStyleSheet("color: #10b981; padding: 2px;")
        
        # Update stats
        stats_text = self._generate_stats_text()
        self.stats_label.setText(stats_text)
        
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
        
        # 1. Draw candlesticks
        candlestick = CandlestickItem(display_data_reset)
        self.plot.addItem(candlestick)
        
        # 2. Draw aggregated zones
        self._draw_zones(aggregated_zones, display_data)
        
        # 3. Draw Camarilla levels
        if camarilla_result:
            self._draw_camarilla_levels(camarilla_result)
            
        # 4. Draw current price line
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
        
        # Calculate Y range to include all elements
        y_min = display_data['low'].min()
        y_max = display_data['high'].max()
        
        # Include zones in range
        for zone in aggregated_zones:
            if zone.price_low < y_min:
                y_min = zone.price_low
            if zone.price_high > y_max:
                y_max = zone.price_high
                
        # Add padding
        y_range = y_max - y_min
        y_min -= y_range * 0.05
        y_max += y_range * 0.05
        
        self.plot.setYRange(y_min, y_max)
        
        # Enable mouse interaction
        self.plot.setMouseEnabled(x=True, y=True)
        
    def _generate_stats_text(self) -> str:
        """Generate summary statistics text"""
        if not self.chart_data:
            return ""
            
        zones = self.chart_data['aggregated_zones']
        hvn_zones = [z for z in zones if z.source_type == 'hvn']
        sd_zones = [z for z in zones if z.source_type == 'supply_demand']
        
        stats = []
        stats.append(f"HVN Zones: {len(hvn_zones)}")
        stats.append(f"S/D Zones: {len(sd_zones)}")
        
        if self.chart_data.get('camarilla_result'):
            cam = self.chart_data['camarilla_result']
            stats.append(f"Pivot: ${cam.central_pivot:.2f}")
            
        return " | ".join(stats)
        
    def _draw_zones(self, zones: List[UnifiedZone], display_data: pd.DataFrame):
        """Draw aggregated zones on chart"""
        for zone in zones:
            # Skip if zone type is hidden
            if zone.source_type == 'hvn' and not self.show_hvn_zones:
                continue
            if zone.source_type == 'supply_demand' and not self.show_sd_zones:
                continue
                
            # Create zone rectangle
            color = QColor(zone.display_color)
            color.setAlpha(zone.opacity)
            
            if zone.display_style == 'dashed':
                pen = pg.mkPen(color, width=2, style=Qt.PenStyle.DashLine)
            else:
                pen = pg.mkPen(color, width=2)
                
            zone_rect = pg.LinearRegionItem(
                values=(zone.price_low, zone.price_high),
                orientation='horizontal',
                brush=pg.mkBrush(color),
                pen=pen,
                movable=False
            )
            
            # Add to plot and store reference
            self.plot.addItem(zone_rect)
            self.zone_items.append({
                'item': zone_rect,
                'zone': zone,
                'type': zone.source_type
            })
            
            # Add zone label
            label_text = f"{zone.zone_name}\n{zone.strength:.0f}%"
            zone_label = pg.TextItem(
                text=label_text,
                color='w',
                anchor=(1, 0.5),
                fill=pg.mkBrush(30, 30, 30, 180)
            )
            
            # Position label
            label_x = len(display_data) - 5
            zone_label.setPos(label_x, zone.center_price)
            self.plot.addItem(zone_label)
            
            # Store label reference
            self.zone_items.append({
                'item': zone_label,
                'zone': zone,
                'type': zone.source_type
            })
            
    def _draw_camarilla_levels(self, camarilla_result):
        """Draw Camarilla pivot levels"""
        if not self.show_camarilla:
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
        
        # Draw resistance levels
        for name, price in camarilla_result.resistance_levels.items():
            style = level_styles.get(name, {})
            line = pg.InfiniteLine(
                pos=price, 
                angle=0, 
                pen=pg.mkPen(
                    style.get('color', '#ffffff'),
                    width=style.get('width', 1),
                    style=style.get('style', Qt.PenStyle.DashLine)
                ),
                label=f'{name}: ${price:.2f}',
                labelOpts={'position': 0.02, 'color': style.get('color', '#ffffff')}
            )
            self.plot.addItem(line)
            self.camarilla_items.append(line)
            
        # Draw support levels
        for name, price in camarilla_result.support_levels.items():
            style = level_styles.get(name, {})
            line = pg.InfiniteLine(
                pos=price, 
                angle=0, 
                pen=pg.mkPen(
                    style.get('color', '#ffffff'),
                    width=style.get('width', 1),
                    style=style.get('style', Qt.PenStyle.DashLine)
                ),
                label=f'{name}: ${price:.2f}',
                labelOpts={'position': 0.02, 'color': style.get('color', '#ffffff')}
            )
            self.plot.addItem(line)
            self.camarilla_items.append(line)
            
        # Draw central pivot
        pivot_line = pg.InfiniteLine(
            pos=camarilla_result.central_pivot, 
            angle=0, 
            pen=pg.mkPen('#a855f7', width=3, style=Qt.PenStyle.DotLine),
            label=f'Pivot: ${camarilla_result.central_pivot:.2f}',
            labelOpts={'position': 0.02, 'color': '#a855f7'}
        )
        self.plot.addItem(pivot_line)
        self.camarilla_items.append(pivot_line)
        
    def toggle_hvn_zones(self, state):
        """Toggle HVN zones visibility"""
        self.show_hvn_zones = state == Qt.CheckState.Checked.value
        self._update_zone_visibility()
        
    def toggle_sd_zones(self, state):
        """Toggle Supply/Demand zones visibility"""
        self.show_sd_zones = state == Qt.CheckState.Checked.value
        self._update_zone_visibility()
        
    def toggle_camarilla(self, state):
        """Toggle Camarilla levels visibility"""
        self.show_camarilla = state == Qt.CheckState.Checked.value
        self._update_camarilla_visibility()
        
    def _update_zone_visibility(self):
        """Update visibility of zone items"""
        for item_data in self.zone_items:
            if item_data['type'] == 'hvn':
                item_data['item'].setVisible(self.show_hvn_zones)
            elif item_data['type'] == 'supply_demand':
                item_data['item'].setVisible(self.show_sd_zones)
                
    def _update_camarilla_visibility(self):
        """Update visibility of Camarilla items"""
        for item in self.camarilla_items:
            item.setVisible(self.show_camarilla)
            
    def refresh_analysis(self):
        """Refresh the analysis"""
        if self.ticker:
            self.load_ticker(self.ticker)


# Export classes
__all__ = ['SummaryChart', 'SummaryDataWorker']