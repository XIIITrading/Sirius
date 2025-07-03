"""
Bid/Ask Ratio Chart Component
Standalone chart widget for visualizing buy/sell pressure with volume
"""

import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPen, QColor

# Configure pyqtgraph for better appearance
pg.setConfigOptions(antialias=True)
pg.setConfigOption('background', '#1e1e1e')
pg.setConfigOption('foreground', '#ffffff')


class BidAskRatioChart(QWidget):
    """
    Chart widget for displaying bid/ask ratio with volume bars.
    Designed to be embedded in the backtest dashboard.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Chart data
        self.timestamps = []
        self.ratios = []
        self.volumes = []
        self.positive_volumes = []
        self.negative_volumes = []
        
        # Setup UI
        self.init_ui()
        
        # Auto-update timer (for future real-time use)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_chart)
        self.auto_update = False
        
    def init_ui(self):
        """Initialize the chart UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create plot widget with 2 plots (ratio and volume)
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground('#1e1e1e')  # Set widget background
        layout.addWidget(self.plot_widget)
        
        # Ratio plot (top, 70% height)
        self.ratio_plot = self.plot_widget.addPlot(row=0, col=0)
        self.ratio_plot.setLabel('left', 'Buy/Sell Pressure')
        self.ratio_plot.setLabel('bottom', 'Time')
        self.ratio_plot.showGrid(x=True, y=True, alpha=0.3)
        self.ratio_plot.setYRange(-1.25, 1.25)
        
        # Add reference lines
        self._add_reference_lines()
        
        # Create ratio line
        self.ratio_line = self.ratio_plot.plot(
            pen=pg.mkPen(color='#00ff00', width=2),
            name='Pressure'
        )
        
        # Volume plot (bottom, 30% height)
        self.volume_plot = self.plot_widget.addPlot(row=1, col=0)
        self.volume_plot.setLabel('left', 'Volume')
        self.volume_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Link X axes
        self.volume_plot.setXLink(self.ratio_plot)
        
        # Create volume bars (we'll use bar graph)
        self.volume_bars_positive = pg.BarGraphItem(
            x=[], height=[], width=0.8, brush='#0d7377'
        )
        self.volume_bars_negative = pg.BarGraphItem(
            x=[], height=[], width=0.8, brush='#ff4444'
        )
        
        self.volume_plot.addItem(self.volume_bars_positive)
        self.volume_plot.addItem(self.volume_bars_negative)
        
        # Set plot heights (70/30 split)
        self.plot_widget.ci.layout.setRowStretchFactor(0, 70)
        self.plot_widget.ci.layout.setRowStretchFactor(1, 30)
        
        # Customize appearance
        self._apply_dark_theme()
        
    def _add_reference_lines(self):
        """Add dotted reference lines to ratio plot"""
        reference_levels = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]
        
        for level in reference_levels:
            # Stronger line for zero
            if level == 0:
                pen = pg.mkPen(color='#ffffff', width=1, style=Qt.PenStyle.DashLine)
            else:
                pen = pg.mkPen(color='#666666', width=1, style=Qt.PenStyle.DotLine)
            
            line = pg.InfiniteLine(
                pos=level,
                angle=0,
                pen=pen,
                movable=False
            )
            self.ratio_plot.addItem(line)
            
    def _apply_dark_theme(self):
        """Apply dark theme to match dashboard"""
        # Set plot backgrounds using ViewBox
        self.ratio_plot.getViewBox().setBackgroundColor('#2b2b2b')
        self.volume_plot.getViewBox().setBackgroundColor('#2b2b2b')
        
        # Axis colors
        for plot in [self.ratio_plot, self.volume_plot]:
            # Axis pen colors
            plot.getAxis('left').setPen(pg.mkPen('#ffffff'))
            plot.getAxis('bottom').setPen(pg.mkPen('#ffffff'))
            
            # Axis label colors
            plot.getAxis('left').setTextPen(pg.mkPen('#ffffff'))
            plot.getAxis('bottom').setTextPen(pg.mkPen('#ffffff'))
            
            # Grid colors are already set via showGrid alpha parameter
    
    def update_from_data(self, chart_data: List[Dict[str, Any]]):
        """
        Standard method name for receiving data from dashboard.
        This is what the dashboard expects to call.
        
        Args:
            chart_data: List of dicts with data format from plugin
        """
        # Delegate to existing update_data method
        self.update_data(chart_data)
    
    def update_data(self, chart_data: List[Dict]):
        """
        Update chart with new data from SimpleDeltaTracker.
        
        Args:
            chart_data: List of dicts with keys:
                - timestamp: ISO format timestamp
                - buy_sell_ratio: Pressure value (-1 to 1)
                - volume: Total volume
                - positive_volume: Volume with positive pressure
                - negative_volume: Volume with negative pressure
        """
        if not chart_data:
            return
        
        # Clear existing data
        self.timestamps.clear()
        self.ratios.clear()
        self.volumes.clear()
        self.positive_volumes.clear()
        self.negative_volumes.clear()
        
        # Process new data
        for i, point in enumerate(chart_data):
            # Convert timestamp to minutes from start
            if i == 0:
                start_time = datetime.fromisoformat(point['timestamp'])
            
            current_time = datetime.fromisoformat(point['timestamp'])
            minutes_from_start = (current_time - start_time).total_seconds() / 60
            
            self.timestamps.append(minutes_from_start)
            self.ratios.append(point['buy_sell_ratio'])
            self.volumes.append(point['volume'])
            self.positive_volumes.append(point.get('positive_volume', 0))
            self.negative_volumes.append(point.get('negative_volume', 0))
        
        # Update chart
        self.update_chart()
        
    def update_chart(self):
        """Refresh the chart display"""
        if not self.timestamps:
            return
        
        # Update ratio line
        self.ratio_line.setData(x=self.timestamps, y=self.ratios)
        
        # Update volume bars
        # For positive pressure bars
        pos_heights = []
        neg_heights = []
        
        for i, ratio in enumerate(self.ratios):
            if ratio >= 0:
                pos_heights.append(self.volumes[i])
                neg_heights.append(0)
            else:
                pos_heights.append(0)
                neg_heights.append(self.volumes[i])
        
        self.volume_bars_positive.setOpts(
            x=self.timestamps,
            height=pos_heights,
            width=0.8
        )
        
        self.volume_bars_negative.setOpts(
            x=self.timestamps,
            height=neg_heights,
            width=0.8
        )
        
        # Auto-scale volume plot
        if self.volumes:
            max_vol = max(self.volumes)
            self.volume_plot.setYRange(0, max_vol * 1.1)
        
        # Set X-axis to show time labels
        self._update_time_axis()
        
    def _update_time_axis(self):
        """Update X-axis to show time labels"""
        if len(self.timestamps) < 2:
            return
        
        # Create time labels every 5 minutes
        ticks = []
        for i in range(0, int(max(self.timestamps)) + 1, 5):
            label = f"-{int(max(self.timestamps) - i)}m"
            ticks.append((i, label))
        
        # Set custom axis
        self.ratio_plot.getAxis('bottom').setTicks([ticks])
        
    def set_real_time_mode(self, enabled: bool, update_interval_ms: int = 1000):
        """
        Enable/disable real-time update mode.
        
        Args:
            enabled: Whether to enable auto-update
            update_interval_ms: Update interval in milliseconds
        """
        self.auto_update = enabled
        
        if enabled:
            self.update_timer.start(update_interval_ms)
        else:
            self.update_timer.stop()
    
    def add_marker(self, minute: float, label: str, color: str = '#ffffff'):
        """
        Add a vertical marker at specific time.
        
        Args:
            minute: Minute from start
            label: Label for marker
            color: Marker color
        """
        line = pg.InfiniteLine(
            pos=minute,
            angle=90,
            pen=pg.mkPen(color=color, width=2, style=Qt.PenStyle.DashLine),
            movable=False
        )
        
        # Add label
        text = pg.TextItem(label, color=color, anchor=(0.5, 1))
        text.setPos(minute, 1.2)
        
        self.ratio_plot.addItem(line)
        self.ratio_plot.addItem(text)
    
    def add_entry_marker(self, entry_time_iso: str):
        """
        Add entry marker based on ISO timestamp.
        Matches the Impact Success pattern.
        
        Args:
            entry_time_iso: Entry time in ISO format
        """
        # This assumes the entry time is at the end of the data (30 minutes)
        self.add_marker(30, "Entry", "#ff0000")
    
    def clear_data(self):
        """Clear all chart data (standard interface)"""
        # Delegate to existing clear method
        self.clear()
    
    def clear(self):
        """Clear all chart data"""
        self.timestamps.clear()
        self.ratios.clear()
        self.volumes.clear()
        self.positive_volumes.clear()
        self.negative_volumes.clear()
        
        self.ratio_line.clear()
        self.volume_bars_positive.setOpts(x=[], height=[])
        self.volume_bars_negative.setOpts(x=[], height=[])
        
    def get_current_ratio(self) -> Optional[float]:
        """Get the most recent ratio value"""
        return self.ratios[-1] if self.ratios else None


# Standalone test function
def test_chart():
    """Test the chart with sample data"""
    import sys
    from PyQt6.QtWidgets import QApplication, QMainWindow
    
    app = QApplication(sys.argv)
    
    # Create main window
    window = QMainWindow()
    window.setWindowTitle("Bid/Ask Ratio Chart Test")
    window.setGeometry(100, 100, 800, 600)
    
    # Create chart
    chart = BidAskRatioChart()
    window.setCentralWidget(chart)
    
    # Generate sample data
    sample_data = []
    base_time = datetime.now()
    
    for i in range(30):
        ratio = np.sin(i * 0.3) * 0.8 + np.random.normal(0, 0.1)
        ratio = max(-1, min(1, ratio))  # Clamp to [-1, 1]
        
        sample_data.append({
            'timestamp': (base_time.replace(second=0, microsecond=0)).isoformat(),
            'buy_sell_ratio': ratio,
            'volume': np.random.randint(50000, 200000),
            'positive_volume': np.random.randint(20000, 100000) if ratio > 0 else 0,
            'negative_volume': np.random.randint(20000, 100000) if ratio < 0 else 0
        })
        
        base_time = base_time.replace(minute=base_time.minute + 1)
    
    # Update chart
    chart.update_data(sample_data)
    
    # Add a marker
    chart.add_marker(15, "Entry", "#ff0000")
    
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    test_chart()