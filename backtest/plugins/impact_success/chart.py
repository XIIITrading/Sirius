"""
Impact Success Chart - Large Order Pressure Visualization
Shows net buy/sell pressure from large orders over time
"""

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ImpactSuccessChart(pg.PlotWidget):
    """
    Chart showing net large order pressure (buy volume - sell volume).
    Single oscillating line that shows market imbalance.
    """
    
    def __init__(self, window_minutes=30, update_interval_ms=1000, show_cumulative=True):
        """
        Initialize Impact Success Chart.
        
        Args:
            window_minutes: Minutes of data to display (default 30)
            update_interval_ms: Update interval in milliseconds
            show_cumulative: Show cumulative pressure vs period pressure
        """
        # Initialize parent WITHOUT DateAxisItem first
        super().__init__()
        
        self.window_minutes = window_minutes
        self.update_interval_ms = update_interval_ms
        self.show_cumulative = show_cumulative
        
        # Data storage - store as minutes from start
        self.minutes_from_start = deque(maxlen=window_minutes * 60)
        self.timestamps = deque(maxlen=window_minutes * 60)  # Keep for reference
        self.net_pressures = deque(maxlen=window_minutes * 60)
        self.cumulative_pressures = deque(maxlen=window_minutes * 60)
        self.buy_volumes = deque(maxlen=window_minutes * 60)
        self.sell_volumes = deque(maxlen=window_minutes * 60)
        
        # Setup the chart
        self.setup_chart()
        
        # Create plot items
        self.create_plot_items()
        
        # Timer for updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(update_interval_ms)
        
        # Track last update time
        self.last_update = datetime.now()
        self.start_timestamp = None
        
    def setup_chart(self):
        """Configure chart appearance and settings"""
        # Set title and labels
        title = "Large Order Cumulative Pressure" if self.show_cumulative else "Large Order Net Pressure"
        self.setTitle(title, color='w', size='12pt')
        self.setLabel('left', 'Volume Delta', color='w', size='10pt')
        self.setLabel('bottom', 'Minutes from Entry', color='w', size='10pt')
        
        # Set background
        self.setBackground('k')
        
        # Enable anti-aliasing
        self.setAntialiasing(True)
        
        # Configure grid
        self.showGrid(x=True, y=True, alpha=0.3)
        
        # Add legend
        self.addLegend(offset=(10, 10))
        
        # Enable auto-range
        self.enableAutoRange('y')
        
    def create_plot_items(self):
        """Create all plot items"""
        # Zero reference line (visual only)
        self.zero_line = pg.InfiniteLine(
            pos=0,
            angle=0,
            pen=pg.mkPen('w', width=2, style=QtCore.Qt.PenStyle.DashLine),
            movable=False
        )
        self.addItem(self.zero_line)
        
        # Main pressure line
        self.pressure_line = self.plot(
            pen=pg.mkPen(color=(255, 255, 255), width=2),
            name='Net Pressure' if not self.show_cumulative else 'Cumulative Pressure'
        )
        
        # Zero baseline curve (for fill)
        self.zero_curve = self.plot(pen=None)  # Invisible line at y=0
        
        # Create fill manually using polygon
        self.fill_item = None  # Will create during update
        
        # Text items for current values
        self.current_text = pg.TextItem(
            text='',
            color=(255, 255, 255),
            anchor=(0, 1),
            fill=(0, 0, 0, 180)
        )
        self.addItem(self.current_text)
        
        self.volume_text = pg.TextItem(
            text='',
            color=(255, 255, 255),
            anchor=(0, 0),
            fill=(0, 0, 0, 180)
        )
        self.addItem(self.volume_text)
        
        # Time reference text
        self.time_text = pg.TextItem(
            text='',
            color=(255, 255, 255),
            anchor=(1, 1),
            fill=(0, 0, 0, 180)
        )
        self.addItem(self.time_text)
        
        # Scatter points for large orders
        self.buy_scatter = pg.ScatterPlotItem(
            size=8,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(0, 255, 0, 120),
            symbol='o'
        )
        self.addItem(self.buy_scatter)
        
        self.sell_scatter = pg.ScatterPlotItem(
            size=8,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(255, 0, 0, 120),
            symbol='o'
        )
        self.addItem(self.sell_scatter)
        
    def update_from_data(self, chart_data: List[Dict[str, Any]]):
        """
        Update chart from pressure data.
        
        Args:
            chart_data: List of dictionaries with:
                - timestamp: ISO format timestamp
                - net_pressure: Buy volume - Sell volume for period
                - cumulative_pressure: Running total
                - buy_volume: Buy volume in period
                - sell_volume: Sell volume in period
        """
        # Clear existing data
        self.minutes_from_start.clear()
        self.timestamps.clear()
        self.net_pressures.clear()
        self.cumulative_pressures.clear()
        self.buy_volumes.clear()
        self.sell_volumes.clear()
        
        if not chart_data:
            return
        
        # Get the start timestamp
        first_timestamp = datetime.fromisoformat(chart_data[0]['timestamp'])
        self.start_timestamp = first_timestamp
        
        # Process new data
        for point in chart_data:
            # Convert timestamp
            if isinstance(point['timestamp'], str):
                timestamp = datetime.fromisoformat(point['timestamp'])
            else:
                timestamp = point['timestamp']
            
            # Calculate minutes from start
            minutes_elapsed = (timestamp - first_timestamp).total_seconds() / 60.0
            
            # Store data
            self.minutes_from_start.append(minutes_elapsed)
            self.timestamps.append(timestamp)
            self.net_pressures.append(point['net_pressure'])
            self.cumulative_pressures.append(point['cumulative_pressure'])
            self.buy_volumes.append(point.get('buy_volume', 0))
            self.sell_volumes.append(point.get('sell_volume', 0))
        
        # Update the display
        self.update_display()
        
        # Set proper x-axis range
        if len(self.minutes_from_start) > 0:
            self.setXRange(-30, 0, padding=0.02)
    
    def update_display(self):
        """Update the chart display"""
        if not self.minutes_from_start:
            return
        
        # Convert to numpy arrays
        # Show as negative minutes (counting back from entry)
        x_data = np.array(self.minutes_from_start) - 30
        
        # Choose which data to display
        if self.show_cumulative:
            y_data = np.array(self.cumulative_pressures)
        else:
            y_data = np.array(self.net_pressures)
        
        # Update main line
        self.pressure_line.setData(x_data, y_data)
        
        # Update zero baseline for fill
        zero_y = np.zeros_like(x_data)
        self.zero_curve.setData(x_data, zero_y)
        
        # Color the line and create fill based on data
        if len(y_data) > 0:
            # Remove old fill if exists
            if self.fill_item is not None:
                self.removeItem(self.fill_item)
                self.fill_item = None
            
            # Create new fill with appropriate color
            latest_value = y_data[-1]
            
            # Determine dominant pressure for coloring
            positive_sum = np.sum(y_data[y_data > 0])
            negative_sum = np.abs(np.sum(y_data[y_data < 0]))
            
            if positive_sum > negative_sum:
                # Overall bullish
                self.pressure_line.setPen(pg.mkPen(color=(0, 255, 0), width=2))
                fill_color = (0, 255, 0, 30)
            elif negative_sum > positive_sum:
                # Overall bearish
                self.pressure_line.setPen(pg.mkPen(color=(255, 0, 0), width=2))
                fill_color = (255, 0, 0, 30)
            else:
                # Neutral
                self.pressure_line.setPen(pg.mkPen(color=(255, 255, 255), width=2))
                fill_color = (255, 255, 255, 30)
            
            # Create fill between curve and zero
            self.fill_item = pg.FillBetweenItem(self.pressure_line, self.zero_curve, brush=fill_color)
            self.addItem(self.fill_item)
        
        # Update scatter points for individual large orders
        # Find points with buy volume
        buy_indices = [i for i, vol in enumerate(self.buy_volumes) if vol > 0]
        if buy_indices:
            buy_x = x_data[buy_indices]
            buy_y = y_data[buy_indices]
            self.buy_scatter.setData(buy_x, buy_y)
        else:
            self.buy_scatter.clear()
        
        # Find points with sell volume
        sell_indices = [i for i, vol in enumerate(self.sell_volumes) if vol > 0]
        if sell_indices:
            sell_x = x_data[sell_indices]
            sell_y = y_data[sell_indices]
            self.sell_scatter.setData(sell_x, sell_y)
        else:
            self.sell_scatter.clear()
        
        # Update text displays with latest values
        if len(x_data) > 0:
            latest_net = self.net_pressures[-1]
            latest_cumulative = self.cumulative_pressures[-1]
            latest_buy = self.buy_volumes[-1]
            latest_sell = self.sell_volumes[-1]
            latest_timestamp = self.timestamps[-1]
            
            # Position text items
            view_range = self.viewRange()
            if view_range:
                x_min, x_max = view_range[0]
                y_min, y_max = view_range[1]
                
                # Current pressure text (top left)
                self.current_text.setPos(x_min + (x_max - x_min) * 0.02, y_max - (y_max - y_min) * 0.05)
                if self.show_cumulative:
                    self.current_text.setText(f'Cumulative: {latest_cumulative:+,} | Current: {latest_net:+,}')
                else:
                    self.current_text.setText(f'Net Pressure: {latest_net:+,}')
                
                # Volume breakdown text (bottom left)
                self.volume_text.setPos(x_min + (x_max - x_min) * 0.02, y_min + (y_max - y_min) * 0.05)
                if latest_buy > 0 or latest_sell > 0:
                    self.volume_text.setText(f'Buy: {latest_buy:,} | Sell: {latest_sell:,}')
                else:
                    self.volume_text.setText('')
                
                # Time reference text (top right)
                self.time_text.setPos(x_max - (x_max - x_min) * 0.02, y_max - (y_max - y_min) * 0.05)
                if self.start_timestamp:
                    start_time_str = self.start_timestamp.strftime('%H:%M')
                    end_time_str = latest_timestamp.strftime('%H:%M')
                    self.time_text.setText(f'{start_time_str} - {end_time_str}')
        
        # Auto-scale Y axis
        if len(y_data) > 1:
            # Add some padding to y-range
            y_min = min(0, y_data.min())  # Include 0 in range
            y_max = max(0, y_data.max())
            y_padding = max(abs(y_max), abs(y_min)) * 0.1
            self.setYRange(y_min - y_padding, y_max + y_padding)
            
        # Update x-axis labels
        self._update_x_axis_labels()
    
    def _update_x_axis_labels(self):
        """Update x-axis to show minutes from entry"""
        # Create custom axis labels
        axis = self.getAxis('bottom')
        
        # Set ticks at 5-minute intervals
        ticks = []
        for minutes in [-30, -25, -20, -15, -10, -5, 0]:
            ticks.append((minutes, f'{minutes}m' if minutes != 0 else 'Entry'))
        
        axis.setTicks([ticks])
    
    def add_marker(self, minute: float, label: str, color: str = '#ffffff'):
        """
        Add a vertical marker at specific time.
        
        Args:
            minute: Minute from start (0-30)
            label: Label for marker
            color: Marker color
        """
        # Convert to our coordinate system (negative minutes from entry)
        marker_x = minute - 30
        
        line = pg.InfiniteLine(
            pos=marker_x,
            angle=90,
            pen=pg.mkPen(color=color, width=2, style=QtCore.Qt.PenStyle.DashLine),
            movable=False
        )
        
        # Find y-position for label
        y_range = self.viewRange()[1] if self.viewRange() else [0, 100]
        y_pos = y_range[1] - (y_range[1] - y_range[0]) * 0.1
        
        # Add label
        text = pg.TextItem(label, color=color, anchor=(0.5, 1))
        text.setPos(marker_x, y_pos)
        
        self.addItem(line)
        self.addItem(text)
    
    def toggle_cumulative(self):
        """Toggle between cumulative and period pressure views"""
        self.show_cumulative = not self.show_cumulative
        title = "Large Order Cumulative Pressure" if self.show_cumulative else "Large Order Net Pressure"
        self.setTitle(title, color='w', size='12pt')
        self.pressure_line.setName('Cumulative Pressure' if self.show_cumulative else 'Net Pressure')
        self.update_display()
    
    def clear_data(self):
        """Clear all data from the chart"""
        self.minutes_from_start.clear()
        self.timestamps.clear()
        self.net_pressures.clear()
        self.cumulative_pressures.clear()
        self.buy_volumes.clear()
        self.sell_volumes.clear()
        
        # Clear plot items
        self.pressure_line.clear()
        self.zero_curve.clear()
        self.buy_scatter.clear()
        self.sell_scatter.clear()
        
        # Clear text
        self.current_text.setText('')
        self.volume_text.setText('')
        self.time_text.setText('')
        
        # Remove fill
        if self.fill_item is not None:
            self.removeItem(self.fill_item)
            self.fill_item = None
    
    def closeEvent(self, event):
        """Clean up when widget is closed"""
        if hasattr(self, 'timer'):
            self.timer.stop()
        super().closeEvent(event)


# Standalone test
if __name__ == '__main__':
    import sys
    from pyqtgraph.Qt import QtWidgets
    
    app = QtWidgets.QApplication(sys.argv)
    
    # Create chart
    chart = ImpactSuccessChart(window_minutes=30, show_cumulative=True)
    chart.show()
    
    # Generate test data
    test_data = []
    current_time = datetime.now()
    cumulative = 0
    
    for i in range(30):
        # Simulate large order imbalances
        buy_vol = np.random.randint(0, 5000) if np.random.random() > 0.5 else 0
        sell_vol = np.random.randint(0, 5000) if np.random.random() > 0.5 else 0
        net = buy_vol - sell_vol
        cumulative += net
        
        test_data.append({
            'timestamp': (current_time - timedelta(minutes=30-i)).isoformat(),
            'net_pressure': net,
            'cumulative_pressure': cumulative,
            'buy_volume': buy_vol,
            'sell_volume': sell_vol,
            'buy_count': 1 if buy_vol > 0 else 0,
            'sell_count': 1 if sell_vol > 0 else 0
        })
    
    # Update chart
    chart.update_from_data(test_data)
    
    # Add entry marker
    chart.add_marker(30, "Entry", "#ffff00")
    
    # Start event loop
    sys.exit(app.exec())