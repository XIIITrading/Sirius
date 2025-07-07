"""
Impact Success Chart - Large Order Pressure Visualization
Shows net buy/sell pressure from large orders over time
"""

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Configure pyqtgraph for better appearance
pg.setConfigOptions(antialias=True)
pg.setConfigOption('background', '#1e1e1e')
pg.setConfigOption('foreground', '#ffffff')


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
        
        # Data storage
        self.minutes_from_entry = deque(maxlen=window_minutes * 60)  # Renamed for clarity
        self.timestamps = deque(maxlen=window_minutes * 60)
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
        
        # Track timestamps
        self.start_timestamp = None
        self.entry_timestamp = None
        
    def setup_chart(self):
        """Configure chart appearance and settings"""
        # Set background to match Buy/Sell Ratio chart
        self.setBackground('#1e1e1e')
        self.getViewBox().setBackgroundColor('#2b2b2b')
        
        # Set title and labels
        title = "Large Order Cumulative Pressure" if self.show_cumulative else "Large Order Net Pressure"
        self.setTitle(title, color='#ffffff', size='12pt')
        self.setLabel('left', 'Volume Delta', color='#ffffff', size='10pt')
        self.setLabel('bottom', 'Minutes from Entry', color='#ffffff', size='10pt')
        
        # Set axis colors
        self.getAxis('left').setPen(pg.mkPen('#ffffff'))
        self.getAxis('bottom').setPen(pg.mkPen('#ffffff'))
        self.getAxis('left').setTextPen(pg.mkPen('#ffffff'))
        self.getAxis('bottom').setTextPen(pg.mkPen('#ffffff'))
        
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
        # Zero reference line
        self.zero_line = pg.InfiniteLine(
            pos=0,
            angle=0,
            pen=pg.mkPen('#ffffff', width=1, style=QtCore.Qt.PenStyle.DashLine),
            movable=False
        )
        self.addItem(self.zero_line)
        
        # Add reference lines
        reference_levels = [-5000, -2500, 2500, 5000]
        for level in reference_levels:
            pen = pg.mkPen(color='#666666', width=1, style=QtCore.Qt.PenStyle.DotLine)
            line = pg.InfiniteLine(
                pos=level,
                angle=0,
                pen=pen,
                movable=False
            )
            self.addItem(line)
        
        # Main pressure line
        self.pressure_line = self.plot(
            pen=pg.mkPen(color='#00ff00', width=2),
            name='Cumulative Pressure' if self.show_cumulative else 'Net Pressure'
        )
        
        # Zero baseline curve (for fill)
        self.zero_curve = self.plot(pen=None)
        
        # Fill item placeholder
        self.fill_item = None
        
        # Text items for current values
        self.current_text = pg.TextItem(
            text='',
            color='#ffffff',
            anchor=(0, 1),
            fill=(0, 0, 0, 180)
        )
        self.addItem(self.current_text)
        
        self.volume_text = pg.TextItem(
            text='',
            color='#ffffff',
            anchor=(0, 0),
            fill=(0, 0, 0, 180)
        )
        self.addItem(self.volume_text)
        
        # Time reference text
        self.time_text = pg.TextItem(
            text='',
            color='#ffffff',
            anchor=(1, 1),
            fill=(0, 0, 0, 180)
        )
        self.addItem(self.time_text)
        
        # Scatter points for large orders
        self.buy_scatter = pg.ScatterPlotItem(
            size=8,
            pen=None,
            brush=QtGui.QColor(13, 115, 119, 120),  # Dashboard teal
            symbol='o'
        )
        self.addItem(self.buy_scatter)
        
        self.sell_scatter = pg.ScatterPlotItem(
            size=8,
            pen=None,
            brush=QtGui.QColor(255, 68, 68, 120),  # Dashboard red
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
        self.minutes_from_entry.clear()
        self.timestamps.clear()
        self.net_pressures.clear()
        self.cumulative_pressures.clear()
        self.buy_volumes.clear()
        self.sell_volumes.clear()
        
        if not chart_data:
            return
        
        # Sort data by timestamp to ensure correct ordering
        chart_data = sorted(chart_data, key=lambda x: datetime.fromisoformat(x['timestamp']))
        
        # Get the start and end timestamps
        self.start_timestamp = datetime.fromisoformat(chart_data[0]['timestamp'])
        self.entry_timestamp = datetime.fromisoformat(chart_data[-1]['timestamp'])
        
        # Debug: Print time range
        time_span_minutes = (self.entry_timestamp - self.start_timestamp).total_seconds() / 60.0
        logger.info(f"Data spans {time_span_minutes:.2f} minutes from {self.start_timestamp} to {self.entry_timestamp}")
        
        # If data doesn't span enough time, we need to spread it out
        if time_span_minutes < 1:  # Less than 1 minute of data
            logger.warning("Data spans less than 1 minute - spreading points evenly across 30 minutes")
            # Spread points evenly across 30 minutes
            for i, point in enumerate(chart_data):
                # Distribute points evenly from -30 to 0
                minutes_from_entry = -30 + (i * 30 / len(chart_data))
                
                # Store data
                self.minutes_from_entry.append(minutes_from_entry)
                self.timestamps.append(datetime.fromisoformat(point['timestamp']))
                self.net_pressures.append(point['net_pressure'])
                self.cumulative_pressures.append(point['cumulative_pressure'])
                self.buy_volumes.append(point.get('buy_volume', 0))
                self.sell_volumes.append(point.get('sell_volume', 0))
        else:
            # Normal processing - data spans reasonable time
            for point in chart_data:
                # Convert timestamp
                if isinstance(point['timestamp'], str):
                    timestamp = datetime.fromisoformat(point['timestamp'])
                else:
                    timestamp = point['timestamp']
                
                # Calculate minutes from entry (negative for past, 0 for entry)
                minutes_from_entry = (timestamp - self.entry_timestamp).total_seconds() / 60.0
                
                # Debug first few points
                if len(self.minutes_from_entry) < 3:
                    logger.debug(f"Point {len(self.minutes_from_entry)}: timestamp={timestamp}, minutes_from_entry={minutes_from_entry:.2f}")
                
                # Store data
                self.minutes_from_entry.append(minutes_from_entry)
                self.timestamps.append(timestamp)
                self.net_pressures.append(point['net_pressure'])
                self.cumulative_pressures.append(point['cumulative_pressure'])
                self.buy_volumes.append(point.get('buy_volume', 0))
                self.sell_volumes.append(point.get('sell_volume', 0))
        
        # Debug: Print x-axis range
        if self.minutes_from_entry:
            logger.info(f"X-axis range: {min(self.minutes_from_entry):.2f} to {max(self.minutes_from_entry):.2f} minutes")
        
        # Update the display
        self.update_display()
        
        # Set proper x-axis range
        self.setXRange(-30, 0, padding=0.02)
        
        # Update x-axis labels
        self._update_x_axis_labels()

    def update_display(self):
        """Update the chart display"""
        if not self.minutes_from_entry:
            return
        
        # Convert to numpy arrays
        x_data = np.array(self.minutes_from_entry)
        
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
            
            # Determine coloring based on display mode
            if self.show_cumulative:
                # For cumulative, use the latest cumulative value
                latest_value = y_data[-1]
                if latest_value > 0:
                    self.pressure_line.setPen(pg.mkPen(color='#0d7377', width=2))
                    fill_color = QtGui.QColor(13, 115, 119, 30)
                elif latest_value < 0:
                    self.pressure_line.setPen(pg.mkPen(color='#ff4444', width=2))
                    fill_color = QtGui.QColor(255, 68, 68, 30)
                else:
                    self.pressure_line.setPen(pg.mkPen(color='#ffffff', width=2))
                    fill_color = QtGui.QColor(255, 255, 255, 30)
            else:
                # For net pressure, use overall balance
                positive_count = np.sum(y_data > 0)
                negative_count = np.sum(y_data < 0)
                
                if positive_count > negative_count:
                    self.pressure_line.setPen(pg.mkPen(color='#0d7377', width=2))
                    fill_color = QtGui.QColor(13, 115, 119, 30)
                elif negative_count > positive_count:
                    self.pressure_line.setPen(pg.mkPen(color='#ff4444', width=2))
                    fill_color = QtGui.QColor(255, 68, 68, 30)
                else:
                    self.pressure_line.setPen(pg.mkPen(color='#ffffff', width=2))
                    fill_color = QtGui.QColor(255, 255, 255, 30)
            
            # Create fill between curve and zero
            self.fill_item = pg.FillBetweenItem(self.pressure_line, self.zero_curve, brush=fill_color)
            self.addItem(self.fill_item)
        
        # Update scatter points for individual large orders
        buy_points_x = []
        buy_points_y = []
        sell_points_x = []
        sell_points_y = []
        
        for i, (buy_vol, sell_vol) in enumerate(zip(self.buy_volumes, self.sell_volumes)):
            if buy_vol > 0:
                buy_points_x.append(x_data[i])
                buy_points_y.append(y_data[i])
            if sell_vol > 0:
                sell_points_x.append(x_data[i])
                sell_points_y.append(y_data[i])
        
        # Update scatter plots
        if buy_points_x:
            self.buy_scatter.setData(buy_points_x, buy_points_y)
        else:
            self.buy_scatter.clear()
        
        if sell_points_x:
            self.sell_scatter.setData(sell_points_x, sell_points_y)
        else:
            self.sell_scatter.clear()
        
        # Update text displays
        if len(x_data) > 0:
            latest_net = self.net_pressures[-1]
            latest_cumulative = self.cumulative_pressures[-1]
            
            # Calculate recent volumes (last 5 points or all if less)
            recent_buy = sum(list(self.buy_volumes)[-5:])
            recent_sell = sum(list(self.sell_volumes)[-5:])
            
            # Position text items
            view_range = self.viewRange()
            if view_range:
                x_min, x_max = view_range[0]
                y_min, y_max = view_range[1]
                
                # Current pressure text
                self.current_text.setPos(x_min + (x_max - x_min) * 0.02, y_max - (y_max - y_min) * 0.05)
                if self.show_cumulative:
                    self.current_text.setText(f'Cumulative: {latest_cumulative:+,} | Latest: {latest_net:+,}')
                else:
                    self.current_text.setText(f'Net Pressure: {latest_net:+,}')
                
                # Volume breakdown text
                self.volume_text.setPos(x_min + (x_max - x_min) * 0.02, y_min + (y_max - y_min) * 0.05)
                self.volume_text.setText(f'Recent Buy: {recent_buy:,} | Recent Sell: {recent_sell:,}')
                
                # Time reference text
                self.time_text.setPos(x_max - (x_max - x_min) * 0.02, y_max - (y_max - y_min) * 0.05)
                if self.start_timestamp and self.entry_timestamp:
                    start_time_str = self.start_timestamp.strftime('%H:%M')
                    end_time_str = self.entry_timestamp.strftime('%H:%M')
                    self.time_text.setText(f'{start_time_str} - {end_time_str}')
        
        # Auto-scale Y axis with padding
        if len(y_data) > 1:
            y_min = min(0, np.min(y_data))
            y_max = max(0, np.max(y_data))
            y_range = y_max - y_min
            
            if y_range > 0:
                padding = y_range * 0.1
            else:
                padding = 100
            
            self.setYRange(y_min - padding, y_max + padding)
    
    def _update_x_axis_labels(self):
        """Update x-axis to show minutes from entry"""
        axis = self.getAxis('bottom')
        
        # Set ticks at 5-minute intervals
        ticks = []
        for minutes in [-30, -25, -20, -15, -10, -5, 0]:
            label = f'{minutes}m' if minutes != 0 else 'Entry'
            ticks.append((minutes, label))
        
        axis.setTicks([ticks])
    
    def add_marker(self, minute: float, label: str, color: str = '#ff0000'):
        """
        Add a vertical marker at specific time.
        
        Args:
            minute: Minute from entry (0 = entry, negative = past)
            label: Label for marker
            color: Marker color (default red)
        """
        line = pg.InfiniteLine(
            pos=minute,
            angle=90,
            pen=pg.mkPen(color=color, width=2, style=QtCore.Qt.PenStyle.DashLine),
            movable=False
        )
        
        # Find y-position for label
        y_range = self.viewRange()[1] if self.viewRange() else [0, 100]
        y_pos = y_range[1] - (y_range[1] - y_range[0]) * 0.1
        
        # Add label
        text = pg.TextItem(label, color=color, anchor=(0.5, 1))
        text.setPos(minute, y_pos)
        
        self.addItem(line)
        self.addItem(text)
    
    def toggle_cumulative(self):
        """Toggle between cumulative and period pressure views"""
        self.show_cumulative = not self.show_cumulative
        title = "Large Order Cumulative Pressure" if self.show_cumulative else "Large Order Net Pressure"
        self.setTitle(title, color='#ffffff', size='12pt')
        self.pressure_line.setName('Cumulative Pressure' if self.show_cumulative else 'Net Pressure')
        self.update_display()
    
    def clear_data(self):
        """Clear all data from the chart"""
        self.minutes_from_entry.clear()
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
    
    # Add entry marker at 0 (entry point)
    chart.add_marker(0, "Entry", "#ff0000")
    
    # Start event loop
    sys.exit(app.exec())