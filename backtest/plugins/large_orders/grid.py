"""
Large Orders Grid Widget
Displays successful large order trades in a sortable, interactive grid
"""

import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QTableWidget, QTableWidgetItem, QVBoxLayout, 
    QHBoxLayout, QWidget, QLabel, QPushButton,
    QHeaderView
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class LargeOrdersGrid(QWidget):
    """
    Grid widget for displaying successful large order trades.
    Provides sorting, filtering, and highlighting capabilities.
    """
    
    # Signal emitted when a row is selected
    order_selected = pyqtSignal(dict)
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Large Orders Grid.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__()
        
        # Default configuration
        self.config = {
            'title': 'Large Orders Grid',
            'max_rows': 30,
            'highlight_threshold': 1.5,  # Highlight orders with impact > 1.5 spreads
            'columns': [
                {'name': 'Timestamp', 'field': 'timestamp', 'width': 150},
                {'name': 'Price', 'field': 'price', 'width': 100},
                {'name': 'Size', 'field': 'size', 'width': 100},
                {'name': 'Side', 'field': 'side', 'width': 80},
                {'name': 'Impact', 'field': 'impact_magnitude', 'width': 100},
                {'name': 'Size vs Avg', 'field': 'size_vs_avg', 'width': 100},
                {'name': 'Volume 1s', 'field': 'volume_1s', 'width': 100}
            ]
        }
        
        # Update config if provided
        if config:
            self.config.update(config)
        
        # Data storage
        self.orders_data = []
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Header with summary
        header_layout = QHBoxLayout()
        
        self.summary_label = QLabel("No data loaded")
        self.summary_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        header_layout.addWidget(self.summary_label)
        
        header_layout.addStretch()
        
        # Export button
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.export_data)
        self.export_button.setMaximumWidth(80)
        header_layout.addWidget(self.export_button)
        
        layout.addLayout(header_layout)
        
        # Create table
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setSortingEnabled(True)
        
        # Set column count and headers
        columns = self.config['columns']
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels([col['name'] for col in columns])
        
        # Set column widths
        for i, col in enumerate(columns):
            if 'width' in col:
                self.table.setColumnWidth(i, col['width'])
        
        # Configure header
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSortIndicatorShown(True)
        
        # Connect selection signal
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        
        layout.addWidget(self.table)
        
        # Statistics panel
        self.stats_layout = QHBoxLayout()
        
        self.buy_count_label = QLabel("Buys: 0")
        self.sell_count_label = QLabel("Sells: 0")
        self.avg_impact_label = QLabel("Avg Impact: 0.00")
        self.total_volume_label = QLabel("Total Volume: 0")
        
        for label in [self.buy_count_label, self.sell_count_label, 
                     self.avg_impact_label, self.total_volume_label]:
            label.setStyleSheet("padding: 5px; background-color: #2b2b2b; border-radius: 3px;")
            self.stats_layout.addWidget(label)
        
        self.stats_layout.addStretch()
        layout.addLayout(self.stats_layout)
        
    def update_from_data(self, orders_data: List[Dict[str, Any]]):
        """
        Update grid with new order data.
        
        Args:
            orders_data: List of order dictionaries
        """
        self.orders_data = orders_data[:self.config['max_rows']]  # Limit rows
        
        # Clear existing data
        self.table.setRowCount(0)
        
        if not self.orders_data:
            self.summary_label.setText("No successful large orders found")
            self._update_statistics()
            return
        
        # Update summary
        self.summary_label.setText(f"Showing {len(self.orders_data)} successful large orders")
        
        # Populate table
        self.table.setRowCount(len(self.orders_data))
        
        for row, order in enumerate(self.orders_data):
            # Get column configuration
            for col, col_config in enumerate(self.config['columns']):
                field = col_config['field']
                value = order.get(field, '')
                
                # Format based on field type
                if field == 'timestamp':
                    if isinstance(value, datetime):
                        display_value = value.strftime('%H:%M:%S.%f')[:-3]
                    else:
                        display_value = str(value)
                elif field == 'price':
                    display_value = f"${value:.2f}" if value else ""
                elif field == 'size':
                    display_value = f"{value:,}" if value else ""
                elif field == 'impact_magnitude':
                    display_value = f"{value:.2f}" if value else "0.00"
                elif field == 'size_vs_avg':
                    display_value = f"{value:.1f}x" if value else ""
                elif field == 'volume_1s':
                    display_value = f"{value:,}" if value else ""
                else:
                    display_value = str(value)
                
                item = QTableWidgetItem(display_value)
                
                # Store original value for sorting
                item.setData(Qt.ItemDataRole.UserRole, value)
                
                # Color coding
                if field == 'side':
                    if value == 'BUY':
                        item.setForeground(QColor(0, 255, 0))
                    elif value == 'SELL':
                        item.setForeground(QColor(255, 0, 0))
                
                elif field == 'impact_magnitude':
                    # Highlight high impact orders
                    if abs(value) >= self.config['highlight_threshold']:
                        item.setBackground(QColor(60, 60, 20))
                        item.setForeground(QColor(255, 255, 0))
                        font = QFont()
                        font.setBold(True)
                        item.setFont(font)
                
                # Center align numeric columns
                if field in ['price', 'size', 'impact_magnitude', 'size_vs_avg', 'volume_1s']:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                
                self.table.setItem(row, col, item)
        
        # Update statistics
        self._update_statistics()
        
        # Auto-resize rows
        self.table.resizeRowsToContents()
        
    def _update_statistics(self):
        """Update statistics labels"""
        if not self.orders_data:
            self.buy_count_label.setText("Buys: 0")
            self.sell_count_label.setText("Sells: 0")
            self.avg_impact_label.setText("Avg Impact: 0.00")
            self.total_volume_label.setText("Total Volume: 0")
            return
        
        # Calculate statistics
        buy_orders = [o for o in self.orders_data if o.get('side') == 'BUY']
        sell_orders = [o for o in self.orders_data if o.get('side') == 'SELL']
        
        buy_count = len(buy_orders)
        sell_count = len(sell_orders)
        
        impacts = [abs(o.get('impact_magnitude', 0)) for o in self.orders_data]
        avg_impact = np.mean(impacts) if impacts else 0
        
        total_volume = sum(o.get('size', 0) for o in self.orders_data)
        
        # Update labels
        self.buy_count_label.setText(f"Buys: {buy_count}")
        self.sell_count_label.setText(f"Sells: {sell_count}")
        self.avg_impact_label.setText(f"Avg Impact: {avg_impact:.2f}")
        self.total_volume_label.setText(f"Total Volume: {total_volume:,}")
        
        # Color code buy/sell labels
        if buy_count > sell_count:
            self.buy_count_label.setStyleSheet(
                "padding: 5px; background-color: #1a4d2e; border-radius: 3px; color: #4ade80;"
            )
            self.sell_count_label.setStyleSheet(
                "padding: 5px; background-color: #2b2b2b; border-radius: 3px;"
            )
        elif sell_count > buy_count:
            self.sell_count_label.setStyleSheet(
                "padding: 5px; background-color: #4d1a1a; border-radius: 3px; color: #f87171;"
            )
            self.buy_count_label.setStyleSheet(
                "padding: 5px; background-color: #2b2b2b; border-radius: 3px;"
            )
        else:
            for label in [self.buy_count_label, self.sell_count_label]:
                label.setStyleSheet("padding: 5px; background-color: #2b2b2b; border-radius: 3px;")
    
    def _on_selection_changed(self):
        """Handle row selection"""
        current_row = self.table.currentRow()
        if 0 <= current_row < len(self.orders_data):
            self.order_selected.emit(self.orders_data[current_row])
    
    def export_data(self):
        """Export grid data to CSV"""
        try:
            from PyQt6.QtWidgets import QFileDialog
            import csv
            
            filename, _ = QFileDialog.getSaveFileName(
                self, 
                "Export Large Orders", 
                f"large_orders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "CSV Files (*.csv)"
            )
            
            if filename:
                with open(filename, 'w', newline='') as csvfile:
                    # Get column names
                    fieldnames = [col['field'] for col in self.config['columns']]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                    
                    # Write header
                    writer.writerow({field: col['name'] for field, col in 
                                   zip(fieldnames, self.config['columns'])})
                    
                    # Write data
                    for order in self.orders_data:
                        # Format timestamp for export
                        if 'timestamp' in order and isinstance(order['timestamp'], datetime):
                            order = order.copy()
                            order['timestamp'] = order['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        writer.writerow(order)
                
                logger.info(f"Exported {len(self.orders_data)} orders to {filename}")
                
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
    
    def clear_data(self):
        """Clear all data from the grid"""
        self.orders_data.clear()
        self.table.setRowCount(0)
        self.summary_label.setText("No data loaded")
        self._update_statistics()


# Standalone test
if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Create grid with test configuration
    config = {
        'title': 'Test Large Orders Grid',
        'highlight_threshold': 1.0
    }
    
    grid = LargeOrdersGrid(config)
    grid.show()
    
    # Generate test data
    test_data = []
    current_time = datetime.now()
    
    for i in range(20):
        side = 'BUY' if np.random.random() > 0.5 else 'SELL'
        impact = np.random.uniform(-2.5, 2.5)
        
        test_data.append({
            'timestamp': current_time - timedelta(minutes=30-i),
            'price': 150.0 + np.random.uniform(-1, 1),
            'size': np.random.randint(1000, 10000),
            'side': side,
            'impact_magnitude': impact if side == 'BUY' else -impact,
            'size_vs_avg': np.random.uniform(1.5, 3.0),
            'volume_1s': np.random.randint(500, 5000),
            'trade_count_1s': np.random.randint(5, 20)
        })
    
    # Update grid
    grid.update_from_data(test_data)
    
    # Connect to see selections
    grid.order_selected.connect(lambda order: print(f"Selected: {order}"))
    
    # Start event loop
    sys.exit(app.exec())