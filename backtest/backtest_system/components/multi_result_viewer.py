"""
Multi-Result Viewer - Displays multiple plugin results in a table
"""

from typing import Dict, Any, List
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor


class MultiResultViewer(QWidget):
    """Widget for displaying multiple plugin results in a table"""
    
    # Signal emitted when a row is selected
    result_selected = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.results = []  # Store full results
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        
        # Results table
        self.results_group = QGroupBox("Plugin Results Summary")
        results_layout = QVBoxLayout(self.results_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels([
            "Plugin", "Direction", "Strength", "Confidence", 
            "Signal Type", "Summary", "Status"
        ])
        
        # Set column widths
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Fixed)
        
        self.results_table.setColumnWidth(0, 200)  # Plugin
        self.results_table.setColumnWidth(1, 100)  # Direction
        self.results_table.setColumnWidth(2, 80)   # Strength
        self.results_table.setColumnWidth(3, 80)   # Confidence
        self.results_table.setColumnWidth(4, 120)  # Signal Type
        self.results_table.setColumnWidth(6, 80)   # Status
        
        # Enable row selection
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.results_table.itemSelectionChanged.connect(self._on_selection_changed)
        
        # Enable sorting
        self.results_table.setSortingEnabled(True)
        
        results_layout.addWidget(self.results_table)
        layout.addWidget(self.results_group)
        
    def clear_results(self):
        """Clear all results"""
        self.results.clear()
        self.results_table.setRowCount(0)
        
    def add_result(self, result: Dict[str, Any]):
        """Add a single result to the table"""
        self.results.append(result)
        
        # Add row to table
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        # Plugin name
        self.results_table.setItem(row, 0, QTableWidgetItem(result.get('plugin_name', 'Unknown')))
        
        # Check for error
        if 'error' in result:
            # Error case
            self.results_table.setItem(row, 1, QTableWidgetItem("ERROR"))
            self.results_table.setItem(row, 2, QTableWidgetItem("-"))
            self.results_table.setItem(row, 3, QTableWidgetItem("-"))
            self.results_table.setItem(row, 4, QTableWidgetItem("-"))
            self.results_table.setItem(row, 5, QTableWidgetItem(str(result['error'])))
            self.results_table.setItem(row, 6, QTableWidgetItem("Failed"))
            
            # Color the row red
            for col in range(7):
                item = self.results_table.item(row, col)
                if item:
                    item.setBackground(QColor(80, 20, 20))
        else:
            # Success case
            signal = result.get('signal', {})
            details = result.get('details', {})
            display_data = result.get('display_data', {})
            
            # Direction
            direction = signal.get('direction', 'NEUTRAL')
            direction_item = QTableWidgetItem(direction)
            
            # Color based on direction
            if direction == 'BULLISH':
                direction_item.setForeground(QColor(81, 207, 102))  # Green
            elif direction == 'BEARISH':
                direction_item.setForeground(QColor(255, 107, 107))  # Red
            else:
                direction_item.setForeground(QColor(134, 142, 150))  # Gray
                
            self.results_table.setItem(row, 1, direction_item)
            
            # Strength
            strength = signal.get('strength', 0)
            strength_item = QTableWidgetItem(f"{strength:.1f}%")
            self.results_table.setItem(row, 2, strength_item)
            
            # Confidence
            confidence = signal.get('confidence', 0)
            confidence_item = QTableWidgetItem(f"{confidence:.1f}%")
            self.results_table.setItem(row, 3, confidence_item)
            
            # Signal Type (from details)
            signal_type = details.get('structure_type', '-')
            self.results_table.setItem(row, 4, QTableWidgetItem(str(signal_type)))
            
            # Summary
            summary = display_data.get('summary', '-')
            self.results_table.setItem(row, 5, QTableWidgetItem(summary))
            
            # Status
            self.results_table.setItem(row, 6, QTableWidgetItem("Success"))
            
            # Highlight strong signals
            if strength >= 70:
                for col in range(7):
                    item = self.results_table.item(row, col)
                    if item:
                        item.setBackground(QColor(30, 50, 30))  # Subtle green
        
        # Auto-resize row height
        self.results_table.resizeRowToContents(row)
        
    def _on_selection_changed(self):
        """Handle row selection"""
        current_row = self.results_table.currentRow()
        if 0 <= current_row < len(self.results):
            self.result_selected.emit(self.results[current_row])