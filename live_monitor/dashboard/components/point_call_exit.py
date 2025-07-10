"""
Point & Call Exit System Module
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QLabel)
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt
from ...styles import PointCallExitStyles, BaseStyles


class PointCallExit(QWidget):
    """Widget for displaying point & call exit signals"""
    
    # Signal emitted when an exit is selected
    exit_selected = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.apply_styles()
        
    def init_ui(self):
        """Initialize the UI"""
        self.setObjectName("point_call_exit_container")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header = QLabel("Point & Call Exit System")
        header.setObjectName("point_call_exit_header")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # Table
        self.table = QTableWidget()
        self.table.setObjectName("exit_signals_table")
        
        # Set up columns
        columns = ["Time", "Type", "Price", "P&L", "Signal", "Urgency"]
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        
        # Configure table
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        
        # Set column widths
        self.table.setColumnWidth(0, 80)   # Time
        self.table.setColumnWidth(1, 60)   # Type
        self.table.setColumnWidth(2, 80)   # Price
        self.table.setColumnWidth(3, 80)   # P&L
        self.table.setColumnWidth(4, 100)  # Signal
        
        # Connect selection signal
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        
        layout.addWidget(self.table)
        
        # Add some placeholder data
        self._add_placeholder_data()
        
    def apply_styles(self):
        """Apply styles to the widget"""
        self.setStyleSheet(PointCallExitStyles.get_stylesheet())
        
    def _add_placeholder_data(self):
        """Add placeholder data for display"""
        placeholder_exits = [
            ("10:30:15", "TARGET", "153.50", "+2.50%", "Target 1 Reached", "Normal"),
            ("10:45:22", "STOP", "149.75", "-0.75%", "Support Break", "Urgent"),
            ("11:15:33", "TRAIL", "152.25", "+1.25%", "Trailing Stop", "Warning"),
        ]
        
        for exit_data in placeholder_exits:
            self.add_exit_signal(*exit_data)
            
    def add_exit_signal(self, time, exit_type, price, pnl, signal, urgency):
        """Add a new exit signal to the table"""
        row = self.table.rowCount()
        self.table.insertRow(row)
        
        # Time
        time_item = QTableWidgetItem(time)
        time_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 0, time_item)
        
        # Type
        type_item = QTableWidgetItem(exit_type)
        type_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 1, type_item)
        
        # Price
        price_item = QTableWidgetItem(f"${price}")
        price_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.table.setItem(row, 2, price_item)
        
        # P&L (with styling)
        pnl_item = QTableWidgetItem(pnl)
        pnl_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if pnl.startswith('+'):
            pnl_item.setData(Qt.ItemDataRole.UserRole, "pnl_positive")
        elif pnl.startswith('-'):
            pnl_item.setData(Qt.ItemDataRole.UserRole, "pnl_negative")
        self.table.setItem(row, 3, pnl_item)
        
        # Signal
        signal_item = QTableWidgetItem(signal)
        self.table.setItem(row, 4, signal_item)
        
        # Urgency (with styling)
        urgency_item = QTableWidgetItem(urgency)
        urgency_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        urgency_lower = urgency.lower()
        if urgency_lower == "urgent":
            urgency_item.setData(Qt.ItemDataRole.UserRole, "exit_urgent")
        elif urgency_lower == "warning":
            urgency_item.setData(Qt.ItemDataRole.UserRole, "exit_warning")
        else:
            urgency_item.setData(Qt.ItemDataRole.UserRole, "exit_normal")
        self.table.setItem(row, 5, urgency_item)
        
    def _on_selection_changed(self):
        """Handle selection changes"""
        selected_items = self.table.selectedItems()
        if selected_items:
            row = selected_items[0].row()
            exit_data = {
                'time': self.table.item(row, 0).text(),
                'type': self.table.item(row, 1).text(),
                'price': self.table.item(row, 2).text().replace('$', ''),
                'pnl': self.table.item(row, 3).text(),
                'signal': self.table.item(row, 4).text(),
                'urgency': self.table.item(row, 5).text(),
            }
            self.exit_selected.emit(exit_data)
            
    @pyqtSlot()
    def clear_signals(self):
        """Clear all signals from the table"""
        self.table.setRowCount(0)
        
    def update_data(self, data):
        """Update table with new data"""
        # Placeholder for future implementation
        pass