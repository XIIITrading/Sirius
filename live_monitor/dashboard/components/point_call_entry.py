"""
Point & Call Entry System Module
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QLabel)
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt
from ...styles import PointCallEntryStyles, BaseStyles


class PointCallEntry(QWidget):
    """Widget for displaying point & call entry signals"""
    
    # Signal emitted when an entry is selected
    entry_selected = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.apply_styles()
        
    def init_ui(self):
        """Initialize the UI"""
        self.setObjectName("point_call_entry_container")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header = QLabel("Point & Call Entry System")
        header.setObjectName("point_call_entry_header")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # Table
        self.table = QTableWidget()
        self.table.setObjectName("entry_signals_table")
        
        # Set up columns
        columns = ["Time", "Type", "Price", "Signal", "Strength", "Notes"]
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        
        # Configure table
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        
        # Set column widths
        self.table.setColumnWidth(0, 80)  # Time
        self.table.setColumnWidth(1, 60)   # Type
        self.table.setColumnWidth(2, 80)   # Price
        self.table.setColumnWidth(3, 100)  # Signal
        self.table.setColumnWidth(4, 80)   # Strength
        
        # Connect selection signal
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        
        layout.addWidget(self.table)
        
        # Add some placeholder data
        self._add_placeholder_data()
        
    def apply_styles(self):
        """Apply styles to the widget"""
        self.setStyleSheet(PointCallEntryStyles.get_stylesheet())
        
    def _add_placeholder_data(self):
        """Add placeholder data for display"""
        placeholder_entries = [
            ("09:30:15", "LONG", "150.25", "HVN Break", "Strong", "Volume confirmation"),
            ("09:45:22", "LONG", "151.50", "Order Block", "Medium", "Retesting support"),
            ("10:15:33", "SHORT", "152.75", "Resistance", "Weak", "Camarilla R3"),
        ]
        
        for entry in placeholder_entries:
            self.add_entry_signal(*entry)
            
    def add_entry_signal(self, time, signal_type, price, signal, strength, notes):
        """Add or update an entry signal in the table"""
        # Extract source identifier from signal description
        # Examples: "M1 EMA LONG Signal" -> "M1_EMA"
        #          "M5 EMA Bullish Crossover" -> "M5_EMA"
        #          "Statistical Trend STRONG LONG" -> "STATISTICAL_TREND"
        source_id = None
        if "M1 EMA" in signal:
            source_id = "M1_EMA"
        elif "M5 EMA" in signal:
            source_id = "M5_EMA"
        elif "M15 EMA" in signal:
            source_id = "M15_EMA"
        elif "Statistical Trend" in signal:
            source_id = "STATISTICAL_TREND"
        
        # Check if we already have a row for this source
        existing_row = None
        if source_id:
            for row in range(self.table.rowCount()):
                # Check the notes column for source identifier
                notes_item = self.table.item(row, 5)
                if notes_item and source_id in notes_item.text():
                    existing_row = row
                    break
        
        # If exists, update it; otherwise add new row
        if existing_row is not None:
            row = existing_row
        else:
            row = self.table.rowCount()
            self.table.insertRow(row)
        
        # Time
        time_item = QTableWidgetItem(time)
        time_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 0, time_item)
        
        # Type (with styling)
        type_item = QTableWidgetItem(signal_type)
        type_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if signal_type == "LONG":
            type_item.setData(Qt.ItemDataRole.UserRole, "entry_type_long")
        else:
            type_item.setData(Qt.ItemDataRole.UserRole, "entry_type_short")
        self.table.setItem(row, 1, type_item)
        
        # Price
        price_item = QTableWidgetItem(f"${price}")
        price_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.table.setItem(row, 2, price_item)
        
        # Signal
        signal_item = QTableWidgetItem(signal)
        self.table.setItem(row, 3, signal_item)
        
        # Strength (with styling)
        strength_item = QTableWidgetItem(strength)
        strength_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        strength_lower = strength.lower()
        if strength_lower == "strong":
            strength_item.setData(Qt.ItemDataRole.UserRole, "signal_strong")
        elif strength_lower == "medium":
            strength_item.setData(Qt.ItemDataRole.UserRole, "signal_medium")
        else:
            strength_item.setData(Qt.ItemDataRole.UserRole, "signal_weak")
        self.table.setItem(row, 4, strength_item)
        
        # Notes - Add source identifier to notes for tracking
        if source_id and source_id not in notes:
            notes = f"{notes} | {source_id}"
        notes_item = QTableWidgetItem(notes)
        self.table.setItem(row, 5, notes_item)
        
    def _on_selection_changed(self):
        """Handle selection changes"""
        selected_items = self.table.selectedItems()
        if selected_items:
            row = selected_items[0].row()
            entry_data = {
                'time': self.table.item(row, 0).text(),
                'type': self.table.item(row, 1).text(),
                'price': self.table.item(row, 2).text().replace('$', ''),
                'signal': self.table.item(row, 3).text(),
                'strength': self.table.item(row, 4).text(),
                'notes': self.table.item(row, 5).text(),
            }
            self.entry_selected.emit(entry_data)
            
    @pyqtSlot()
    def clear_signals(self):
        """Clear all signals from the table"""
        self.table.setRowCount(0)
        
    def update_data(self, data):
        """Update table with new data"""
        # Placeholder for future implementation
        pass