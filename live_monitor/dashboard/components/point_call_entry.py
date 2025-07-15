from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QTableWidget, QTableWidgetItem, QPushButton, QHeaderView)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor


class PointCallEntry(QWidget):
    """Point & Call Entry System Widget"""
    
    entry_selected = pyqtSignal(dict)  # Emits entry data when selected
    
    def __init__(self):
        super().__init__()
        self.signal_rows = {}  # Track which row each signal source is on
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Header
        header_layout = QHBoxLayout()
        header_label = QLabel("📊 Entry Signals")
        header_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #4CAF50;
                padding: 5px;
            }
        """)
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        
        # Clear button with inline style
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_signals)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #424242;
                color: white;
                border: 1px solid #555;
                padding: 5px 10px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #555;
                border-color: #777;
            }
            QPushButton:pressed {
                background-color: #333;
            }
        """)
        header_layout.addWidget(self.clear_btn)
        
        layout.addLayout(header_layout)
        
        # Entry signals table
        self.entry_table = QTableWidget()
        self.entry_table.setColumnCount(6)
        self.entry_table.setHorizontalHeaderLabels([
            "Time", "Type", "Price", "Signal", "Strength", "Notes"
        ])
        
        # Set column widths
        self.entry_table.setColumnWidth(0, 60)   # Time
        self.entry_table.setColumnWidth(1, 50)   # Type
        self.entry_table.setColumnWidth(2, 70)   # Price
        self.entry_table.setColumnWidth(3, 200)  # Signal
        self.entry_table.setColumnWidth(4, 60)   # Strength
        self.entry_table.setColumnWidth(5, 150)  # Notes
        
        # Style the table with inline CSS
        self.entry_table.setStyleSheet("""
            QTableWidget {
                background-color: #2a2a2a;
                border: 1px solid #444;
                gridline-color: #444;
                color: #ffffff;
            }
            QTableWidget::item {
                padding: 5px;
                border: none;
            }
            QTableWidget::item:selected {
                background-color: #3a3a3a;
            }
            QHeaderView::section {
                background-color: #333;
                color: #ffffff;
                padding: 5px;
                border: 1px solid #444;
                font-weight: bold;
            }
            QTableWidget::item:alternate {
                background-color: #252525;
            }
        """)
        
        # Set additional table properties
        self.entry_table.setAlternatingRowColors(True)
        self.entry_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.entry_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.entry_table.horizontalHeader().setStretchLastSection(True)
        self.entry_table.verticalHeader().setVisible(False)
        
        # Connect selection signal
        self.entry_table.itemSelectionChanged.connect(self._on_selection_changed)
        
        layout.addWidget(self.entry_table)
    
    def add_or_update_signal(self, signal_data: dict):
        """Add new signal or update existing one based on source"""
        # Extract source from signal description
        source = self._extract_source(signal_data.get('signal', ''))
        
        if source and source in self.signal_rows:
            # Update existing row
            row = self.signal_rows[source]
            self._update_row(row, signal_data)
        else:
            # Add new row
            row = self.entry_table.rowCount()
            self.entry_table.insertRow(row)
            self._update_row(row, signal_data)
            
            # Track this source
            if source:
                self.signal_rows[source] = row
    
    def _extract_source(self, signal_desc: str) -> str:
        """Extract source identifier from signal description"""
        # Map signal descriptions to sources
        if "M1 EMA" in signal_desc:
            return "M1_EMA"
        elif "M5 EMA" in signal_desc:
            return "M5_EMA"
        elif "M15 EMA" in signal_desc:
            return "M15_EMA"
        elif "M1 Trend" in signal_desc and "M15" not in signal_desc:
            return "M1_TREND"
        elif "M5 Trend:" in signal_desc:
            return "M5_TREND"
        elif "M15 Trend:" in signal_desc:
            return "M15_TREND"
        elif "M1 MS:" in signal_desc:
            return "M1_MSTRUCT"
        elif "M5 MS:" in signal_desc:
            return "M5_MSTRUCT"
        elif "M15 MS:" in signal_desc:
            return "M15_MSTRUCT"
        else:
            return signal_desc  # Use full description as fallback
    
    def _update_row(self, row: int, signal_data: dict):
        """Update a specific row with signal data"""
        # Time
        time_item = QTableWidgetItem(signal_data.get('time', ''))
        time_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.entry_table.setItem(row, 0, time_item)
        
        # Type (LONG/SHORT)
        signal_type = signal_data.get('signal_type', '')
        type_item = QTableWidgetItem(signal_type)
        type_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Color based on type
        if signal_type == 'LONG':
            type_item.setForeground(QColor('#4CAF50'))
        else:
            type_item.setForeground(QColor('#f44336'))
        
        self.entry_table.setItem(row, 1, type_item)
        
        # Price
        price_item = QTableWidgetItem(signal_data.get('price', ''))
        price_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.entry_table.setItem(row, 2, price_item)
        
        # Signal
        signal_item = QTableWidgetItem(signal_data.get('signal', ''))
        self.entry_table.setItem(row, 3, signal_item)
        
        # Strength
        strength = signal_data.get('strength', '')
        strength_item = QTableWidgetItem(strength)
        strength_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Color based on strength
        if strength == 'Strong':
            strength_item.setForeground(QColor('#4CAF50'))
        elif strength == 'Medium':
            strength_item.setForeground(QColor('#FF9800'))
        else:
            strength_item.setForeground(QColor('#9E9E9E'))
        
        self.entry_table.setItem(row, 4, strength_item)
        
        # Notes
        notes_item = QTableWidgetItem(signal_data.get('notes', ''))
        self.entry_table.setItem(row, 5, notes_item)
        
        # Store full data in first item
        time_item.setData(Qt.ItemDataRole.UserRole, signal_data)
    
    def clear_signals(self):
        """Clear all signals from the table"""
        self.entry_table.setRowCount(0)
        self.signal_rows.clear()
    
    def _on_selection_changed(self):
        """Handle row selection"""
        selected_items = self.entry_table.selectedItems()
        if selected_items:
            row = selected_items[0].row()
            data_item = self.entry_table.item(row, 0)
            if data_item:
                signal_data = data_item.data(Qt.ItemDataRole.UserRole)
                if signal_data:
                    self.entry_selected.emit(signal_data)
    
    # Add this method for backwards compatibility
    def add_entry_signal(self, time, signal_type, price, signal, strength, notes):
        """Legacy method - redirects to add_or_update_signal"""
        signal_data = {
            'time': time,
            'signal_type': signal_type,
            'price': price,
            'signal': signal,
            'strength': strength,
            'notes': notes
        }
        self.add_or_update_signal(signal_data)