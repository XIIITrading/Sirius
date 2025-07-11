# live_monitor/dashboard/components/hvn_table.py
"""
HVN Ranges Table Component
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QLabel)
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt6.QtGui import QColor

from ...styles import BaseStyles

logger = logging.getLogger(__name__)


class HVNTableWidget(QWidget):
    """Widget for displaying HVN ranges in a table"""
    
    # Signals
    hvn_zone_clicked = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_price = 0.0
        self.m15_atr = 0.0
        self.hvn_zones = []
        
        self.init_ui()
        self.apply_styles()
        
    def init_ui(self):
        """Initialize the UI"""
        self.setObjectName("hvn_table_container")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header = QLabel("HVN Ranges")
        header.setObjectName("table_header")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # Info bar
        self.info_label = QLabel("No data")
        self.info_label.setStyleSheet(f"color: {BaseStyles.TEXT_SECONDARY}; padding: 5px;")
        layout.addWidget(self.info_label)
        
        # Table
        self.table = QTableWidget()
        self.table.setObjectName("hvn_table")
        
        # Set up columns
        columns = ["Zone #", "Price High", "Price Low", "Strength %", "Within M15 ATR"]
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        
        # Configure table
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        
        # Set column widths
        self.table.setColumnWidth(0, 80)   # Zone #
        self.table.setColumnWidth(1, 100)  # Price High
        self.table.setColumnWidth(2, 100)  # Price Low
        self.table.setColumnWidth(3, 100)  # Strength %
        self.table.setColumnWidth(4, 120)  # Within ATR
        
        # Connect selection signal
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        
        layout.addWidget(self.table)
        
    def apply_styles(self):
        """Apply styles to the widget"""
        self.setStyleSheet(f"""
            #hvn_table_container {{
                background-color: {BaseStyles.BACKGROUND_SECONDARY};
                border: 1px solid {BaseStyles.BORDER_COLOR};
                border-radius: 5px;
            }}
            
            #table_header {{
                font-size: {BaseStyles.FONT_SIZE_LARGE};
                font-weight: bold;
                padding: 10px;
                background-color: {BaseStyles.BACKGROUND_TERTIARY};
                border-bottom: 1px solid {BaseStyles.BORDER_COLOR};
                color: {BaseStyles.TEXT_PRIMARY};
            }}
            
            #hvn_table {{
                background-color: {BaseStyles.BACKGROUND_SECONDARY};
                alternate-background-color: {BaseStyles.BACKGROUND_TERTIARY};
                gridline-color: {BaseStyles.BORDER_COLOR};
                border: none;
            }}
            
            #hvn_table::item {{
                padding: 5px;
            }}
            
            #hvn_table::item:selected {{
                background-color: {BaseStyles.ACCENT_PRIMARY};
            }}
            
            QHeaderView::section {{
                background-color: {BaseStyles.BACKGROUND_TERTIARY};
                border: 1px solid {BaseStyles.BORDER_COLOR};
                padding: 5px;
                font-weight: bold;
                color: {BaseStyles.TEXT_PRIMARY};
            }}
        """)
        
    def update_hvn_zones(self, zones: List[Dict], current_price: float, m15_atr: float = 0.0):
        """Update HVN zones display"""
        self.current_price = current_price
        self.m15_atr = m15_atr
        self.hvn_zones = zones
        
        # Clear table
        self.table.setRowCount(0)
        
        # Update info label
        self.info_label.setText(
            f"Current Price: ${current_price:.2f} | "
            f"M15 ATR: ${m15_atr:.2f} | "
            f"Zones: {len(zones)}"
        )
        
        # Add zones to table
        for i, zone in enumerate(zones):
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            # Zone #
            zone_item = QTableWidgetItem(f"HVN {i+1}")
            zone_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 0, zone_item)
            
            # Price High
            high_item = QTableWidgetItem(f"${zone['price_high']:.2f}")
            high_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 1, high_item)
            
            # Price Low
            low_item = QTableWidgetItem(f"${zone['price_low']:.2f}")
            low_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 2, low_item)
            
            # Strength %
            strength = zone.get('strength', 0)
            strength_item = QTableWidgetItem(f"{strength:.1f}%")
            strength_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # Color code strength
            if strength >= 80:
                strength_item.setForeground(QColor('#10b981'))  # Strong
            elif strength >= 50:
                strength_item.setForeground(QColor('#f59e0b'))  # Medium
            else:
                strength_item.setForeground(QColor('#ef4444'))  # Weak
                
            self.table.setItem(row, 3, strength_item)
            
            # Within M15 ATR
            center_price = zone.get('center_price', (zone['price_high'] + zone['price_low']) / 2)
            distance = abs(center_price - current_price)
            within_atr = distance <= m15_atr if m15_atr > 0 else False
            
            atr_item = QTableWidgetItem("Yes" if within_atr else "No")
            atr_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            
            if within_atr:
                atr_item.setForeground(QColor('#10b981'))
                # Highlight the entire row
                for col in range(self.table.columnCount()):
                    item = self.table.item(row, col)
                    if item:
                        item.setBackground(QColor(16, 185, 129, 30))  # Light green background
            else:
                atr_item.setForeground(QColor('#ef4444'))
                
            self.table.setItem(row, 4, atr_item)
            
        logger.info(f"Updated HVN table with {len(zones)} zones")
        
    def _on_selection_changed(self):
        """Handle selection changes"""
        selected_items = self.table.selectedItems()
        if selected_items:
            row = selected_items[0].row()
            if row < len(self.hvn_zones):
                self.hvn_zone_clicked.emit(self.hvn_zones[row])
                
    def clear_zones(self):
        """Clear all zones"""
        self.table.setRowCount(0)
        self.hvn_zones.clear()
        self.info_label.setText("No data")