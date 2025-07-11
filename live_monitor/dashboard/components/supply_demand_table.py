# live_monitor/dashboard/components/supply_demand_table.py
"""
Supply/Demand Table Component
Displays user-input supply/demand zones from Supabase
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QLabel,
                             QHBoxLayout, QPushButton)
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt6.QtGui import QColor

from ...styles import BaseStyles

logger = logging.getLogger(__name__)


class SupplyDemandTableWidget(QWidget):
    """Widget for displaying user-input supply/demand zones in a table"""
    
    # Signals
    zone_clicked = pyqtSignal(dict)
    add_zone_requested = pyqtSignal()
    refresh_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_price = 0.0
        self.m15_atr = 0.0
        self.zones = []
        
        self.init_ui()
        self.apply_styles()
        
    def init_ui(self):
        """Initialize the UI"""
        self.setObjectName("supply_demand_table_container")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header with controls
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(10, 5, 10, 5)
        
        header = QLabel("Supply / Demand Zones")
        header.setObjectName("table_header")
        header_layout.addWidget(header)
        
        header_layout.addStretch()
        
        # Add zone button
        add_btn = QPushButton("Add Zone")
        add_btn.setObjectName("small_button")
        add_btn.clicked.connect(self.add_zone_requested.emit)
        header_layout.addWidget(add_btn)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setObjectName("small_button")
        refresh_btn.clicked.connect(self.refresh_requested.emit)
        header_layout.addWidget(refresh_btn)
        
        layout.addWidget(header_widget)
        
        # Info bar
        self.info_label = QLabel("No zones loaded")
        self.info_label.setStyleSheet(f"color: {BaseStyles.TEXT_SECONDARY}; padding: 5px;")
        layout.addWidget(self.info_label)
        
        # Table
        self.table = QTableWidget()
        self.table.setObjectName("supply_demand_table")
        
        # Set up columns
        columns = ["Zone #", "Type", "Price High", "Price Low", "Within M15 ATR"]
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        
        # Configure table
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        
        # Set column widths
        self.table.setColumnWidth(0, 80)   # Zone #
        self.table.setColumnWidth(1, 80)   # Type
        self.table.setColumnWidth(2, 100)  # Price High
        self.table.setColumnWidth(3, 100)  # Price Low
        self.table.setColumnWidth(4, 120)  # Within ATR
        
        # Connect selection signal
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        
        layout.addWidget(self.table)
        
    def apply_styles(self):
        """Apply styles to the widget"""
        self.setStyleSheet(f"""
            #supply_demand_table_container {{
                background-color: {BaseStyles.BACKGROUND_SECONDARY};
                border: 1px solid {BaseStyles.BORDER_COLOR};
                border-radius: 5px;
            }}
            
            #table_header {{
                font-size: {BaseStyles.FONT_SIZE_LARGE};
                font-weight: bold;
                color: {BaseStyles.TEXT_PRIMARY};
            }}
            
            #small_button {{
                background-color: {BaseStyles.ACCENT_PRIMARY};
                border: none;
                padding: 4px 8px;
                border-radius: 3px;
                font-size: {BaseStyles.FONT_SIZE_SMALL};
                color: white;
            }}
            
            #small_button:hover {{
                background-color: {BaseStyles.ACCENT_HOVER};
            }}
            
            #supply_demand_table {{
                background-color: {BaseStyles.BACKGROUND_SECONDARY};
                alternate-background-color: {BaseStyles.BACKGROUND_TERTIARY};
                gridline-color: {BaseStyles.BORDER_COLOR};
                border: none;
            }}
            
            #supply_demand_table::item {{
                padding: 5px;
            }}
            
            #supply_demand_table::item:selected {{
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
        
    def update_zones(self, supply_zones: List[Dict], demand_zones: List[Dict], 
                     current_price: float, m15_atr: float = 0.0):
        """Update supply and demand zones display"""
        self.current_price = current_price
        self.m15_atr = m15_atr
        
        # Combine zones with type
        self.zones = []
        for zone in supply_zones:
            zone['zone_type'] = 'supply'
            self.zones.append(zone)
        for zone in demand_zones:
            zone['zone_type'] = 'demand'
            self.zones.append(zone)
        
        # Clear table
        self.table.setRowCount(0)
        
        # Update info label
        self.info_label.setText(
            f"Current Price: ${current_price:.2f} | "
            f"Supply: {len(supply_zones)} | "
            f"Demand: {len(demand_zones)}"
        )
        
        # Add zones to table
        for i, zone in enumerate(self.zones):
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            # Zone #
            zone_type = zone['zone_type'].upper()
            zone_item = QTableWidgetItem(f"{zone_type[0]}{i+1}")
            zone_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 0, zone_item)
            
            # Type
            type_item = QTableWidgetItem(zone_type.capitalize())
            type_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if zone['zone_type'] == 'supply':
                type_item.setForeground(QColor('#ef4444'))  # Red
            else:
                type_item.setForeground(QColor('#10b981'))  # Green
            self.table.setItem(row, 1, type_item)
            
            # Price High
            high_item = QTableWidgetItem(f"${zone['price_high']:.2f}")
            high_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 2, high_item)
            
            # Price Low
            low_item = QTableWidgetItem(f"${zone['price_low']:.2f}")
            low_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 3, low_item)
            
            # Within M15 ATR
            center_price = zone.get('center_price', (zone['price_high'] + zone['price_low']) / 2)
            distance = abs(center_price - current_price)
            within_atr = distance <= m15_atr if m15_atr > 0 else False
            
            atr_item = QTableWidgetItem("Yes" if within_atr else "No")
            atr_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            
            if within_atr:
                atr_item.setForeground(QColor('#10b981'))
                # Highlight the entire row based on zone type
                color = QColor(239, 68, 68, 30) if zone['zone_type'] == 'supply' else QColor(16, 185, 129, 30)
                for col in range(self.table.columnCount()):
                    item = self.table.item(row, col)
                    if item:
                        item.setBackground(color)
            else:
                atr_item.setForeground(QColor('#888888'))
                
            self.table.setItem(row, 4, atr_item)
            
        logger.info(f"Updated S/D table with {len(self.zones)} zones")
        
    def _on_selection_changed(self):
        """Handle selection changes"""
        selected_items = self.table.selectedItems()
        if selected_items:
            row = selected_items[0].row()
            if row < len(self.zones):
                self.zone_clicked.emit(self.zones[row])
                
    def clear_zones(self):
        """Clear all zones"""
        self.table.setRowCount(0)
        self.zones.clear()
        self.info_label.setText("No zones loaded")