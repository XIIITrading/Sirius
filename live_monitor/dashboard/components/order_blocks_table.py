# live_monitor/dashboard/components/order_blocks_table.py
"""
Order Blocks Table Component
Uses supply_demand.py calculations
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


class OrderBlocksTableWidget(QWidget):
    """Widget for displaying order blocks in a table"""
    
    # Signals
    block_clicked = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_price = 0.0
        self.m15_atr = 0.0
        self.order_blocks = []
        
        self.init_ui()
        self.apply_styles()
        
    def init_ui(self):
        """Initialize the UI"""
        self.setObjectName("order_blocks_table_container")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header = QLabel("Order Blocks")
        header.setObjectName("table_header")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # Info bar
        self.info_label = QLabel("No data")
        self.info_label.setStyleSheet(f"color: {BaseStyles.TEXT_SECONDARY}; padding: 5px;")
        layout.addWidget(self.info_label)
        
        # Table
        self.table = QTableWidget()
        self.table.setObjectName("order_blocks_table")
        
        # Set up columns
        columns = ["Block #", "Type", "Price High", "Price Low", "Status", "Within M15 ATR"]
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        
        # Configure table
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        
        # Set column widths
        self.table.setColumnWidth(0, 80)   # Block #
        self.table.setColumnWidth(1, 80)   # Type
        self.table.setColumnWidth(2, 100)  # Price High
        self.table.setColumnWidth(3, 100)  # Price Low
        self.table.setColumnWidth(4, 80)   # Status
        self.table.setColumnWidth(5, 120)  # Within ATR
        
        # Connect selection signal
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        
        layout.addWidget(self.table)
        
    def apply_styles(self):
        """Apply styles to the widget"""
        self.setStyleSheet(f"""
            #order_blocks_table_container {{
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
            
            #order_blocks_table {{
                background-color: {BaseStyles.BACKGROUND_SECONDARY};
                alternate-background-color: {BaseStyles.BACKGROUND_TERTIARY};
                gridline-color: {BaseStyles.BORDER_COLOR};
                border: none;
            }}
            
            #order_blocks_table::item {{
                padding: 5px;
            }}
            
            #order_blocks_table::item:selected {{
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
        
    def update_order_blocks(self, order_blocks: List[Dict], current_price: float, m15_atr: float = 0.0):
        """Update order blocks display"""
        self.current_price = current_price
        self.m15_atr = m15_atr
        self.order_blocks = order_blocks
        
        # Clear table
        self.table.setRowCount(0)
        
        # Count bullish and bearish
        bullish_count = sum(1 for b in order_blocks if b.get('block_type') == 'bullish')
        bearish_count = sum(1 for b in order_blocks if b.get('block_type') == 'bearish')
        
        # Update info label
        self.info_label.setText(
            f"Current Price: ${current_price:.2f} | "
            f"Bullish: {bullish_count} | "
            f"Bearish: {bearish_count}"
        )
        
        # Add blocks to table
        bull_counter = 0
        bear_counter = 0
        
        for block in order_blocks:
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            # Block #
            if block['block_type'] == 'bullish':
                bull_counter += 1
                block_num = f"BULL {bull_counter}"
            else:
                bear_counter += 1
                block_num = f"BEAR {bear_counter}"
                
            block_item = QTableWidgetItem(block_num)
            block_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 0, block_item)
            
            # Type
            type_item = QTableWidgetItem(block['block_type'].capitalize())
            type_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if block['block_type'] == 'bullish':
                type_item.setForeground(QColor('#10b981'))  # Green
            else:
                type_item.setForeground(QColor('#ef4444'))  # Red
            self.table.setItem(row, 1, type_item)
            
            # Price High (Top)
            high_item = QTableWidgetItem(f"${block['top']:.2f}")
            high_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 2, high_item)
            
            # Price Low (Bottom)
            low_item = QTableWidgetItem(f"${block['bottom']:.2f}")
            low_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 3, low_item)
            
            # Status
            is_breaker = block.get('is_breaker', False)
            status_item = QTableWidgetItem("Broken" if is_breaker else "Valid")
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            
            if is_breaker:
                status_item.setForeground(QColor('#f59e0b'))  # Orange for broken
            else:
                status_item.setForeground(QColor('#10b981'))  # Green for valid
                
            self.table.setItem(row, 4, status_item)
            
            # Within M15 ATR
            center_price = block.get('center', (block['top'] + block['bottom']) / 2)
            distance = abs(center_price - current_price)
            within_atr = distance <= m15_atr if m15_atr > 0 else False
            
            atr_item = QTableWidgetItem("Yes" if within_atr else "No")
            atr_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            
            if within_atr:
                atr_item.setForeground(QColor('#10b981'))
                # Highlight the entire row based on block type
                color = QColor(239, 68, 68, 30) if block['block_type'] == 'bearish' else QColor(16, 185, 129, 30)
                for col in range(self.table.columnCount()):
                    item = self.table.item(row, col)
                    if item:
                        item.setBackground(color)
            else:
                atr_item.setForeground(QColor('#888888'))
                
            self.table.setItem(row, 5, atr_item)
            
        logger.info(f"Updated Order Blocks table with {len(order_blocks)} blocks")
        
    def _on_selection_changed(self):
        """Handle selection changes"""
        selected_items = self.table.selectedItems()
        if selected_items:
            row = selected_items[0].row()
            if row < len(self.order_blocks):
                self.block_clicked.emit(self.order_blocks[row])
                
    def clear_blocks(self):
        """Clear all blocks"""
        self.table.setRowCount(0)
        self.order_blocks.clear()
        self.info_label.setText("No data")