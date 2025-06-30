"""
Result Viewer - Displays plugin results without transformation
"""

from typing import Dict, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QTableWidget, QTableWidgetItem, QTextEdit, QLabel
)
from PyQt6.QtCore import Qt


class ResultViewer(QWidget):
    """Widget for displaying plugin results"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        
        # Signal summary
        self.summary_group = QGroupBox("Signal Summary")
        summary_layout = QVBoxLayout(self.summary_group)
        
        self.summary_label = QLabel("No results yet")
        self.summary_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        summary_layout.addWidget(self.summary_label)
        
        self.description_label = QLabel("")
        self.description_label.setWordWrap(True)
        summary_layout.addWidget(self.description_label)
        
        layout.addWidget(self.summary_group)
        
        # Details table
        self.details_group = QGroupBox("Signal Details")
        details_layout = QVBoxLayout(self.details_group)
        
        self.details_table = QTableWidget()
        self.details_table.setColumnCount(2)
        self.details_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.details_table.horizontalHeader().setStretchLastSection(True)
        details_layout.addWidget(self.details_table)
        
        layout.addWidget(self.details_group)
        
        # Raw data viewer (optional)
        self.raw_group = QGroupBox("Raw Plugin Output")
        raw_layout = QVBoxLayout(self.raw_group)
        
        self.raw_text = QTextEdit()
        self.raw_text.setReadOnly(True)
        self.raw_text.setMaximumHeight(150)
        raw_layout.addWidget(self.raw_text)
        
        layout.addWidget(self.raw_group)
        self.raw_group.setVisible(False)  # Hidden by default
        
    def display_result(self, result: Dict[str, Any]):
        """Display plugin result"""
        # Check for error
        if 'error' in result:
            self.summary_label.setText(f"Error: {result['error']}")
            self.summary_label.setStyleSheet("color: #ff6b6b; font-size: 14pt; font-weight: bold;")
            self.description_label.setText("")
            self.details_table.setRowCount(0)
            return
            
        # Display summary
        display_data = result.get('display_data', {})
        signal = result.get('signal', {})
        
        # Summary text
        summary = display_data.get('summary', 'No Signal')
        direction = signal.get('direction', 'NEUTRAL')
        strength = signal.get('strength', 0)
        
        self.summary_label.setText(f"{summary} - {direction} ({strength:.0f}%)")
        
        # Color based on direction
        if direction == 'BULLISH':
            self.summary_label.setStyleSheet("color: #51cf66; font-size: 14pt; font-weight: bold;")
        elif direction == 'BEARISH':
            self.summary_label.setStyleSheet("color: #ff6b6b; font-size: 14pt; font-weight: bold;")
        else:
            self.summary_label.setStyleSheet("color: #868e96; font-size: 14pt; font-weight: bold;")
            
        # Description
        description = display_data.get('description', '')
        self.description_label.setText(description)
        
        # Details table
        table_data = display_data.get('table_data', [])
        self.details_table.setRowCount(len(table_data))
        
        for i, (key, value) in enumerate(table_data):
            self.details_table.setItem(i, 0, QTableWidgetItem(str(key)))
            self.details_table.setItem(i, 1, QTableWidgetItem(str(value)))
            
        # Raw data (for debugging)
        import json
        self.raw_text.setText(json.dumps(result, indent=2, default=str))