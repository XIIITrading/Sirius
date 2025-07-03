"""
Result Viewer - Displays plugin results without transformation
Enhanced with chart support for plugins that provide chart data
"""

import logging
import importlib
from typing import Dict, Any, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QTableWidget, QTableWidgetItem, QTextEdit, QLabel,
    QSplitter
)
from PyQt6.QtCore import Qt

logger = logging.getLogger(__name__)


class ResultViewer(QWidget):
    """Widget for displaying plugin results with optional chart support"""
    
    def __init__(self):
        super().__init__()
        self.current_chart = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        main_layout = QVBoxLayout(self)
        
        # Create horizontal splitter for result details and chart
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - traditional result display
        self.result_widget = QWidget()
        result_layout = QVBoxLayout(self.result_widget)
        
        # Signal summary
        self.summary_group = QGroupBox("Signal Summary")
        summary_layout = QVBoxLayout(self.summary_group)
        
        self.summary_label = QLabel("No results yet")
        self.summary_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        summary_layout.addWidget(self.summary_label)
        
        self.description_label = QLabel("")
        self.description_label.setWordWrap(True)
        summary_layout.addWidget(self.description_label)
        
        result_layout.addWidget(self.summary_group)
        
        # Details table
        self.details_group = QGroupBox("Signal Details")
        details_layout = QVBoxLayout(self.details_group)
        
        self.details_table = QTableWidget()
        self.details_table.setColumnCount(2)
        self.details_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.details_table.horizontalHeader().setStretchLastSection(True)
        details_layout.addWidget(self.details_table)
        
        result_layout.addWidget(self.details_group)
        
        # Raw data viewer (optional)
        self.raw_group = QGroupBox("Raw Plugin Output")
        raw_layout = QVBoxLayout(self.raw_group)
        
        self.raw_text = QTextEdit()
        self.raw_text.setReadOnly(True)
        self.raw_text.setMaximumHeight(150)
        raw_layout.addWidget(self.raw_text)
        
        result_layout.addWidget(self.raw_group)
        self.raw_group.setVisible(False)  # Hidden by default
        
        # Add left widget to splitter
        self.splitter.addWidget(self.result_widget)
        
        # Right side - chart container
        self.chart_container = QGroupBox("Chart Visualization")
        self.chart_layout = QVBoxLayout(self.chart_container)
        self.splitter.addWidget(self.chart_container)
        
        # Initially hide chart container
        self.chart_container.setVisible(False)
        
        # Add splitter to main layout
        main_layout.addWidget(self.splitter)
        
        # Set initial splitter sizes (60/40 split when chart is shown)
        self.splitter.setSizes([600, 400])
        
    def display_result(self, result: Dict[str, Any]):
        """Display plugin result with optional chart"""
        # Clear any existing chart first
        self._clear_chart()
        
        # Check for error
        if 'error' in result:
            self.summary_label.setText(f"Error: {result['error']}")
            self.summary_label.setStyleSheet("color: #ff6b6b; font-size: 14pt; font-weight: bold;")
            self.description_label.setText("")
            self.details_table.setRowCount(0)
            self._hide_chart()
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
            
        # Handle chart if present
        chart_config = display_data.get('chart_widget')
        if chart_config:
            self._display_chart(chart_config)
        else:
            self._hide_chart()
            
        # Raw data (for debugging)
        import json
        self.raw_text.setText(json.dumps(result, indent=2, default=str))
    
    def _display_chart(self, chart_config: Dict[str, Any]):
        """Display a chart based on configuration"""
        try:
            # Import the chart module dynamically
            module_name = chart_config.get('module')
            chart_class_name = chart_config.get('type')
            
            if not module_name or not chart_class_name:
                logger.error("Invalid chart configuration")
                return
                
            # Import and create chart
            module = importlib.import_module(module_name)
            ChartClass = getattr(module, chart_class_name)
            
            # Create chart instance
            self.current_chart = ChartClass()
            self.chart_layout.addWidget(self.current_chart)
            
            # Update chart with data - check which method to use
            chart_data = chart_config.get('data', [])
            
            # Use the appropriate update method based on what the chart supports
            if hasattr(self.current_chart, 'update_from_data'):
                # For charts that expect pre-formatted data (like Impact Success)
                self.current_chart.update_from_data(chart_data)
            elif hasattr(self.current_chart, 'update_data'):
                # For legacy charts
                self.current_chart.update_data(chart_data)
            else:
                logger.error(f"Chart {chart_class_name} has no update method")
                self._hide_chart()
                return
            
            # Add entry marker if specified and chart supports it
            if chart_config.get('entry_time') and hasattr(self.current_chart, 'add_entry_marker'):
                self.current_chart.add_entry_marker(chart_config['entry_time'])
            elif chart_config.get('entry_time') and hasattr(self.current_chart, 'add_marker'):
                # For charts using add_marker method (legacy)
                self.current_chart.add_marker(30, "Entry", "#ff0000")
            
            # Show chart container
            self.chart_container.setVisible(True)
            
            # Adjust splitter to show both sides
            self.splitter.setSizes([500, 500])
            
        except Exception as e:
            logger.error(f"Error displaying chart: {e}")
            import traceback
            traceback.print_exc()
            self._hide_chart()
    
    def _clear_chart(self):
        """Clear current chart"""
        if self.current_chart:
            self.chart_layout.removeWidget(self.current_chart)
            self.current_chart.deleteLater()
            self.current_chart = None
    
    def _hide_chart(self):
        """Hide chart container"""
        self._clear_chart()
        self.chart_container.setVisible(False)
        # Reset splitter to full width for results
        self.splitter.setSizes([1000, 0])