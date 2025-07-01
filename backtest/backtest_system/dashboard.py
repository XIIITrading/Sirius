"""
Backtest Dashboard with Multi-Plugin Support
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QDateTimeEdit, QComboBox,
    QGroupBox, QSplitter, QTableWidget, QTableWidgetItem,
    QTextEdit, QProgressBar, QMessageBox, QCheckBox,
    QListWidget, QListWidgetItem, QAbstractItemView
)
from PyQt6.QtCore import Qt, QDateTime, pyqtSignal, QObject

from .components.plugin_runner import PluginRunner
from .components.result_viewer import ResultViewer
from .components.multi_result_viewer import MultiResultViewer

logger = logging.getLogger(__name__)


class BacktestDashboard(QMainWindow):
    """Main dashboard window with multi-plugin support"""
    
    def __init__(self):
        super().__init__()
        self.plugin_runner = PluginRunner()
        self.single_result_viewer = ResultViewer()
        self.multi_result_viewer = MultiResultViewer()
        self.init_ui()
        self.apply_dark_theme()
        
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("Modular Backtest System")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Top controls
        controls_widget = self._create_controls()
        main_layout.addWidget(controls_widget)
        
        # Create splitter for results
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Multi-result viewer (for multiple plugins)
        self.splitter.addWidget(self.multi_result_viewer)
        
        # Single result viewer (for detailed view)
        self.splitter.addWidget(self.single_result_viewer)
        
        # Set initial sizes (60% for multi, 40% for single)
        self.splitter.setSizes([540, 360])
        
        main_layout.addWidget(self.splitter)
        
        # Status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)
        
        # Initially hide single result viewer
        self.single_result_viewer.setVisible(False)
        
    def _create_controls(self) -> QWidget:
        """Create control panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # First row - inputs
        input_layout = QHBoxLayout()
        
        # Symbol input
        input_layout.addWidget(QLabel("Symbol:"))
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("TSLA")
        self.symbol_input.setText("TSLA")
        self.symbol_input.setMaximumWidth(100)
        input_layout.addWidget(self.symbol_input)
        
        # DateTime input
        input_layout.addWidget(QLabel("Entry Time (UTC):"))
        self.datetime_input = QDateTimeEdit()
        self.datetime_input.setDateTime(QDateTime.currentDateTimeUtc().addSecs(-3600))
        self.datetime_input.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.datetime_input.setCalendarPopup(True)
        input_layout.addWidget(self.datetime_input)
        
        # Direction
        input_layout.addWidget(QLabel("Direction:"))
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["LONG", "SHORT"])
        input_layout.addWidget(self.direction_combo)
        
        input_layout.addStretch()
        layout.addLayout(input_layout)
        
        # Second row - plugin selection and run
        plugin_layout = QHBoxLayout()
        
        # Plugin selection list
        plugin_layout.addWidget(QLabel("Plugins:"))
        self.plugin_list = QListWidget()
        self.plugin_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.plugin_list.setMaximumHeight(100)
        self.plugin_list.setMinimumWidth(300)
        
        # Add plugins to list
        for plugin_name in self.plugin_runner.get_available_plugins():
            item = QListWidgetItem(plugin_name)
            self.plugin_list.addItem(item)
            # Select first item by default
            if self.plugin_list.count() == 1:
                item.setSelected(True)
        
        plugin_layout.addWidget(self.plugin_list)
        
        # Buttons
        button_layout = QVBoxLayout()
        
        # Select all/none buttons
        select_button_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self._select_all_plugins)
        self.select_none_btn = QPushButton("Select None")
        self.select_none_btn.clicked.connect(self._select_no_plugins)
        select_button_layout.addWidget(self.select_all_btn)
        select_button_layout.addWidget(self.select_none_btn)
        button_layout.addLayout(select_button_layout)
        
        # Run button
        self.run_button = QPushButton("Run Selected Plugins")
        self.run_button.clicked.connect(self.run_analysis)
        button_layout.addWidget(self.run_button)
        
        plugin_layout.addLayout(button_layout)
        plugin_layout.addStretch()
        
        layout.addLayout(plugin_layout)
        
        return widget
    
    def _select_all_plugins(self):
        """Select all plugins"""
        for i in range(self.plugin_list.count()):
            self.plugin_list.item(i).setSelected(True)
            
    def _select_no_plugins(self):
        """Deselect all plugins"""
        for i in range(self.plugin_list.count()):
            self.plugin_list.item(i).setSelected(False)
        
    def run_analysis(self):
        """Run the selected plugin analyses"""
        # Get inputs
        symbol = self.symbol_input.text().strip().upper()
        if not symbol:
            QMessageBox.warning(self, "Input Error", "Please enter a symbol")
            return
            
        # Get selected plugins
        selected_items = self.plugin_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Selection Error", "Please select at least one plugin")
            return
            
        selected_plugins = [item.text() for item in selected_items]
            
        entry_time = self.datetime_input.dateTime().toPyDateTime()
        entry_time = entry_time.replace(tzinfo=timezone.utc)
        
        direction = self.direction_combo.currentText()
        
        # Update status
        self.status_label.setText(f"Running {len(selected_plugins)} plugins for {symbol}...")
        self.run_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(selected_plugins))
        self.progress_bar.setValue(0)
        
        # Clear previous results
        self.multi_result_viewer.clear_results()
        self.single_result_viewer.display_result({})
        self.single_result_viewer.setVisible(False)
        
        # Run plugins asynchronously
        asyncio.create_task(self._run_plugins_async(selected_plugins, symbol, entry_time, direction))
        
    async def _run_plugins_async(self, plugin_names: List[str], symbol: str, 
                               entry_time: datetime, direction: str):
        """Run multiple plugins asynchronously"""
        try:
            results = []
            
            # Run plugins one by one to show progress
            for i, plugin_name in enumerate(plugin_names):
                # Update progress
                self.progress_bar.setValue(i)
                self.status_label.setText(f"Running {plugin_name}...")
                
                # Run the plugin
                result = await self.plugin_runner.run_single_plugin(
                    plugin_name, symbol, entry_time, direction
                )
                results.append(result)
                
                # Add to multi-viewer immediately
                self.multi_result_viewer.add_result(result)
            
            # Update final status
            self.progress_bar.setValue(len(plugin_names))
            
            # Count successful results
            successful = sum(1 for r in results if 'error' not in r)
            self.status_label.setText(
                f"Completed: {successful}/{len(plugin_names)} plugins successful"
            )
            
            # Connect row selection to show details
            self.multi_result_viewer.result_selected.connect(self._show_detailed_result)
                
        except Exception as e:
            logger.error(f"Error running plugins: {e}")
            self.status_label.setText(f"Error: {str(e)}")
            QMessageBox.critical(self, "Analysis Error", str(e))
            
        finally:
            self.run_button.setEnabled(True)
            self.progress_bar.setVisible(False)
    
    def _show_detailed_result(self, result: Dict[str, Any]):
        """Show detailed view of selected result"""
        self.single_result_viewer.display_result(result)
        self.single_result_viewer.setVisible(True)
        # Adjust splitter to show both views
        self.splitter.setSizes([400, 500])
            
    def apply_dark_theme(self):
        """Apply dark theme to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QGroupBox {
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #0d7377;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
            QPushButton:pressed {
                background-color: #0a5d61;
            }
            QPushButton:disabled {
                background-color: #4a4a4a;
                color: #888888;
            }
            QLineEdit, QComboBox, QDateTimeEdit {
                background-color: #2b2b2b;
                border: 1px solid #444;
                padding: 5px;
                border-radius: 3px;
            }
            QListWidget {
                background-color: #2b2b2b;
                border: 1px solid #444;
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: #0d7377;
            }
            QTableWidget {
                background-color: #2b2b2b;
                alternate-background-color: #323232;
                gridline-color: #444;
            }
            QTableWidget::item:selected {
                background-color: #0d7377;
            }
            QTextEdit {
                background-color: #2b2b2b;
                border: 1px solid #444;
            }
            QProgressBar {
                border: 1px solid #444;
                border-radius: 3px;
                text-align: center;
                background-color: #2b2b2b;
            }
            QProgressBar::chunk {
                background-color: #0d7377;
                border-radius: 3px;
            }
        """)