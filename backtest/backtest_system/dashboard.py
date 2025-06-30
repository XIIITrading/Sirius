"""
Simplified Backtest Dashboard
Only handles UI and plugin execution - no data transformation
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
    QTextEdit, QProgressBar, QMessageBox, QCheckBox
)
from PyQt6.QtCore import Qt, QDateTime, pyqtSignal, QObject

from .components.plugin_runner import PluginRunner
from .components.result_viewer import ResultViewer

logger = logging.getLogger(__name__)


class BacktestDashboard(QMainWindow):
    """Main dashboard window - simplified to just run plugins and display results"""
    
    def __init__(self):
        super().__init__()
        self.plugin_runner = PluginRunner()
        self.result_viewer = ResultViewer()
        self.init_ui()
        self.apply_dark_theme()
        
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("Modular Backtest System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Top controls
        controls_widget = self._create_controls()
        main_layout.addWidget(controls_widget)
        
        # Results area
        main_layout.addWidget(self.result_viewer)
        
        # Status bar
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)
        
    def _create_controls(self) -> QWidget:
        """Create control panel"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Symbol input
        layout.addWidget(QLabel("Symbol:"))
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("AAPL")
        self.symbol_input.setText("AAPL")
        self.symbol_input.setMaximumWidth(100)
        layout.addWidget(self.symbol_input)
        
        # DateTime input
        layout.addWidget(QLabel("Entry Time (UTC):"))
        self.datetime_input = QDateTimeEdit()
        self.datetime_input.setDateTime(QDateTime.currentDateTimeUtc().addSecs(-3600))
        self.datetime_input.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.datetime_input.setCalendarPopup(True)
        layout.addWidget(self.datetime_input)
        
        # Direction
        layout.addWidget(QLabel("Direction:"))
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["LONG", "SHORT"])
        layout.addWidget(self.direction_combo)
        
        # Plugin selection
        layout.addWidget(QLabel("Plugin:"))
        self.plugin_combo = QComboBox()
        self.plugin_combo.addItems(self.plugin_runner.get_available_plugins())
        layout.addWidget(self.plugin_combo)
        
        # Run button
        self.run_button = QPushButton("Run Analysis")
        self.run_button.clicked.connect(self.run_analysis)
        layout.addWidget(self.run_button)
        
        layout.addStretch()
        
        return widget
        
    def run_analysis(self):
        """Run the selected plugin analysis"""
        # Get inputs
        symbol = self.symbol_input.text().strip().upper()
        if not symbol:
            QMessageBox.warning(self, "Input Error", "Please enter a symbol")
            return
            
        entry_time = self.datetime_input.dateTime().toPyDateTime()
        entry_time = entry_time.replace(tzinfo=timezone.utc)
        
        direction = self.direction_combo.currentText()
        plugin_name = self.plugin_combo.currentText()
        
        # Update status
        self.status_label.setText(f"Running {plugin_name} for {symbol}...")
        self.run_button.setEnabled(False)
        
        # Run plugin asynchronously
        asyncio.create_task(self._run_plugin_async(plugin_name, symbol, entry_time, direction))
        
    async def _run_plugin_async(self, plugin_name: str, symbol: str, 
                               entry_time: datetime, direction: str):
        """Run plugin analysis asynchronously"""
        try:
            # Run the plugin
            result = await self.plugin_runner.run_single_plugin(
                plugin_name, symbol, entry_time, direction
            )
            
            # Display result
            self.result_viewer.display_result(result)
            
            # Update status
            if 'error' in result:
                self.status_label.setText(f"Error: {result['error']}")
            else:
                signal = result['signal']
                self.status_label.setText(
                    f"Analysis complete: {signal['direction']} "
                    f"(Strength: {signal['strength']:.0f}%)"
                )
                
        except Exception as e:
            logger.error(f"Error running plugin: {e}")
            self.status_label.setText(f"Error: {str(e)}")
            QMessageBox.critical(self, "Analysis Error", str(e))
            
        finally:
            self.run_button.setEnabled(True)
            
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
            QTableWidget {
                background-color: #2b2b2b;
                alternate-background-color: #323232;
                gridline-color: #444;
            }
            QTextEdit {
                background-color: #2b2b2b;
                border: 1px solid #444;
            }
        """)