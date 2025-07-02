"""
Backtest Dashboard with Simplified Plugin Selection and Custom Ordering
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from enum import Enum

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QDateTimeEdit, QComboBox,
    QGroupBox, QSplitter, QTableWidget, QTableWidgetItem,
    QTextEdit, QProgressBar, QMessageBox, QCheckBox,
    QListWidget, QListWidgetItem, QAbstractItemView,
    QScrollArea
)
from PyQt6.QtCore import Qt, QDateTime, pyqtSignal, QObject

from .components.plugin_runner import PluginRunner
from .components.result_viewer import ResultViewer
from .components.multi_result_viewer import MultiResultViewer

# Import PolygonDataManager
from backtest.data.polygon_data_manager import PolygonDataManager

logger = logging.getLogger(__name__)

# Define the order in which plugins should run
PLUGIN_RUN_ORDER = [
    "1-Min EMA Crossover",
    "5-Min EMA Crossover", 
    "15-Min EMA Crossover",
    "1-Min Market Structure",
    "5-Min Market Structure",
    "15-Min Market Structure",
    "1-Min Bid/Ask Analysis",
    "Bid/Ask Imbalance Analysis",
    "Tick Flow Analysis"
]


class PluginStatus(Enum):
    """Plugin execution status"""
    PENDING = "Pending"
    RUNNING = "Running"
    COMPLETED = "Completed"
    ERROR = "Error"


class ProgressSignals(QObject):
    """Signals for progress updates"""
    plugin_started = pyqtSignal(str)  # plugin_name
    plugin_progress = pyqtSignal(str, int, str)  # plugin_name, percentage, message
    plugin_completed = pyqtSignal(str, dict)  # plugin_name, result
    plugin_error = pyqtSignal(str, str)  # plugin_name, error_message


class BacktestDashboard(QMainWindow):
    """Main dashboard with simplified plugin selection and progress tracking"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize data manager first
        logger.info("Initializing PolygonDataManager...")
        self.data_manager = PolygonDataManager(disable_polygon_cache=True)
        
        # Pass data manager to plugin runner
        logger.info("Initializing PluginRunner with data manager...")
        self.plugin_runner = PluginRunner(data_manager=self.data_manager)
        
        self.single_result_viewer = ResultViewer()
        self.multi_result_viewer = MultiResultViewer()
        
        # Progress tracking
        self.progress_signals = ProgressSignals()
        self.plugin_status = {}  # Track status of each plugin
        self.progress_widgets = {}  # Progress bars for each plugin
        
        # Get available plugins and sort them according to our order
        self.available_plugins = self._get_ordered_plugins()
        
        self.init_ui()
        self.apply_dark_theme()
        self.connect_progress_signals()
        
    def _get_ordered_plugins(self) -> List[str]:
        """Get available plugins sorted according to PLUGIN_RUN_ORDER"""
        available = self.plugin_runner.get_available_plugins()
        
        # First, add plugins in the specified order if they exist
        ordered = []
        for plugin_name in PLUGIN_RUN_ORDER:
            if plugin_name in available:
                ordered.append(plugin_name)
        
        # Then add any remaining plugins not in the order list
        for plugin_name in available:
            if plugin_name not in ordered:
                ordered.append(plugin_name)
                
        return ordered
        
    def connect_progress_signals(self):
        """Connect progress signals to UI updates"""
        self.progress_signals.plugin_started.connect(self.on_plugin_started)
        self.progress_signals.plugin_progress.connect(self.on_plugin_progress)
        self.progress_signals.plugin_completed.connect(self.on_plugin_completed)
        self.progress_signals.plugin_error.connect(self.on_plugin_error)
        
    def init_ui(self):
        """Initialize the UI with simplified controls"""
        self.setWindowTitle("Backtest Analysis System")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Top controls (single row now)
        controls_widget = self._create_controls()
        main_layout.addWidget(controls_widget)
        
        # Progress tracking widget
        self.progress_widget = self._create_progress_widget()
        main_layout.addWidget(self.progress_widget)
        
        # Create splitter for results
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Multi-result viewer (for multiple plugins)
        self.splitter.addWidget(self.multi_result_viewer)
        
        # Single result viewer (for detailed view)
        self.splitter.addWidget(self.single_result_viewer)
        
        # Set initial sizes
        self.splitter.setSizes([400, 300])
        
        main_layout.addWidget(self.splitter)
        
        # Bottom status and report section
        bottom_layout = QHBoxLayout()
        
        # Overall status bar
        self.status_label = QLabel("Ready")
        bottom_layout.addWidget(self.status_label)
        
        bottom_layout.addStretch()
        
        # Report buttons
        self.generate_report_button = QPushButton("Generate Data Report")
        self.generate_report_button.clicked.connect(self.generate_data_report)
        self.generate_report_button.setEnabled(False)
        bottom_layout.addWidget(self.generate_report_button)
        
        self.clear_cache_button = QPushButton("Clear Cache")
        self.clear_cache_button.clicked.connect(self.clear_cache)
        bottom_layout.addWidget(self.clear_cache_button)
        
        bottom_widget = QWidget()
        bottom_widget.setLayout(bottom_layout)
        main_layout.addWidget(bottom_widget)
        
        # Initially hide single result viewer and progress widget
        self.single_result_viewer.setVisible(False)
        self.progress_widget.setVisible(False)
        
    def _create_controls(self) -> QWidget:
        """Create simplified control panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Single row with all controls
        control_layout = QHBoxLayout()
        
        # Symbol input
        control_layout.addWidget(QLabel("Symbol:"))
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("TSLA")
        self.symbol_input.setText("TSLA")
        self.symbol_input.setMaximumWidth(80)
        control_layout.addWidget(self.symbol_input)
        
        # DateTime input
        control_layout.addWidget(QLabel("Entry Time (UTC):"))
        self.datetime_input = QDateTimeEdit()
        self.datetime_input.setDateTime(QDateTime.currentDateTimeUtc().addSecs(-3600))
        self.datetime_input.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.datetime_input.setCalendarPopup(True)
        control_layout.addWidget(self.datetime_input)
        
        # Direction
        control_layout.addWidget(QLabel("Direction:"))
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["LONG", "SHORT"])
        self.direction_combo.setMaximumWidth(100)
        control_layout.addWidget(self.direction_combo)
        
        # Plugin selection dropdown
        control_layout.addWidget(QLabel("Analysis:"))
        self.plugin_combo = QComboBox()
        self.plugin_combo.setMinimumWidth(250)
        
        # Add "All Plugins" option first
        self.plugin_combo.addItem("All Plugins")
        self.plugin_combo.insertSeparator(1)
        
        # Add individual plugins
        for plugin_name in self.available_plugins:
            self.plugin_combo.addItem(plugin_name)
            
        control_layout.addWidget(self.plugin_combo)
        
        # Run button
        self.run_button = QPushButton("Run Analysis")
        self.run_button.clicked.connect(self.run_analysis)
        self.run_button.setMinimumWidth(120)
        control_layout.addWidget(self.run_button)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # Optional: Show plugin order info
        if len(self.available_plugins) > 1:
            order_label = QLabel(f"Plugin execution order: {' → '.join(self.available_plugins[:3])}...")
            order_label.setStyleSheet("color: #888; font-size: 11px;")
            layout.addWidget(order_label)
        
        return widget
    
    def _create_progress_widget(self) -> QWidget:
        """Create widget for tracking plugin progress"""
        widget = QGroupBox("Analysis Progress")
        layout = QVBoxLayout(widget)
        
        # Container for individual plugin progress bars
        self.progress_container = QWidget()
        self.progress_layout = QVBoxLayout(self.progress_container)
        self.progress_layout.setSpacing(5)
        
        # Scroll area for many plugins
        scroll = QScrollArea()
        scroll.setWidget(self.progress_container)
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(200)
        
        layout.addWidget(scroll)
        
        return widget
    
    def _create_plugin_progress_bar(self, plugin_name: str) -> QWidget:
        """Create progress bar widget for a single plugin"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 2, 5, 2)
        
        # Plugin name
        name_label = QLabel(plugin_name)
        name_label.setMinimumWidth(200)
        name_label.setMaximumWidth(200)
        layout.addWidget(name_label)
        
        # Progress bar
        progress_bar = QProgressBar()
        progress_bar.setMinimum(0)
        progress_bar.setMaximum(100)
        progress_bar.setTextVisible(True)
        progress_bar.setMinimumWidth(150)
        layout.addWidget(progress_bar)
        
        # Status label
        status_label = QLabel("Pending")
        status_label.setMinimumWidth(80)
        status_label.setMaximumWidth(80)
        status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(status_label)
        
        # Details label
        details_label = QLabel("")
        details_label.setMinimumWidth(300)
        layout.addWidget(details_label)
        
        layout.addStretch()
        
        # Store references
        self.progress_widgets[plugin_name] = {
            'widget': widget,
            'progress_bar': progress_bar,
            'status_label': status_label,
            'details_label': details_label
        }
        
        return widget
    
    def on_plugin_started(self, plugin_name: str):
        """Handle plugin started signal"""
        self.plugin_status[plugin_name] = PluginStatus.RUNNING
        
        if plugin_name not in self.progress_widgets:
            widget = self._create_plugin_progress_bar(plugin_name)
            self.progress_layout.addWidget(widget)
        
        widgets = self.progress_widgets[plugin_name]
        widgets['status_label'].setText("Running")
        widgets['status_label'].setStyleSheet("color: #14a085; font-weight: bold;")
        widgets['progress_bar'].setValue(0)
        widgets['details_label'].setText("Initializing...")
    
    def on_plugin_progress(self, plugin_name: str, percentage: int, message: str):
        """Handle plugin progress update"""
        if plugin_name in self.progress_widgets:
            widgets = self.progress_widgets[plugin_name]
            widgets['progress_bar'].setValue(percentage)
            widgets['details_label'].setText(message)
    
    def on_plugin_completed(self, plugin_name: str, result: dict):
        """Handle plugin completion"""
        self.plugin_status[plugin_name] = PluginStatus.COMPLETED
        
        if plugin_name in self.progress_widgets:
            widgets = self.progress_widgets[plugin_name]
            widgets['status_label'].setText("✓ Complete")
            widgets['status_label'].setStyleSheet("color: #0d7377; font-weight: bold;")
            widgets['progress_bar'].setValue(100)
            
            # Show summary in details
            signal_dir = result.get('signal', {}).get('direction', 'N/A')
            confidence = result.get('signal', {}).get('confidence', 0)
            strength = result.get('signal', {}).get('strength', 0)
            widgets['details_label'].setText(
                f"{signal_dir} - Strength: {strength:.0f}%, Confidence: {confidence:.0f}%"
            )
    
    def on_plugin_error(self, plugin_name: str, error_message: str):
        """Handle plugin error"""
        self.plugin_status[plugin_name] = PluginStatus.ERROR
        
        if plugin_name in self.progress_widgets:
            widgets = self.progress_widgets[plugin_name]
            widgets['status_label'].setText("✗ Error")
            widgets['status_label'].setStyleSheet("color: #ff4444; font-weight: bold;")
            widgets['details_label'].setText(
                error_message[:50] + "..." if len(error_message) > 50 else error_message
            )
    
    def run_analysis(self):
        """Run the selected analysis (all plugins or single)"""
        # Get inputs
        symbol = self.symbol_input.text().strip().upper()
        if not symbol:
            QMessageBox.warning(self, "Input Error", "Please enter a symbol")
            return
        
        # Get selected analysis
        selected_analysis = self.plugin_combo.currentText()
        
        # Determine which plugins to run
        if selected_analysis == "All Plugins":
            selected_plugins = self.available_plugins.copy()
        else:
            selected_plugins = [selected_analysis]
            
        entry_time = self.datetime_input.dateTime().toPyDateTime()
        entry_time = entry_time.replace(tzinfo=timezone.utc)
        
        direction = self.direction_combo.currentText()
        
        # Clear previous progress widgets
        for plugin_name in list(self.progress_widgets.keys()):
            widget_info = self.progress_widgets[plugin_name]
            widget_info['widget'].deleteLater()
            del self.progress_widgets[plugin_name]
        
        # Reset plugin status
        self.plugin_status.clear()
        
        # Update status
        plugin_text = f"{len(selected_plugins)} plugins" if len(selected_plugins) > 1 else selected_plugins[0]
        self.status_label.setText(f"Running {plugin_text} for {symbol}...")
        self.run_button.setEnabled(False)
        self.generate_report_button.setEnabled(False)
        
        # Show progress widget
        self.progress_widget.setVisible(True)
        
        # Clear previous results
        self.multi_result_viewer.clear_results()
        self.single_result_viewer.display_result({})
        self.single_result_viewer.setVisible(False)
        
        # Run plugins asynchronously
        asyncio.create_task(self._run_plugins_async(selected_plugins, symbol, entry_time, direction))
        
    async def _run_plugins_async(self, plugin_names: List[str], symbol: str, 
                               entry_time: datetime, direction: str):
        """Run plugins asynchronously in the specified order"""
        try:
            results = []
            
            # Create progress callback
            def progress_callback(plugin_name: str, percentage: int, message: str):
                self.progress_signals.plugin_progress.emit(plugin_name, percentage, message)
            
            # Run plugins in order
            for i, plugin_name in enumerate(plugin_names):
                # Emit start signal
                self.progress_signals.plugin_started.emit(plugin_name)
                
                # Update overall status
                if len(plugin_names) > 1:
                    self.status_label.setText(
                        f"Running {plugin_name} ({i+1}/{len(plugin_names)})..."
                    )
                else:
                    self.status_label.setText(f"Running {plugin_name}...")
                
                try:
                    # Run the plugin with progress callback
                    result = await self.plugin_runner.run_single_plugin(
                        plugin_name, symbol, entry_time, direction,
                        progress_callback=lambda pct, msg: progress_callback(plugin_name, pct, msg)
                    )
                    
                    results.append(result)
                    
                    # Add to multi-viewer immediately
                    self.multi_result_viewer.add_result(result)
                    
                    # Emit completion signal
                    self.progress_signals.plugin_completed.emit(plugin_name, result)
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error in {plugin_name}: {error_msg}")
                    self.progress_signals.plugin_error.emit(plugin_name, error_msg)
                    
                    # Add error result
                    error_result = {
                        'plugin_name': plugin_name,
                        'timestamp': entry_time,
                        'error': error_msg,
                        'signal': {'direction': 'ERROR', 'strength': 0, 'confidence': 0}
                    }
                    results.append(error_result)
            
            # Generate data report automatically after all plugins run
            try:
                logger.info("Generating data report after plugin execution...")
                json_file, summary_file = self.data_manager.generate_data_report()
                
                # Show report location in status
                report_msg = f" | Report: {Path(summary_file).name}"
                
                # Check for critical issues in the report
                with open(summary_file, 'r') as f:
                    report_content = f.read()
                    if "CRITICAL ISSUES FOUND" in report_content:
                        report_msg += " ⚠️"
                
            except Exception as e:
                logger.error(f"Failed to generate data report: {e}")
                report_msg = ""
            
            # Update final status
            successful = sum(1 for r in results if 'error' not in r)
            failed = len(results) - successful
            
            if len(plugin_names) == 1:
                if successful == 1:
                    self.status_label.setText(f"Analysis complete for {symbol}{report_msg}")
                else:
                    self.status_label.setText(f"Analysis failed for {symbol}{report_msg}")
            else:
                status_msg = f"Completed: {successful}/{len(plugin_names)} plugins successful"
                if failed > 0:
                    status_msg += f" ({failed} failed)"
                status_msg += report_msg
                self.status_label.setText(status_msg)
            
            # Enable report button
            self.generate_report_button.setEnabled(True)
            
            # Auto-show single result if only one plugin was run
            if len(results) == 1 and 'error' not in results[0]:
                self._show_detailed_result(results[0])
            
            # Connect row selection to show details
            self.multi_result_viewer.result_selected.connect(self._show_detailed_result)
                
        except Exception as e:
            logger.error(f"Error running plugins: {e}")
            self.status_label.setText(f"Error: {str(e)}")
            QMessageBox.critical(self, "Analysis Error", str(e))
            
        finally:
            self.run_button.setEnabled(True)
            # Hide progress widget if single plugin
            if len(plugin_names) == 1:
                self.progress_widget.setVisible(False)
    
    def _show_detailed_result(self, result: Dict[str, Any]):
        """Show detailed view of selected result"""
        self.single_result_viewer.display_result(result)
        self.single_result_viewer.setVisible(True)
        # Adjust splitter to show both views
        self.splitter.setSizes([300, 400])
    
    def generate_data_report(self):
        """Generate data manager report manually"""
        try:
            json_file, summary_file = self.data_manager.generate_data_report()
            
            # Show message with report location
            msg = f"Data report generated:\n\nSummary: {summary_file}\nJSON: {json_file}"
            
            # Check for issues in the report
            with open(summary_file, 'r') as f:
                report_content = f.read()
                if "CRITICAL ISSUES FOUND" in report_content:
                    msg += "\n\n⚠️ Critical issues found in data requests!"
            
            QMessageBox.information(self, "Report Generated", msg)
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            QMessageBox.critical(self, "Report Error", f"Failed to generate report: {str(e)}")
    
    def clear_cache(self):
        """Clear data cache"""
        reply = QMessageBox.question(
            self, 
            "Clear Cache", 
            "Are you sure you want to clear all cached data?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.data_manager.clear_cache()
                
                # Show cache stats after clearing
                stats = self.data_manager.get_cache_stats()
                msg = f"Cache cleared successfully!\n\n"
                msg += f"Memory cache: {stats['memory_cache']['cached_items']} items\n"
                msg += f"File cache: {stats['file_cache']['total_files']} files"
                
                QMessageBox.information(self, "Cache Cleared", msg)
                
            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")
                QMessageBox.critical(self, "Cache Error", f"Failed to clear cache: {str(e)}")
            
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
            QComboBox QAbstractItemView {
                background-color: #2b2b2b;
                selection-background-color: #0d7377;
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
            QScrollArea {
                background-color: #2b2b2b;
                border: 1px solid #444;
            }
        """)