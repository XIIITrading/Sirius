"""
Backtest Dashboard with Enhanced Layout
- Header: Input controls
- Middle: Results grid (left) | Charts (right)
- Bottom: Three analysis grids (HVN, Order Block, Supply/Demand)
"""

import asyncio
import logging
import importlib
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
    QScrollArea, QTabWidget, QHeaderView
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
    "1-Min Statistical Trend",
    "5-Min Statistical Trend",
    "15-Min Statistical Trend",
    "1-Min Bid/Ask Analysis",
    "Bid/Ask Imbalance Analysis",
    "Tick Flow Analysis",
     "Bid/Ask Ratio Tracker"
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


class ChartPlaceholder(QWidget):
    """Placeholder widget for charts"""
    def __init__(self, title: str):
        super().__init__()
        layout = QVBoxLayout(self)
        label = QLabel(title)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border: 1px solid #444;
                padding: 20px;
                font-size: 14px;
                color: #888;
            }
        """)
        layout.addWidget(label)


class AnalysisGrid(QTableWidget):
    """Base class for analysis grids"""
    def __init__(self, title: str):
        super().__init__()
        self.title = title
        self.init_ui()
        
    def init_ui(self):
        """Initialize the grid UI"""
        # Set some default properties
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.horizontalHeader().setStretchLastSection(True)
        
        # Add placeholder data
        self.setColumnCount(4)
        self.setRowCount(5)
        self.setHorizontalHeaderLabels(['Column 1', 'Column 2', 'Column 3', 'Column 4'])
        
        # Add some placeholder data
        for i in range(5):
            for j in range(4):
                self.setItem(i, j, QTableWidgetItem(f"{self.title} {i},{j}"))


class BacktestDashboard(QMainWindow):
    """Main dashboard with enhanced layout"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize data manager first
        logger.info("Initializing PolygonDataManager...")
        self.data_manager = PolygonDataManager(disable_polygon_cache=True)
        
        # Pass data manager to plugin runner
        logger.info("Initializing PluginRunner with data manager...")
        self.plugin_runner = PluginRunner(data_manager=self.data_manager)
        
        # Create result viewers
        self.result_detail_viewer = ResultViewer()
        self.multi_result_viewer = MultiResultViewer()
        
        # Progress tracking
        self.progress_signals = ProgressSignals()
        self.plugin_status = {}
        self.progress_widgets = {}
        
        # Chart containers
        self.chart_widgets = {}
        
        # Get available plugins and sort them according to our order
        self.available_plugins = self._get_ordered_plugins()
        
        # Analysis grids
        self.hvn_grid = None
        self.order_block_grid = None
        self.supply_demand_grid = None
        
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
        """Initialize the UI with new layout"""
        self.setWindowTitle("Backtest Analysis System - Enhanced")
        self.setGeometry(50, 50, 1600, 1000)
        
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        
        # 1. HEADER SECTION - Input controls
        header_widget = self._create_header_section()
        main_layout.addWidget(header_widget)
        
        # 2. PROGRESS WIDGET (shown during analysis)
        self.progress_widget = self._create_progress_widget()
        self.progress_widget.setVisible(False)
        main_layout.addWidget(self.progress_widget)
        
        # 3. MIDDLE SECTION - Results and Charts
        middle_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - Results grid
        results_container = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout(results_container)
        results_layout.addWidget(self.multi_result_viewer)
        middle_splitter.addWidget(results_container)
        
        # Right side - Charts
        charts_container = self._create_charts_section()
        middle_splitter.addWidget(charts_container)
        
        # Set proportions (40% results, 60% charts)
        middle_splitter.setSizes([600, 900])
        main_layout.addWidget(middle_splitter, 3)  # Give more space to middle section
        
        # 4. BOTTOM SECTION - Analysis Grids
        bottom_widget = self._create_bottom_grids()
        main_layout.addWidget(bottom_widget, 2)  # Give less space to bottom section
        
        # 5. STATUS BAR
        status_widget = self._create_status_bar()
        main_layout.addWidget(status_widget)
        
        # Connect result selection to detail viewer
        self.multi_result_viewer.result_selected.connect(self._show_detailed_result)
        
    def _create_header_section(self) -> QWidget:
        """Create the header section with input controls"""
        header = QGroupBox("Analysis Configuration")
        header.setMaximumHeight(150)  # Increased from 120
        layout = QVBoxLayout(header)
        
        # Main controls row
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(15)
        
        # Symbol input
        symbol_group = QWidget()
        symbol_layout = QVBoxLayout(symbol_group)
        symbol_layout.setSpacing(2)
        symbol_label = QLabel("Symbol")
        symbol_label.setStyleSheet("font-weight: bold;")
        symbol_layout.addWidget(symbol_label)
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("TSLA")
        self.symbol_input.setText("TSLA")
        self.symbol_input.setMinimumHeight(30)  # Set minimum height
        self.symbol_input.setMaximumWidth(100)
        self.symbol_input.setStyleSheet("font-size: 14px; padding: 5px;")
        symbol_layout.addWidget(self.symbol_input)
        controls_layout.addWidget(symbol_group)
        
        # DateTime input
        time_group = QWidget()
        time_layout = QVBoxLayout(time_group)
        time_layout.setSpacing(2)
        time_label = QLabel("Entry Time (UTC)")
        time_label.setStyleSheet("font-weight: bold;")
        time_layout.addWidget(time_label)
        self.datetime_input = QDateTimeEdit()
        self.datetime_input.setDateTime(QDateTime.currentDateTimeUtc().addSecs(-3600))
        self.datetime_input.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.datetime_input.setCalendarPopup(True)
        self.datetime_input.setMinimumHeight(30)  # Set minimum height
        self.datetime_input.setMinimumWidth(200)  # Set minimum width for date
        self.datetime_input.setStyleSheet("font-size: 14px; padding: 5px;")
        time_layout.addWidget(self.datetime_input)
        controls_layout.addWidget(time_group)
        
        # Direction
        direction_group = QWidget()
        direction_layout = QVBoxLayout(direction_group)
        direction_layout.setSpacing(2)
        direction_label = QLabel("Direction")
        direction_label.setStyleSheet("font-weight: bold;")
        direction_layout.addWidget(direction_label)
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["LONG", "SHORT"])
        self.direction_combo.setMinimumHeight(30)  # Set minimum height
        self.direction_combo.setMaximumWidth(100)
        self.direction_combo.setStyleSheet("font-size: 14px; padding: 5px;")
        direction_layout.addWidget(self.direction_combo)
        controls_layout.addWidget(direction_group)
        
        # Plugin selection
        plugin_group = QWidget()
        plugin_layout = QVBoxLayout(plugin_group)
        plugin_layout.setSpacing(2)
        plugin_label = QLabel("Analysis Type")
        plugin_label.setStyleSheet("font-weight: bold;")
        plugin_layout.addWidget(plugin_label)
        self.plugin_combo = QComboBox()
        self.plugin_combo.setMinimumWidth(300)  # Increased from 250
        self.plugin_combo.setMinimumHeight(30)  # Set minimum height
        self.plugin_combo.setStyleSheet("font-size: 14px; padding: 5px;")
        self.plugin_combo.addItem("All Plugins")
        self.plugin_combo.insertSeparator(1)
        for plugin_name in self.available_plugins:
            self.plugin_combo.addItem(plugin_name)
        plugin_layout.addWidget(self.plugin_combo)
        controls_layout.addWidget(plugin_group)
        
        # Run button
        self.run_button = QPushButton("Run Analysis")
        self.run_button.clicked.connect(self.run_analysis)
        self.run_button.setMinimumWidth(120)
        self.run_button.setMinimumHeight(45)  # Increased from 40
        self.run_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                padding: 8px;
            }
        """)
        controls_layout.addWidget(self.run_button)
        
        controls_layout.addStretch()
        
        # Additional controls row
        extra_controls = QHBoxLayout()
        
        # Cache controls
        self.clear_cache_button = QPushButton("Clear Cache")
        self.clear_cache_button.clicked.connect(self.clear_cache)
        self.clear_cache_button.setMinimumHeight(28)
        extra_controls.addWidget(self.clear_cache_button)
        
        # Report button
        self.generate_report_button = QPushButton("Generate Report")
        self.generate_report_button.clicked.connect(self.generate_data_report)
        self.generate_report_button.setEnabled(False)
        self.generate_report_button.setMinimumHeight(28)
        extra_controls.addWidget(self.generate_report_button)
        
        extra_controls.addStretch()
        
        layout.addLayout(controls_layout)
        layout.addLayout(extra_controls)
        
        return header
    
    def _create_charts_section(self) -> QWidget:
        """Create the charts section with three visible charts in a grid"""
        charts_group = QGroupBox("Chart Analysis")
        main_layout = QVBoxLayout(charts_group)
        
        # Create a widget to hold the charts
        charts_widget = QWidget()
        charts_layout = QHBoxLayout(charts_widget)
        charts_layout.setSpacing(10)
        
        # Chart 1: Buy/Sell Ratio Chart
        buy_sell_container = QGroupBox("Buy/Sell Ratio")
        buy_sell_layout = QVBoxLayout(buy_sell_container)
        self.buy_sell_chart = ChartPlaceholder("Buy/Sell Ratio Chart")
        buy_sell_layout.addWidget(self.buy_sell_chart)
        charts_layout.addWidget(buy_sell_container, 1)  # stretch factor 1
        
        # Chart 2: Large Order Viewer
        large_order_container = QGroupBox("Large Orders")
        large_order_layout = QVBoxLayout(large_order_container)
        self.large_order_chart = ChartPlaceholder("Large Order Viewer")
        large_order_layout.addWidget(self.large_order_chart)
        charts_layout.addWidget(large_order_container, 1)  # stretch factor 1
        
        # Chart 3: Additional Analysis
        additional_container = QGroupBox("Additional Analysis")
        additional_layout = QVBoxLayout(additional_container)
        self.additional_chart = ChartPlaceholder("Additional Analysis Chart")
        additional_layout.addWidget(self.additional_chart)
        charts_layout.addWidget(additional_container, 1)  # stretch factor 1
        
        main_layout.addWidget(charts_widget)
        
        return charts_group
    
    def _create_bottom_grids(self) -> QWidget:
        """Create the bottom section with three analysis grids"""
        bottom_widget = QWidget()
        layout = QHBoxLayout(bottom_widget)
        layout.setSpacing(10)
        
        # HVN Zone Grid
        hvn_group = QGroupBox("HVN Zone Grid")
        hvn_layout = QVBoxLayout(hvn_group)
        self.hvn_grid = AnalysisGrid("HVN")
        hvn_layout.addWidget(self.hvn_grid)
        layout.addWidget(hvn_group)
        
        # Order Block Grid
        order_group = QGroupBox("Order Block Grid")
        order_layout = QVBoxLayout(order_group)
        self.order_block_grid = AnalysisGrid("Order Block")
        order_layout.addWidget(self.order_block_grid)
        layout.addWidget(order_group)
        
        # Supply and Demand Grid
        supply_group = QGroupBox("Supply & Demand Grid")
        supply_layout = QVBoxLayout(supply_group)
        self.supply_demand_grid = AnalysisGrid("S&D")
        supply_layout.addWidget(self.supply_demand_grid)
        layout.addWidget(supply_group)
        
        return bottom_widget
    
    def _create_status_bar(self) -> QWidget:
        """Create status bar widget"""
        status_widget = QWidget()
        layout = QHBoxLayout(status_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        return status_widget
        
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
        scroll.setMaximumHeight(150)
        
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
        details_label.setMinimumWidth(200)
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
            
        # Update chart if this is a chart plugin
        self._update_chart_if_applicable(plugin_name, result)
    
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
    
    def _update_chart_if_applicable(self, plugin_name: str, result: dict):
        """Update chart if plugin provides chart data"""
        display_data = result.get('display_data', {})
        chart_config = display_data.get('chart_widget')
        
        if chart_config and plugin_name == "Bid/Ask Ratio Tracker":
            # Replace placeholder with actual chart
            try:
                # Import the chart module
                module = importlib.import_module(chart_config['module'])
                ChartClass = getattr(module, chart_config['type'])
                
                # Create chart instance
                chart = ChartClass()
                chart.update_data(chart_config['data'])
                
                # Add entry marker
                if 'entry_time' in chart_config:
                    chart.add_marker(30, "Entry", "#ff0000")
                
                # Find the buy/sell container and replace its content
                buy_sell_container = self.buy_sell_chart.parent()
                if buy_sell_container:
                    # Remove placeholder
                    layout = buy_sell_container.layout()
                    layout.removeWidget(self.buy_sell_chart)
                    self.buy_sell_chart.deleteLater()
                    
                    # Add real chart
                    self.buy_sell_chart = chart
                    layout.addWidget(chart)
                    
                    # Store reference
                    self.chart_widgets['buy_sell_ratio'] = chart
                
            except Exception as e:
                logger.error(f"Error loading chart: {e}")
    
    def run_analysis(self):
        """Run the selected analysis"""
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
        
        # Run plugins asynchronously
        asyncio.create_task(self._run_plugins_async(selected_plugins, symbol, entry_time, direction))
        
    async def _run_plugins_async(self, plugin_names: List[str], symbol: str, 
                               entry_time: datetime, direction: str):
        """Run plugins asynchronously"""
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
            
            # Update final status
            successful = sum(1 for r in results if 'error' not in r)
            failed = len(results) - successful
            
            if len(plugin_names) == 1:
                if successful == 1:
                    self.status_label.setText(f"Analysis complete for {symbol}")
                else:
                    self.status_label.setText(f"Analysis failed for {symbol}")
            else:
                status_msg = f"Completed: {successful}/{len(plugin_names)} plugins successful"
                if failed > 0:
                    status_msg += f" ({failed} failed)"
                self.status_label.setText(status_msg)
            
            # Enable report button
            self.generate_report_button.setEnabled(True)
                
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
        # Create a dialog or update a detail panel
        # For now, we'll log it
        logger.info(f"Selected result: {result.get('plugin_name')}")
    
    def generate_data_report(self):
        """Generate data manager report"""
        try:
            json_file, summary_file = self.data_manager.generate_data_report()
            
            # Show message with report location
            msg = f"Data report generated:\n\nSummary: {summary_file}\nJSON: {json_file}"
            
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
                font-size: 14px;
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
                font-size: 13px;
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
                padding: 6px;
                border-radius: 3px;
                font-size: 13px;
            }
            QComboBox::drop-down {
                width: 20px;
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
            QTabWidget::pane {
                border: 1px solid #444;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #2b2b2b;
                padding: 8px 16px;
                margin-right: 2px;
                border: 1px solid #444;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background-color: #0d7377;
            }
            QTabBar::tab:hover {
                background-color: #323232;
            }
            QLabel {
                font-size: 13px;
            }
        """)