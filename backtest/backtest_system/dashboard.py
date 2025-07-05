# backtest/backtest_system/dashboard.py
"""
Backtest Dashboard with Enhanced Layout
- Header: Input controls
- Middle: Results grid (left) | Charts (right)
- Bottom: Three analysis grids (HVN, Order Block, Supply/Demand)
- Footer: Status bar with progress tracking
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
    "Bid/Ask Ratio Tracker",
    "Impact Success",
    "Large Orders Grid"
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
        self.data_manager = PolygonDataManager()
        
        # Pass data manager to plugin runner
        logger.info("Initializing PluginRunner with data manager...")
        self.plugin_runner = PluginRunner(data_manager=self.data_manager)
        
        # Create result viewers
        self.result_detail_viewer = ResultViewer()
        self.multi_result_viewer = MultiResultViewer()
        
        # Progress tracking
        self.progress_signals = ProgressSignals()
        self.plugin_status = {}
        self.current_plugin_progress = None  # Track current plugin being executed
        
        # Chart containers - updated for new layout
        self.chart_widgets = {}
        self.large_orders_grid_chart = None
        self.buy_sell_chart = None
        self.impact_success_chart = None
        
        # Store for chart data that might get corrupted by signals
        self.pending_chart_updates = {}
        self.pending_grid_updates = {}
        
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
        
        # 2. MIDDLE SECTION - Results and Charts (removed progress widget)
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
        
        # 3. BOTTOM SECTION - Analysis Grids
        bottom_widget = self._create_bottom_grids()
        main_layout.addWidget(bottom_widget, 2)  # Give less space to bottom section
        
        # 4. STATUS BAR with progress
        status_widget = self._create_status_bar_with_progress()
        main_layout.addWidget(status_widget)
        
        # Connect result selection to detail viewer
        self.multi_result_viewer.result_selected.connect(self._show_detailed_result)
        
    def _create_header_section(self) -> QWidget:
        """Create the header section with input controls"""
        header = QGroupBox("Analysis Configuration")
        header.setMaximumHeight(150)
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
        self.symbol_input.setMinimumHeight(30)
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
        self.datetime_input.setMinimumHeight(30)
        self.datetime_input.setMinimumWidth(200)
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
        self.direction_combo.setMinimumHeight(30)
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
        self.plugin_combo.setMinimumWidth(300)
        self.plugin_combo.setMinimumHeight(30)
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
        self.run_button.setMinimumHeight(45)
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
        
        # Chart 1: Large Orders Grid (LEFT)
        large_orders_container = QGroupBox("Large Orders Grid")
        large_orders_layout = QVBoxLayout(large_orders_container)
        self.large_orders_grid_chart = ChartPlaceholder("Large Orders Grid")
        large_orders_layout.addWidget(self.large_orders_grid_chart)
        charts_layout.addWidget(large_orders_container, 1)
        
        # Chart 2: Buy/Sell Ratio Chart (MIDDLE)
        buy_sell_container = QGroupBox("Buy/Sell Ratio")
        buy_sell_layout = QVBoxLayout(buy_sell_container)
        self.buy_sell_chart = ChartPlaceholder("Buy/Sell Ratio Chart")
        buy_sell_layout.addWidget(self.buy_sell_chart)
        charts_layout.addWidget(buy_sell_container, 1)
        
        # Chart 3: Impact Success (RIGHT)
        impact_container = QGroupBox("Impact Success")
        impact_layout = QVBoxLayout(impact_container)
        self.impact_success_chart = ChartPlaceholder("Impact Success Chart")
        impact_layout.addWidget(self.impact_success_chart)
        charts_layout.addWidget(impact_container, 1)
        
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
    
    def _create_status_bar_with_progress(self) -> QWidget:
        """Create status bar widget with integrated progress tracking"""
        status_widget = QWidget()
        status_widget.setMaximumHeight(50)
        layout = QHBoxLayout(status_widget)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setMinimumWidth(200)
        layout.addWidget(self.status_label)
        
        # Plugin name label (hidden by default)
        self.plugin_name_label = QLabel("")
        self.plugin_name_label.setMinimumWidth(250)
        self.plugin_name_label.setStyleSheet("font-weight: bold; color: #0d7377;")
        self.plugin_name_label.setVisible(False)
        layout.addWidget(self.plugin_name_label)
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumWidth(200)
        self.progress_bar.setMaximumHeight(25)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Progress details label (hidden by default)
        self.progress_details_label = QLabel("")
        self.progress_details_label.setMinimumWidth(300)
        self.progress_details_label.setVisible(False)
        layout.addWidget(self.progress_details_label)
        
        layout.addStretch()
        
        return status_widget
        
    def on_plugin_started(self, plugin_name: str):
        """Handle plugin started signal"""
        self.plugin_status[plugin_name] = PluginStatus.RUNNING
        self.current_plugin_progress = plugin_name
        
        # Show progress elements
        self.plugin_name_label.setText(f"Running: {plugin_name}")
        self.plugin_name_label.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.progress_details_label.setText("Initializing...")
        self.progress_details_label.setVisible(True)
    
    def on_plugin_progress(self, plugin_name: str, percentage: int, message: str):
        """Handle plugin progress update"""
        if plugin_name == self.current_plugin_progress:
            self.progress_bar.setValue(percentage)
            self.progress_details_label.setText(message)
    
    def on_plugin_completed(self, plugin_name: str, result: dict):
        """Handle plugin completion"""
        logger.info(f"=== on_plugin_completed called for {plugin_name} ===")
        
        self.plugin_status[plugin_name] = PluginStatus.COMPLETED
        
        # Update progress if this is the current plugin
        if plugin_name == self.current_plugin_progress:
            self.progress_bar.setValue(100)
            
            # Show summary
            signal_dir = result.get('signal', {}).get('direction', 'N/A')
            confidence = result.get('signal', {}).get('confidence', 0)
            strength = result.get('signal', {}).get('strength', 0)
            self.progress_details_label.setText(
                f"Complete - {signal_dir} (Strength: {strength:.0f}%, Confidence: {confidence:.0f}%)"
            )
            
        # Check if we have pending chart data for this plugin
        if plugin_name in self.pending_chart_updates:
            logger.info(f"Found pending chart update for {plugin_name}")
            result_with_chart = result.copy()
            if 'display_data' not in result_with_chart:
                result_with_chart['display_data'] = {}
            result_with_chart['display_data']['chart_widget'] = self.pending_chart_updates.pop(plugin_name)
            self._update_display_if_applicable(plugin_name, result_with_chart)
        elif plugin_name in self.pending_grid_updates:
            logger.info(f"Found pending grid update for {plugin_name}")
            result_with_grid = result.copy()
            if 'display_data' not in result_with_grid:
                result_with_grid['display_data'] = {}
            result_with_grid['display_data']['grid_widget'] = self.pending_grid_updates.pop(plugin_name)
            self._update_display_if_applicable(plugin_name, result_with_grid)
        else:
            self._update_display_if_applicable(plugin_name, result)
    
    def on_plugin_error(self, plugin_name: str, error_message: str):
        """Handle plugin error"""
        self.plugin_status[plugin_name] = PluginStatus.ERROR
        
        if plugin_name == self.current_plugin_progress:
            self.progress_details_label.setText(
                f"Error: {error_message[:50]}..." if len(error_message) > 50 else f"Error: {error_message}"
            )
            self.progress_details_label.setStyleSheet("color: #ff4444;")
    
    def _hide_progress_elements(self):
        """Hide all progress elements in the status bar"""
        self.plugin_name_label.setVisible(False)
        self.progress_bar.setVisible(False)
        self.progress_details_label.setVisible(False)
        self.progress_details_label.setStyleSheet("")  # Reset style
        self.current_plugin_progress = None
    
    def _update_display_if_applicable(self, plugin_name: str, result: dict):
        """Update chart or grid if plugin provides display data"""
        logger.info(f"Checking display update for {plugin_name}")
        
        display_data = result.get('display_data', {})
        
        # Check for chart widget
        chart_config = display_data.get('chart_widget')
        if chart_config:
            self._handle_chart_display(plugin_name, chart_config)
        
        # Check for grid widget
        grid_config = display_data.get('grid_widget')
        if grid_config:
            self._handle_grid_display(plugin_name, grid_config)
    
    def _handle_grid_display(self, plugin_name: str, grid_config: dict):
        """Handle grid display for plugins that provide grid data"""
        logger.info(f"Found grid config for {plugin_name}")
        
        # Mapping for grid plugins
        target_mapping = {
            "Large Orders Grid": ('large_orders_grid_chart', 'large_orders_grid')
        }
        
        if plugin_name not in target_mapping:
            logger.warning(f"No grid target mapping for plugin: {plugin_name}")
            return
        
        target_attr, grid_key = target_mapping[plugin_name]
        
        try:
            # Get the target container
            target_placeholder = getattr(self, target_attr)
            target_container = target_placeholder.parent()
            
            if not target_container:
                logger.error(f"No parent container for {target_attr}")
                return
            
            # Import the grid module and class
            module = importlib.import_module(grid_config['module'])
            GridClass = getattr(module, grid_config['type'])
            
            logger.info(f"Successfully imported {GridClass.__name__} from {grid_config['module']}")
            
            # Create grid instance
            grid = GridClass(grid_config.get('config', {}))
            
            # Update grid with data
            grid_data = grid_config.get('data', [])
            if grid_data:
                grid.update_from_data(grid_data)
                logger.info(f"Updated grid with {len(grid_data)} items")
            
            # Replace placeholder with real grid
            layout = target_container.layout()
            if not layout:
                logger.error("Target container has no layout")
                return
            
            # Remove old placeholder
            layout.removeWidget(target_placeholder)
            target_placeholder.deleteLater()
            
            # Add real grid
            setattr(self, target_attr, grid)
            layout.addWidget(grid)
            
            # Store reference
            self.chart_widgets[grid_key] = grid
            
            logger.info(f"✅ Successfully loaded {plugin_name} grid")
            
        except Exception as e:
            logger.error(f"Error loading grid for {plugin_name}: {e}")
            import traceback
            traceback.print_exc()
    
    def _handle_chart_display(self, plugin_name: str, chart_config: dict):
        """Handle chart display for plugins that provide chart data"""
        logger.info(f"Found chart config for {plugin_name}")
        
        # Mapping for chart plugins
        target_mapping = {
            "Bid/Ask Ratio Tracker": ('buy_sell_chart', 'buy_sell_ratio'),
            "Impact Success": ('impact_success_chart', 'impact_success')
        }
        
        if plugin_name not in target_mapping:
            logger.warning(f"No chart target mapping for plugin: {plugin_name}")
            return
        
        target_chart_attr, chart_key = target_mapping[plugin_name]
        
        # Get the target container
        try:
            target_placeholder = getattr(self, target_chart_attr)
            target_container = target_placeholder.parent()
            
            if not target_container:
                logger.error(f"No parent container for {target_chart_attr}")
                return
                
        except AttributeError as e:
            logger.error(f"Could not find chart attribute {target_chart_attr}: {e}")
            return
        
        try:
            # Import the chart module and class
            module = importlib.import_module(chart_config['module'])
            ChartClass = getattr(module, chart_config['type'])
            
            logger.info(f"Successfully imported {ChartClass.__name__} from {chart_config['module']}")
            
            # Create chart instance
            chart = ChartClass()
            
            # Check if data exists
            chart_data = chart_config.get('data', [])
            logger.info(f"Updating chart with {len(chart_data)} data points")
            
            # Pass the data directly to the chart
            if chart_data and hasattr(chart, 'update_from_data'):
                chart.update_from_data(chart_data)
                logger.info("Called update_from_data successfully")
            elif chart_data and hasattr(chart, 'update_data'):
                chart.update_data(chart_data)
                logger.info("Called update_data successfully")
            else:
                logger.warning(f"No data or update method not found")
            
            # Add entry marker if the chart supports it
            if chart_config.get('entry_time'):
                if hasattr(chart, 'add_entry_marker'):
                    chart.add_entry_marker(chart_config['entry_time'])
                elif hasattr(chart, 'add_marker'):
                    chart.add_marker(30, "Entry", "#ff0000")
                logger.info("Added entry marker")
            
            # Replace placeholder with real chart
            layout = target_container.layout()
            if not layout:
                logger.error("Target container has no layout")
                return
                
            # Remove old chart
            layout.removeWidget(target_placeholder)
            target_placeholder.deleteLater()
            
            # Add real chart
            setattr(self, target_chart_attr, chart)
            layout.addWidget(chart)
            
            # Store reference
            self.chart_widgets[chart_key] = chart
            
            logger.info(f"✅ Successfully loaded {plugin_name} chart")
            
        except Exception as e:
            logger.error(f"Error loading chart for {plugin_name}: {e}")
            import traceback
            traceback.print_exc()
    
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
        
        # Reset plugin status and pending updates
        self.plugin_status.clear()
        self.pending_chart_updates.clear()
        self.pending_grid_updates.clear()
        
        # Update status
        plugin_text = f"{len(selected_plugins)} plugins" if len(selected_plugins) > 1 else selected_plugins[0]
        self.status_label.setText(f"Running {plugin_text} for {symbol}...")
        self.run_button.setEnabled(False)
        self.generate_report_button.setEnabled(False)
        
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
                        f"Running plugin {i+1}/{len(plugin_names)}: {plugin_name}"
                    )
                
                try:
                    # Run the plugin with progress callback
                    result = await self.plugin_runner.run_single_plugin(
                        plugin_name, symbol, entry_time, direction,
                        progress_callback=lambda pct, msg: progress_callback(plugin_name, pct, msg)
                    )
                    
                    results.append(result)
                    
                    # Add to multi-viewer immediately
                    self.multi_result_viewer.add_result(result)
                    
                    # Check if result has chart data and store it separately
                    if 'display_data' in result and 'chart_widget' in result['display_data']:
                        chart_config = result['display_data']['chart_widget']
                        if chart_config and 'data' in chart_config and len(chart_config['data']) > 0:
                            logger.info(f"Storing chart data for {plugin_name} ({len(chart_config['data'])} points)")
                            self.pending_chart_updates[plugin_name] = chart_config
                    
                    # Check if result has grid data and store it separately
                    if 'display_data' in result and 'grid_widget' in result['display_data']:
                        grid_config = result['display_data']['grid_widget']
                        if grid_config and 'data' in grid_config and len(grid_config['data']) > 0:
                            logger.info(f"Storing grid data for {plugin_name} ({len(grid_config['data'])} items)")
                            self.pending_grid_updates[plugin_name] = grid_config
                    
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
                    self.multi_result_viewer.add_result(error_result)
                
                # Small delay between plugins to allow UI updates
                await asyncio.sleep(0.1)
            
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
            
            # Hide progress elements after a short delay
            await asyncio.sleep(2)
            self._hide_progress_elements()
                
        except Exception as e:
            logger.error(f"Error running plugins: {e}")
            self.status_label.setText(f"Error: {str(e)}")
            QMessageBox.critical(self, "Analysis Error", str(e))
            self._hide_progress_elements()
            
        finally:
            self.run_button.setEnabled(True)
    
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