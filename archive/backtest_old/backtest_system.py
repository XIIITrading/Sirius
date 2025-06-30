# backtest/backtest_system.py
"""
Main entry point for the modular backtest system.
Provides command-line and GUI interfaces for running backtests.
"""

import sys
import os
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import argparse

# CRITICAL: Load .env file BEFORE any imports that need credentials
from dotenv import load_dotenv
# Load .env from parent directory (Sirius root)
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# Verify environment variables are loaded
print(f"Loading .env from: {env_path}")
print(f"SUPABASE_URL exists: {'SUPABASE_URL' in os.environ}")
print(f"POLYGON_API_KEY exists: {'POLYGON_API_KEY' in os.environ}")

# Fix imports for running from backtest directory
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QDateTimeEdit, QComboBox,
    QGroupBox, QSplitter, QTextEdit, QProgressBar, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
    QDialog, QDialogButtonBox, QSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDateTime
from PyQt6.QtGui import QFont, QColor

# Import core components with relative imports
from core.engine import BacktestEngine, BacktestConfig
from data.polygon_data_manager import PolygonDataManager
from core.result_store import BacktestResultStore, BacktestResult
from storage.supabase_storage import prepare_bars_for_storage

# Import plugin loader
from plugins.plugin_loader import PluginLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BacktestWorker(QThread):
    """Worker thread for running backtests"""
    
    # Signals
    started = pyqtSignal()
    progress = pyqtSignal(int, str)
    result_ready = pyqtSignal(dict, object)  # Changed to pass both dict and BacktestResult
    error = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, engine: BacktestEngine):
        super().__init__()
        self.engine = engine
        self.config = None
        
    def set_config(self, config: BacktestConfig):
        """Set backtest configuration"""
        self.config = config
        
    def run(self):
        """Run the backtest"""
        self.started.emit()
        
        try:
            # Create event loop for async operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run backtest
            self.progress.emit(10, "Starting backtest...")
            result = loop.run_until_complete(self.engine.run_backtest(self.config))
            
            # Convert result to dict for display
            result_dict = result.to_dict()
            result_dict['summary'] = result.get_summary()
            
            # Pass both the dict and the full result object
            self.result_ready.emit(result_dict, result)
            self.progress.emit(100, "Backtest complete")
            
        except Exception as e:
            logger.error(f"Backtest error: {e}", exc_info=True)
            self.error.emit(str(e))
            
        finally:
            self.finished.emit()
            loop.close()


class PushWorker(QThread):
    """Worker thread for pushing results to Supabase"""
    
    finished = pyqtSignal(str)  # Emits UID on success
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, engine: BacktestEngine, result: BacktestResult):
        super().__init__()
        self.engine = engine
        self.result = result
        
    def run(self):
        """Push the result to Supabase"""
        try:
            # Create event loop for async operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            self.progress.emit("Generating UID...")
            
            # Generate UID
            uid = self.engine.supabase_storage.generate_uid(self.result.config)
            
            self.progress.emit(f"Checking if {uid} exists...")
            
            # Check if already exists
            exists = loop.run_until_complete(
                self.engine.supabase_storage.check_uid_exists(uid)
            )
            
            if exists:
                self.progress.emit(f"Overwriting existing UID: {uid}")
            
            self.progress.emit("Preparing data for storage...")
            
            # Get the bars for storage
            if self.result.bars_for_storage is None:
                # Prepare bars if not already done
                if self.result.historical_bars is not None and not self.result.forward_data.empty:
                    bars_df = prepare_bars_for_storage(
                        self.result.historical_bars,
                        self.result.forward_data,
                        self.result.config.entry_time
                    )
                else:
                    raise Exception("Bar data not available for storage")
            else:
                bars_df = self.result.bars_for_storage
            
            self.progress.emit("Storing to Supabase...")
            
            # Store to Supabase
            storage_result = loop.run_until_complete(
                self.engine.supabase_storage.store_backtest_data(
                    uid=uid,
                    config=self.result.config,
                    bars_df=bars_df,
                    results=self.result
                )
            )
            
            if storage_result.success:
                self.finished.emit(uid)
            else:
                self.error.emit(storage_result.error or "Unknown storage error")
                
        except Exception as e:
            self.error.emit(str(e))
        finally:
            loop.close()


class MarketStructureDebugDialog(QDialog):
    """Dialog for market structure debug settings"""
    
    def __init__(self, parent=None, symbol="", entry_time=None):
        super().__init__(parent)
        self.setWindowTitle("Market Structure Debug Settings")
        self.setModal(True)
        self.resize(400, 300)
        
        layout = QVBoxLayout(self)
        
        # Symbol
        symbol_layout = QHBoxLayout()
        symbol_layout.addWidget(QLabel("Symbol:"))
        self.symbol_input = QLineEdit(symbol)
        symbol_layout.addWidget(self.symbol_input)
        layout.addLayout(symbol_layout)
        
        # Entry time
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Entry Time:"))
        self.time_input = QDateTimeEdit()
        if entry_time:
            self.time_input.setDateTime(QDateTime.fromSecsSinceEpoch(int(entry_time.timestamp())))
        else:
            self.time_input.setDateTime(QDateTime.currentDateTimeUtc())
        self.time_input.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.time_input.setCalendarPopup(True)
        time_layout.addWidget(self.time_input)
        layout.addLayout(time_layout)
        
        # Lookback hours
        lookback_layout = QHBoxLayout()
        lookback_layout.addWidget(QLabel("Lookback Hours:"))
        self.lookback_spin = QSpinBox()
        self.lookback_spin.setMinimum(1)
        self.lookback_spin.setMaximum(24)
        self.lookback_spin.setValue(2)
        lookback_layout.addWidget(self.lookback_spin)
        layout.addLayout(lookback_layout)
        
        # Fractal length
        fractal_layout = QHBoxLayout()
        fractal_layout.addWidget(QLabel("Fractal Length:"))
        self.fractal_spin = QSpinBox()
        self.fractal_spin.setMinimum(2)
        self.fractal_spin.setMaximum(10)
        self.fractal_spin.setValue(5)
        fractal_layout.addWidget(self.fractal_spin)
        layout.addLayout(fractal_layout)
        
        # Timeframe selection
        timeframe_layout = QHBoxLayout()
        timeframe_layout.addWidget(QLabel("Timeframe:"))
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(["1-Min", "5-Min", "15-Min"])
        timeframe_layout.addWidget(self.timeframe_combo)
        layout.addLayout(timeframe_layout)
        
        # Save results checkbox
        self.save_checkbox = QCheckBox("Save debug results to file")
        self.save_checkbox.setChecked(False)
        layout.addWidget(self.save_checkbox)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class BacktestDashboard(QMainWindow):
    """Main dashboard window for backtesting"""
    
    def __init__(self):
        super().__init__()
        self.engine = None
        self.worker = None
        self.registered_adapters = []  # Track registered adapter names
        self.last_result = None  # Store the last BacktestResult object
        self.last_result_dict = None  # Store the dict version
        self.push_worker = None
        self.plugin_loader = None  # Plugin loader
        self.init_engine()
        self.init_ui()
        self.apply_dark_theme()
        
    def init_engine(self):
        """Initialize backtest engine with plugin system"""
        try:
            # Load plugins first
            self.plugin_loader = PluginLoader()
            plugins = self.plugin_loader.load_all_plugins()
            plugin_registry = self.plugin_loader.get_registry()
            
            # Create data manager
            self.data_manager = PolygonDataManager()
            
            # Create engine with plugin registry
            self.engine = BacktestEngine(
                data_manager=self.data_manager,
                plugin_registry=plugin_registry
            )
            logger.info("Backtest engine initialized with plugin system")
            
        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            # Show error dialog
            QMessageBox.critical(None, "Initialization Error", 
                            f"Failed to initialize backtest engine:\n{str(e)}\n\n"
                            "Please check your .env file has SUPABASE_URL, SUPABASE_KEY, and POLYGON_API_KEY")
            raise
        
        # Register adapters from plugins
        try:
            adapter_configs = self.plugin_loader.get_adapter_configs()
            
            for adapter_name, config in adapter_configs.items():
                # Create adapter instance
                adapter_class = config['adapter_class']
                adapter_config = config['adapter_config']
                
                adapter = adapter_class(**adapter_config)
                self.engine.register_adapter(adapter_name, adapter)
                self.registered_adapters.append(adapter_name)
            
            logger.info(f"Registered {len(self.registered_adapters)} adapters from plugins")
            
        except Exception as e:
            logger.error(f"Failed to register adapters: {e}")
            
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("Modular Backtest System - Polygon Data")
        self.setGeometry(100, 100, 1400, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Top controls
        controls_widget = self._create_controls()
        main_layout.addWidget(controls_widget)
        
        # Main content area
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Results
        results_widget = self._create_results_panel()
        self.splitter.addWidget(results_widget)
        
        # Right panel - Chart placeholder
        chart_widget = self._create_chart_panel()
        self.splitter.addWidget(chart_widget)
        
        self.splitter.setSizes([700, 700])
        main_layout.addWidget(self.splitter)
        
        # Status bar
        self.status_bar = QProgressBar()
        self.status_bar.setVisible(False)
        main_layout.addWidget(self.status_bar)
        
        self.status_label = QLabel("Ready to run backtest")
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
        
        # Adapter selection
        layout.addWidget(QLabel("Calculation:"))
        self.adapter_combo = QComboBox()
        # Add "All Calculations" as first option
        self.adapter_combo.addItem("All Calculations")
        # Then add individual adapters
        self.adapter_combo.addItems(self.registered_adapters)
        layout.addWidget(self.adapter_combo)
        
        # Store to Supabase checkbox
        self.store_supabase_checkbox = QCheckBox("Auto-store to Supabase")
        self.store_supabase_checkbox.setChecked(False)  # Default to off
        layout.addWidget(self.store_supabase_checkbox)
        
        # Debug mode checkbox
        self.debug_checkbox = QCheckBox("Debug Mode")
        self.debug_checkbox.setChecked(False)
        self.debug_checkbox.setToolTip("Enable detailed logging for troubleshooting")
        layout.addWidget(self.debug_checkbox)
        
        # Run button
        self.run_button = QPushButton("Run Backtest")
        self.run_button.clicked.connect(self.run_backtest)
        layout.addWidget(self.run_button)
        
        # Push to Supabase button
        self.push_button = QPushButton("Push to Supabase")
        self.push_button.clicked.connect(self.push_to_supabase)
        self.push_button.setEnabled(False)  # Disabled until we have results
        self.push_button.setStyleSheet("""
            QPushButton:disabled {
                background-color: #4a4a4a;
                color: #888888;
            }
            QPushButton:enabled {
                background-color: #14a085;
            }
            QPushButton:enabled:hover {
                background-color: #1abc9c;
            }
        """)
        layout.addWidget(self.push_button)
        
        # Debug button
        self.debug_button = QPushButton("Debug Market Structure")
        self.debug_button.clicked.connect(self.debug_market_structure)
        self.debug_button.setStyleSheet("""
            QPushButton {
                background-color: #e67e22;
            }
            QPushButton:hover {
                background-color: #f39c12;
            }
        """)
        layout.addWidget(self.debug_button)
        
        # Cache stats button
        self.cache_button = QPushButton("Cache Stats")
        self.cache_button.clicked.connect(self.show_cache_stats)
        layout.addWidget(self.cache_button)
        
        layout.addStretch()
        
        return widget
        
    def _create_results_panel(self) -> QWidget:
        """Create results display panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Summary group
        summary_group = QGroupBox("Backtest Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(2)
        self.summary_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.summary_table.horizontalHeader().setStretchLastSection(True)
        summary_layout.addWidget(self.summary_table)
        
        layout.addWidget(summary_group)
        
        # Signals group
        signals_group = QGroupBox("Entry Signals")
        signals_layout = QVBoxLayout(signals_group)
        
        self.signals_table = QTableWidget()
        self.signals_table.setColumnCount(5)
        self.signals_table.setHorizontalHeaderLabels([
            "Calculation", "Direction", "Strength", "Confidence", "Metadata"
        ])
        self.signals_table.horizontalHeader().setStretchLastSection(True)
        signals_layout.addWidget(self.signals_table)
        
        layout.addWidget(signals_group)
        
        # Aggregated Signal group (shown when running all calculations)
        self.aggregated_group = QGroupBox("Aggregated Signal (Point & Call)")
        aggregated_layout = QVBoxLayout(self.aggregated_group)
        
        self.aggregated_text = QTextEdit()
        self.aggregated_text.setReadOnly(True)
        self.aggregated_text.setMaximumHeight(150)
        aggregated_layout.addWidget(self.aggregated_text)
        
        layout.addWidget(self.aggregated_group)
        self.aggregated_group.setVisible(False)  # Hidden by default
        
        return widget
        
    def _create_chart_panel(self) -> QWidget:
        """Create chart panel (placeholder)"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        chart_group = QGroupBox("Price Chart")
        chart_layout = QVBoxLayout(chart_group)
        
        # Placeholder for chart
        placeholder = QLabel("Chart visualization will be implemented here")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("QLabel { background-color: #2b2b2b; padding: 20px; }")
        chart_layout.addWidget(placeholder)
        
        layout.addWidget(chart_group)
        
        # Forward analysis
        analysis_group = QGroupBox("Forward Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setMaximumHeight(200)
        analysis_layout.addWidget(self.analysis_text)
        
        layout.addWidget(analysis_group)
        
        return widget
        
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
            QLineEdit, QComboBox, QDateTimeEdit, QSpinBox {
                background-color: #2b2b2b;
                border: 1px solid #444;
                padding: 5px;
                border-radius: 3px;
            }
            QCheckBox {
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #444;
                background-color: #2b2b2b;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                background-color: #0d7377;
            }
            QTableWidget {
                background-color: #2b2b2b;
                alternate-background-color: #323232;
                gridline-color: #444;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #323232;
                padding: 5px;
                border: 1px solid #444;
            }
            QTextEdit {
                background-color: #2b2b2b;
                border: 1px solid #444;
            }
            QProgressBar {
                border: 1px solid #444;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0d7377;
                border-radius: 3px;
            }
        """)
        
    def run_backtest(self):
        """Run the backtest"""
        # Get parameters
        symbol = self.symbol_input.text().strip().upper()
        if not symbol:
            QMessageBox.warning(self, "Input Error", "Please enter a symbol")
            return
            
        # Get datetime in UTC
        entry_time = self.datetime_input.dateTime().toPyDateTime()
        entry_time = entry_time.replace(tzinfo=timezone.utc)
        
        direction = self.direction_combo.currentText()
        selected_adapter = self.adapter_combo.currentText()
        store_to_supabase = self.store_supabase_checkbox.isChecked()
        
        # Enable debug mode if checked
        if self.debug_checkbox.isChecked():
            self.engine.enable_debug_mode()
            # Also set up console logging
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            
            # Get root logger and set level
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.DEBUG)
            
            # Remove existing handlers to avoid duplicates
            for handler in root_logger.handlers[:]:
                if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                    root_logger.removeHandler(handler)
            
            root_logger.addHandler(console_handler)
        else:
            # Reset logging level
            logging.getLogger().setLevel(logging.INFO)
        
        # Determine which calculations to run
        if selected_adapter == "All Calculations":
            # Empty list means run all registered adapters
            enabled_calculations = []
            self.aggregated_group.setVisible(True)
        else:
            # Run only the selected adapter
            enabled_calculations = [selected_adapter]
            self.aggregated_group.setVisible(False)
        
        # Create config with Supabase storage option
        config = BacktestConfig(
            symbol=symbol,
            entry_time=entry_time,
            direction=direction,
            historical_lookback_hours=2,
            forward_bars=60,
            enabled_calculations=enabled_calculations,
            store_to_supabase=store_to_supabase
        )
        
        # Clear previous results
        self.clear_results()
        
        # Update status
        status_msg = f"Running "
        if selected_adapter == "All Calculations":
            status_msg += f"all {len(self.registered_adapters)} calculations"
        else:
            status_msg += f"{selected_adapter}"
        
        if store_to_supabase:
            status_msg += " (will auto-store to Supabase)"
        
        if self.debug_checkbox.isChecked():
            status_msg += " [DEBUG MODE]"
        
        self.status_label.setText(status_msg + "...")
        
        # Disable controls
        self.run_button.setEnabled(False)
        self.status_bar.setVisible(True)
        
        # Create and start worker
        self.worker = BacktestWorker(self.engine)
        self.worker.set_config(config)
        
        # Connect signals
        self.worker.progress.connect(self.update_progress)
        self.worker.result_ready.connect(self.display_results)
        self.worker.error.connect(self.handle_error)
        self.worker.finished.connect(self.on_finished)
        
        # Start
        self.worker.start()
        
    def clear_results(self):
        """Clear all result displays"""
        self.summary_table.setRowCount(0)
        self.signals_table.setRowCount(0)
        self.analysis_text.clear()
        self.aggregated_text.clear()
        self.last_result = None
        self.last_result_dict = None
        self.push_button.setEnabled(False)
        
    def update_progress(self, value: int, message: str):
        """Update progress bar and status"""
        self.status_bar.setValue(value)
        self.status_label.setText(message)
        
    def display_results(self, results_dict: Dict, result_obj: BacktestResult):
        """Display backtest results"""
        # Store both versions
        self.last_result_dict = results_dict
        self.last_result = result_obj
        
        # Enable push button if we have results and Supabase is available
        if self.engine.supabase_enabled and self.last_result:
            self.push_button.setEnabled(True)
            self.push_button.setToolTip("Push these results to Supabase")
        else:
            self.push_button.setEnabled(False)
            if not self.engine.supabase_enabled:
                self.push_button.setToolTip("Supabase storage not available")
        
        # Display summary
        summary = results_dict.get('summary', {})
        self.summary_table.setRowCount(len(summary))
        
        for i, (key, value) in enumerate(summary.items()):
            self.summary_table.setItem(i, 0, QTableWidgetItem(str(key)))
            self.summary_table.setItem(i, 1, QTableWidgetItem(str(value)))
            
        # Display signals
        signals = results_dict.get('entry_signals', [])
        self.signals_table.setRowCount(len(signals))
        
        for i, signal in enumerate(signals):
            self.signals_table.setItem(i, 0, QTableWidgetItem(signal['name']))
            self.signals_table.setItem(i, 1, QTableWidgetItem(signal['direction']))
            self.signals_table.setItem(i, 2, QTableWidgetItem(f"{signal['strength']:.1f}"))
            self.signals_table.setItem(i, 3, QTableWidgetItem(f"{signal['confidence']:.1f}"))
            
            # Format metadata for display
            metadata = signal.get('metadata', {})
            if 'reason' in metadata:
                metadata_str = metadata['reason']
            else:
                # Show key metrics from metadata
                key_metrics = []
                if 'ema_9' in metadata:
                    key_metrics.append(f"EMA9: {metadata['ema_9']:.2f}")
                if 'ema_21' in metadata:
                    key_metrics.append(f"EMA21: {metadata['ema_21']:.2f}")
                if 'structure_type' in metadata:
                    key_metrics.append(f"Type: {metadata['structure_type']}")
                if 'last_high_fractal' in metadata and metadata['last_high_fractal']:
                    key_metrics.append(f"H: {metadata['last_high_fractal']:.2f}")
                if 'last_low_fractal' in metadata and metadata['last_low_fractal']:
                    key_metrics.append(f"L: {metadata['last_low_fractal']:.2f}")
                metadata_str = ", ".join(key_metrics) if key_metrics else str(metadata)
                
            self.signals_table.setItem(i, 4, QTableWidgetItem(metadata_str))
        
        # Display aggregated signal if running all calculations
        if self.adapter_combo.currentText() == "All Calculations":
            aggregated = results_dict.get('aggregated_signal', {})
            if aggregated:
                agg_text = f"Consensus Direction: {aggregated.get('consensus_direction', 'N/A')}\n"
                agg_text += f"Agreement Score: {aggregated.get('agreement_score', 0):.1f}%\n"
                agg_text += f"Vote Breakdown: {aggregated.get('vote_breakdown', {})}\n"
                agg_text += f"Average Strength: {aggregated.get('average_strength', 0):.1f}\n"
                agg_text += f"Average Confidence: {aggregated.get('average_confidence', 0):.1f}\n"
                agg_text += f"Participating Calculations: {aggregated.get('participating_calculations', 0)}/{aggregated.get('total_calculations', 0)}"
                self.aggregated_text.setText(agg_text)
        
        # Display forward analysis
        forward = results_dict.get('forward_analysis', {})
        analysis_text = f"Entry Price: ${forward.get('entry_price', 0):.2f}\n"
        analysis_text += f"Exit Price: ${forward.get('exit_price', 0):.2f}\n"
        analysis_text += f"P&L: {forward.get('final_pnl', 0):.2f}%\n"
        analysis_text += f"Max Favorable: {forward.get('max_favorable_move', 0):.2f}%\n"
        analysis_text += f"Max Adverse: {forward.get('max_adverse_move', 0):.2f}%\n"
        
        # Add signal accuracy if available
        accuracy = forward.get('signal_accuracy', {})
        if accuracy:
            analysis_text += f"\nSignal Accuracy:\n"
            analysis_text += f"Consensus Matched User: {'Yes' if accuracy.get('consensus_matched_user') else 'No'}\n"
            analysis_text += f"Trade Profitable: {'Yes' if accuracy.get('profitable') else 'No'}\n"
            analysis_text += f"Signal Aligned with Outcome: {'Yes' if accuracy.get('signal_aligned_with_outcome') else 'No'}"
            
        self.analysis_text.setText(analysis_text)
        
    def push_to_supabase(self):
        """Push the last backtest result to Supabase"""
        if not self.last_result:
            QMessageBox.warning(self, "No Results", "No backtest results to push")
            return
            
        if not self.engine.supabase_enabled:
            QMessageBox.warning(self, "Supabase Unavailable", 
                              "Supabase storage is not available. Check your credentials.")
            return
        
        # Disable button during push
        self.push_button.setEnabled(False)
        self.push_button.setText("Pushing...")
        
        # Create worker to push in background
        self.push_worker = PushWorker(self.engine, self.last_result)
        self.push_worker.finished.connect(self.on_push_finished)
        self.push_worker.error.connect(self.on_push_error)
        self.push_worker.progress.connect(self.on_push_progress)
        self.push_worker.start()
    
    def on_push_progress(self, message: str):
        """Update status during push"""
        self.status_label.setText(message)
    
    def on_push_finished(self, uid: str):
        """Handle successful push to Supabase"""
        self.push_button.setText("Push to Supabase")
        self.push_button.setEnabled(True)
        
        QMessageBox.information(self, "Success", 
                               f"Results successfully pushed to Supabase!\n\nUID: {uid}")
        
        # Update status
        self.status_label.setText(f"Pushed to Supabase with UID: {uid}")
    
    def on_push_error(self, error_msg: str):
        """Handle push error"""
        self.push_button.setText("Push to Supabase")
        self.push_button.setEnabled(True)
        
        QMessageBox.critical(self, "Push Error", 
                            f"Failed to push to Supabase:\n{error_msg}")
        
        self.status_label.setText("Push failed")
        
    def handle_error(self, error_msg: str):
        """Handle backtest error"""
        QMessageBox.critical(self, "Backtest Error", error_msg)
        self.status_label.setText(f"Error: {error_msg}")
        
    def on_finished(self):
        """Handle backtest completion"""
        self.run_button.setEnabled(True)
        self.status_bar.setVisible(False)
        if not self.last_result:
            self.status_label.setText("Backtest failed")
        else:
            self.status_label.setText("Backtest complete")
    
    def debug_market_structure(self):
        """Open debug window for market structure"""
        # Get current values
        symbol = self.symbol_input.text().strip().upper()
        entry_time = self.datetime_input.dateTime().toPyDateTime()
        
        # Show dialog for debug settings
        dialog = MarketStructureDebugDialog(self, symbol, entry_time)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Get values from dialog
            symbol = dialog.symbol_input.text().strip().upper()
            entry_time = dialog.time_input.dateTime().toPyDateTime()
            entry_time = entry_time.replace(tzinfo=timezone.utc)
            lookback_hours = dialog.lookback_spin.value()
            fractal_length = dialog.fractal_spin.value()
            timeframe = dialog.timeframe_combo.currentText()
            save_results = dialog.save_checkbox.isChecked()
            
            # Update status
            self.status_label.setText(f"Running {timeframe} market structure debug for {symbol}...")
            
            # Run debug in separate thread to avoid blocking UI
            import threading
            
            def run_debug():
                try:
                    # Create new event loop for thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # Import appropriate debugger based on timeframe
                    if timeframe == "1-Min":
                        from debug.debug_m1_market_structure import M1MarketStructureDebugger
                        debugger = M1MarketStructureDebugger(
                            symbol=symbol,
                            entry_time=entry_time,
                            lookback_minutes=int(lookback_hours * 60),
                            fractal_length=fractal_length
                        )
                    elif timeframe == "5-Min":
                        from debug.debug_m5_market_structure import M5MarketStructureDebugger
                        debugger = M5MarketStructureDebugger(
                            symbol=symbol,
                            entry_time=entry_time,
                            lookback_minutes=int(lookback_hours * 60),
                            fractal_length=min(fractal_length, 3)  # Max 3 for 5-min
                        )
                    elif timeframe == "15-Min":
                        from debug.debug_m15_market_structure import M15MarketStructureDebugger
                        debugger = M15MarketStructureDebugger(
                            symbol=symbol,
                            entry_time=entry_time,
                            lookback_minutes=int(lookback_hours * 60),
                            fractal_length=min(fractal_length, 2)  # Max 2 for 15-min
                        )
                    
                    # Run debug
                    results = loop.run_until_complete(debugger.run_debug())
                    
                    # Save results if requested
                    if save_results:
                        output_path = debugger.save_debug_report(results)
                        logger.info(f"Debug results saved to: {output_path}")
                    
                    # Update status on completion
                    self.status_label.setText(f"Debug complete for {symbol} ({timeframe})")
                    
                except Exception as e:
                    logger.error(f"Debug error: {e}", exc_info=True)
                    self.status_label.setText(f"Debug error: {str(e)}")
                finally:
                    loop.close()
            
            # Start debug thread
            debug_thread = threading.Thread(target=run_debug)
            debug_thread.start()
        
    def show_cache_stats(self):
        """Show cache statistics"""
        if self.data_manager:
            stats = self.data_manager.get_cache_stats()
            
            msg = "=== Cache Statistics ===\n\n"
            
            # Memory cache
            mem = stats['memory_cache']
            msg += f"Memory Cache:\n"
            msg += f"  Hits: {mem['hits']}\n"
            msg += f"  Misses: {mem['misses']}\n"
            msg += f"  Hit Rate: {mem['hit_rate']:.1f}%\n"
            msg += f"  Cached Items: {mem['cached_items']}\n\n"
            
            # File cache
            file = stats['file_cache']
            msg += f"File Cache:\n"
            msg += f"  Total Files: {file['total_files']}\n"
            msg += f"  Total Size: {file['total_size_mb']:.2f} MB\n"
            msg += f"  Metadata Entries: {file['metadata_entries']}\n\n"
            
            # API stats
            api = stats['api_stats']
            msg += f"API Statistics:\n"
            msg += f"  API Calls: {api['api_calls']}\n"
            msg += f"  Cache Hits: {api['cache_hits']}\n"
            msg += f"  Total Requests: {api['total_requests']}\n"
            msg += f"  Overall Hit Rate: {api['cache_hit_rate']:.1f}%"
            
            QMessageBox.information(self, "Cache Statistics", msg)


def run_gui():
    """Run the GUI application"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    dashboard = BacktestDashboard()
    dashboard.show()
    
    sys.exit(app.exec())


def run_cli(args):
    """Run backtest from command line"""
    # Load plugins
    plugin_loader = PluginLoader()
    plugins = plugin_loader.load_all_plugins()
    plugin_registry = plugin_loader.get_registry()
    
    # Create data manager
    data_manager = PolygonDataManager()
    
    # Create engine with plugin registry
    engine = BacktestEngine(
        data_manager=data_manager,
        plugin_registry=plugin_registry
    )
    
    # Register adapters from plugins
    adapter_configs = plugin_loader.get_adapter_configs()
    
    for adapter_name, config in adapter_configs.items():
        adapter_class = config['adapter_class']
        adapter_config = config['adapter_config']
        
        adapter = adapter_class(**adapter_config)
        engine.register_adapter(adapter_name, adapter)
    
    # Parse entry time
    entry_time = datetime.strptime(args.entry_time, "%Y-%m-%d %H:%M:%S")
    entry_time = entry_time.replace(tzinfo=timezone.utc)
    
    # Determine which calculations to run
    if args.calculation == "all":
        enabled_calculations = []  # Empty means all
    else:
        enabled_calculations = [args.calculation] if args.calculation else []
    
    # Create config with optional Supabase storage
    config = BacktestConfig(
        symbol=args.symbol,
        entry_time=entry_time,
        direction=args.direction,
        historical_lookback_hours=args.lookback_hours,
        forward_bars=args.forward_bars,
        enabled_calculations=enabled_calculations,
        store_to_supabase=args.store_supabase if hasattr(args, 'store_supabase') else False
    )
    
    # Run backtest
    print(f"Running backtest for {args.symbol} at {entry_time}")
    if args.calculation == "all":
        print("Running ALL calculations")
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(engine.run_backtest(config))
        
        # Display results
        print("\n=== Backtest Results ===")
        summary = result.get_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")
            
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\nResults saved to {args.output}")
            
        # Show cache stats
        print("\n=== Cache Statistics ===")
        stats = data_manager.get_cache_stats()
        api_stats = stats['api_stats']
        print(f"API Calls: {api_stats['api_calls']}")
        print(f"Cache Hits: {api_stats['cache_hits']}")
        print(f"Cache Hit Rate: {api_stats['cache_hit_rate']:.1f}%")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    finally:
        loop.close()
        
    return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Modular Backtest System")
    parser.add_argument("--gui", action="store_true", help="Run GUI mode (default)")
    parser.add_argument("--cli", action="store_true", help="Run CLI mode")
    
    # CLI arguments
    parser.add_argument("--symbol", type=str, help="Stock symbol")
    parser.add_argument("--entry-time", type=str, help="Entry time (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--direction", type=str, choices=["LONG", "SHORT"], help="Trade direction")
    parser.add_argument("--calculation", type=str, default="all", help="Calculation to run (or 'all')")
    parser.add_argument("--lookback-hours", type=int, default=2, help="Historical lookback hours")
    parser.add_argument("--forward-bars", type=int, default=60, help="Forward bars to analyze")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--store-supabase", action="store_true", help="Store results to Supabase")
    
    args = parser.parse_args()
    
    # Determine mode
    if args.cli and all([args.symbol, args.entry_time, args.direction]):
        return run_cli(args)
    else:
        return run_gui()


if __name__ == "__main__":
    sys.exit(main())