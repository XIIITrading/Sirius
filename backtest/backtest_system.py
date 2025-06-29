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
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDateTime
from PyQt6.QtGui import QFont, QColor

# Import core components with relative imports
from core.engine import BacktestEngine, BacktestConfig
from core.data_manager import BacktestDataManager
from core.result_store import BacktestResultStore
from adapters.dummy_adapter import DummyAdapter
from adapters.indicators.m1_ema_back_adapter import M1EMABackAdapter

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
    result_ready = pyqtSignal(dict)
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
            
            self.result_ready.emit(result_dict)
            self.progress.emit(100, "Backtest complete")
            
        except Exception as e:
            logger.error(f"Backtest error: {e}", exc_info=True)
            self.error.emit(str(e))
            
        finally:
            self.finished.emit()
            loop.close()


class BacktestDashboard(QMainWindow):
    """Main dashboard window for backtesting"""
    
    def __init__(self):
        super().__init__()
        self.engine = None
        self.worker = None
        self.registered_adapters = []  # Track registered adapter names
        self.init_engine()
        self.init_ui()
        self.apply_dark_theme()
        
    def init_engine(self):
        """Initialize backtest engine with real data"""
        try:
            # Create engine - it will use BacktestDataManager which connects to Supabase
            self.engine = BacktestEngine()
            logger.info("Backtest engine initialized with Supabase data manager")
            
        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            # Show error dialog
            QMessageBox.critical(None, "Initialization Error", 
                               f"Failed to initialize backtest engine:\n{str(e)}\n\n"
                               "Please check your .env file has SUPABASE_URL and SUPABASE_KEY")
            raise
        
        # Register real adapters
        try:
            # Register M1 EMA adapter
            m1_ema_adapter = M1EMABackAdapter(buffer_size=100)
            self.engine.register_adapter("m1_ema_crossover", m1_ema_adapter)
            self.registered_adapters.append("m1_ema_crossover")
            
            # Also keep dummy for testing
            dummy_adapter = DummyAdapter()
            self.engine.register_adapter("dummy_test", dummy_adapter)
            self.registered_adapters.append("dummy_test")
            
            logger.info(f"Registered adapters: {', '.join(self.registered_adapters)}")
            
        except Exception as e:
            logger.error(f"Failed to register adapters: {e}")
            
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("Modular Backtest System - Real Data")
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
        
        # Run button
        self.run_button = QPushButton("Run Backtest")
        self.run_button.clicked.connect(self.run_backtest)
        layout.addWidget(self.run_button)
        
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
        
        # Determine which calculations to run
        if selected_adapter == "All Calculations":
            # Empty list means run all registered adapters
            enabled_calculations = []
            self.aggregated_group.setVisible(True)
        else:
            # Run only the selected adapter
            enabled_calculations = [selected_adapter]
            self.aggregated_group.setVisible(False)
        
        # Create config
        config = BacktestConfig(
            symbol=symbol,
            entry_time=entry_time,
            direction=direction,
            historical_lookback_hours=2,
            forward_bars=60,
            enabled_calculations=enabled_calculations
        )
        
        # Clear previous results
        self.clear_results()
        
        # Update status
        if selected_adapter == "All Calculations":
            self.status_label.setText(f"Running all {len(self.registered_adapters)} calculations...")
        else:
            self.status_label.setText(f"Running {selected_adapter}...")
        
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
        
    def update_progress(self, value: int, message: str):
        """Update progress bar and status"""
        self.status_bar.setValue(value)
        self.status_label.setText(message)
        
    def display_results(self, results: Dict):
        """Display backtest results"""
        # Display summary
        summary = results.get('summary', {})
        self.summary_table.setRowCount(len(summary))
        
        for i, (key, value) in enumerate(summary.items()):
            self.summary_table.setItem(i, 0, QTableWidgetItem(str(key)))
            self.summary_table.setItem(i, 1, QTableWidgetItem(str(value)))
            
        # Display signals
        signals = results.get('entry_signals', [])
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
                metadata_str = ", ".join(key_metrics) if key_metrics else str(metadata)
                
            self.signals_table.setItem(i, 4, QTableWidgetItem(metadata_str))
        
        # Display aggregated signal if running all calculations
        if self.adapter_combo.currentText() == "All Calculations":
            aggregated = results.get('aggregated_signal', {})
            if aggregated:
                agg_text = f"Consensus Direction: {aggregated.get('consensus_direction', 'N/A')}\n"
                agg_text += f"Agreement Score: {aggregated.get('agreement_score', 0):.1f}%\n"
                agg_text += f"Vote Breakdown: {aggregated.get('vote_breakdown', {})}\n"
                agg_text += f"Average Strength: {aggregated.get('average_strength', 0):.1f}\n"
                agg_text += f"Average Confidence: {aggregated.get('average_confidence', 0):.1f}\n"
                agg_text += f"Participating Calculations: {aggregated.get('participating_calculations', 0)}/{aggregated.get('total_calculations', 0)}"
                self.aggregated_text.setText(agg_text)
        
        # Display forward analysis
        forward = results.get('forward_analysis', {})
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
        
    def handle_error(self, error_msg: str):
        """Handle backtest error"""
        QMessageBox.critical(self, "Backtest Error", error_msg)
        self.status_label.setText(f"Error: {error_msg}")
        
    def on_finished(self):
        """Handle backtest completion"""
        self.run_button.setEnabled(True)
        self.status_bar.setVisible(False)
        self.status_label.setText("Backtest complete")


def run_gui():
    """Run the GUI application"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    dashboard = BacktestDashboard()
    dashboard.show()
    
    sys.exit(app.exec())


def run_cli(args):
    """Run backtest from command line"""
    # Create engine
    engine = BacktestEngine()
    
    # Register adapters
    m1_ema_adapter = M1EMABackAdapter()
    engine.register_adapter("m1_ema_crossover", m1_ema_adapter)
    
    dummy_adapter = DummyAdapter()
    engine.register_adapter("dummy_test", dummy_adapter)
    
    # Parse entry time
    entry_time = datetime.strptime(args.entry_time, "%Y-%m-%d %H:%M:%S")
    entry_time = entry_time.replace(tzinfo=timezone.utc)
    
    # Determine which calculations to run
    if args.calculation == "all":
        enabled_calculations = []  # Empty means all
    else:
        enabled_calculations = [args.calculation] if args.calculation else []
    
    # Create config
    config = BacktestConfig(
        symbol=args.symbol,
        entry_time=entry_time,
        direction=args.direction,
        historical_lookback_hours=args.lookback_hours,
        forward_bars=args.forward_bars,
        enabled_calculations=enabled_calculations
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
                json.dump(result.to_dict(), f, indent=2, default=str)
            print(f"\nResults saved to {args.output}")
            
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
    
    args = parser.parse_args()
    
    # Determine mode
    if args.cli and all([args.symbol, args.entry_time, args.direction]):
        return run_cli(args)
    else:
        return run_gui()


if __name__ == "__main__":
    sys.exit(main())