# modules/ui/dashboards/entry_dashboard.py
"""
Module: Entry Dashboard for Multi-Timeframe Statistical Trend and Order Flow Analysis
Purpose: Real-time display of trend signals and volume/order flow analysis
UI Framework: PyQt6 with PyQtGraph
Features: WebSocket integration, Conditional formatting, Live updates, Historical data loading
"""

import sys
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import asyncio
import logging
from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QSplitter, QFrame, QGridLayout,
    QProgressBar, QComboBox, QTabWidget, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, pyqtSlot
from PyQt6.QtGui import QFont, QColor, QBrush, QPalette
import pyqtgraph as pg
import numpy as np

# Setup path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
dashboards_dir = current_dir
ui_dir = os.path.dirname(dashboards_dir)
modules_dir = os.path.dirname(ui_dir)
vega_root = os.path.dirname(modules_dir)

if vega_root not in sys.path:
    sys.path.insert(0, vega_root)

# Import trend calculators
from modules.calculations.trend.statistical_trend_1min import StatisticalTrend1Min, ScalperSignal
from modules.calculations.trend.statistical_trend_5min import StatisticalTrend5Min, PositionSignal5Min
from modules.calculations.trend.statistical_trend_15min import StatisticalTrend15Min, MarketRegimeSignal

# Import volume/order flow calculators
from modules.calculations.order_flow.trade_size_distro import TradeSizeDistribution, TradeSizeSignal, Trade
from modules.calculations.volume.tick_flow import TickFlowAnalyzer
from modules.calculations.volume.volume_analysis_1min import VolumeAnalysis1Min
from modules.calculations.volume.market_context import MarketContext

# Import Polygon data fetcher
from polygon import DataFetcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrendWorker(QThread):
    """Background thread for running all calculations"""
    # Signals for UI updates - Trend
    signal_1min = pyqtSignal(object)  # ScalperSignal
    signal_5min = pyqtSignal(object)  # PositionSignal5Min
    signal_15min = pyqtSignal(object)  # MarketRegimeSignal
    
    # Signals for UI updates - Volume/Order Flow
    trade_size_signal = pyqtSignal(object)  # TradeSizeSignal
    tick_flow_signal = pyqtSignal(object)   # VolumeSignal from tick_flow
    volume_1min_signal = pyqtSignal(object) # VolumeSignal from volume_1min
    market_context_signal = pyqtSignal(object) # VolumeSignal from market_context
    
    # General signals
    error_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)  # Progress percentage
    
    def __init__(self):
        super().__init__()
        self.symbol = None
        self.running = False
        
        # Trend calculators
        self.calc_1min = StatisticalTrend1Min(
            micro_lookback=3,
            short_lookback=5,
            medium_lookback=10,
            calculation_interval=15
        )
        
        self.calc_5min = StatisticalTrend5Min(
            short_lookback=3,
            medium_lookback=5,
            long_lookback=10,
            calculation_interval=30
        )
        
        self.calc_15min = StatisticalTrend15Min(
            short_lookback=3,
            medium_lookback=5,
            long_lookback=10,
            calculation_interval=60
        )
        
        # Volume/Order Flow calculators
        self.trade_size_calc = TradeSizeDistribution(
            large_trade_threshold=1000,
            buffer_size=500,
            historical_lookback=20,
            zscore_threshold=2.0,
            premarket_adjustment=True
        )
        
        self.tick_flow_calc = TickFlowAnalyzer(
            buffer_size=200,
            large_trade_multiplier=3.0,
            momentum_threshold=60.0,
            min_trades_required=50
        )
        
        self.volume_1min_calc = VolumeAnalysis1Min(
            lookback_bars=14,
            absorption_threshold=0.1,
            efficiency_threshold=0.5,
            volume_imbalance_threshold=65.0
        )
        
        self.market_context_calc = MarketContext(
            lookback_bars=10,
            vwap_deviation_threshold=0.5,
            volume_surge_threshold=1.5,
            delta_imbalance_threshold=100000
        )
        
        # Data fetcher
        self.fetcher = DataFetcher()
        
        # Event loop for async operations
        self.loop = None
        
        # Track last bar times to handle aggregation
        self.last_1min_bar = None
        self.aggregating_5min = {'open': None, 'high': None, 'low': None, 'close': None, 'volume': 0, 'bar_count': 0}
        self.aggregating_15min = {'open': None, 'high': None, 'low': None, 'close': None, 'volume': 0, 'bar_count': 0}
        
    def set_symbol(self, symbol: str):
        """Set the symbol to monitor"""
        self.symbol = symbol.upper()
        
    async def load_historical_and_start(self):
        """Load historical data first, then start monitoring"""
        if not self.symbol:
            return
            
        try:
            # Step 1: Load historical data
            self.status_signal.emit(f"Loading historical data for {self.symbol}...")
            self.progress_signal.emit(10)
            
            # Get current time
            end_time = datetime.now()
            
            # Load 1-minute data (need at least 10 minutes for medium lookback)
            self.status_signal.emit("Loading 1-minute data...")
            start_1min = end_time - timedelta(minutes=30)
            
            df_1min = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.fetcher.fetch_data(
                    symbol=self.symbol,
                    timeframe='1min',
                    start_date=start_1min,
                    end_date=end_time,
                    use_cache=False
                )
            )
            
            if not df_1min.empty:
                # Feed historical data to 1-min calculator
                last_signal = None
                for idx, row in df_1min.iterrows():
                    signal = self.calc_1min.update_price(
                        symbol=self.symbol,
                        price=row['close'],
                        volume=row['volume'],
                        timestamp=idx
                    )
                    if signal:
                        last_signal = signal
                        
                # Emit the last valid signal
                if last_signal:
                    self.signal_1min.emit(last_signal)
                    self.status_signal.emit(f"✓ 1-min: Loaded {len(df_1min)} bars")
                else:
                    self.status_signal.emit(f"⚠️ 1-min: Loaded {len(df_1min)} bars but no signal yet")
            
            self.progress_signal.emit(30)
            
            # Load 5-minute data
            self.status_signal.emit("Loading 5-minute data...")
            start_5min = end_time - timedelta(hours=2)
            
            df_5min = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.fetcher.fetch_data(
                    symbol=self.symbol,
                    timeframe='5min',
                    start_date=start_5min,
                    end_date=end_time,
                    use_cache=False
                )
            )
            
            if not df_5min.empty:
                # Feed historical data to 5-min calculator
                last_signal = None
                for idx, row in df_5min.iterrows():
                    signal = self.calc_5min.update_bar(
                        symbol=self.symbol,
                        open_price=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume'],
                        timestamp=idx
                    )
                    if signal:
                        last_signal = signal
                        
                if last_signal:
                    self.signal_5min.emit(last_signal)
                    
                self.status_signal.emit(f"✓ Loaded {len(df_5min)} 5-min bars")
            
            self.progress_signal.emit(60)
            
            # Load 15-minute data
            self.status_signal.emit("Loading 15-minute data...")
            start_15min = end_time - timedelta(hours=4)
            
            df_15min = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.fetcher.fetch_data(
                    symbol=self.symbol,
                    timeframe='15min',
                    start_date=start_15min,
                    end_date=end_time,
                    use_cache=False
                )
            )
            
            if not df_15min.empty:
                # Feed historical data to 15-min calculator
                last_signal = None
                for idx, row in df_15min.iterrows():
                    signal = self.calc_15min.update_bar(
                        symbol=self.symbol,
                        open_price=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume'],
                        timestamp=idx
                    )
                    if signal:
                        last_signal = signal
                        
                if last_signal:
                    self.signal_15min.emit(last_signal)
                    
                self.status_signal.emit(f"✓ Loaded {len(df_15min)} 15-min bars")
            
            self.progress_signal.emit(80)
            
            # Initialize volume/order flow calculators with historical trades if available
            # Note: In production, you'd load actual historical trade data here
            self.status_signal.emit("Initializing order flow analyzers...")
            
            # Step 2: Start WebSocket monitoring with both channels
            self.status_signal.emit(f"Starting real-time monitoring for {self.symbol}...")
            
            # Import WebSocket client
            from polygon import PolygonWebSocketClient
            
            # Create single WebSocket client
            self.ws_client = PolygonWebSocketClient()
            await self.ws_client.connect()
            
            # Subscribe to BOTH minute aggregates AND trades
            await self.ws_client.subscribe(
                symbols=[self.symbol],
                channels=['AM', 'T'],  # Aggregate Minutes AND Trades
                callback=self._handle_unified_websocket_data
            )
            
            self.progress_signal.emit(100)
            self.status_signal.emit(f"Connected - Monitoring {self.symbol} (Trends + Order Flow)")
            
            # Start periodic calculation loops for each timeframe
            self.calc_1min_task = asyncio.create_task(self._run_1min_calculations())
            self.calc_5min_task = asyncio.create_task(self._run_5min_calculations())
            self.calc_15min_task = asyncio.create_task(self._run_15min_calculations())
            
            # Listen for WebSocket data
            await self.ws_client.listen()
                
        except Exception as e:
            self.error_signal.emit(f"Error: {str(e)}")
            logger.error(f"Monitoring error: {e}", exc_info=True)
            
    async def _handle_unified_websocket_data(self, data: Dict):
        """Handle incoming WebSocket data for all calculators"""
        try:
            event_type = data.get('event_type')
            symbol = data.get('symbol')
            
            if event_type == 'aggregate' and symbol == self.symbol:
                # Existing minute bar handling for trend calculators
                timestamp = datetime.fromtimestamp(data['timestamp'] / 1000, tz=timezone.utc)
                
                # Always update 1-minute with new data
                signal_1min = self.calc_1min.update_price(
                    symbol=symbol,
                    price=data['close'],
                    volume=data['volume'],
                    timestamp=timestamp
                )
                
                if signal_1min:
                    self.signal_1min.emit(signal_1min)
                    logger.info(f"1-min signal: {signal_1min.signal}")
                
                # Aggregate for 5-minute bars
                minute = timestamp.minute
                if minute % 5 == 0 and self.aggregating_5min['bar_count'] > 0:
                    # Complete 5-min bar
                    signal_5min = self.calc_5min.update_bar(
                        symbol=symbol,
                        open_price=self.aggregating_5min['open'],
                        high=self.aggregating_5min['high'],
                        low=self.aggregating_5min['low'],
                        close=self.aggregating_5min['close'],
                        volume=self.aggregating_5min['volume'],
                        timestamp=timestamp
                    )
                    if signal_5min:
                        self.signal_5min.emit(signal_5min)
                        logger.info(f"5-min signal: {signal_5min.signal}")
                    
                    # Reset aggregation
                    self.aggregating_5min = {'open': data['open'], 'high': data['high'], 
                                           'low': data['low'], 'close': data['close'], 
                                           'volume': data['volume'], 'bar_count': 1}
                else:
                    # Update aggregation
                    if self.aggregating_5min['bar_count'] == 0:
                        self.aggregating_5min = {'open': data['open'], 'high': data['high'], 
                                               'low': data['low'], 'close': data['close'], 
                                               'volume': data['volume'], 'bar_count': 1}
                    else:
                        self.aggregating_5min['high'] = max(self.aggregating_5min['high'], data['high'])
                        self.aggregating_5min['low'] = min(self.aggregating_5min['low'], data['low'])
                        self.aggregating_5min['close'] = data['close']
                        self.aggregating_5min['volume'] += data['volume']
                        self.aggregating_5min['bar_count'] += 1
                
                # Aggregate for 15-minute bars
                if minute % 15 == 0 and self.aggregating_15min['bar_count'] > 0:
                    # Complete 15-min bar
                    signal_15min = self.calc_15min.update_bar(
                        symbol=symbol,
                        open_price=self.aggregating_15min['open'],
                        high=self.aggregating_15min['high'],
                        low=self.aggregating_15min['low'],
                        close=self.aggregating_15min['close'],
                        volume=self.aggregating_15min['volume'],
                        timestamp=timestamp
                    )
                    if signal_15min:
                        self.signal_15min.emit(signal_15min)
                        logger.info(f"15-min signal: {signal_15min.regime}")
                    
                    # Reset aggregation
                    self.aggregating_15min = {'open': data['open'], 'high': data['high'], 
                                            'low': data['low'], 'close': data['close'], 
                                            'volume': data['volume'], 'bar_count': 1}
                else:
                    # Update aggregation
                    if self.aggregating_15min['bar_count'] == 0:
                        self.aggregating_15min = {'open': data['open'], 'high': data['high'], 
                                                'low': data['low'], 'close': data['close'], 
                                                'volume': data['volume'], 'bar_count': 1}
                    else:
                        self.aggregating_15min['high'] = max(self.aggregating_15min['high'], data['high'])
                        self.aggregating_15min['low'] = min(self.aggregating_15min['low'], data['low'])
                        self.aggregating_15min['close'] = data['close']
                        self.aggregating_15min['volume'] += data['volume']
                        self.aggregating_15min['bar_count'] += 1
                        
            elif event_type == 'trade' and symbol == self.symbol:
                # NEW: Route trade data to all volume/order flow calculators
                
                # Trade Size Distribution
                trade_obj = Trade(
                    symbol=symbol,
                    price=data['price'],
                    size=data['size'],
                    timestamp=datetime.fromtimestamp(data['timestamp'] / 1000, tz=timezone.utc)
                )
                signal = self.trade_size_calc.process_trade(trade_obj)
                if signal and (signal.bull_score >= 2 or signal.bear_score >= 2):
                    self.trade_size_signal.emit(signal)
                
                # Tick Flow Analysis
                signal = self.tick_flow_calc.process_trade(symbol, data)
                if signal:
                    self.tick_flow_signal.emit(signal)
                
                # 1-Minute Volume Analysis
                self.volume_1min_calc.process_trade(symbol, data)
                
                # 15-Minute Market Context
                self.market_context_calc.process_trade(symbol, data)
                
        except Exception as e:
            logger.error(f"Error handling WebSocket data: {e}", exc_info=True)
            
    async def _run_1min_calculations(self):
        """Run periodic 1-minute calculations"""
        while self.running:
            try:
                await asyncio.sleep(15)  # Every 15 seconds
                
                # Get latest signal
                signals = self.calc_1min.get_batch_analysis([self.symbol])
                if self.symbol in signals and signals[self.symbol]:
                    self.signal_1min.emit(signals[self.symbol])
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in 1-min calculation loop: {e}")
                
    async def _run_5min_calculations(self):
        """Run periodic 5-minute calculations"""
        while self.running:
            try:
                await asyncio.sleep(30)  # Every 30 seconds
                
                # Force recalculation with latest data
                if hasattr(self.calc_5min, 'latest_signals') and self.symbol in self.calc_5min.latest_signals:
                    signal = self.calc_5min.latest_signals[self.symbol]
                    if signal:
                        self.signal_5min.emit(signal)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in 5-min calculation loop: {e}")
                
    async def _run_15min_calculations(self):
        """Run periodic 15-minute calculations"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Every 60 seconds
                
                # Force recalculation with latest data
                if hasattr(self.calc_15min, 'latest_signals') and self.symbol in self.calc_15min.latest_signals:
                    signal = self.calc_15min.latest_signals[self.symbol]
                    if signal:
                        self.signal_15min.emit(signal)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in 15-min calculation loop: {e}")
            
    async def stop_monitoring(self):
        """Stop all monitoring"""
        try:
            self.running = False
            
            # Cancel calculation tasks
            if hasattr(self, 'calc_1min_task'):
                self.calc_1min_task.cancel()
            if hasattr(self, 'calc_5min_task'):
                self.calc_5min_task.cancel()
            if hasattr(self, 'calc_15min_task'):
                self.calc_15min_task.cancel()
                
            # Disconnect WebSocket
            if hasattr(self, 'ws_client'):
                await self.ws_client.disconnect()
                
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
            
    def run(self):
        """Run the worker thread"""
        self.running = True
        
        # Create new event loop for this thread
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self.load_historical_and_start())
        except Exception as e:
            self.error_signal.emit(f"Worker error: {str(e)}")
        finally:
            self.loop.close()
            
    def stop(self):
        """Stop the worker thread"""
        self.running = False
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.stop_monitoring(), self.loop)


class TrendTable(QTableWidget):
    """Custom table widget with conditional formatting"""
    
    def __init__(self, columns: List[str]):
        super().__init__()
        self.setColumnCount(len(columns))
        self.setHorizontalHeaderLabels(columns)
        
        # Styling
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().hide()
        
        # Set column widths
        header = self.horizontalHeader()
        for i in range(len(columns)):
            if i == 0:  # First column (label)
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.Fixed)
                self.setColumnWidth(i, 150)
            else:
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
                
    def set_row_data(self, row: int, label: str, value: str, 
                    color: Optional[QColor] = None, bold: bool = False):
        """Set data for a row with formatting"""
        # Ensure row exists
        if row >= self.rowCount():
            self.setRowCount(row + 1)
            
        # Label
        label_item = QTableWidgetItem(label)
        label_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if bold:
            font = label_item.font()
            font.setBold(True)
            label_item.setFont(font)
        self.setItem(row, 0, label_item)
        
        # Value
        value_item = QTableWidgetItem(str(value))
        value_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if color:
            value_item.setForeground(QBrush(color))
        if bold:
            font = value_item.font()
            font.setBold(True)
            value_item.setFont(font)
        self.setItem(row, 1, value_item)


class TimeframeWidget(QGroupBox):
    """Widget for displaying a single timeframe analysis"""
    
    def __init__(self, title: str, timeframe: str):
        super().__init__(title)
        self.timeframe = timeframe
        self.has_initial_signal = False
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        
        # Signal indicator
        self.signal_frame = QFrame()
        self.signal_frame.setFrameStyle(QFrame.Shape.Box)
        self.signal_frame.setFixedHeight(60)
        signal_layout = QVBoxLayout(self.signal_frame)
        
        self.signal_label = QLabel("LOADING HISTORICAL DATA...")
        self.signal_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.signal_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        signal_layout.addWidget(self.signal_label)
        
        self.confidence_label = QLabel("")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        signal_layout.addWidget(self.confidence_label)
        
        layout.addWidget(self.signal_frame)
        
        # Data table
        if self.timeframe == "1min":
            columns = ["Metric", "Value"]
            self.table = TrendTable(columns)
            self.table.setRowCount(10)
        elif self.timeframe == "5min":
            columns = ["Metric", "Value"]
            self.table = TrendTable(columns)
            self.table.setRowCount(8)
        else:  # 15min
            columns = ["Metric", "Value"]
            self.table = TrendTable(columns)
            self.table.setRowCount(9)
            
        layout.addWidget(self.table)
        
        # Additional info area
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setMaximumHeight(50)
        layout.addWidget(self.info_label)
        
        self.setLayout(layout)
        
        # Set loading state colors
        self.signal_frame.setStyleSheet("background-color: #2a2a3a;")
        self.signal_label.setStyleSheet("color: #ffcc00;")
        
    def update_1min_signal(self, signal: ScalperSignal):
        """Update 1-minute signal display"""
        if not signal:
            return
            
        self.has_initial_signal = True
        
        # Update signal indicator
        self.signal_label.setText(signal.signal)
        self.confidence_label.setText(f"Confidence: {signal.confidence:.1f}%")
        
        # Color based on signal
        if "STRONG BUY" in signal.signal:
            self.signal_frame.setStyleSheet("background-color: #0a4f0a;")
            self.signal_label.setStyleSheet("color: #00ff00;")
        elif "BUY" in signal.signal:
            self.signal_frame.setStyleSheet("background-color: #0a3a0a;")
            self.signal_label.setStyleSheet("color: #00cc00;")
        elif "STRONG SELL" in signal.signal:
            self.signal_frame.setStyleSheet("background-color: #4f0a0a;")
            self.signal_label.setStyleSheet("color: #ff0000;")
        elif "SELL" in signal.signal:
            self.signal_frame.setStyleSheet("background-color: #3a0a0a;")
            self.signal_label.setStyleSheet("color: #cc0000;")
        else:
            self.signal_frame.setStyleSheet("background-color: #2a2a2a;")
            self.signal_label.setStyleSheet("color: #cccccc;")
        
        # Update table
        row = 0
        self.table.set_row_data(row, "Price", f"${signal.price:.2f}", bold=True)
        row += 1
        
        self.table.set_row_data(row, "Signal", signal.signal,
                               self._get_signal_color(signal.signal), bold=True)
        row += 1
        
        self.table.set_row_data(row, "Strength", f"{signal.strength:.1f}%",
                               self._get_strength_color(signal.strength))
        row += 1
        
        self.table.set_row_data(row, "Target Hold", signal.target_hold)
        row += 1
        
        # Timeframe data
        if signal.micro_trend:
            color = self._get_direction_color(signal.micro_trend['direction'])
            self.table.set_row_data(row, "3-min Trend", 
                                   f"{signal.micro_trend['direction'].upper()} ({signal.micro_trend['momentum']:.2f}%)",
                                   color)
        row += 1
        
        if signal.short_trend:
            color = self._get_direction_color(signal.short_trend['direction'])
            self.table.set_row_data(row, "5-min Trend",
                                   f"{signal.short_trend['direction'].upper()} ({signal.short_trend['strength']:.1f}%)",
                                   color)
        row += 1
        
        if signal.medium_trend:
            color = self._get_direction_color(signal.medium_trend['direction'])
            self.table.set_row_data(row, "10-min Trend",
                                   f"{signal.medium_trend['direction'].upper()} ({signal.medium_trend['strength']:.1f}%)",
                                   color)
        row += 1
        
        self.table.set_row_data(row, "Reason", signal.reason)
        row += 1
        
        # Fill remaining rows
        while row < self.table.rowCount():
            self.table.set_row_data(row, "", "")
            row += 1
        
        # Update info
        update_type = "Initial" if not self.has_initial_signal else "Update"
        self.info_label.setText(f"{update_type}: {datetime.now().strftime('%H:%M:%S')}")
        
    def update_5min_signal(self, signal: PositionSignal5Min):
        """Update 5-minute signal display"""
        if not signal:
            return
            
        self.has_initial_signal = True
        
        # Update signal indicator
        self.signal_label.setText(f"{signal.signal} - {signal.bias}")
        self.confidence_label.setText(f"Confidence: {signal.confidence:.1f}%")
        
        # Color based on bias
        if signal.bias == "BULLISH":
            self.signal_frame.setStyleSheet("background-color: #0a3a0a;")
            self.signal_label.setStyleSheet("color: #00cc00;")
        elif signal.bias == "BEARISH":
            self.signal_frame.setStyleSheet("background-color: #3a0a0a;")
            self.signal_label.setStyleSheet("color: #cc0000;")
        else:
            self.signal_frame.setStyleSheet("background-color: #2a2a2a;")
            self.signal_label.setStyleSheet("color: #cccccc;")
        
        # Update table
        row = 0
        self.table.set_row_data(row, "Price", f"${signal.price:.2f}", bold=True)
        row += 1
        
        self.table.set_row_data(row, "Market State", signal.market_state,
                               self._get_state_color(signal.market_state), bold=True)
        row += 1
        
        self.table.set_row_data(row, "Bias", signal.bias,
                               self._get_direction_color(signal.bias.lower()))
        row += 1
        
        self.table.set_row_data(row, "Strength", f"{signal.strength:.1f}%",
                               self._get_strength_color(signal.strength))
        row += 1
        
        # Timeframes
        if signal.short_trend:
            self.table.set_row_data(row, "15-min", 
                                   f"{signal.short_trend['direction'].upper()} (VWAP {signal.short_trend['vwap_position']:+.2f}%)")
        row += 1
        
        if signal.medium_trend:
            self.table.set_row_data(row, "25-min",
                                   f"{signal.medium_trend['direction'].upper()} ({signal.medium_trend['strength']:.1f}%)")
        row += 1
        
        if signal.long_trend:
            self.table.set_row_data(row, "50-min",
                                   f"{signal.long_trend['direction'].upper()} ({signal.long_trend['strength']:.1f}%)")
        row += 1
        
        self.table.set_row_data(row, "Recommendation", signal.recommendation)
        
        # Update info
        update_type = "Initial" if not self.has_initial_signal else "Update"
        self.info_label.setText(f"{update_type}: {datetime.now().strftime('%H:%M:%S')}")
        
    def update_15min_signal(self, signal: MarketRegimeSignal):
        """Update 15-minute signal display"""
        if not signal:
            return
            
        self.has_initial_signal = True
        
        # Update signal indicator
        self.signal_label.setText(signal.regime)
        self.confidence_label.setText(f"Daily Bias: {signal.daily_bias}")
        
        # Color based on regime
        if "BULL" in signal.regime:
            self.signal_frame.setStyleSheet("background-color: #0a3a0a;")
            self.signal_label.setStyleSheet("color: #00cc00;")
        elif "BEAR" in signal.regime:
            self.signal_frame.setStyleSheet("background-color: #3a0a0a;")
            self.signal_label.setStyleSheet("color: #cc0000;")
        elif "RANGE" in signal.regime:
            self.signal_frame.setStyleSheet("background-color: #2a2a3a;")
            self.signal_label.setStyleSheet("color: #cccc00;")
        else:
            self.signal_frame.setStyleSheet("background-color: #2a2a2a;")
            self.signal_label.setStyleSheet("color: #cccccc;")
        
        # Update table
        row = 0
        self.table.set_row_data(row, "Price", f"${signal.price:.2f}", bold=True)
        row += 1
        
        self.table.set_row_data(row, "Regime", signal.regime,
                               self._get_regime_color(signal.regime), bold=True)
        row += 1
        
        self.table.set_row_data(row, "Volatility", signal.volatility_state,
                               self._get_volatility_color(signal.volatility_state))
        row += 1
        
        self.table.set_row_data(row, "Strength", f"{signal.strength:.1f}%",
                               self._get_strength_color(signal.strength))
        row += 1
        
        # Key levels
        if signal.key_levels:
            self.table.set_row_data(row, "Resistance", f"${signal.key_levels.get('recent_high', 0):.2f}")
            row += 1
            self.table.set_row_data(row, "Support", f"${signal.key_levels.get('recent_low', 0):.2f}")
            row += 1
            self.table.set_row_data(row, "VWAP", f"${signal.key_levels.get('vwap', 0):.2f}")
            row += 1
        
        # Timeframes summary
        trend_summary = []
        if signal.short_trend:
            trend_summary.append(f"45m:{signal.short_trend['direction'][:1].upper()}")
        if signal.medium_trend:
            trend_summary.append(f"75m:{signal.medium_trend['direction'][:1].upper()}")
        if signal.long_trend:
            trend_summary.append(f"150m:{signal.long_trend['direction'][:1].upper()}")
        
        self.table.set_row_data(row, "Trends", " | ".join(trend_summary))
        row += 1
        
        # Trading notes (truncated)
        notes = signal.trading_notes[:80] + "..." if len(signal.trading_notes) > 80 else signal.trading_notes
        self.table.set_row_data(row, "Notes", notes)
        
        # Update info with full notes
        self.info_label.setText(f"Trading Notes: {signal.trading_notes}")
        
    def _get_signal_color(self, signal: str) -> QColor:
        """Get color for signal type"""
        if "STRONG BUY" in signal:
            return QColor(0, 255, 0)
        elif "BUY" in signal:
            return QColor(0, 200, 0)
        elif "STRONG SELL" in signal:
            return QColor(255, 0, 0)
        elif "SELL" in signal:
            return QColor(200, 0, 0)
        else:
            return QColor(150, 150, 150)
            
    def _get_direction_color(self, direction: str) -> QColor:
        """Get color for direction"""
        if "bull" in direction.lower():
            return QColor(0, 200, 0)
        elif "bear" in direction.lower():
            return QColor(200, 0, 0)
        else:
            return QColor(150, 150, 150)
            
    def _get_strength_color(self, strength: float) -> QColor:
        """Get color for strength value"""
        if strength >= 70:
            return QColor(0, 255, 0)
        elif strength >= 50:
            return QColor(255, 255, 0)
        elif strength >= 30:
            return QColor(255, 150, 0)
        else:
            return QColor(150, 150, 150)
            
    def _get_state_color(self, state: str) -> QColor:
        """Get color for market state"""
        if state == "TRENDING":
            return QColor(0, 200, 200)
        elif state == "VOLATILE":
            return QColor(255, 150, 0)
        elif state == "CONSOLIDATING":
            return QColor(150, 150, 255)
        else:
            return QColor(150, 150, 150)
            
    def _get_regime_color(self, regime: str) -> QColor:
        """Get color for market regime"""
        if "BULL" in regime:
            return QColor(0, 200, 0)
        elif "BEAR" in regime:
            return QColor(200, 0, 0)
        elif "RANGE" in regime:
            return QColor(200, 200, 0)
        else:
            return QColor(150, 150, 150)
            
    def _get_volatility_color(self, volatility: str) -> QColor:
        """Get color for volatility state"""
        if volatility == "EXTREME":
            return QColor(255, 0, 0)
        elif volatility == "HIGH":
            return QColor(255, 150, 0)
        elif volatility == "LOW":
            return QColor(0, 150, 255)
        else:
            return QColor(150, 150, 150)


class OrderFlowWidget(QGroupBox):
    """Widget for displaying order flow/volume analysis"""
    
    def __init__(self, title: str, signal_type: str):
        super().__init__(title)
        self.signal_type = signal_type
        self.has_initial_signal = False
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        
        # Signal indicator
        self.signal_frame = QFrame()
        self.signal_frame.setFrameStyle(QFrame.Shape.Box)
        self.signal_frame.setFixedHeight(60)
        signal_layout = QVBoxLayout(self.signal_frame)
        
        self.signal_label = QLabel("WAITING FOR DATA...")
        self.signal_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.signal_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        signal_layout.addWidget(self.signal_label)
        
        self.strength_label = QLabel("")
        self.strength_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        signal_layout.addWidget(self.strength_label)
        
        layout.addWidget(self.signal_frame)
        
        # Data table
        columns = ["Metric", "Value"]
        self.table = TrendTable(columns)
        
        # Set different row counts based on signal type
        if self.signal_type == "trade_size":
            self.table.setRowCount(12)
        elif self.signal_type == "tick_flow":
            self.table.setRowCount(10)
        elif self.signal_type == "volume_1min":
            self.table.setRowCount(8)
        else:  # market_context
            self.table.setRowCount(10)
            
        layout.addWidget(self.table)
        
        # Additional info area
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setMaximumHeight(50)
        layout.addWidget(self.info_label)
        
        self.setLayout(layout)
        
        # Set loading state colors
        self.signal_frame.setStyleSheet("background-color: #2a2a3a;")
        self.signal_label.setStyleSheet("color: #999999;")
        
    def update_trade_size_signal(self, signal: TradeSizeSignal):
        """Update trade size distribution display"""
        if not signal:
            return
            
        self.has_initial_signal = True
        
        # Update signal indicator
        self.signal_label.setText(signal.signal_type)
        self.strength_label.setText(f"Confidence: {signal.confidence:.1%}")
        
        # Color based on signal
        if signal.bull_score == 2:
            self.signal_frame.setStyleSheet("background-color: #0a4f0a;")
            self.signal_label.setStyleSheet("color: #00ff00;")
        elif signal.bull_score == 1:
            self.signal_frame.setStyleSheet("background-color: #0a3a0a;")
            self.signal_label.setStyleSheet("color: #00cc00;")
        elif signal.bear_score == 2:
            self.signal_frame.setStyleSheet("background-color: #4f0a0a;")
            self.signal_label.setStyleSheet("color: #ff0000;")
        elif signal.bear_score == 1:
            self.signal_frame.setStyleSheet("background-color: #3a0a0a;")
            self.signal_label.setStyleSheet("color: #cc0000;")
        else:
            self.signal_frame.setStyleSheet("background-color: #2a2a2a;")
            self.signal_label.setStyleSheet("color: #cccccc;")
        
        # Update table
        row = 0
        self.table.set_row_data(row, "Price", f"${signal.current_price:.2f}", bold=True)
        row += 1
        
        # Main signal
        bull_bear = f"Bull: {signal.bull_score} / Bear: {signal.bear_score}"
        color = QColor(0, 255, 0) if signal.bull_score > signal.bear_score else QColor(255, 0, 0)
        self.table.set_row_data(row, "Signal", bull_bear, color, bold=True)
        row += 1
        
        # Components
        c = signal.components
        self.table.set_row_data(row, "VW Ratio", f"{c.volume_weighted_ratio:.1%}",
                               self._get_ratio_color(c.volume_weighted_ratio))
        row += 1
        
        self.table.set_row_data(row, "Z-Score", f"{c.zscore:.2f}",
                               self._get_zscore_color(c.zscore))
        row += 1
        
        self.table.set_row_data(row, "Direction", f"{c.large_direction_ratio:+.2f}",
                               self._get_direction_value_color(c.large_direction_ratio))
        row += 1
        
        self.table.set_row_data(row, "Inst Score", f"{c.institutional_score:.0%}")
        row += 1
        
        self.table.set_row_data(row, "Avg Size", f"{c.avg_trade_size:.0f}")
        row += 1
        
        self.table.set_row_data(row, "Large Vol", f"{c.large_volume:,.0f}")
        row += 1
        
        self.table.set_row_data(row, "Total Vol", f"{c.total_volume:,.0f}")
        row += 1
        
        self.table.set_row_data(row, "Blocks", f"{c.block_sequences}")
        row += 1
        
        self.table.set_row_data(row, "Icebergs", f"{c.iceberg_candidates}")
        row += 1
        
        self.table.set_row_data(row, "Trades", f"{signal.trade_count}")
        
        # Update info
        self.info_label.setText(f"Reason: {signal.reason}")
        
    def update_tick_flow_signal(self, signal):
        """Update tick flow display"""
        if not signal:
            return
            
        self.has_initial_signal = True
        
        # Update signal indicator
        self.signal_label.setText(signal.signal)
        self.strength_label.setText(f"Strength: {signal.strength:.0f}%")
        
        # Color based on signal
        if signal.signal == "BULLISH":
            self.signal_frame.setStyleSheet("background-color: #0a3a0a;")
            self.signal_label.setStyleSheet("color: #00cc00;")
        elif signal.signal == "BEARISH":
            self.signal_frame.setStyleSheet("background-color: #3a0a0a;")
            self.signal_label.setStyleSheet("color: #cc0000;")
        else:
            self.signal_frame.setStyleSheet("background-color: #2a2a2a;")
            self.signal_label.setStyleSheet("color: #cccccc;")
        
        # Update table
        m = signal.metrics
        row = 0
        
        self.table.set_row_data(row, "Buy Volume", f"{m['buy_volume_pct']:.1f}%",
                               self._get_buy_pct_color(m['buy_volume_pct']))
        row += 1
        
        self.table.set_row_data(row, "Buy/Sell", f"{m['buy_trades']}/{m['sell_trades']}")
        row += 1
        
        self.table.set_row_data(row, "Volume", f"{m['buy_volume']:,.0f} / {m['sell_volume']:,.0f}")
        row += 1
        
        self.table.set_row_data(row, "Large Buys", f"{m['large_buy_trades']}")
        row += 1
        
        self.table.set_row_data(row, "Large Sells", f"{m['large_sell_trades']}")
        row += 1
        
        self.table.set_row_data(row, "Trade Rate", f"{m['trade_rate']:.1f}/sec")
        row += 1
        
        self.table.set_row_data(row, "Momentum", f"{m['momentum_score']:+.1f}",
                               self._get_momentum_color(m['momentum_score']))
        row += 1
        
        self.table.set_row_data(row, "Price Trend", m['price_trend'].upper())
        row += 1
        
        self.table.set_row_data(row, "Avg Size", f"{m['avg_trade_size']:.0f}")
        row += 1
        
        self.table.set_row_data(row, "Total Trades", f"{m['total_trades']}")
        
        # Update info
        self.info_label.setText(f"Reason: {signal.reason}")
        
    def update_volume_1min_signal(self, signal):
        """Update 1-minute volume analysis display"""
        if not signal:
            return
            
        self.has_initial_signal = True
        
        # Update signal indicator
        self.signal_label.setText(signal.signal)
        self.strength_label.setText(f"Strength: {signal.strength:.0f}%")
        
        # Color based on signal
        if signal.signal == "BULLISH":
            self.signal_frame.setStyleSheet("background-color: #0a3a0a;")
            self.signal_label.setStyleSheet("color: #00cc00;")
        elif signal.signal == "BEARISH":
            self.signal_frame.setStyleSheet("background-color: #3a0a0a;")
            self.signal_label.setStyleSheet("color: #cc0000;")
        else:
            self.signal_frame.setStyleSheet("background-color: #2a2a2a;")
            self.signal_label.setStyleSheet("color: #cccccc;")
        
        # Update table
        m = signal.metrics
        row = 0
        
        self.table.set_row_data(row, "Buy Volume", f"{m['buy_volume_pct']:.1f}%",
                               self._get_buy_pct_color(m['buy_volume_pct']))
        row += 1
        
        self.table.set_row_data(row, "Efficiency", f"{m['volume_efficiency']:.2f}")
        row += 1
        
        self.table.set_row_data(row, "Rel Volume", f"{m['relative_volume']:.1f}x")
        row += 1
        
        self.table.set_row_data(row, "Bar Strength", f"{m['bar_strength']:.0f}",
                               self._get_strength_color(m['bar_strength']))
        row += 1
        
        absorption = "YES" if m['absorption_detected'] else "NO"
        color = QColor(255, 150, 0) if m['absorption_detected'] else QColor(150, 150, 150)
        self.table.set_row_data(row, "Absorption", absorption, color)
        row += 1
        
        self.table.set_row_data(row, "Volume Trend", m['volume_trend'].upper())
        row += 1
        
        # Fill remaining rows
        while row < self.table.rowCount():
            self.table.set_row_data(row, "", "")
            row += 1
        
        # Update info
        self.info_label.setText(f"Reason: {signal.reason}")
        
    def update_market_context_signal(self, signal):
        """Update market context display"""
        if not signal:
            return
            
        self.has_initial_signal = True
        
        # Update signal indicator
        self.signal_label.setText(signal.signal)
        self.strength_label.setText(f"Strength: {signal.strength:.0f}%")
        
        # Color based on signal
        if signal.signal == "BULLISH":
            self.signal_frame.setStyleSheet("background-color: #0a3a0a;")
            self.signal_label.setStyleSheet("color: #00cc00;")
        elif signal.signal == "BEARISH":
            self.signal_frame.setStyleSheet("background-color: #3a0a0a;")
            self.signal_label.setStyleSheet("color: #cc0000;")
        else:
            self.signal_frame.setStyleSheet("background-color: #2a2a2a;")
            self.signal_label.setStyleSheet("color: #cccccc;")
        
        # Update table
        m = signal.metrics
        row = 0
        
        self.table.set_row_data(row, "Price vs VWAP", f"{m['price_vs_vwap']:+.2f}%",
                               self._get_vwap_color(m['price_vs_vwap']))
        row += 1
        
        self.table.set_row_data(row, "Cum Delta", f"{m['cumulative_delta']:+,.0f}",
                               self._get_delta_color(m['cumulative_delta']))
        row += 1
        
        self.table.set_row_data(row, "Rel Volume", f"{m['relative_volume']:.1f}x")
        row += 1
        
        self.table.set_row_data(row, "Volume Trend", m['volume_trend'].upper())
        row += 1
        
        self.table.set_row_data(row, "Market Phase", m['market_phase'].upper())
        row += 1
        
        self.table.set_row_data(row, "Regime", m['regime'].upper(),
                               self._get_regime_color(m['regime']))
        row += 1
        
        self.table.set_row_data(row, "Trend Str", f"{m['trend_strength']:.0f}",
                               self._get_strength_color(m['trend_strength']))
        row += 1
        
        self.table.set_row_data(row, "Inst Activity", f"{m['institutional_activity']:.0f}%")
        row += 1
        
        # Fill remaining rows
        while row < self.table.rowCount():
            self.table.set_row_data(row, "", "")
            row += 1
        
        # Update info
        self.info_label.setText(f"Reason: {signal.reason}")
        
    def _get_ratio_color(self, ratio: float) -> QColor:
        """Get color for volume ratio"""
        if ratio > 0.3:
            return QColor(255, 0, 0)  # Very high
        elif ratio > 0.2:
            return QColor(255, 150, 0)  # High
        elif ratio > 0.15:
            return QColor(255, 255, 0)  # Normal
        else:
            return QColor(150, 150, 150)  # Low
            
    def _get_zscore_color(self, zscore: float) -> QColor:
        """Get color for z-score"""
        if abs(zscore) > 2:
            return QColor(255, 0, 0)  # Significant
        elif abs(zscore) > 1:
            return QColor(255, 150, 0)  # Notable
        else:
            return QColor(150, 150, 150)  # Normal
            
    def _get_direction_value_color(self, value: float) -> QColor:
        """Get color for direction value"""
        if value > 0.3:
            return QColor(0, 255, 0)
        elif value > 0:
            return QColor(0, 200, 0)
        elif value < -0.3:
            return QColor(255, 0, 0)
        elif value < 0:
            return QColor(200, 0, 0)
        else:
            return QColor(150, 150, 150)
            
    def _get_buy_pct_color(self, pct: float) -> QColor:
        """Get color for buy percentage"""
        if pct > 65:
            return QColor(0, 255, 0)
        elif pct > 55:
            return QColor(0, 200, 0)
        elif pct < 35:
            return QColor(255, 0, 0)
        elif pct < 45:
            return QColor(200, 0, 0)
        else:
            return QColor(150, 150, 150)
            
    def _get_momentum_color(self, momentum: float) -> QColor:
        """Get color for momentum score"""
        if momentum > 50:
            return QColor(0, 255, 0)
        elif momentum > 20:
            return QColor(0, 200, 0)
        elif momentum < -50:
            return QColor(255, 0, 0)
        elif momentum < -20:
            return QColor(200, 0, 0)
        else:
            return QColor(150, 150, 150)
            
    def _get_strength_color(self, strength: float) -> QColor:
        """Get color for strength value"""
        if strength >= 70:
            return QColor(0, 255, 0)
        elif strength >= 50:
            return QColor(255, 255, 0)
        elif strength >= 30:
            return QColor(255, 150, 0)
        else:
            return QColor(150, 150, 150)
            
    def _get_vwap_color(self, vwap_diff: float) -> QColor:
        """Get color for VWAP difference"""
        if vwap_diff > 0.5:
            return QColor(0, 255, 0)
        elif vwap_diff > 0:
            return QColor(0, 200, 0)
        elif vwap_diff < -0.5:
            return QColor(255, 0, 0)
        elif vwap_diff < 0:
            return QColor(200, 0, 0)
        else:
            return QColor(150, 150, 150)
            
    def _get_delta_color(self, delta: float) -> QColor:
        """Get color for cumulative delta"""
        if delta > 100000:
            return QColor(0, 255, 0)
        elif delta > 0:
            return QColor(0, 200, 0)
        elif delta < -100000:
            return QColor(255, 0, 0)
        elif delta < 0:
            return QColor(200, 0, 0)
        else:
            return QColor(150, 150, 150)
            
    def _get_regime_color(self, regime: str) -> QColor:
        """Get color for market regime"""
        if regime.lower() == "trending":
            return QColor(0, 200, 200)
        elif regime.lower() == "volatile":
            return QColor(255, 150, 0)
        else:
            return QColor(150, 150, 150)


class EntryDashboard(QMainWindow):
    """Main dashboard window"""
    
    def __init__(self):
        super().__init__()
        self.worker = None
        self.current_symbol = None
        self.init_ui()
        self.apply_dark_theme()
        
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("Entry Dashboard - Trends & Order Flow Analysis")
        self.setGeometry(100, 100, 1600, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Top controls
        controls_layout = QHBoxLayout()
        
        # Symbol input
        self.symbol_label = QLabel("Symbol:")
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("Enter symbol (e.g., TSLA)")
        self.symbol_input.setMaximumWidth(150)
        self.symbol_input.returnPressed.connect(self.start_monitoring)
        
        # Start/Stop button
        self.start_btn = QPushButton("Start Monitoring")
        self.start_btn.clicked.connect(self.start_monitoring)
        self.start_btn.setMinimumWidth(150)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #00cc00; font-weight: bold;")
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        
        controls_layout.addWidget(self.symbol_label)
        controls_layout.addWidget(self.symbol_input)
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.status_label)
        controls_layout.addWidget(self.progress_bar)
        controls_layout.addStretch()
        
        main_layout.addLayout(controls_layout)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(line)
        
        # Main content area - Tab widget
        self.tab_widget = QTabWidget()
        
        # Tab 1: Trend Analysis
        trend_widget = QWidget()
        trend_layout = QHBoxLayout(trend_widget)
        
        # 1-minute widget
        self.widget_1min = TimeframeWidget("1-Minute Scalper Signals", "1min")
        trend_layout.addWidget(self.widget_1min)
        
        # 5-minute widget
        self.widget_5min = TimeframeWidget("5-Minute Position Signals", "5min")
        trend_layout.addWidget(self.widget_5min)
        
        # 15-minute widget
        self.widget_15min = TimeframeWidget("15-Minute Market Regime", "15min")
        trend_layout.addWidget(self.widget_15min)
        
        self.tab_widget.addTab(trend_widget, "Trend Analysis")
        
        # Tab 2: Order Flow Analysis
        order_flow_widget = QWidget()
        order_flow_layout = QGridLayout(order_flow_widget)
        
        # Trade Size Distribution
        self.widget_trade_size = OrderFlowWidget("Trade Size Distribution", "trade_size")
        order_flow_layout.addWidget(self.widget_trade_size, 0, 0)
        
        # Tick Flow
        self.widget_tick_flow = OrderFlowWidget("Tick Flow Analysis", "tick_flow")
        order_flow_layout.addWidget(self.widget_tick_flow, 0, 1)
        
        # 1-Min Volume
        self.widget_volume_1min = OrderFlowWidget("1-Min Volume Analysis", "volume_1min")
        order_flow_layout.addWidget(self.widget_volume_1min, 1, 0)
        
        # Market Context
        self.widget_market_context = OrderFlowWidget("15-Min Market Context", "market_context")
        order_flow_layout.addWidget(self.widget_market_context, 1, 1)
        
        self.tab_widget.addTab(order_flow_widget, "Order Flow Analysis")
        
        main_layout.addWidget(self.tab_widget)
        
        # Bottom info bar
        self.info_bar = QLabel("Enter a symbol and click 'Start Monitoring' to begin")
        self.info_bar.setStyleSheet("background-color: #1a1a1a; padding: 10px;")
        main_layout.addWidget(self.info_bar)
        
    def apply_dark_theme(self):
        """Apply dark theme to the application"""
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
        dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(60, 60, 60))
        dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        
        self.setPalette(dark_palette)
        
        # Additional styling
        self.setStyleSheet("""
            QGroupBox {
                border: 2px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                font-size: 14px;
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
            QTableWidget {
                gridline-color: #444;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #3a3a3a;
                padding: 5px;
                border: 1px solid #444;
                font-weight: bold;
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
            QTabWidget::pane {
                border: 1px solid #444;
                background-color: #2a2a2a;
            }
            QTabBar::tab {
                background-color: #3a3a3a;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #0d7377;
            }
        """)
        
    def start_monitoring(self):
        """Start or stop monitoring"""
        if self.worker and self.worker.isRunning():
            # Stop monitoring
            self.stop_monitoring()
        else:
            # Start monitoring
            symbol = self.symbol_input.text().strip().upper()
            if not symbol:
                self.status_label.setText("Please enter a symbol")
                self.status_label.setStyleSheet("color: #ff0000;")
                return
                
            self.current_symbol = symbol
            self.status_label.setText(f"Starting {symbol}...")
            self.status_label.setStyleSheet("color: #ffcc00;")
            
            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Create and start worker
            self.worker = TrendWorker()
            self.worker.set_symbol(symbol)
            
            # Connect trend signals
            self.worker.signal_1min.connect(self.update_1min)
            self.worker.signal_5min.connect(self.update_5min)
            self.worker.signal_15min.connect(self.update_15min)
            
            # Connect order flow signals
            self.worker.trade_size_signal.connect(self.update_trade_size)
            self.worker.tick_flow_signal.connect(self.update_tick_flow)
            self.worker.volume_1min_signal.connect(self.update_volume_1min)
            self.worker.market_context_signal.connect(self.update_market_context)
            
            # Connect general signals
            self.worker.error_signal.connect(self.handle_error)
            self.worker.status_signal.connect(self.update_status)
            self.worker.progress_signal.connect(self.update_progress)
            
            # Start worker
            self.worker.start()
            
            # Update UI
            self.start_btn.setText("Stop Monitoring")
            self.symbol_input.setEnabled(False)
            self.info_bar.setText(f"Loading historical data for {symbol}...")
            
    def stop_monitoring(self):
        """Stop monitoring"""
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
            
        # Update UI
        self.start_btn.setText("Start Monitoring")
        self.symbol_input.setEnabled(True)
        self.status_label.setText("Stopped")
        self.status_label.setStyleSheet("color: #cccccc;")
        self.info_bar.setText("Monitoring stopped")
        self.progress_bar.setVisible(False)
        
    # Trend update slots
    @pyqtSlot(object)
    def update_1min(self, signal: ScalperSignal):
        """Update 1-minute display"""
        self.widget_1min.update_1min_signal(signal)
        
    @pyqtSlot(object)
    def update_5min(self, signal: PositionSignal5Min):
        """Update 5-minute display"""
        self.widget_5min.update_5min_signal(signal)
        
    @pyqtSlot(object)
    def update_15min(self, signal: MarketRegimeSignal):
        """Update 15-minute display"""
        self.widget_15min.update_15min_signal(signal)
        
    # Order flow update slots
    @pyqtSlot(object)
    def update_trade_size(self, signal: TradeSizeSignal):
        """Update trade size distribution display"""
        self.widget_trade_size.update_trade_size_signal(signal)
        
    @pyqtSlot(object)
    def update_tick_flow(self, signal):
        """Update tick flow display"""
        self.widget_tick_flow.update_tick_flow_signal(signal)
        
    @pyqtSlot(object)
    def update_volume_1min(self, signal):
        """Update 1-minute volume display"""
        self.widget_volume_1min.update_volume_1min_signal(signal)
        
    @pyqtSlot(object)
    def update_market_context(self, signal):
        """Update market context display"""
        self.widget_market_context.update_market_context_signal(signal)
        
    @pyqtSlot(str)
    def handle_error(self, error_msg: str):
        """Handle error messages"""
        self.status_label.setText(f"Error: {error_msg}")
        self.status_label.setStyleSheet("color: #ff0000;")
        self.progress_bar.setVisible(False)
        logger.error(f"Dashboard error: {error_msg}")
        
    @pyqtSlot(str)
    def update_status(self, status_msg: str):
        """Update status message"""
        self.status_label.setText(status_msg)
        if "Connected" in status_msg:
            self.status_label.setStyleSheet("color: #00cc00;")
            self.progress_bar.setVisible(False)
            self.info_bar.setText(f"Monitoring {self.current_symbol} - Trends + Order Flow Analysis")
        else:
            self.status_label.setStyleSheet("color: #ffcc00;")
            
    @pyqtSlot(int)
    def update_progress(self, value: int):
        """Update progress bar"""
        self.progress_bar.setValue(value)
        
    def closeEvent(self, event):
        """Handle window close"""
        if self.worker and self.worker.isRunning():
            self.stop_monitoring()
        event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    dashboard = EntryDashboard()
    dashboard.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()