# modules/ui/dashboards/backtesting_dashboard.py
"""
Module: Backtesting Dashboard for Entry Algorithm Analysis
Purpose: Analyze historical entry points with calculations and price action
UI Framework: PyQt6 with PyQtGraph
Features: Point-in-time calculations, candlestick charting, no look-ahead bias, Claude AI integration
Performance: Optimized for high-volume symbols with concurrent trade fetching
"""

import sys
import os
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import json

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QDateEdit, QTimeEdit,
    QGroupBox, QSplitter, QFrame, QGridLayout, QMessageBox,
    QProgressBar, QTableWidget, QHeaderView, QScrollArea, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDate, QTime, QDateTime, pyqtSlot
from PyQt6.QtGui import QFont, QColor, QPalette, QPen, QIcon
import pyqtgraph as pg

# Setup path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
dashboards_dir = current_dir
ui_dir = os.path.dirname(dashboards_dir)
modules_dir = os.path.dirname(ui_dir)
vega_root = os.path.dirname(modules_dir)

if vega_root not in sys.path:
    sys.path.insert(0, vega_root)

# Import display widgets from entry dashboard
from modules.ui.dashboards.entry_dashboard import (
    TimeframeWidget, OrderFlowWidget, TrendTable
)

# Import all calculators
from modules.calculations.trend.statistical_trend_1min import StatisticalTrend1Min
from modules.calculations.trend.statistical_trend_5min import StatisticalTrend5Min
from modules.calculations.trend.statistical_trend_15min import StatisticalTrend15Min
from modules.calculations.order_flow.trade_size_distro import TradeSizeDistribution, TradeSizeSignal, Trade
from modules.calculations.order_flow.bid_ask_imbal import BidAskImbalance, Quote  # NEW IMPORT
from modules.calculations.volume.tick_flow import TickFlowAnalyzer
from modules.calculations.volume.volume_analysis_1min import VolumeAnalysis1Min
from modules.calculations.volume.market_context import MarketContext

# Import data fetcher
from polygon import DataFetcher

# Import Claude integration modules
from modules.integrations.claude_dialog import ClaudeConversationDialog
from modules.utils.result_formatter import BacktestResultFormatter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedDataFetcher(DataFetcher):
    """Enhanced data fetcher with trade and quote data support"""
    
    def __init__(self):
        super().__init__()
        self.session = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Load API key from environment if not already set
        if not hasattr(self, 'api_key') or not self.api_key:
            import os
            from dotenv import load_dotenv
            
            # Load .env file from root directory
            env_path = os.path.join(vega_root, '.env')
            load_dotenv(env_path)
            
            self.api_key = os.getenv('POLYGON_API_KEY')
            if not self.api_key:
                raise ValueError("POLYGON_API_KEY not found in environment variables")
            
            logger.info("Loaded Polygon API key from environment")
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
        self.executor.shutdown(wait=False)
    
    async def _fetch_trade_chunk(self, session: aiohttp.ClientSession, symbol: str,
                            start_ns: int, end_ns: int, limit: int = 50000) -> List[Dict]:
        """Fetch a single chunk of trades"""
        trades = []
        base_url = f"https://api.polygon.io/v3/trades/{symbol}"
        next_url = base_url
        
        # Ensure timestamps are integers without scientific notation
        params = {
            'timestamp.gte': str(int(start_ns)),  # Convert to string to avoid scientific notation
            'timestamp.lte': str(int(end_ns)),    # Convert to string to avoid scientific notation
            'limit': limit,
            'sort': 'timestamp',
            'order': 'asc',
            'apiKey': self.api_key
        }
        
        try:
            while next_url:
                async with session.get(next_url, params=params if next_url == base_url else None) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Error fetching trades: {response.status} - {error_text}")
                        break
                        
                    data = await response.json()
                    
                    if 'results' in data:
                        trades.extend(data['results'])
                    
                    # Check for next page
                    next_url = data.get('next_url')
                    if next_url:
                        next_url = f"{next_url}&apiKey={self.api_key}"
                        params = None  # Don't use params on subsequent requests
                        
        except Exception as e:
            logger.error(f"Error in chunk fetch: {e}")
            
        return trades
    
    async def _fetch_quote_chunk(self, session: aiohttp.ClientSession, symbol: str,
                            start_ns: int, end_ns: int, limit: int = 50000) -> List[Dict]:
        """Fetch a single chunk of quotes"""
        quotes = []
        base_url = f"https://api.polygon.io/v3/quotes/{symbol}"
        next_url = base_url
        
        params = {
            'timestamp.gte': str(int(start_ns)),
            'timestamp.lte': str(int(end_ns)),
            'limit': limit,
            'sort': 'timestamp',
            'order': 'asc',
            'apiKey': self.api_key
        }
        
        try:
            while next_url:
                async with session.get(next_url, params=params if next_url == base_url else None) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Error fetching quotes: {response.status} - {error_text}")
                        break
                        
                    data = await response.json()
                    
                    if 'results' in data:
                        quotes.extend(data['results'])
                    
                    # Check for next page
                    next_url = data.get('next_url')
                    if next_url:
                        next_url = f"{next_url}&apiKey={self.api_key}"
                        params = None
                        
        except Exception as e:
            logger.error(f"Error in quote chunk fetch: {e}")
            
        return quotes
    
    async def fetch_trades_concurrent(self, symbol: str, start_date: datetime, 
                                    end_date: datetime, chunk_minutes: int = 15,
                                    progress_callback: Optional[callable] = None) -> List[Dict]:
        """
        Fetch trades using concurrent requests for faster loading
        
        Args:
            symbol: Stock symbol
            start_date: Start datetime (UTC)
            end_date: End datetime (UTC)
            chunk_minutes: Minutes per chunk (default 15)
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of trade dictionaries
        """
        # Convert to nanoseconds
        start_ns = int(start_date.timestamp() * 1e9)
        end_ns = int(end_date.timestamp() * 1e9)
        
        # Create time chunks
        time_chunks = []
        chunk_duration_ns = chunk_minutes * 60 * 1e9
        current_start_ns = start_ns
        
        while current_start_ns < end_ns:
            chunk_end_ns = min(current_start_ns + chunk_duration_ns, end_ns)
            time_chunks.append((current_start_ns, chunk_end_ns))
            current_start_ns = chunk_end_ns
        
        logger.info(f"Fetching trades in {len(time_chunks)} chunks of {chunk_minutes} minutes each")
        
        # Use existing session or create temporary one
        session_created = False
        if not self.session:
            self.session = aiohttp.ClientSession()
            session_created = True
        
        try:
            # Create tasks for concurrent fetching
            tasks = []
            for i, (chunk_start, chunk_end) in enumerate(time_chunks):
                task = self._fetch_trade_chunk(self.session, symbol, chunk_start, chunk_end)
                tasks.append(task)
            
            # Execute with progress tracking
            all_trades = []
            completed = 0
            
            # Process in batches to avoid overwhelming the API
            batch_size = 5  # Process 5 chunks at a time
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i:i + batch_size]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Chunk fetch error: {result}")
                    elif result:
                        all_trades.extend(result)
                
                completed += len(batch_tasks)
                if progress_callback:
                    progress_pct = (completed / len(time_chunks)) * 100
                    await progress_callback(f"Fetched {completed}/{len(time_chunks)} trade chunks", progress_pct)
                
                # Small delay between batches to be API-friendly
                if i + batch_size < len(tasks):
                    await asyncio.sleep(0.1)
            
            # Sort all trades by timestamp
            all_trades.sort(key=lambda x: x['participant_timestamp'])
            
            # Convert to expected format
            formatted_trades = []
            for trade in all_trades:
                formatted_trades.append({
                    'symbol': symbol,
                    'price': trade['price'],
                    'size': trade['size'],
                    'timestamp': trade['participant_timestamp'],  # nanoseconds
                    'conditions': trade.get('conditions', []),
                    'exchange': trade.get('exchange', 0)
                })
            
            logger.info(f"Total trades fetched: {len(formatted_trades):,}")
            return formatted_trades
            
        finally:
            if session_created and self.session:
                await self.session.close()
                self.session = None
    
    async def fetch_quotes_concurrent(self, symbol: str, start_date: datetime, 
                                    end_date: datetime, chunk_minutes: int = 15,
                                    progress_callback: Optional[callable] = None) -> List[Dict]:
        """
        Fetch quotes using concurrent requests
        
        Args:
            symbol: Stock symbol
            start_date: Start datetime (UTC)
            end_date: End datetime (UTC)
            chunk_minutes: Minutes per chunk (default 15)
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of quote dictionaries
        """
        # Convert to nanoseconds
        start_ns = int(start_date.timestamp() * 1e9)
        end_ns = int(end_date.timestamp() * 1e9)
        
        # Create time chunks
        time_chunks = []
        chunk_duration_ns = chunk_minutes * 60 * 1e9
        current_start_ns = start_ns
        
        while current_start_ns < end_ns:
            chunk_end_ns = min(current_start_ns + chunk_duration_ns, end_ns)
            time_chunks.append((current_start_ns, chunk_end_ns))
            current_start_ns = chunk_end_ns
        
        logger.info(f"Fetching quotes in {len(time_chunks)} chunks of {chunk_minutes} minutes each")
        
        # Use existing session or create temporary one
        session_created = False
        if not self.session:
            self.session = aiohttp.ClientSession()
            session_created = True
        
        try:
            # Create tasks for concurrent fetching
            tasks = []
            for i, (chunk_start, chunk_end) in enumerate(time_chunks):
                task = self._fetch_quote_chunk(self.session, symbol, chunk_start, chunk_end)
                tasks.append(task)
            
            # Execute with progress tracking
            all_quotes = []
            completed = 0
            
            # Process in batches
            batch_size = 5
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i:i + batch_size]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Quote chunk fetch error: {result}")
                    elif result:
                        all_quotes.extend(result)
                
                completed += len(batch_tasks)
                if progress_callback:
                    progress_pct = (completed / len(time_chunks)) * 100
                    await progress_callback(f"Fetched {completed}/{len(time_chunks)} quote chunks", progress_pct)
                
                if i + batch_size < len(tasks):
                    await asyncio.sleep(0.1)
            
            # Sort all quotes by timestamp
            all_quotes.sort(key=lambda x: x['participant_timestamp'])
            
            # Convert to expected format
            formatted_quotes = []
            for quote in all_quotes:
                formatted_quotes.append({
                    'symbol': symbol,
                    'bid': quote['bid_price'],
                    'ask': quote['ask_price'],
                    'bid_size': quote['bid_size'],
                    'ask_size': quote['ask_size'],
                    'timestamp': quote['participant_timestamp'],  # nanoseconds
                    'exchange': quote.get('exchange', 0)
                })
            
            logger.info(f"Total quotes fetched: {len(formatted_quotes):,}")
            return formatted_quotes
            
        finally:
            if session_created and self.session:
                await self.session.close()
                self.session = None
    
    async def fetch_trades(self, symbol: str, start_date: datetime, 
                          end_date: datetime, limit: int = 50000) -> List[Dict]:
        """
        Fetch trade data from Polygon.io (backward compatible method)
        """
        # Use concurrent method for better performance
        return await self.fetch_trades_concurrent(symbol, start_date, end_date)


class CandlestickChart(pg.GraphicsLayoutWidget):
    # Custom candlestick chart widget
    
    def __init__(self):
        super().__init__()
        self.setBackground('k')  # Black background
        
        # Create plot
        self.plot = self.addPlot(row=0, col=0)
        self.plot.setLabel('left', 'Price', units='$')
        self.plot.setLabel('bottom', 'Time')
        self.plot.showGrid(x=False, y=False, alpha=0)  # Disable gridlines
        
        # Add volume plot below
        self.volume_plot = self.addPlot(row=1, col=0)
        self.volume_plot.setLabel('left', 'Volume')
        self.volume_plot.showGrid(x=False, y=False, alpha=0)  # Disable gridlines
        
        # Link x-axes
        self.volume_plot.setXLink(self.plot)
        
        # Set height ratio (price chart gets more space)
        self.ci.layout.setRowStretchFactor(0, 3)
        self.ci.layout.setRowStretchFactor(1, 1)
        
        # Candlestick data
        self.candlesticks = []
        self.entry_marker = None
        
        # Crosshair
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.plot.addItem(self.vLine, ignoreBounds=True)
        self.plot.addItem(self.hLine, ignoreBounds=True)
        
        # Text label for crosshair
        self.label = pg.TextItem(anchor=(0, 1))
        self.plot.addItem(self.label)
        
        # Mouse tracking
        self.proxy = pg.SignalProxy(self.plot.scene().sigMouseMoved, 
                                   rateLimit=60, slot=self.mouse_moved)
        
    def plot_candlesticks(self, df: pd.DataFrame, entry_index: int):
        # Plot candlestick data with entry point marked
        # Clear previous data
        self.plot.clear()
        self.volume_plot.clear()
        self.candlesticks = []
        
        # Re-add crosshair
        self.plot.addItem(self.vLine, ignoreBounds=True)
        self.plot.addItem(self.hLine, ignoreBounds=True)
        self.plot.addItem(self.label)
        
        # Plot candlesticks
        for i, (idx, row) in enumerate(df.iterrows()):
            # Determine color
            if row['close'] >= row['open']:
                color = 'g'  # Green for up
                brush = pg.mkBrush('g')
                volume_brush = pg.mkBrush(0, 255, 0, 100)  # Translucent green
            else:
                color = 'r'  # Red for down
                brush = pg.mkBrush('r')
                volume_brush = pg.mkBrush(255, 0, 0, 100)  # Translucent red
            
            # Create candlestick
            high_low = pg.PlotDataItem(
                [i, i], [row['low'], row['high']], 
                pen=pg.mkPen(color, width=1)
            )
            self.plot.addItem(high_low)
            
            # Body
            body_height = abs(row['close'] - row['open'])
            if body_height > 0:
                body = pg.BarGraphItem(
                    x=[i], 
                    height=[body_height],
                    width=0.8, 
                    y=[min(row['open'], row['close'])],
                    brush=brush
                )
                self.plot.addItem(body)
            
            # Volume bar
            volume_bar = pg.BarGraphItem(
                x=[i],
                height=[row['volume']],
                width=0.8,
                brush=volume_brush
            )
            self.volume_plot.addItem(volume_bar)
            
            self.candlesticks.append({
                'time': idx,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            })
        
        # Mark entry point with white triangle
        if 0 <= entry_index < len(df):
            entry_row = df.iloc[entry_index]
            entry_high = entry_row['high']
            
            # Create triangle marker above the candle
            triangle_size = (df['high'].max() - df['low'].min()) * 0.02  # 2% of price range
            triangle_y = entry_high + triangle_size * 2  # Position above high
            
            # Create scatter plot for triangle
            self.entry_marker = pg.ScatterPlotItem(
                x=[entry_index],
                y=[triangle_y],
                symbol='t',  # Triangle
                size=10,
                brush=pg.mkBrush('w'),  # White fill
                pen=pg.mkPen('w', width=1)  # White outline
            )
            self.plot.addItem(self.entry_marker)
            
            # REMOVED: Entry line and price line
        
        # Set x-axis to show time labels
        self.plot.setXRange(-0.5, len(df) - 0.5)
        self.volume_plot.setXRange(-0.5, len(df) - 0.5)
        
        # Create time axis labels
        time_strings = []
        for i in range(0, len(df), max(1, len(df) // 10)):  # Show ~10 labels
            time_strings.append((i, df.index[i].strftime('%H:%M')))
        
        self.plot.getAxis('bottom').setTicks([time_strings])
        self.volume_plot.getAxis('bottom').setTicks([time_strings])
        
        # Hide x-axis labels on price chart (only show on volume)
        self.plot.getAxis('bottom').setStyle(showValues=False)
        
    def mouse_moved(self, evt):
        # Handle mouse movement for crosshair
        pos = evt[0]
        if self.plot.sceneBoundingRect().contains(pos):
            mouse_point = self.plot.vb.mapSceneToView(pos)
            index = int(mouse_point.x())
            
            if 0 <= index < len(self.candlesticks):
                candle = self.candlesticks[index]
                self.vLine.setPos(mouse_point.x())
                self.hLine.setPos(mouse_point.y())
                
                # Update label
                text = f"Time: {candle['time'].strftime('%H:%M:%S')}\n"
                text += f"O: ${candle['open']:.2f} H: ${candle['high']:.2f}\n"
                text += f"L: ${candle['low']:.2f} C: ${candle['close']:.2f}\n"
                text += f"Vol: {candle['volume']:,.0f}"
                self.label.setText(text)
                self.label.setPos(mouse_point.x(), mouse_point.y())


class BacktestWorker(QThread):
    """Worker thread for backtesting calculations"""
    
    # Signals for results
    calculation_complete = pyqtSignal(dict)  # All calculation results
    chart_data_ready = pyqtSignal(object, int)  # DataFrame and entry index
    error_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    trade_progress = pyqtSignal(str, int)  # Trade loading progress
    quote_progress = pyqtSignal(str, int)  # Quote loading progress
    
    def __init__(self):
        super().__init__()
        self.symbol = None
        self.entry_time = None
        
        # Initialize all calculators
        self.calc_1min = StatisticalTrend1Min()
        self.calc_5min = StatisticalTrend5Min()
        self.calc_15min = StatisticalTrend15Min()
        self.trade_size_calc = TradeSizeDistribution()
        self.bid_ask_calc = BidAskImbalance()  # NEW CALCULATOR
        self.tick_flow_calc = TickFlowAnalyzer()
        self.volume_1min_calc = VolumeAnalysis1Min()
        self.market_context_calc = MarketContext()
        
        # Enhanced data fetcher
        self.fetcher = EnhancedDataFetcher()
        
    def set_parameters(self, symbol: str, entry_time: datetime):
        """Set backtest parameters"""
        self.symbol = symbol.upper()
        self.entry_time = entry_time.replace(tzinfo=timezone.utc)
        
    async def _trade_fetch_progress(self, message: str, progress: float):
        """Callback for trade fetch progress"""
        self.trade_progress.emit(message, int(progress))
        
    async def _quote_fetch_progress(self, message: str, progress: float):
        """Callback for quote fetch progress"""
        self.quote_progress.emit(message, int(progress))
        
    async def run_backtest(self):
        """Run the backtest with optimized trade and quote fetching"""
        try:
            results = {}
            
            # Calculate data requirements
            # Need data from 2 hours before entry for calculations
            calc_start = self.entry_time - timedelta(hours=2)
            # Need 15 minutes before + 60 minutes after for chart
            chart_start = self.entry_time - timedelta(minutes=15)
            chart_end = self.entry_time + timedelta(minutes=60)
            
            self.status_signal.emit(f"Loading historical data for {self.symbol}...")
            self.progress_signal.emit(10)
            
            # Fetch 1-minute data for entire range
            df_all = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.fetcher.fetch_data(
                    symbol=self.symbol,
                    timeframe='1min',
                    start_date=calc_start,
                    end_date=chart_end,
                    use_cache=True
                )
            )
            
            if df_all.empty:
                raise ValueError(f"No data available for {self.symbol} at {self.entry_time}")
            
            # Split data for calculations (up to entry) and chart (full range)
            df_calc = df_all[df_all.index <= self.entry_time]
            df_chart = df_all[(df_all.index >= chart_start) & (df_all.index <= chart_end)]
            
            # Find entry index in chart data
            entry_index = None
            for i, idx in enumerate(df_chart.index):
                if idx >= self.entry_time:
                    entry_index = i
                    break
            
            if entry_index is None:
                entry_index = 15  # Default to middle if exact time not found
            
            self.progress_signal.emit(30)
            
            # Process data through calculators UP TO entry time only
            self.status_signal.emit("Running trend calculations...")
            
            # 1. Statistical Trends - feed 1-min bars
            last_1min_signal = None
            last_5min_signal = None
            last_15min_signal = None
            
            for idx, row in df_calc.iterrows():
                if idx > self.entry_time:
                    break  # Don't use future data
                    
                # 1-minute trend
                signal = self.calc_1min.update_price(
                    symbol=self.symbol,
                    price=row['close'],
                    volume=row['volume'],
                    timestamp=idx
                )
                if signal:
                    last_1min_signal = signal
            
            results['trend_1min'] = last_1min_signal
            self.progress_signal.emit(40)
            
            # 2. Process 5-min and 15-min aggregated data
            # Create 5-min bars
            df_5min = df_calc.resample('5min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            for idx, row in df_5min.iterrows():
                if idx > self.entry_time:
                    break
                    
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
                    last_5min_signal = signal
            
            results['trend_5min'] = last_5min_signal
            self.progress_signal.emit(50)
            
            # Create 15-min bars
            df_15min = df_calc.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            for idx, row in df_15min.iterrows():
                if idx > self.entry_time:
                    break
                    
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
                    last_15min_signal = signal
            
            results['trend_15min'] = last_15min_signal
            self.progress_signal.emit(60)
            
            # 3. Fetch and process REAL trade data with concurrent fetching
            self.status_signal.emit("Fetching trade data from Polygon (optimized)...")
            
            # Use async context manager for proper session handling
            async with self.fetcher as fetcher:
                # Determine optimal chunk size based on symbol liquidity
                # High volume symbols need smaller chunks
                avg_volume = df_calc['volume'].mean()
                if avg_volume > 10_000_000:  # Very high volume (SPY, TSLA, etc.)
                    chunk_minutes = 5
                elif avg_volume > 1_000_000:  # High volume
                    chunk_minutes = 10
                else:  # Normal volume
                    chunk_minutes = 15
                
                # Fetch trades
                trades = await fetcher.fetch_trades_concurrent(
                    symbol=self.symbol,
                    start_date=calc_start,
                    end_date=self.entry_time,
                    chunk_minutes=chunk_minutes,
                    progress_callback=self._trade_fetch_progress
                )
                
                # NEW: Fetch quotes for bid/ask imbalance
                self.status_signal.emit("Fetching quote data from Polygon...")
                quotes = await fetcher.fetch_quotes_concurrent(
                    symbol=self.symbol,
                    start_date=calc_start,
                    end_date=self.entry_time,
                    chunk_minutes=chunk_minutes,
                    progress_callback=self._quote_fetch_progress
                )
            
            # Process quotes first
            if quotes:
                self.status_signal.emit(f"Processing {len(quotes):,} quotes...")
                quote_count = 0
                
                for quote_data in quotes:
                    # Convert timestamp from nanoseconds to datetime
                    quote_time = datetime.fromtimestamp(quote_data['timestamp'] / 1e9, tz=timezone.utc)
                    
                    # Skip if after entry time
                    if quote_time > self.entry_time:
                        continue
                    
                    # Create Quote object
                    quote_obj = Quote(
                        symbol=self.symbol,
                        bid=quote_data['bid'],
                        ask=quote_data['ask'],
                        bid_size=quote_data['bid_size'],
                        ask_size=quote_data['ask_size'],
                        timestamp=quote_time
                    )
                    
                    # Process quote
                    self.bid_ask_calc.process_quote(quote_obj)
                    quote_count += 1
                    
                    # Update progress periodically
                    if quote_count % 10000 == 0:
                        progress_pct = int((quote_count / len(quotes)) * 10) + 65  # 65-75% range
                        self.progress_signal.emit(progress_pct)
                
                logger.info(f"Processed {quote_count:,} quotes")
            
            if trades:
                self.status_signal.emit(f"Processing {len(trades):,} trades...")
                total_trades = len(trades)
                
                # Process trades in optimized batches
                # Larger batches for better performance
                batch_size = 10000
                processed_count = 0
                
                for batch_start in range(0, total_trades, batch_size):
                    batch_end = min(batch_start + batch_size, total_trades)
                    batch = trades[batch_start:batch_end]
                    
                    # Update progress
                    progress_pct = int((batch_end / total_trades) * 15) + 75  # 75-90% range
                    self.progress_signal.emit(progress_pct)
                    self.trade_progress.emit(
                        f"Processing trades {batch_start:,}-{batch_end:,} of {total_trades:,}", 
                        progress_pct
                    )
                    
                    # Process each trade in the batch
                    for trade_data in batch:
                        # Convert timestamp from nanoseconds to datetime
                        trade_time = datetime.fromtimestamp(trade_data['timestamp'] / 1e9, tz=timezone.utc)
                        
                        # Skip if after entry time
                        if trade_time > self.entry_time:
                            continue
                        
                        processed_count += 1
                        
                        # Convert to format expected by calculators
                        formatted_trade = {
                            'symbol': trade_data['symbol'],
                            'price': trade_data['price'],
                            'size': trade_data['size'],
                            'timestamp': int(trade_data['timestamp'] / 1e6),  # Convert to milliseconds
                            'conditions': trade_data.get('conditions', []),
                            'exchange': trade_data.get('exchange', 0)
                        }
                        
                        # Trade Size Distribution
                        trade_obj = Trade(
                            symbol=self.symbol,
                            price=trade_data['price'],
                            size=trade_data['size'],
                            timestamp=trade_time
                        )
                        self.trade_size_calc.process_trade(trade_obj)
                        
                        # Bid/Ask Imbalance
                        bid_ask_signal = self.bid_ask_calc.process_trade(trade_obj)
                        
                        # Tick Flow
                        self.tick_flow_calc.process_trade(self.symbol, formatted_trade)
                        
                        # Volume Analysis (aggregates into bars)
                        self.volume_1min_calc.process_trade(self.symbol, formatted_trade)
                        self.market_context_calc.process_trade(self.symbol, formatted_trade)
                    
                    # Small delay to keep UI responsive
                    await asyncio.sleep(0.001)
                
                self.status_signal.emit(f"Processed {processed_count:,} trades successfully")
                
                # Log summary statistics
                time_range_hours = (self.entry_time - calc_start).total_seconds() / 3600
                logger.info(f"Trade processing complete:")
                logger.info(f"  - Total trades fetched: {total_trades:,}")
                logger.info(f"  - Trades processed: {processed_count:,}")
                logger.info(f"  - Time range: {time_range_hours:.1f} hours")
                logger.info(f"  - Avg trades/min: {processed_count / (time_range_hours * 60):.0f}")
                
            else:
                self.status_signal.emit("No trade data available - order flow signals will be empty")
            
            # Get final signals
            results['trade_size'] = self.trade_size_calc.latest_signals.get(self.symbol)
            results['bid_ask'] = self.bid_ask_calc.latest_signals.get(self.symbol)  # NEW
            results['tick_flow'] = self.tick_flow_calc.get_current_analysis(self.symbol)
            results['volume_1min'] = self.volume_1min_calc.get_current_analysis(self.symbol)
            results['market_context'] = self.market_context_calc.get_current_analysis(self.symbol)
            
            self.progress_signal.emit(95)
            
            # Emit results
            self.calculation_complete.emit(results)
            self.chart_data_ready.emit(df_chart, entry_index)
            
            self.progress_signal.emit(100)
            self.status_signal.emit(f"Backtest complete for {self.symbol} at {self.entry_time.strftime('%Y-%m-%d %H:%M UTC')}")
            
        except Exception as e:
            self.error_signal.emit(f"Backtest error: {str(e)}")
            logger.error(f"Backtest error: {e}", exc_info=True)
            
    def run(self):
        """Run the worker thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self.run_backtest())
        except Exception as e:
            self.error_signal.emit(f"Worker error: {str(e)}")
        finally:
            loop.close()


class BacktestingDashboard(QMainWindow):
    """Main backtesting dashboard window"""
    
    def __init__(self):
        super().__init__()
        self.worker = None
        self.latest_results = None  # Store latest backtest results
        self.latest_context = None  # Store context for Claude
        self.init_ui()
        self.apply_dark_theme()
        
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("Backtesting Dashboard - Historical Entry Analysis")
        self.setGeometry(100, 100, 1600, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Top controls
        controls_widget = QWidget()
        controls_widget.setMaximumHeight(80)
        controls_layout = QGridLayout(controls_widget)
        
        # Input fields
        controls_layout.addWidget(QLabel("Symbol:"), 0, 0)
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("e.g., TSLA")
        self.symbol_input.setMaximumWidth(100)
        controls_layout.addWidget(self.symbol_input, 0, 1)
        
        controls_layout.addWidget(QLabel("Date:"), 0, 2)
        self.date_input = QDateEdit()
        self.date_input.setCalendarPopup(True)
        self.date_input.setDate(QDate.currentDate())
        self.date_input.setDisplayFormat("yyyy-MM-dd")
        controls_layout.addWidget(self.date_input, 0, 3)
        
        controls_layout.addWidget(QLabel("Time (UTC):"), 0, 4)  # Added UTC label
        self.time_input = QTimeEdit()
        self.time_input.setDisplayFormat("HH:mm:ss")
        self.time_input.setTime(QTime(14, 30, 0))  # Default to 14:30 UTC (9:30 AM ET)
        controls_layout.addWidget(self.time_input, 0, 5)
        
        # Run button
        self.run_btn = QPushButton("Run Backtest")
        self.run_btn.clicked.connect(self.run_backtest)
        self.run_btn.setMinimumWidth(120)
        controls_layout.addWidget(self.run_btn, 0, 6)
        
        # Claude export button - initially disabled
        self.claude_btn = QPushButton("Export to Claude")
        self.claude_btn.clicked.connect(self.export_to_claude)
        self.claude_btn.setMinimumWidth(120)
        self.claude_btn.setEnabled(False)
        self.claude_btn.setStyleSheet("""
            QPushButton:disabled {
                background-color: #4a4a4a;
                color: #888888;
            }
        """)
        controls_layout.addWidget(self.claude_btn, 0, 7)
        
        # Status and progress
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #00cc00;")
        controls_layout.addWidget(self.status_label, 1, 0, 1, 4)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar, 1, 4, 1, 4)
        
        controls_layout.setColumnStretch(8, 1)
        main_layout.addWidget(controls_widget)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(line)
        
        # Main content - Splitter with calculations and chart
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - Calculations (scrollable)
        left_scroll = QScrollArea()
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Trend Analysis Section
        trend_group = QGroupBox("Trend Analysis")
        trend_layout = QVBoxLayout(trend_group)
        
        self.widget_1min = TimeframeWidget("1-Minute Scalper", "1min")
        self.widget_1min.setMaximumHeight(350)
        trend_layout.addWidget(self.widget_1min)
        
        self.widget_5min = TimeframeWidget("5-Minute Position", "5min")
        self.widget_5min.setMaximumHeight(350)
        trend_layout.addWidget(self.widget_5min)
        
        self.widget_15min = TimeframeWidget("15-Minute Regime", "15min")
        self.widget_15min.setMaximumHeight(350)
        trend_layout.addWidget(self.widget_15min)
        
        left_layout.addWidget(trend_group)
        
        # Order Flow Section
        flow_group = QGroupBox("Order Flow Analysis")
        flow_layout = QGridLayout(flow_group)
        
        self.widget_trade_size = OrderFlowWidget("Trade Size Distribution", "trade_size")
        self.widget_trade_size.setMaximumHeight(250)
        flow_layout.addWidget(self.widget_trade_size, 0, 0)
        
        self.widget_bid_ask = OrderFlowWidget("Bid/Ask Imbalance", "bid_ask")  # NEW WIDGET
        self.widget_bid_ask.setMaximumHeight(250)
        flow_layout.addWidget(self.widget_bid_ask, 0, 1)
        
        self.widget_tick_flow = OrderFlowWidget("Tick Flow", "tick_flow")
        self.widget_tick_flow.setMaximumHeight(250)
        flow_layout.addWidget(self.widget_tick_flow, 1, 0)
        
        self.widget_volume_1min = OrderFlowWidget("1-Min Volume", "volume_1min")
        self.widget_volume_1min.setMaximumHeight(250)
        flow_layout.addWidget(self.widget_volume_1min, 1, 1)
        
        self.widget_market_context = OrderFlowWidget("Market Context", "market_context")
        self.widget_market_context.setMaximumHeight(250)
        flow_layout.addWidget(self.widget_market_context, 2, 0, 1, 2)  # Span 2 columns
        
        left_layout.addWidget(flow_group)
        left_layout.addStretch()
        
        left_scroll.setWidget(left_widget)
        left_scroll.setWidgetResizable(True)
        self.splitter.addWidget(left_scroll)
        
        # Right side - Chart
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        
        chart_label = QLabel("1-Minute Chart (15 before → 60 after entry)")
        chart_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chart_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        chart_layout.addWidget(chart_label)
        
        self.chart = CandlestickChart()
        chart_layout.addWidget(self.chart)
        
        self.splitter.addWidget(chart_widget)
        
        # Set splitter sizes (60% calculations, 40% chart)
        self.splitter.setSizes([960, 640])
        
        # Add splitter with stretch to take up most space
        main_layout.addWidget(self.splitter, 1)  # stretch factor of 1
        
        # Bottom info bar - Fixed height
        self.info_bar = QLabel("Enter symbol, date, and time (UTC), then click 'Run Backtest'")
        self.info_bar.setMaximumHeight(30)  # Fixed maximum height
        self.info_bar.setStyleSheet("background-color: #1a1a1a; padding: 5px; font-size: 12px;")
        main_layout.addWidget(self.info_bar, 0)  # stretch factor of 0
        
    def apply_dark_theme(self):
        """Apply dark theme"""
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
                padding-top: 10px;
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
                font-size: 11px;
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
            QDateEdit, QTimeEdit, QLineEdit, QComboBox {
                background-color: #3a3a3a;
                border: 1px solid #555;
                padding: 5px;
                border-radius: 3px;
            }
            QDateEdit::drop-down, QTimeEdit::drop-down, QComboBox::drop-down {
                background-color: #555;
            }
        """)
        
    def run_backtest(self):
        """Run the backtest"""
        # Validate inputs
        symbol = self.symbol_input.text().strip().upper()
        if not symbol:
            QMessageBox.warning(self, "Input Error", "Please enter a symbol")
            return
            
        # Combine date and time
        date = self.date_input.date().toPyDate()
        input_time = self.time_input.time().toPyTime()
        
        # Create datetime in UTC
        entry_time = datetime.combine(date, input_time)
        entry_time = entry_time.replace(tzinfo=timezone.utc)  # Always UTC
        
        # Check if market is open at that time (UTC hours)
        # NYSE market hours: 9:30 AM - 4:00 PM ET
        # In UTC: 14:30 - 21:00 (EST) or 13:30 - 20:00 (EDT)
        utc_hour = input_time.hour
        utc_minute = input_time.minute
        
        if entry_time.weekday() >= 5:  # Weekend
            QMessageBox.warning(self, "Input Error", "Selected date is a weekend")
            return
            
        # Check market hours in UTC (conservative check for both EST and EDT)
        market_open_utc = (utc_hour == 13 and utc_minute >= 30) or (utc_hour == 14 and utc_minute >= 30) or utc_hour > 14
        market_close_utc = utc_hour < 21
        
        if not (market_open_utc and market_close_utc):
            reply = QMessageBox.question(
                self, "Outside Market Hours",
                "Selected time appears to be outside regular market hours (13:30-21:00 UTC). Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        # Store context for Claude
        self.latest_context = {
            'symbol': symbol,
            'entry_time': entry_time.strftime('%Y-%m-%d %H:%M:%S UTC'),
            'timeframe': '1min'
        }
        
        # Update UI
        self.status_label.setText(f"Running backtest for {symbol}...")
        self.status_label.setStyleSheet("color: #ffcc00;")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.run_btn.setEnabled(False)
        self.claude_btn.setEnabled(False)  # Disable Claude button during backtest
        
        # Clear previous results
        self._clear_displays()
        self.latest_results = None
        
        # Create and start worker
        self.worker = BacktestWorker()
        self.worker.set_parameters(symbol, entry_time)
        
        # Connect signals
        self.worker.calculation_complete.connect(self.display_results)
        self.worker.chart_data_ready.connect(self.display_chart)
        self.worker.error_signal.connect(self.handle_error)
        self.worker.status_signal.connect(self.update_status)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.trade_progress.connect(self.update_trade_progress)
        self.worker.quote_progress.connect(self.update_quote_progress)  # NEW
        
        # Start worker
        self.worker.start()
        
        # Update info bar
        self.info_bar.setText(
            f"Backtesting {symbol} at {entry_time.strftime('%Y-%m-%d %H:%M:%S UTC')} - "
            f"Fetching real trade and quote data from Polygon.io"
        )
        
    def _clear_displays(self):
        """Clear all display widgets"""
        # Reset signal frames to loading state
        for widget in [self.widget_1min, self.widget_5min, self.widget_15min]:
            widget.signal_label.setText("CALCULATING...")
            widget.signal_frame.setStyleSheet("background-color: #2a2a3a;")
            widget.signal_label.setStyleSheet("color: #ffcc00;")
            widget.table.clearContents()
            
        for widget in [self.widget_trade_size, self.widget_bid_ask, self.widget_tick_flow, 
                      self.widget_volume_1min, self.widget_market_context]:
            widget.signal_label.setText("CALCULATING...")
            widget.signal_frame.setStyleSheet("background-color: #2a2a3a;")
            widget.signal_label.setStyleSheet("color: #999999;")
            widget.table.clearContents()
            
    @pyqtSlot(dict)
    def display_results(self, results: Dict):
        """Display calculation results"""
        # Store results for Claude export
        self.latest_results = results
        
        # Trend signals
        if results.get('trend_1min'):
            self.widget_1min.update_1min_signal(results['trend_1min'])
        else:
            self.widget_1min.signal_label.setText("NO SIGNAL")
            self.widget_1min.signal_label.setStyleSheet("color: #666666;")
            
        if results.get('trend_5min'):
            self.widget_5min.update_5min_signal(results['trend_5min'])
        else:
            self.widget_5min.signal_label.setText("NO SIGNAL")
            self.widget_5min.signal_label.setStyleSheet("color: #666666;")
            
        if results.get('trend_15min'):
            self.widget_15min.update_15min_signal(results['trend_15min'])
        else:
            self.widget_15min.signal_label.setText("NO SIGNAL")
            self.widget_15min.signal_label.setStyleSheet("color: #666666;")
        
        # Order flow signals
        if results.get('trade_size'):
            self.widget_trade_size.update_trade_size_signal(results['trade_size'])
        else:
            self.widget_trade_size.signal_label.setText("NO DATA")
            
        if results.get('bid_ask'):  # NEW
            self.widget_bid_ask.update_bid_ask_signal(results['bid_ask'])
        else:
            self.widget_bid_ask.signal_label.setText("NO DATA")
            
        if results.get('tick_flow'):
            self.widget_tick_flow.update_tick_flow_signal(results['tick_flow'])
        else:
            self.widget_tick_flow.signal_label.setText("NO DATA")
            
        if results.get('volume_1min'):
            self.widget_volume_1min.update_volume_1min_signal(results['volume_1min'])
        else:
            self.widget_volume_1min.signal_label.setText("NO DATA")
            
        if results.get('market_context'):
            self.widget_market_context.update_market_context_signal(results['market_context'])
        else:
            self.widget_market_context.signal_label.setText("NO DATA")
            
    @pyqtSlot(object, int)
    def display_chart(self, df: pd.DataFrame, entry_index: int):
        """Display candlestick chart"""
        self.chart.plot_candlesticks(df, entry_index)
        
    @pyqtSlot(str)
    def handle_error(self, error_msg: str):
        """Handle errors"""
        self.status_label.setText(f"Error: {error_msg}")
        self.status_label.setStyleSheet("color: #ff0000;")
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.claude_btn.setEnabled(False)
        QMessageBox.critical(self, "Backtest Error", error_msg)
        
    @pyqtSlot(str)
    def update_status(self, status: str):
        """Update status"""
        self.status_label.setText(status)
        if "complete" in status.lower():
            self.status_label.setStyleSheet("color: #00cc00;")
            self.progress_bar.setVisible(False)
            self.run_btn.setEnabled(True)
            # Enable Claude button when backtest is complete
            if self.latest_results:
                self.claude_btn.setEnabled(True)
                self.info_bar.setText(
                    f"Backtest complete! Click 'Export to Claude' to get AI analysis and optimization recommendations."
                )
            
    @pyqtSlot(int)
    def update_progress(self, value: int):
        """Update progress"""
        self.progress_bar.setValue(value)
        
    @pyqtSlot(str, int)
    def update_trade_progress(self, message: str, progress: int):
        """Update trade loading progress"""
        self.status_label.setText(message)
        self.progress_bar.setValue(progress)
        
    @pyqtSlot(str, int)
    def update_quote_progress(self, message: str, progress: int):
        """Update quote loading progress"""
        self.status_label.setText(message)
        # Don't update progress bar as it's handled by trades
        
    def export_to_claude(self):
        """Export results to Claude for analysis"""
        if not self.latest_results or not self.latest_context:
            QMessageBox.warning(self, "No Results", "Please run a backtest first")
            return
            
        try:
            # Create Claude dialog with results
            self.claude_dialog = ClaudeConversationDialog(
                parent=self,
                initial_results=self.latest_results,
                context=self.latest_context
            )
            
            # Show dialog
            self.claude_dialog.show()
            
            # Update status
            self.info_bar.setText("Claude analysis window opened - analyzing backtest results...")
            
        except Exception as e:
            QMessageBox.critical(self, "Claude Error", f"Failed to open Claude dialog: {str(e)}")
            logger.error(f"Claude export error: {e}", exc_info=True)


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    dashboard = BacktestingDashboard()
    dashboard.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()