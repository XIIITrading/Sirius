# backtest/plugins/cum_delta/test.py
"""
Test for Cumulative Delta Analysis Plugin
"""

import asyncio
import argparse
from datetime import datetime, timedelta, timezone
import logging
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# Add parent paths
current_file = Path(__file__).resolve()
sirius_dir = current_file.parent.parent.parent.parent
sys.path.insert(0, str(sirius_dir))

from modules.calculations.order_flow.cum_delta import DeltaFlowAnalyzer, Trade, Quote
from backtest.data.polygon_data_manager import PolygonDataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class CumDeltaTest:
    def __init__(self):
        self.analyzer = DeltaFlowAnalyzer()
        self.data_manager = PolygonDataManager()
        
    async def run_test(self, symbol: str, test_time: datetime, direction: str, 
                      lookback_minutes: int = 45, plot_chart: bool = True):  # CHANGED: Default to 45
        """Run cumulative delta analysis test"""
        
        print(f"\n{'='*60}")
        print(f"CUMULATIVE DELTA ANALYSIS TEST")
        print(f"Symbol: {symbol}")
        print(f"Test Time: {test_time}")
        print(f"Direction: {direction}")
        print(f"Lookback: {lookback_minutes} minutes")
        print(f"{'='*60}\n")
        
        # Calculate time range
        start_time = test_time - timedelta(minutes=lookback_minutes)
        
        # Check for market open reset
        market_open = test_time.replace(hour=14, minute=30, second=0, microsecond=0)  # 9:30 AM ET in UTC
        reset_at_open = start_time < market_open < test_time
        
        # Fetch data
        print(f"Fetching trades from {start_time} to {test_time}...")
        trades_df = await self.data_manager.load_trades(symbol, start_time, test_time)
        
        print(f"Fetching quotes from {start_time} to {test_time}...")
        quotes_df = await self.data_manager.load_quotes(symbol, start_time, test_time)
        
        if trades_df.empty:
            print("ERROR: No trade data available")
            return
            
        print(f"Processing {len(trades_df)} trades with {len(quotes_df)} quotes...")
        
        # Initialize symbol
        self.analyzer.initialize_symbol(symbol)
        
        # Reset at market open if needed
        if reset_at_open:
            self.analyzer.reset_session(symbol, market_open)
            print(f"Session reset scheduled at market open: {market_open}")
        
        # Enable warmup mode for better performance
        self.analyzer.warmup_config['enabled'] = True
        
        # Use the analyzer's warmup method for efficiency
        print("Using optimized warmup processing...")
        last_signal = self.analyzer.warmup_with_trades(
            symbol=symbol,
            trades_df=trades_df,
            quotes_df=quotes_df,
            entry_time=test_time,
            progress_callback=lambda pct, msg: print(f"  [{pct:3d}%] {msg}")
        )
        
        # Get final analysis
        if last_signal:
            self._display_results(last_signal, direction)
            
            # Get time series for plotting
            time_series = self.analyzer.get_delta_time_series(symbol, lookback_minutes)
            
            if plot_chart and time_series:
                self._plot_delta_chart(symbol, time_series, test_time)
        else:
            print("No signals generated")
    
    def _display_results(self, signal, direction: str):
        """Display the analysis results"""
        print(f"\n{'='*50}")
        print("CUMULATIVE DELTA ANALYSIS RESULTS")
        print(f"{'='*50}")
        
        # Signal summary
        print(f"\nSIGNAL: {signal.signal_type}")
        print(f"Strength: {signal.signal_strength}")
        print(f"Bull Score: {signal.bull_score} | Bear Score: {signal.bear_score}")
        print(f"Confidence: {signal.confidence:.2%}")
        print(f"Reason: {signal.reason}")
        
        # Components
        comp = signal.components
        print(f"\nCUMULATIVE METRICS:")
        print(f"  Cumulative Delta: {comp.cumulative_delta:+,}")
        print(f"  Delta Rate: {comp.delta_rate:+.0f} per minute")
        print(f"  Delta Volatility: {comp.delta_volatility:.2f}")
        
        print(f"\nEFFICIENCY METRICS:")
        print(f"  Price Efficiency: {comp.efficiency:.3f}")
        print(f"  Directional Efficiency: {comp.directional_efficiency:.2%}")
        print(f"  Absorption Score: {comp.absorption_score:.2%}")
        
        print(f"\nTIMEFRAME DELTAS:")
        for tf, delta in comp.timeframe_deltas.items():
            print(f"  {tf}: {delta:+,}")
        
        # Divergences
        if comp.divergences:
            print(f"\nDIVERGENCES DETECTED:")
            for div in comp.divergences:
                print(f"  - {div['type'].upper()} on {div['timeframe']}: {div['description']}")
        
        # Warnings
        if signal.warnings:
            print(f"\nWARNINGS:")
            for warning in signal.warnings:
                print(f"  ⚠️ {warning}")
        
        # Trade alignment
        print(f"\nTRADE ALIGNMENT:")
        print(f"Intended Direction: {direction}")
        
        if direction == "LONG" and signal.bull_score >= 1:
            print("✅ ALIGNED - Delta supports LONG trade")
        elif direction == "SHORT" and signal.bear_score >= 1:
            print("✅ ALIGNED - Delta supports SHORT trade")
        else:
            print("⚠️  WARNING - Delta does not support intended direction")
        
        print(f"\n{'='*50}\n")
    
    def _plot_delta_chart(self, symbol: str, time_series, test_time: datetime):
        """Plot cumulative delta chart using PyQtGraph - clean line chart"""
        if not time_series:
            return
        
        try:
            import pyqtgraph as pg
            from PyQt6.QtWidgets import QApplication
            from PyQt6.QtCore import Qt
            import numpy as np
            
            # Ensure QApplication exists
            app = QApplication.instance()
            if app is None:
                app = QApplication([])
            
            # Convert to arrays
            timestamps = []
            cumulative_deltas = []
            
            for ts in time_series:
                timestamps.append(ts.timestamp.timestamp())
                cumulative_deltas.append(ts.cumulative_delta)
            
            x = np.array(timestamps)
            y = np.array(cumulative_deltas)
            
            # Create plot window
            pg.setConfigOptions(background='k', foreground='w')
            win = pg.GraphicsLayoutWidget(show=True)
            win.setWindowTitle(f"{symbol} - Cumulative Delta")
            win.resize(1000, 600)
            
            # Create plot
            plot = win.addPlot(title="Cumulative Delta Analysis")
            plot.setLabel('left', 'Cumulative Delta', units='')
            plot.setLabel('bottom', 'Time')
            plot.showGrid(x=True, y=True, alpha=0.3)
            
            # Add reference lines
            plot.addLine(y=0, pen=pg.mkPen('w', width=2))  # Zero line - white
            plot.addLine(y=5000, pen=pg.mkPen((100, 100, 100), width=1, style=Qt.PenStyle.DashLine))
            plot.addLine(y=-5000, pen=pg.mkPen((100, 100, 100), width=1, style=Qt.PenStyle.DashLine))
            
            # Add threshold labels
            text1 = pg.TextItem("+5000", color=(100, 100, 100))
            text1.setPos(x[0], 5000)
            plot.addItem(text1)
            
            text2 = pg.TextItem("-5000", color=(100, 100, 100))
            text2.setPos(x[0], -5000)
            plot.addItem(text2)
            
            # Main cumulative delta line - blue
            line = plot.plot(x, y, pen=pg.mkPen('#00BFFF', width=3), name='Cumulative Delta')
            
            # Add subtle fill
            fill = pg.FillBetweenItem(line, plot.plot(x, np.zeros_like(y), pen=None), 
                                    brush=pg.mkBrush(0, 191, 255, 30))
            plot.addItem(fill)
            
            # Entry time marker
            entry_line = pg.InfiniteLine(pos=test_time.timestamp(), angle=90, 
                                        pen=pg.mkPen('#FF1493', width=2, style=Qt.PenStyle.DashLine),
                                        label='Entry', labelOpts={'position': 0.95})
            plot.addItem(entry_line)
            
            # Format x-axis as time
            axis = pg.DateAxisItem(orientation='bottom')
            plot.setAxisItems({'bottom': axis})
            
            # Add value display on hover
            vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=1, style=Qt.PenStyle.DotLine))
            hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('y', width=1, style=Qt.PenStyle.DotLine))
            plot.addItem(vLine, ignoreBounds=True)
            plot.addItem(hLine, ignoreBounds=True)
            
            text = pg.TextItem(anchor=(1, 1))
            plot.addItem(text)
            
            def mouseMoved(evt):
                pos = evt[0]
                if plot.sceneBoundingRect().contains(pos):
                    mousePoint = plot.vb.mapSceneToView(pos)
                    idx = np.searchsorted(x, mousePoint.x())
                    if 0 <= idx < len(x):
                        vLine.setPos(mousePoint.x())
                        hLine.setPos(mousePoint.y())
                        dt = datetime.fromtimestamp(x[idx])
                        text.setText(f"Time: {dt.strftime('%H:%M:%S')}\nDelta: {y[idx]:+,}")
                        text.setPos(mousePoint.x(), mousePoint.y())
            
            proxy = pg.SignalProxy(plot.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)
            
            # Show
            if app.exec is not None:
                app.exec()
                
        except ImportError:
            print("PyQtGraph not installed. Install with: pip install pyqtgraph PyQt6")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Cumulative Delta Analysis Test'
    )
    
    parser.add_argument(
        '-s', '--symbol',
        type=str,
        required=True,
        help='Stock symbol (e.g., AAPL, TSLA, SPY)'
    )
    
    parser.add_argument(
        '-t', '--time',
        type=str,
        default=None,
        help='Analysis time in format "YYYY-MM-DD HH:MM:SS"'
    )
    
    parser.add_argument(
        '-d', '--direction',
        type=str,
        choices=['LONG', 'SHORT'],
        default='LONG',
        help='Trade direction (default: LONG)'
    )
    
    parser.add_argument(
        '-l', '--lookback',
        type=int,
        default=45,  # CHANGED: Default to 45 minutes
        help='Lookback period in minutes (default: 45)'
    )
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable chart plotting'
    )
    
    return parser.parse_args()


async def main():
    """Run the test with CLI arguments"""
    args = parse_arguments()
    
    # Parse datetime
    if args.time:
        try:
            test_time = datetime.strptime(args.time, "%Y-%m-%d %H:%M:%S")
            test_time = test_time.replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"ERROR: Invalid datetime format: {args.time}")
            print("Please use format: YYYY-MM-DD HH:MM:SS")
            return
    else:
        test_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        print(f"No time specified, using: {test_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # Create tester and run
    tester = CumDeltaTest()
    
    try:
        await tester.run_test(
            symbol=args.symbol.upper(),
            test_time=test_time,
            direction=args.direction,
            lookback_minutes=args.lookback,
            plot_chart=not args.no_plot
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())