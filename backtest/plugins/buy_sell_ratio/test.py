"""
Test for Bid/Ask Ratio Plugin with Chart Visualization
"""

import asyncio
import argparse
from datetime import datetime, timedelta, timezone
import logging
import sys
from pathlib import Path

# Add parent paths
current_file = Path(__file__).resolve()
sirius_dir = current_file.parent.parent.parent.parent
sys.path.insert(0, str(sirius_dir))

from modules.calculations.order_flow.buy_sell_ratio import SimpleDeltaTracker, Trade
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


class BidAskRatioTest:
    def __init__(self):
        self.data_manager = PolygonDataManager()
        self.chart_data = None  # Store for chart display
        
    async def run_test(self, symbol: str, test_time: datetime, direction: str, show_chart: bool = False):
        """Run bid/ask ratio analysis test - pure passthrough"""
        
        print(f"\n{'='*80}")
        print(f"BID/ASK RATIO TRACKER TEST")
        print(f"Symbol: {symbol}")
        print(f"Test Time: {test_time}")
        print(f"Direction: {direction}")
        print(f"{'='*80}\n")
        
        # Initialize tracker
        tracker = SimpleDeltaTracker(window_minutes=30)
        
        print(f"Configuration:")
        print(f"- Window: 30 minutes")
        print(f"- Output range: -1 to +1")
        print()
        
        # Fetch data
        data_start_time = test_time - timedelta(minutes=35)  # Extra 5 minutes
        
        print(f"Fetching data from {data_start_time} to {test_time}...")
        trades_df = await self.data_manager.load_trades(symbol, data_start_time, test_time)
        quotes_df = await self.data_manager.load_quotes(symbol, data_start_time, test_time)
        
        if trades_df.empty:
            print("ERROR: No trade data available")
            return
            
        print(f"\nData Summary:")
        print(f"- Trades loaded: {len(trades_df)}")
        print(f"- Quotes loaded: {len(quotes_df)}")
        
        # Process quotes
        print(f"\nProcessing quotes...")
        quote_count = 0
        for timestamp, quote_data in quotes_df.iterrows():
            tracker.update_quote(
                symbol=symbol,
                bid=float(quote_data['bid']),
                ask=float(quote_data['ask']),
                timestamp=timestamp.to_pydatetime() if hasattr(timestamp, 'to_pydatetime') else timestamp
            )
            quote_count += 1
            
            if quote_count % 5000 == 0:
                print(f"  Processed {quote_count} quotes...")
        
        print(f"Total quotes processed: {quote_count}")
        
        # Process trades
        print(f"\nProcessing trades...")
        completed_bars = []
        trades_processed = 0
        
        for timestamp, trade_data in trades_df.iterrows():
            # Create trade object
            trade = Trade(
                symbol=symbol,
                price=float(trade_data['price']),
                size=int(trade_data['size']),
                timestamp=timestamp.to_pydatetime() if hasattr(timestamp, 'to_pydatetime') else timestamp
            )
            
            # Process trade
            completed_bar = tracker.process_trade(trade)
            if completed_bar:
                completed_bars.append(completed_bar)
                
            trades_processed += 1
            
            if trades_processed % 5000 == 0:
                print(f"  Processed {trades_processed} trades, {len(completed_bars)} minute bars...")
        
        print(f"\nTotal trades processed: {trades_processed}")
        print(f"Minute bars completed: {len(completed_bars)}")
        
        # Get results from tracker
        chart_data = tracker.get_chart_data(symbol)
        self.chart_data = chart_data  # Store for chart display
        latest_ratio = tracker.get_latest_ratio(symbol)
        summary_stats = tracker.get_summary_stats(symbol)
        
        # Display results
        print(f"\n{'='*60}")
        print("ANALYSIS RESULTS")
        print(f"{'='*60}")
        
        if latest_ratio is not None:
            print(f"\nCurrent Weighted Pressure: {latest_ratio:+.3f}")
            print(f"Average Pressure: {summary_stats['avg_ratio']:+.3f}")
            print(f"Max Pressure: {summary_stats['max_ratio']:+.3f}")
            print(f"Min Pressure: {summary_stats['min_ratio']:+.3f}")
            print(f"Total Volume: {summary_stats['total_volume']:,.0f}")
            print(f"Minutes Tracked: {summary_stats['minutes_tracked']}")
            
            # Classification stats if available
            if 'classification_rate' in summary_stats:
                print(f"Classification Rate: {summary_stats['classification_rate']:.1%}")
            
            # Sync stats if available
            if 'sync_stats' in summary_stats:
                sync = summary_stats['sync_stats']
                print(f"\nSync Statistics:")
                print(f"  Total Trades: {sync.get('total_trades', 0)}")
                print(f"  Synced Trades: {sync.get('synced_trades', 0)}")
                print(f"  Failed Syncs: {sync.get('failed_syncs', 0)}")
            
            # Direction alignment
            print(f"\nTrade Direction: {direction}")
            if direction == "LONG" and latest_ratio > 0:
                print("✅ ALIGNED - Order flow supports LONG trade")
            elif direction == "SHORT" and latest_ratio < 0:
                print("✅ ALIGNED - Order flow supports SHORT trade")
            else:
                print("⚠️  WARNING - Order flow contradicts intended direction")
        
        # Show recent minute bars
        if completed_bars:
            print(f"\nRecent Minute Bars (last 10):")
            print(f"{'Time':<10} {'Pressure':>10} {'Pos Vol':>12} {'Neg Vol':>12} {'Total Vol':>12} {'Classified':>12}")
            print("-" * 80)
            
            for bar in completed_bars[-10:]:
                print(f"{bar.timestamp.strftime('%H:%M:%S'):<10} "
                      f"{bar.weighted_pressure:>10.3f} "
                      f"{bar.positive_volume:>12,.0f} "
                      f"{bar.negative_volume:>12,.0f} "
                      f"{bar.total_volume:>12,.0f} "
                      f"{bar.classified_trades:>12}")
        
        # Show chart data visualization
        if chart_data:
            print(f"\nChart Data Points: {len(chart_data)}")
            recent_chart = [p for p in chart_data[-30:]]
            
            if recent_chart:
                print(f"\nPressure Trend (5-minute intervals):")
                for i in range(0, len(recent_chart), 5):
                    point = recent_chart[i]
                    time_str = datetime.fromisoformat(point['timestamp']).strftime('%H:%M')
                    pressure = point['buy_sell_ratio']
                    bar = '█' * int(abs(pressure) * 20)
                    
                    if pressure >= 0:
                        print(f"{time_str} | {' '*20}|{bar:<20} {pressure:+.3f}")
                    else:
                        print(f"{time_str} | {bar:>20}|{' '*20} {pressure:+.3f}")
        
        print(f"\n{'='*80}\n")
        
        # Show chart if requested
        if show_chart and self.chart_data:
            self.display_chart(symbol, test_time, direction)
    
    def display_chart(self, symbol: str, test_time: datetime, direction: str):
        """Display the chart with the analyzed data"""
        from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
        from PyQt6.QtCore import Qt
        
        # Import the chart
        sys.path.append(str(Path(__file__).parent))
        from chart import BidAskRatioChart
        
        app = QApplication(sys.argv)
        
        # Create main window
        window = QMainWindow()
        window.setWindowTitle(f"Bid/Ask Ratio Chart - {symbol} - {test_time.strftime('%Y-%m-%d %H:%M:%S')}")
        window.setGeometry(100, 100, 1200, 800)
        
        # Create central widget with layout
        central_widget = QWidget()
        window.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add title label
        title = QLabel(f"{symbol} - {direction} - {test_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Create and add chart
        chart = BidAskRatioChart()
        layout.addWidget(chart)
        
        # Update chart with data
        chart.update_data(self.chart_data)
        
        # Add entry marker at the 30-minute mark (end of data)
        chart.add_marker(30, "Entry", "#ff0000")
        
        # Apply dark theme to window
        window.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QLabel {
                background-color: #1e1e1e;
                color: #ffffff;
            }
        """)
        
        window.show()
        
        # Keep the window open
        app.exec()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Bid/Ask Ratio Analysis Test'
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
        '-c', '--chart',
        action='store_true',
        help='Display chart after analysis'
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
    tester = BidAskRatioTest()
    
    try:
        await tester.run_test(
            symbol=args.symbol.upper(),
            test_time=test_time,
            direction=args.direction,
            show_chart=args.chart
        )
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())