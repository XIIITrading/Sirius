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

# Import the plugin functions
from backtest.plugins.buy_sell_ratio import run_analysis_with_progress
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
        # Create data manager for testing
        self.data_manager = PolygonDataManager()
        self.chart_data = None  # Store for chart display
        self.test_progress = 0
        
    def progress_callback(self, percentage: int, message: str):
        """Progress callback for testing"""
        if percentage > self.test_progress:
            self.test_progress = percentage
            print(f"[{percentage:3d}%] {message}")
        
    async def run_test(self, symbol: str, test_time: datetime, direction: str, show_chart: bool = False):
        """Run bid/ask ratio analysis test using the plugin interface"""
        
        print(f"\n{'='*80}")
        print(f"BID/ASK RATIO PLUGIN TEST")
        print(f"Symbol: {symbol}")
        print(f"Entry Time: {test_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Direction: {direction}")
        print(f"{'='*80}\n")
        
        # Reset progress
        self.test_progress = 0
        
        try:
            # Call the plugin with progress tracking
            result = await run_analysis_with_progress(
                self.data_manager,
                symbol=symbol,
                entry_time=test_time,
                direction=direction,
                progress_callback=self.progress_callback
            )
            
            # Check for errors
            if 'error' in result:
                print(f"\nERROR: {result['error']}")
                return
            
            # Display results
            print(f"\n{'='*60}")
            print("ANALYSIS RESULTS")
            print(f"{'='*60}")
            
            # Signal assessment
            signal = result['signal']
            print(f"\nSignal Assessment:")
            print(f"  Direction: {signal['direction']}")
            print(f"  Strength: {signal['strength']:.1f}%")
            print(f"  Confidence: {signal['confidence']:.1f}%")
            
            # Details
            details = result['details']
            print(f"\nMetrics:")
            print(f"  Current Ratio: {details['current_ratio']:+.3f}")
            print(f"  Average Ratio: {details['average_ratio']:+.3f}")
            print(f"  Max Ratio: {details['max_ratio']:+.3f}")
            print(f"  Min Ratio: {details['min_ratio']:+.3f}")
            print(f"  Total Volume: {details['total_volume']:,.0f}")
            print(f"  Minutes Tracked: {details['minutes_tracked']}")
            
            # Alignment
            print(f"\nTrade Direction: {direction}")
            if details['aligned']:
                print("✅ ALIGNED - Order flow supports trade direction")
            else:
                print("⚠️  WARNING - Order flow contradicts intended direction")
            
            # Display data
            display = result['display_data']
            print(f"\nSummary: {display['summary']}")
            print(f"Description: {display['description']}")
            
            # Recent bars table
            if 'recent_bars' in display:
                recent_bars = display['recent_bars']
                print(f"\nRecent Minute Bars:")
                # Print headers
                headers = recent_bars['headers']
                print(f"{headers[0]:<10} {headers[1]:>10} {headers[2]:>12} {headers[3]:>12} {headers[4]:>12}")
                print("-" * 60)
                # Print rows
                for row in recent_bars['rows']:
                    print(f"{row[0]:<10} {row[1]:>10} {row[2]:>12} {row[3]:>12} {row[4]:>12}")
            
            # Store chart data for visualization
            if 'chart_widget' in display:
                self.chart_data = display['chart_widget']['data']
                
                # Show pressure trend
                if self.chart_data and len(self.chart_data) >= 5:
                    print(f"\nPressure Trend (5-minute samples):")
                    
                    # Sample every 5 minutes
                    sample_indices = list(range(0, len(self.chart_data), 5))
                    if len(self.chart_data) - 1 not in sample_indices:
                        sample_indices.append(len(self.chart_data) - 1)
                    
                    for idx in sample_indices:
                        point = self.chart_data[idx]
                        time_str = datetime.fromisoformat(point['timestamp']).strftime('%H:%M')
                        pressure = point['buy_sell_ratio']
                        bar = '█' * int(abs(pressure) * 20)
                        
                        # Calculate minutes from entry
                        point_time = datetime.fromisoformat(point['timestamp'])
                        mins_from_entry = int((point_time - test_time).total_seconds() / 60)
                        
                        if pressure >= 0:
                            print(f"{time_str} ({mins_from_entry:3d}m) | {' '*20}|{bar:<20} {pressure:+.3f}")
                        else:
                            print(f"{time_str} ({mins_from_entry:3d}m) | {bar:>20}|{' '*20} {pressure:+.3f}")
            
            print(f"\n{'='*80}\n")
            
            # Show chart if requested
            if show_chart and self.chart_data:
                self.display_chart(symbol, test_time, direction, display['chart_widget']['entry_time'])
                
        except Exception as e:
            print(f"\nERROR during test: {e}")
            import traceback
            traceback.print_exc()
    
    def display_chart(self, symbol: str, test_time: datetime, direction: str, entry_time_iso: str):
        """Display the chart with the analyzed data"""
        from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
        from PyQt6.QtCore import Qt
        
        # Import the chart
        sys.path.append(str(Path(__file__).parent))
        from chart import BidAskRatioChart
        
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        # Create main window
        window = QMainWindow()
        window.setWindowTitle(f"Bid/Ask Ratio Chart - {symbol} - Entry: {test_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        window.setGeometry(100, 100, 1200, 800)
        
        # Create central widget with layout
        central_widget = QWidget()
        window.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add title label with time window info
        start_time = test_time - timedelta(minutes=30)
        title_text = (f"{symbol} - {direction} Trade Analysis\n"
                     f"Window: {start_time.strftime('%H:%M:%S')} to {test_time.strftime('%H:%M:%S')} UTC (30 minutes)")
        title = QLabel(title_text)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Create and add chart
        chart = BidAskRatioChart()
        layout.addWidget(chart)
        
        # Update chart with data using the standard interface
        chart.update_from_data(self.chart_data)
        
        # Add entry marker
        chart.add_entry_marker(entry_time_iso)
        
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
        description='Bid/Ask Ratio Plugin Test - Tests the plugin interface for order flow analysis'
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
        help='Entry time in format "YYYY-MM-DD HH:MM:SS" (analyzes 30 minutes before this time)'
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
        help='Display interactive chart after analysis'
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
        # Default to current time (will analyze 30 minutes before now)
        test_time = datetime.now(timezone.utc)
        print(f"No time specified, using current time: {test_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"Will analyze 30 minutes before this time")
    
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