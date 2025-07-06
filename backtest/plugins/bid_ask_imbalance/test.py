"""
Test for Bid/Ask Imbalance Analysis Plugin
Simplified production version for dashboard integration
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

# Import from the consolidated module
from backtest.plugins.bid_ask_imbalance import run_analysis_with_progress, get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SimpleTable:
    """Simple table formatter without external dependencies"""
    
    @staticmethod
    def format_table(data, headers=None, col_widths=None):
        """Format data as a simple ASCII table"""
        if not data:
            return ""
        
        # Calculate column widths
        if col_widths is None:
            col_widths = []
            if headers:
                for i, header in enumerate(headers):
                    max_width = len(str(header))
                    for row in data:
                        if i < len(row):
                            max_width = max(max_width, len(str(row[i])))
                    col_widths.append(max_width + 2)
            else:
                num_cols = len(data[0]) if data else 0
                for i in range(num_cols):
                    max_width = 0
                    for row in data:
                        if i < len(row):
                            max_width = max(max_width, len(str(row[i])))
                    col_widths.append(max_width + 2)
        
        # Build separator
        separator = "+" + "+".join("-" * width for width in col_widths) + "+"
        
        lines = [separator]
        
        # Add headers if provided
        if headers:
            header_line = "|"
            for i, header in enumerate(headers):
                width = col_widths[i] if i < len(col_widths) else 10
                header_line += f" {str(header):<{width-2}} |"
            lines.append(header_line)
            lines.append(separator)
        
        # Add data rows
        for row in data:
            row_line = "|"
            for i, cell in enumerate(row):
                width = col_widths[i] if i < len(col_widths) else 10
                row_line += f" {str(cell):<{width-2}} |"
            lines.append(row_line)
        
        lines.append(separator)
        return "\n".join(lines)


class BidAskImbalanceTest:
    """Test class for Bid/Ask Imbalance plugin"""
    
    def __init__(self):
        self.config = get_config()
        
    async def run_test(self, symbol: str, test_time: datetime, direction: str):
        """Run bid/ask imbalance analysis test"""
        
        print(f"\n{'='*80}")
        print(f"BID/ASK IMBALANCE ANALYSIS TEST")
        print(f"Symbol: {symbol}")
        print(f"Test Time: {test_time}")
        print(f"Direction: {direction}")
        print(f"{'='*80}\n")
        
        print(f"Configuration:")
        print(f"- Total trades tracked: {self.config['imbalance_lookback']}")
        print(f"- Trades per bar: {self.config['trades_per_bar']}")
        print(f"- Number of bars: {self.config['imbalance_lookback'] // self.config['trades_per_bar']}")
        print(f"- Quote sync tolerance: {self.config['quote_sync_tolerance_ms']}ms")
        print(f"- Count interval: Every {self.config['count_sample_interval']} trades")
        print(f"- Max trades to process: {self.config['max_trades_to_process']:,}")
        print()
        
        # Progress callback
        def progress_callback(percentage: int, message: str):
            print(f"[{percentage:3d}%] {message}")
        
        try:
            # Run analysis
            result = await run_analysis_with_progress(
                symbol=symbol,
                entry_time=test_time,
                direction=direction,
                progress_callback=progress_callback
            )
            
            # Check for errors
            if 'error' in result:
                print(f"\nERROR: {result['error']}")
                return
            
            # Display results
            self._display_results(result)
            
            # Display performance stats
            if 'performance_stats' in result:
                self._display_performance_stats(result['performance_stats'])
                
        except Exception as e:
            print(f"\nERROR during test: {e}")
            import traceback
            traceback.print_exc()
    
    def _display_results(self, result: dict):
        """Display the analysis results"""
        print(f"\n{'='*60}")
        print("ANALYSIS RESULTS")
        print(f"{'='*60}")
        
        # Signal summary
        signal = result.get('signal', {})
        print(f"\nSIGNAL SUMMARY:")
        print(f"Direction: {signal.get('direction', 'UNKNOWN')}")
        print(f"Strength: {signal.get('strength', 0):.0f}%")
        print(f"Confidence: {signal.get('confidence', 0):.0f}%")
        
        # Display data summary
        display_data = result.get('display_data', {})
        print(f"\nSummary: {display_data.get('summary', 'N/A')}")
        print(f"Description: {display_data.get('description', 'N/A')}")
        
        # Display main metrics table
        if 'table_data' in display_data:
            print(f"\nKEY METRICS:")
            table = SimpleTable.format_table(
                display_data['table_data'],
                headers=['Metric', 'Value']
            )
            print(table)
        
        # Display bar indices
        bar_indices = display_data.get('bar_indices', {})
        if bar_indices.get('rows'):
            print(f"\nBAR INDEX BREAKDOWN:")
            print("Shows imbalance evolution (100 trades per bar)")
            
            # Show first 10 and last 10 bars if more than 20
            rows = bar_indices['rows']
            if len(rows) > 20:
                display_rows = rows[:10] + [['...', '...', '...', '...', '...', '...', '...']] + rows[-10:]
            else:
                display_rows = rows
                
            table = SimpleTable.format_table(
                display_rows,
                headers=bar_indices.get('headers', [])
            )
            print(table)
            
            # Bar trend analysis
            if len(rows) >= 3:
                print("\nBAR TREND ANALYSIS:")
                recent_avg = sum(float(row[1].strip('%+')) for row in rows[:3]) / 3
                older_avg = sum(float(row[1].strip('%+')) for row in rows[3:6]) / 3 if len(rows) >= 6 else recent_avg
                
                if recent_avg > older_avg + 10:
                    print("📈 BULLISH ACCELERATION - Imbalance strengthening")
                elif recent_avg < older_avg - 10:
                    print("📉 BEARISH ACCELERATION - Imbalance strengthening")
                elif abs(recent_avg - older_avg) < 5:
                    print("➡️ STEADY FLOW - Consistent imbalance")
                else:
                    print("⚠️ MOMENTUM SLOWING - Imbalance weakening")
                    
                print(f"Recent bars average: {recent_avg:+.1f}%")
                print(f"Older bars average: {older_avg:+.1f}%")
    
    def _display_performance_stats(self, perf_stats: dict):
        """Display performance statistics"""
        print(f"\nPERFORMANCE STATISTICS:")
        perf_data = [
            ['Original Trades', f"{perf_stats.get('original_trades', 0):,}"],
            ['Processed Trades', f"{perf_stats.get('processed_trades', 0):,}"],
            ['Sampling Ratio', f"{perf_stats.get('sampling_ratio', 1):.2f}"],
            ['Count Interval', f"Every {perf_stats.get('count_interval', 'N/A')} trades"],
            ['Signals Generated', perf_stats.get('signals_generated', 0)],
            ['Sync Rate', f"{perf_stats.get('sync_rate', 0):.1f}%"],
            ['Sync Tolerance', f"{perf_stats.get('sync_tolerance_ms', 0)}ms"],
        ]
        table = SimpleTable.format_table(perf_data, headers=['Metric', 'Value'])
        print(table)
        
        # Analysis quality indicator
        sync_rate = perf_stats.get('sync_rate', 0)
        print("\nANALYSIS QUALITY:")
        if sync_rate >= 80:
            print("✅ EXCELLENT - High sync rate achieved")
        elif sync_rate >= 60:
            print("👍 GOOD - Acceptable sync rate")
        elif sync_rate >= 40:
            print("⚠️ FAIR - Moderate sync rate")
        else:
            print("❌ POOR - Low sync rate, results may be unreliable")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Bid/Ask Imbalance Analysis Test'
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
    
    # Run test
    tester = BidAskImbalanceTest()
    await tester.run_test(
        symbol=args.symbol.upper(),
        test_time=test_time,
        direction=args.direction
    )


if __name__ == "__main__":
    asyncio.run(main())