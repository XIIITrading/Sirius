# backtest/plugins/tick_flow/test.py
"""
Test for Tick Flow Analysis Plugin
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

# Import the plugin
from backtest.plugins.tick_flow import run_analysis, PLUGIN_NAME, PLUGIN_VERSION, CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


async def test_tick_flow(symbol: str, test_time: datetime, direction: str):
    """Test tick flow analysis plugin"""
    
    print(f"\n{'='*60}")
    print(f"TICK FLOW ANALYSIS PLUGIN TEST")
    print(f"Plugin: {PLUGIN_NAME} v{PLUGIN_VERSION}")
    print(f"Symbol: {symbol}")
    print(f"Test Time: {test_time}")
    print(f"Direction: {direction}")
    print(f"Config: {CONFIG}")
    print(f"{'='*60}\n")
    
    try:
        # Run the analysis
        print("Running analysis...")
        result = await run_analysis(symbol, test_time, direction)
        
        # Check for errors
        if 'error' in result:
            print(f"\nERROR: {result['error']}")
            return
        
        # Display results
        display_results(result)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


def display_results(result: dict):
    """Display the analysis results"""
    
    print(f"\n{'='*60}")
    print("ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    # Signal
    signal = result['signal']
    print(f"\nSIGNAL:")
    print(f"Direction: {signal['direction']}")
    print(f"Strength: {signal['strength']:.1f}%")
    print(f"Confidence: {signal['confidence']:.1f}%")
    
    # Details
    details = result['details']
    print(f"\nDETAILS:")
    print(f"Trades Analyzed: {details['trades_analyzed']}")
    print(f"Buy Volume %: {details['buy_volume_pct']:.1f}%")
    print(f"Momentum Score: {details['momentum_score']:+.1f}")
    print(f"Large Buy Trades: {details['large_buy_trades']}")
    print(f"Large Sell Trades: {details['large_sell_trades']}")
    print(f"Trade Rate: {details['trade_rate']:.1f} trades/sec")
    print(f"Price Trend: {details['price_trend']}")
    print(f"Aligned with Direction: {'✅ Yes' if details['aligned'] else '❌ No'}")
    
    # Display data
    display = result['display_data']
    print(f"\nSUMMARY: {display['summary']}")
    print(f"DESCRIPTION: {display['description']}")
    
    # Table data
    if display['table_data']:
        print(f"\nMETRICS:")
        print("-" * 40)
        for row in display['table_data']:
            print(f"{row[0]:<20} | {row[1]:>15}")
    
    # Signal history
    if display.get('signal_history') and display['signal_history']['rows']:
        print(f"\nSIGNAL HISTORY (Last 10):")
        print("-" * 60)
        headers = display['signal_history']['headers']
        print(f"{headers[0]:^12} | {headers[1]:^10} | {headers[2]:>8} | {headers[3]:>6}")
        print("-" * 60)
        
        for row in display['signal_history']['rows']:
            print(f"{row[0]:^12} | {row[1]:^10} | {row[2]:>8} | {row[3]:>6}")
    
    # Chart markers
    if display.get('chart_markers'):
        print(f"\nCHART MARKERS:")
        for marker in display['chart_markers']:
            print(f"- {marker['label']} ({marker['color']})")
    
    print(f"\n{'='*60}\n")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Test Tick Flow Analysis Plugin'
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
    await test_tick_flow(
        symbol=args.symbol.upper(),
        test_time=test_time,
        direction=args.direction
    )


if __name__ == "__main__":
    asyncio.run(main())