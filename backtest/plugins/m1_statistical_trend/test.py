# backtest/plugins/m1_statistical_trend/test.py
"""
Test for 1-Minute Statistical Trend Plugin
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

from backtest.plugins.m1_statistical_trend import run_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


async def test_statistical_trend(symbol: str, test_time: datetime, direction: str):
    """Run statistical trend analysis test"""
    
    print(f"\n{'='*60}")
    print(f"1-MINUTE STATISTICAL TREND ANALYSIS")
    print(f"Symbol: {symbol}")
    print(f"Test Time: {test_time}")
    print(f"Direction: {direction}")
    print(f"{'='*60}\n")
    
    try:
        # Run the analysis
        print("Running statistical trend analysis...")
        result = await run_analysis(symbol, test_time, direction)
        
        if 'error' in result:
            print(f"ERROR: {result['error']}")
            return
        
        # Display results
        print(f"\nANALYSIS COMPLETE")
        print(f"{'='*60}")
        
        # Main signal
        details = result['details']
        print(f"\nTRADING SIGNAL: {details['trading_signal']}")
        print(f"Confidence: {result['signal']['confidence']:.0f}%")
        print(f"Strength: {result['signal']['strength']:.0f}%")
        print(f"Target Hold: {details['target_hold']}")
        print(f"Reason: {details['reason']}")
        
        # Alignment
        if details['aligned']:
            print(f"\n✅ SIGNAL ALIGNED with {direction} trade")
        else:
            print(f"\n⚠️ SIGNAL NOT ALIGNED with {direction} trade")
        
        # Timeframe analysis
        print(f"\nTIMEFRAME ANALYSIS:")
        if details['micro_trend']:
            micro = details['micro_trend']
            print(f"3-min: {micro.get('direction', 'N/A').upper()} "
                  f"(Momentum: {micro.get('momentum', 0):.2f}%, "
                  f"Strength: {micro.get('strength', 0):.0f}%)")
        
        if details['short_trend']:
            short = details['short_trend']
            print(f"5-min: {short.get('direction', 'N/A').upper()} "
                  f"(Strength: {short.get('strength', 0):.0f}%, "
                  f"Score: {short.get('score', 0):.3f})")
        
        if details['medium_trend']:
            medium = details['medium_trend']
            print(f"10-min: {medium.get('direction', 'N/A').upper()} "
                  f"(Strength: {medium.get('strength', 0):.0f}%, "
                  f"Score: {medium.get('score', 0):.3f})")
        
        # Signal progression
        if 'signal_progression' in details:
            print(f"\n{details['signal_progression']}")
        
        # Signal history
        display_data = result['display_data']
        if 'signal_history' in display_data:
            history = display_data['signal_history']
            print(f"\nSIGNAL HISTORY (last {details['signals_analyzed']} candles):")
            print("-" * 85)
            
            # Headers
            headers = history['headers']
            print(f"{headers[0]:^10} | {headers[1]:^8} | {headers[2]:^15} | {headers[3]:^6} | {headers[4]:^5} | {headers[5]:^15} | {headers[6]:^6}")
            print("-" * 85)
            
            # Rows
            for row in history['rows']:
                print(f"{row[0]:^10} | {row[1]:^8} | {row[2]:^15} | {row[3]:^6} | {row[4]:^5} | {row[5]:^15} | {row[6]:^6}")
        
        # Summary table
        print(f"\nSUMMARY:")
        print("-" * 40)
        for row in display_data['table_data']:
            print(f"{row[0]:.<20} {row[1]}")
        
        print(f"\n{'='*60}\n")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='1-Minute Statistical Trend Analysis Test'
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
    await test_statistical_trend(
        symbol=args.symbol.upper(),
        test_time=test_time,
        direction=args.direction
    )


if __name__ == "__main__":
    asyncio.run(main())