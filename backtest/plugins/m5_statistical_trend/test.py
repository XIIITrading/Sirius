# backtest/plugins/m5_statistical_trend/test.py
"""
Test for 5-Minute Statistical Trend Plugin (Modernized)
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

# Import the plugin and calculation module
import backtest.plugins.m5_statistical_trend as m5_plugin
from backtest.data.polygon_data_manager import PolygonDataManager
from backtest.calculations.trend.statistical_trend_5min import StatisticalTrend5Min

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


async def test_single_analysis(symbol: str, test_time: datetime, direction: str):
    """Test single point analysis"""
    
    print(f"\n{'='*60}")
    print(f"5-MINUTE STATISTICAL TREND ANALYSIS (MODERNIZED)")
    print(f"Symbol: {symbol}")
    print(f"Test Time: {test_time}")
    print(f"Direction: {direction}")
    print(f"{'='*60}\n")
    
    try:
        # Initialize data manager
        data_manager = PolygonDataManager()
        m5_plugin.set_data_manager(data_manager)
        
        # Run analysis
        print("Running analysis...")
        result = await m5_plugin.run_analysis(symbol, test_time, direction)
        
        if 'error' in result:
            print(f"ERROR: {result['error']}")
            return
        
        # Display results
        print(f"\nANALYSIS RESULTS")
        print(f"{'='*60}")
        
        details = result['details']
        print(f"\nSignal: {details['signal']}")
        print(f"Bias: {details['bias']}")
        print(f"Confidence: {result['signal']['confidence']:.0f}%")
        print(f"Strength: {result['signal']['strength']:.0f}%")
        print(f"Trend Strength: {details['trend_strength']:.2f}%")
        print(f"Volatility-Adjusted: {details['volatility_adjusted_strength']:.2f}x")
        print(f"Price: ${details['price']:.2f}")
        
        # Alignment
        if details['aligned']:
            print(f"\n✅ BIAS ALIGNED with {direction} trade")
        else:
            print(f"\n⚠️ BIAS NOT ALIGNED with {direction} trade")
        
        # Display table
        print(f"\nDETAILED METRICS:")
        print("-" * 50)
        for row in result['display_data']['table_data']:
            print(f"{row[0]:.<25} {row[1]}")
        
        print(f"\n{'='*60}\n")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


async def test_live_monitoring(symbol: str, duration_minutes: int = 5):
    """Test live monitoring mode"""
    
    print(f"\n{'='*60}")
    print(f"LIVE MONITORING MODE - {symbol}")
    print(f"Duration: {duration_minutes} minutes")
    print(f"{'='*60}\n")
    
    # Initialize data manager
    data_manager = PolygonDataManager()
    m5_plugin.set_data_manager(data_manager)
    
    end_time = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
    
    while datetime.now(timezone.utc) < end_time:
        current_time = datetime.now(timezone.utc)
        
        try:
            # Run analysis for current time
            result = await m5_plugin.run_analysis(symbol, current_time, 'LONG')
            
            if 'error' not in result:
                details = result['details']
                print(f"{current_time.strftime('%H:%M:%S')} - "
                      f"{details['signal']} | "
                      f"{details['bias']} | "
                      f"Confidence: {result['signal']['confidence']:.0f}% | "
                      f"Strength: {details['volatility_adjusted_strength']:.2f}x")
            else:
                print(f"{current_time.strftime('%H:%M:%S')} - ERROR: {result['error']}")
                
        except Exception as e:
            print(f"{current_time.strftime('%H:%M:%S')} - Exception: {e}")
        
        # Wait 30 seconds before next update
        await asyncio.sleep(30)
    
    print("\nMonitoring complete")


async def test_batch_analysis(symbol: str, start_time: datetime, hours: int = 1):
    """Test batch analysis over time range"""
    
    print(f"\n{'='*60}")
    print(f"BATCH ANALYSIS - {symbol}")
    print(f"Start: {start_time}")
    print(f"Duration: {hours} hours")
    print(f"{'='*60}\n")
    
    # Initialize data manager
    data_manager = PolygonDataManager()
    m5_plugin.set_data_manager(data_manager)
    
    # Analyze every 5 minutes
    current = start_time
    end_time = start_time + timedelta(hours=hours)
    
    results = []
    
    while current <= end_time:
        try:
            result = await m5_plugin.run_analysis(symbol, current, 'LONG')
            
            if 'error' not in result:
                details = result['details']
                results.append({
                    'time': current,
                    'signal': details['signal'],
                    'bias': details['bias'],
                    'confidence': result['signal']['confidence'],
                    'strength': details['volatility_adjusted_strength']
                })
                print(f".", end="", flush=True)
            else:
                print(f"E", end="", flush=True)
                
        except Exception:
            print(f"X", end="", flush=True)
        
        current += timedelta(minutes=5)
    
    # Summary
    print(f"\n\nANALYSIS SUMMARY ({len(results)} successful)")
    print("-" * 80)
    print(f"{'Time':^10} | {'Signal':^20} | {'Bias':^10} | {'Confidence':^10} | {'Strength':^10}")
    print("-" * 80)
    
    for r in results[-10:]:  # Show last 10
        print(f"{r['time'].strftime('%H:%M'):^10} | "
              f"{r['signal']:^20} | "
              f"{r['bias']:^10} | "
              f"{r['confidence']:^10.0f}% | "
              f"{r['strength']:^10.2f}x")


async def test_direct_calculation(symbol: str, test_time: datetime):
    """Test the calculation module directly without plugin"""
    
    print(f"\n{'='*60}")
    print(f"DIRECT CALCULATION TEST - {symbol}")
    print(f"Test Time: {test_time}")
    print(f"{'='*60}\n")
    
    try:
        # Initialize data manager
        data_manager = PolygonDataManager()
        
        # Create analyzer
        analyzer = StatisticalTrend5Min(lookback_periods=10)
        
        # Calculate time range
        start_time = test_time - timedelta(minutes=analyzer.lookback_periods * 5 + 10)
        
        print(f"Fetching data from {start_time} to {test_time}")
        
        # Fetch data
        bars_df = await data_manager.load_bars(
            symbol=symbol,
            start_time=start_time,
            end_time=test_time,
            timeframe='5min'
        )
        
        if bars_df is None or bars_df.empty:
            print("ERROR: No data available")
            return
        
        print(f"Fetched {len(bars_df)} bars")
        
        # Run analysis
        signal = analyzer.analyze(symbol, bars_df, test_time)
        
        # Display results
        print(f"\nCALCULATION RESULTS:")
        print(f"Signal: {signal.signal}")
        print(f"Bias: {signal.bias}")
        print(f"Confidence: {signal.confidence:.0f}%")
        print(f"Trend Strength: {signal.trend_strength:.2f}%")
        print(f"Volatility-Adjusted: {signal.volatility_adjusted_strength:.2f}x")
        print(f"Volume Confirmation: {'Yes' if signal.volume_confirmation else 'No'}")
        print(f"Price: ${signal.price:.2f}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='5-Minute Statistical Trend Analysis Test (Modernized)'
    )
    
    parser.add_argument(
        '-s', '--symbol',
        type=str,
        required=True,
        help='Stock symbol (e.g., AAPL, TSLA, SPY)'
    )
    
    parser.add_argument(
        '-m', '--mode',
        type=str,
        choices=['single', 'live', 'batch', 'direct'],
        default='single',
        help='Test mode (default: single)'
    )
    
    parser.add_argument(
        '-t', '--time',
        type=str,
        default=None,
        help='Analysis time for single/batch mode (YYYY-MM-DD HH:MM:SS)'
    )
    
    parser.add_argument(
        '-d', '--direction',
        type=str,
        choices=['LONG', 'SHORT'],
        default='LONG',
        help='Trade direction (default: LONG)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=5,
        help='Duration in minutes for live mode (default: 5)'
    )
    
    parser.add_argument(
        '--hours',
        type=int,
        default=1,
        help='Hours of data for batch mode (default: 1)'
    )
    
    return parser.parse_args()


async def main():
    """Run the test with CLI arguments"""
    args = parse_arguments()
    
    # Parse datetime if provided
    if args.time:
        try:
            test_time = datetime.strptime(args.time, "%Y-%m-%d %H:%M:%S")
            test_time = test_time.replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"ERROR: Invalid datetime format: {args.time}")
            print("Please use format: YYYY-MM-DD HH:MM:SS")
            return
    else:
        # Default to current time for live mode, market hours for others
        if args.mode == 'live':
            test_time = datetime.now(timezone.utc)
        else:
            test_time = datetime.now(timezone.utc).replace(hour=14, minute=30, second=0)
            print(f"No time specified, using: {test_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # Run appropriate test mode
    if args.mode == 'single':
        await test_single_analysis(args.symbol.upper(), test_time, args.direction)
    elif args.mode == 'live':
        await test_live_monitoring(args.symbol.upper(), args.duration)
    elif args.mode == 'batch':
        await test_batch_analysis(args.symbol.upper(), test_time, args.hours)
    elif args.mode == 'direct':
        await test_direct_calculation(args.symbol.upper(), test_time)


if __name__ == "__main__":
    asyncio.run(main())