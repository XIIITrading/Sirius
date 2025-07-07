# backtest/plugins/m15_statistical_trend/test.py
"""
Test module for M15 Statistical Trend Analyzer
Run with: python test.py -s AAPL -t "2025-01-15 10:30:00" -d LONG
"""

import asyncio
import argparse
import sys
import os
from datetime import datetime, timedelta, timezone
import logging
from typing import Dict, Any

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
plugins_dir = os.path.dirname(current_dir)
backtest_dir = os.path.dirname(plugins_dir)
project_root = os.path.dirname(backtest_dir)
sys.path.insert(0, project_root)

# Now we can import from the correct paths
from backtest.data.polygon_data_manager import PolygonDataManager
from backtest.calculations.trend.statistical_trend_15min import StatisticalTrend15MinSimplified
from backtest.plugins.m15_statistical_trend import run_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_datetime(dt_str: str) -> datetime:
    """Parse datetime string to timezone-aware datetime"""
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%Y-%m-%d',
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(dt_str, fmt)
            # Make timezone aware (UTC)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    
    raise ValueError(f"Could not parse datetime: {dt_str}")


async def test_m15_statistical_trend(symbol: str, entry_time: datetime, direction: str):
    """Test M15 statistical trend analyzer for a specific symbol and time"""
    print("=== M15 STATISTICAL TREND ANALYSIS ===")
    print(f"Symbol: {symbol}")
    print(f"Entry Time: {entry_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Direction: {direction}")
    print(f"Current UTC time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
    
    try:
        # Run the analysis
        print("Running 15-minute statistical trend analysis...")
        result = await run_analysis(symbol, entry_time, direction)
        
        if 'error' in result:
            print(f"\nError during analysis: {result['error']}")
            return
        
        # Extract results
        signal = result['signal']
        details = result['details']
        display = result['display_data']
        
        print(f"\n{'='*60}")
        print(f"SIGNAL GENERATED")
        print(f"{'='*60}")
        print(f"Market Regime: {details['regime']}")
        print(f"Daily Bias: {details['daily_bias']}")
        print(f"Signal Direction: {signal['direction']}")
        print(f"Confidence: {signal['confidence']:.0f}%")
        print(f"Trend Strength: {signal['strength']:.1f}%")
        
        # Display metrics
        print(f"\nMetrics:")
        print(f"  Volatility State: {details['volatility_state']}")
        print(f"  Volume Trend: {details['volume_trend']}")
        print(f"  Vol-Adjusted Strength: {details['volatility_adjusted_strength']:.2f}")
        print(f"  Current Price: ${details['price']:.2f}")
        
        # Direction alignment check
        print(f"\n{'='*60}")
        print(f"TRADE ALIGNMENT")
        print(f"{'='*60}")
        
        if details['aligned']:
            print(f"✅ 15-minute regime aligns with {direction} direction")
        else:
            print(f"⚠️ 15-minute regime conflicts with {direction} direction")
        
        # Display summary and description
        print(f"\nSUMMARY: {display['summary']}")
        print(f"DESCRIPTION: {display['description']}")
        
        # Display detailed table
        print(f"\n{'='*60}")
        print("DETAILED ANALYSIS")
        print(f"{'='*60}")
        for row in display['table_data']:
            print(f"{row[0]:.<30} {row[1]}")
        
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()


async def test_batch_analysis(symbol: str, test_times: list):
    """Run analysis on multiple trades"""
    print(f"\n=== M15 BATCH ANALYSIS ===")
    print(f"Symbol: {symbol}")
    print(f"Testing {len(test_times)} trades\n")
    
    results = []
    
    for i, (entry_time, direction) in enumerate(test_times, 1):
        print(f"\n{'='*60}")
        print(f"TRADE {i}/{len(test_times)}")
        print(f"{'='*60}")
        print(f"Entry Time: {entry_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Direction: {direction}")
        
        try:
            result = await run_analysis(symbol, entry_time, direction)
            results.append({
                'time': entry_time,
                'direction': direction,
                'result': result
            })
            
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                details = result['details']
                signal = result['signal']
                
                print(f"\nResult:")
                print(f"  Regime: {details['regime']}")
                print(f"  Signal: {signal['direction']} ({signal['confidence']:.0f}% confidence)")
                print(f"  Aligned: {'YES' if details['aligned'] else 'NO'}")
                
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                'time': entry_time,
                'direction': direction,
                'result': {'error': str(e)}
            })
        
        await asyncio.sleep(0.5)  # Small delay between requests
    
    # Print summary
    print(f"\n{'='*60}")
    print("BATCH SUMMARY")
    print(f"{'='*60}")
    print(f"Total Trades: {len(results)}")
    
    # Count successful analyses
    successful = [r for r in results if 'error' not in r['result']]
    print(f"Successful Analyses: {len(successful)}")
    
    if successful:
        # Count aligned trades
        aligned = sum(1 for r in successful 
                     if r['result']['details'].get('aligned', False))
        print(f"Aligned Trades: {aligned}/{len(successful)} ({aligned/len(successful)*100:.0f}%)")
        
        # Group by regime
        regimes = {}
        for r in successful:
            regime = r['result']['details']['regime']
            regimes[regime] = regimes.get(regime, 0) + 1
        
        print("\nRegime Distribution:")
        for regime, count in regimes.items():
            print(f"  {regime}: {count} ({count/len(successful)*100:.0f}%)")


async def test_live_monitoring(symbol: str, duration_minutes: int = 30):
    """Monitor 15-minute statistical trend in real-time (simulated)"""
    print(f"\n=== M15 LIVE MONITORING MODE ===")
    print(f"Symbol: {symbol}")
    print(f"Duration: {duration_minutes} minutes")
    print("Note: This uses historical data to simulate live 15-minute monitoring\n")
    
    # Start from a recent time
    current_time = datetime.now(timezone.utc) - timedelta(hours=24)  # Yesterday
    end_time = current_time + timedelta(minutes=duration_minutes)
    
    last_analysis = None
    
    while current_time < end_time:
        # Only check every 15 minutes (when a new 15-min candle completes)
        if current_time.minute % 15 == 0:
            print(f"\n[{current_time.strftime('%H:%M:%S')}] New 15-min candle complete...")
            
            try:
                # Run analysis for LONG direction (monitoring mode)
                result = await run_analysis(symbol, current_time, 'LONG')
                
                if 'error' not in result:
                    details = result['details']
                    signal = result['signal']
                    
                    # Check if regime changed
                    regime_changed = (last_analysis and 
                                    last_analysis['details']['regime'] != details['regime'])
                    
                    if regime_changed:
                        print(f"  *** REGIME CHANGE: {last_analysis['details']['regime']} → {details['regime']} ***")
                    
                    print(f"  Current Regime: {details['regime']}")
                    print(f"  Signal: {signal['direction']} (Strength: {signal['strength']:.1f}%)")
                    print(f"  Volatility: {details['volatility_state']}")
                    
                    last_analysis = result
                else:
                    print(f"  Error: {result['error']}")
                    
            except Exception as e:
                print(f"  Error: {e}")
        
        # Advance time by 1 minute
        current_time += timedelta(minutes=1)
        await asyncio.sleep(0.1)  # Small delay for readability
    
    print("\nMonitoring complete")


def main():
    parser = argparse.ArgumentParser(
        description="Test M15 Statistical Trend Analyzer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-s", "--symbol",
        type=str,
        default="AAPL",
        help="Stock symbol to analyze"
    )
    
    parser.add_argument(
        "-t", "--time",
        type=str,
        default=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        help="Entry time in format 'YYYY-MM-DD HH:MM:SS'"
    )
    
    parser.add_argument(
        "-d", "--direction",
        type=str,
        choices=["LONG", "SHORT"],
        default="LONG",
        help="Trade direction"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch analysis with multiple test trades"
    )
    
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live monitoring mode (simulated)"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Duration for live monitoring in minutes"
    )
    
    args = parser.parse_args()
    
    # Parse entry time
    try:
        entry_time = parse_datetime(args.time)
    except ValueError as e:
        print(f"Error parsing time: {e}")
        print("Please use format: YYYY-MM-DD HH:MM:SS")
        sys.exit(1)
    
    # Run appropriate test
    if args.live:
        asyncio.run(test_live_monitoring(args.symbol.upper(), args.duration))
    elif args.batch:
        # Create test scenarios
        base_time = entry_time
        test_trades = [
            (base_time, 'LONG'),
            (base_time.replace(hour=14, minute=30), 'LONG'),
            (base_time.replace(hour=15, minute=30), 'SHORT'),
            (base_time.replace(hour=13, minute=0), 'SHORT'),
            (base_time.replace(hour=17, minute=30), 'LONG'),
        ]
        asyncio.run(test_batch_analysis(args.symbol.upper(), test_trades))
    else:
        asyncio.run(test_m15_statistical_trend(args.symbol.upper(), entry_time, args.direction))


if __name__ == "__main__":
    main()