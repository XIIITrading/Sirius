# backtest/plugins/m1_market_structure/test.py
"""
Test module for M1 Market Structure Analyzer
Run with: python test.py -s AAPL -t "2025-01-15 10:30:00" -d LONG
"""

import asyncio
import argparse
import sys
import os
from datetime import datetime, timedelta, timezone
import pandas as pd

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
plugins_dir = os.path.dirname(current_dir)
backtest_dir = os.path.dirname(plugins_dir)
project_root = os.path.dirname(backtest_dir)
sys.path.insert(0, project_root)

# Now we can import from the correct paths
from backtest.data import PolygonDataManager
from backtest.calculations.market_structure.m1_market_structure import MarketStructureAnalyzer


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


async def test_market_structure(symbol: str, entry_time: datetime, direction: str):
    """Test market structure analyzer for a specific symbol and time"""
    print("=== M1 MARKET STRUCTURE ANALYSIS ===")
    print(f"Symbol: {symbol}")
    print(f"Entry Time: {entry_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Direction: {direction}")
    print(f"Current UTC time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
    
    # Create analyzer
    analyzer = MarketStructureAnalyzer(
        fractal_length=5,
        buffer_size=200,
        min_candles_required=21,
        bars_needed=200  # Request exactly 200 bars
    )
    
    # Create data manager
    data_manager = PolygonDataManager()
    data_manager.set_current_plugin("M1_MarketStructure_Test")
    
    print(f"Analyzer configured to request {analyzer.get_required_bars()} bars")
    
    try:
        # Calculate how far back we need to go to get required bars
        # This is an estimate - we request more time to ensure we get enough bars
        bars_needed = analyzer.get_required_bars()
        
        # Add buffer for weekends/holidays (roughly 3x to be safe)
        estimated_minutes = bars_needed * 3
        start_time = entry_time - timedelta(minutes=estimated_minutes)
        
        print(f"Fetching data for {symbol}")
        print(f"Entry time: {entry_time}")
        print(f"Requesting from: {start_time} (estimated)")
        
        # Fetch bars
        bars_df = await data_manager.load_bars(
            symbol=symbol,
            start_time=start_time,
            end_time=entry_time,
            timeframe='1min'
        )
        
        if bars_df.empty:
            print("No data returned!")
            return
        
        print(f"Received {len(bars_df)} bars")
        print(f"Actual range: {bars_df.index.min()} to {bars_df.index.max()}")
        
        # Trim to exact number of bars needed
        if len(bars_df) > bars_needed:
            # Take the most recent N bars up to entry time
            bars_df = bars_df.iloc[-bars_needed:]
            print(f"Trimmed to {len(bars_df)} most recent bars")
        
        # Process bars up to entry time
        signal = analyzer.process_bars_dataframe(symbol, bars_df, entry_time)
        
        if signal:
            print(f"\n{'='*60}")
            print(f"SIGNAL GENERATED")
            print(f"{'='*60}")
            print(f"Direction: {signal.signal}")
            print(f"Type: {signal.structure_type}")
            print(f"Strength: {signal.strength:.0f}%")
            print(f"Reason: {signal.reason}")
            
            # Display metrics
            print(f"\nMetrics:")
            m = signal.metrics
            print(f"  Current Trend: {m['current_trend']}")
            if m['last_high_fractal']:
                print(f"  Last High Fractal: ${m['last_high_fractal']:.2f}")
            if m['last_low_fractal']:
                print(f"  Last Low Fractal: ${m['last_low_fractal']:.2f}")
            print(f"  Total Fractals: {m['fractal_count']}")
            print(f"  Structure Breaks: {m['structure_breaks']}")
            print(f"  Trend Changes: {m['trend_changes']}")
            print(f"  Candles Processed: {m['candles_processed']}")
            
            # Direction alignment check
            print(f"\n{'='*60}")
            print(f"TRADE ALIGNMENT")
            print(f"{'='*60}")
            
            if signal.signal == 'BULL' and direction == 'LONG':
                print("✓ Signal aligns with LONG direction")
            elif signal.signal == 'BEAR' and direction == 'SHORT':
                print("✓ Signal aligns with SHORT direction")
            else:
                print(f"⚠ Signal ({signal.signal}) conflicts with {direction} direction")
            
        else:
            current = analyzer.get_current_analysis(symbol)
            if current:
                print(f"\nNo new signal. Current state:")
                print(f"  Trend: {current.signal}")
                print(f"  Last break: {current.metrics['last_break_type'] or 'None'}")
            else:
                print("\nNo signal generated and insufficient data for analysis")
        
        # Show statistics
        stats = analyzer.get_statistics()
        print(f"\n{'='*60}")
        print("ANALYZER STATISTICS")
        print(f"{'='*60}")
        print(f"Candles processed: {stats['candles_processed']}")
        print(f"Fractals detected: {stats['fractals_detected']}")
        print(f"Signals generated: {stats['signals_generated']}")
        
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()


async def test_live_monitoring(symbol: str, duration_minutes: int = 5):
    """Monitor market structure in real-time (simulated)"""
    print(f"\n=== LIVE MONITORING MODE ===")
    print(f"Symbol: {symbol}")
    print(f"Duration: {duration_minutes} minutes")
    print("Note: This uses historical data to simulate live monitoring\n")
    
    # Create analyzer
    analyzer = MarketStructureAnalyzer(bars_needed=200)
    data_manager = PolygonDataManager()
    data_manager.set_current_plugin("M1_MarketStructure_Live")
    
    # Start from a recent time
    current_time = datetime.now(timezone.utc) - timedelta(hours=24)  # Yesterday
    end_time = current_time + timedelta(minutes=duration_minutes)
    
    while current_time < end_time:
        print(f"\n[{current_time.strftime('%H:%M:%S')}] Checking...")
        
        try:
            # Fetch data up to current time
            start_time = current_time - timedelta(minutes=300)
            
            bars_df = await data_manager.load_bars(
                symbol=symbol,
                start_time=start_time,
                end_time=current_time,
                timeframe='1min'
            )
            
            if not bars_df.empty and len(bars_df) >= analyzer.get_required_bars():
                # Process latest bars
                recent_bars = bars_df.iloc[-analyzer.get_required_bars():]
                signal = analyzer.process_bars_dataframe(symbol, recent_bars, current_time)
                
                if signal:
                    print(f"  *** NEW SIGNAL: {signal.signal} - {signal.structure_type} ***")
                    print(f"  Strength: {signal.strength:.0f}%")
                    print(f"  {signal.reason}")
                else:
                    current_state = analyzer.get_current_analysis(symbol)
                    if current_state:
                        print(f"  Current trend: {current_state.signal}")
            
        except Exception as e:
            print(f"  Error: {e}")
        
        # Advance time by 1 minute
        current_time += timedelta(minutes=1)
        await asyncio.sleep(0.5)  # Small delay for readability
    
    print("\nMonitoring complete")


def main():
    parser = argparse.ArgumentParser(
        description="Test M1 Market Structure Analyzer",
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
        "--live",
        action="store_true",
        help="Run in live monitoring mode (simulated)"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
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
        asyncio.run(test_live_monitoring(args.symbol, args.duration))
    else:
        asyncio.run(test_market_structure(args.symbol, entry_time, args.direction))


if __name__ == "__main__":
    main()