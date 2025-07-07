# backtest/plugins/m1_statistical_trend/test.py
"""
Test module for M1 Statistical Trend Analyzer
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
from backtest.calculations.trend.statistical_trend_1min import (
    StatisticalTrend1MinSimplified
)


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


async def test_statistical_trend(symbol: str, entry_time: datetime, direction: str):
    """Test statistical trend analyzer for a specific symbol and time"""
    print("=== M1 STATISTICAL TREND ANALYSIS ===")
    print(f"Symbol: {symbol}")
    print(f"Entry Time: {entry_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Direction: {direction}")
    print(f"Current UTC time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
    
    # Create analyzer
    analyzer = StatisticalTrend1MinSimplified(lookback_periods=10)
    
    # Create data manager
    data_manager = PolygonDataManager()
    data_manager.set_current_plugin("M1_StatisticalTrend_Test")
    
    print(f"Analyzer configured for {analyzer.lookback_periods} lookback periods")
    
    try:
        # Calculate start time for data fetch
        # Need at least lookback_periods of data before entry_time
        start_time = entry_time - timedelta(minutes=20)  # Get extra for safety
        
        print(f"Fetching data from {start_time} to {entry_time}")
        
        # Fetch bars using load_bars (matching m1_market_structure pattern)
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
        print(f"Data range: {bars_df.index.min()} to {bars_df.index.max()}")
        
        # Trim to just what we need if we got too much
        if len(bars_df) > analyzer.lookback_periods:
            bars_df = bars_df.iloc[-analyzer.lookback_periods:]
            print(f"Trimmed to {len(bars_df)} bars")
        
        # Analyze
        signal = analyzer.analyze(symbol, bars_df, entry_time)
        
        print(f"\n{'='*60}")
        print(f"SIGNAL GENERATED")
        print(f"{'='*60}")
        print(f"Signal: {signal.signal}")
        print(f"Confidence: {signal.confidence:.1f}%")
        print(f"Trend Strength: {signal.trend_strength:.2f}%")
        print(f"Volatility Adjusted Strength: {signal.volatility_adjusted_strength:.2f}")
        print(f"Volume Confirmation: {'Yes' if signal.volume_confirmation else 'No'}")
        
        # Direction alignment check
        print(f"\n{'='*60}")
        print(f"TRADE ALIGNMENT")
        print(f"{'='*60}")
        
        alignment_map = {
            'LONG': ['STRONG BUY', 'BUY', 'WEAK BUY'],
            'SHORT': ['STRONG SELL', 'SELL', 'WEAK SELL']
        }
        
        if signal.signal in alignment_map.get(direction, []):
            print(f"✓ Signal ({signal.signal}) aligns with {direction} direction")
        else:
            print(f"⚠ Signal ({signal.signal}) conflicts with {direction} direction")
        
        # Show thresholds
        print(f"\n{'='*60}")
        print("SIGNAL THRESHOLDS")
        print(f"{'='*60}")
        print("Strong Signal: Trend > 2.0x volatility")
        print("Normal Signal: Trend > 1.0x volatility")
        print("Weak Signal: Trend > 0.5x volatility")
        print("Neutral: Trend < 0.5x volatility")
        
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()


async def test_live_monitoring(symbol: str, duration_minutes: int = 5):
    """Monitor statistical trends in real-time (simulated)"""
    print(f"\n=== LIVE MONITORING MODE ===")
    print(f"Symbol: {symbol}")
    print(f"Duration: {duration_minutes} minutes")
    print("Note: This uses historical data to simulate live monitoring\n")
    
    # Create analyzer
    analyzer = StatisticalTrend1MinSimplified(lookback_periods=10)
    data_manager = PolygonDataManager()
    data_manager.set_current_plugin("M1_StatisticalTrend_Live")
    
    # Start from a recent time
    current_time = datetime.now(timezone.utc) - timedelta(hours=24)  # Yesterday
    end_time = current_time + timedelta(minutes=duration_minutes)
    
    last_signal = None
    
    while current_time < end_time:
        print(f"\n[{current_time.strftime('%H:%M:%S')}] Checking...")
        
        try:
            # Get data
            start_time = current_time - timedelta(minutes=20)
            
            bars_df = await data_manager.load_bars(
                symbol=symbol,
                start_time=start_time,
                end_time=current_time,
                timeframe='1min'
            )
            
            if not bars_df.empty and len(bars_df) >= analyzer.lookback_periods:
                # Take most recent bars
                recent_bars = bars_df.iloc[-analyzer.lookback_periods:]
                
                # Analyze
                signal = analyzer.analyze(symbol, recent_bars, current_time)
                
                # Check for signal change
                if last_signal is None or signal.signal != last_signal.signal:
                    print(f"  *** NEW SIGNAL: {signal.signal} ***")
                    print(f"  Confidence: {signal.confidence:.0f}%")
                    print(f"  Strength: {signal.trend_strength:.1f}%")
                    last_signal = signal
                else:
                    print(f"  Current: {signal.signal} ({signal.confidence:.0f}%)")
            
        except Exception as e:
            print(f"  Error: {e}")
        
        # Advance time by 1 minute
        current_time += timedelta(minutes=1)
        await asyncio.sleep(0.5)  # Small delay for readability
    
    print("\nMonitoring complete")


async def test_batch_analysis(symbol: str, start_time: datetime, duration_hours: int = 2):
    """Test multiple time points to see signal patterns"""
    print(f"\n=== BATCH ANALYSIS ===")
    print(f"Symbol: {symbol}")
    print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Duration: {duration_hours} hours")
    
    analyzer = StatisticalTrend1MinSimplified(lookback_periods=10)
    data_manager = PolygonDataManager()
    data_manager.set_current_plugin("M1_StatisticalTrend_Batch")
    
    # Load all data for the period at once
    data_start = start_time - timedelta(minutes=20)
    data_end = start_time + timedelta(hours=duration_hours)
    
    print("Loading data for batch analysis...")
    all_bars = await data_manager.load_bars(
        symbol=symbol,
        start_time=data_start,
        end_time=data_end,
        timeframe='1min'
    )
    
    if all_bars.empty:
        print("No data available for batch analysis")
        return
    
    print(f"Loaded {len(all_bars)} bars total")
    
    current_time = start_time
    end_time = start_time + timedelta(hours=duration_hours)
    
    signals = []
    
    print("\nTime                  Signal         Confidence  Strength  Vol-Adj")
    print("-" * 70)
    
    while current_time < end_time:
        try:
            # Get subset of bars up to current time
            bars_subset = all_bars[all_bars.index <= current_time]
            
            if len(bars_subset) >= analyzer.lookback_periods:
                # Take the most recent bars
                recent_bars = bars_subset.iloc[-analyzer.lookback_periods:]
                
                signal = analyzer.analyze(symbol, recent_bars, current_time)
                signals.append((current_time, signal))
                
                print(f"{current_time.strftime('%H:%M:%S')}  "
                      f"{signal.signal:<12}  "
                      f"{signal.confidence:>8.1f}%  "
                      f"{signal.trend_strength:>7.2f}%  "
                      f"{signal.volatility_adjusted_strength:>7.2f}")
        except:
            pass
        
        current_time += timedelta(minutes=5)
    
    # Summary
    if signals:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        
        signal_counts = {}
        for _, sig in signals:
            signal_counts[sig.signal] = signal_counts.get(sig.signal, 0) + 1
        
        print("Signal Distribution:")
        for sig_type, count in sorted(signal_counts.items()):
            pct = count / len(signals) * 100
            print(f"  {sig_type:<12}: {count:>3} ({pct:>5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Test M1 Statistical Trend Analyzer",
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
        "--batch",
        action="store_true",
        help="Run batch analysis over time period"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Duration for live monitoring in minutes or batch in hours"
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
    elif args.batch:
        asyncio.run(test_batch_analysis(args.symbol, entry_time, args.duration))
    else:
        asyncio.run(test_statistical_trend(args.symbol, entry_time, args.direction))


if __name__ == "__main__":
    main()