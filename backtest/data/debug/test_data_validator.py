# backtest/data/debug/test_data_validator.py
"""
Test script for DataValidator module
Run with: python -m backtest.data.debug.test_data_validator -s AAPL -t "2025-01-15 10:30:00"
"""

import argparse
import asyncio
import sys
import os
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.dirname(current_dir)
backtest_dir = os.path.dirname(data_dir)
sys.path.insert(0, backtest_dir)

from data.data_validator import DataValidator, ValidationLevel
from data.polygon_data_manager import PolygonDataManager
from data.debug.test_utils import parse_datetime, generate_mock_bars, generate_mock_trades, generate_mock_quotes


async def test_validator_with_real_data(symbol: str, entry_time: datetime):
    """Test validator with real market data"""
    print("=" * 80)
    print("TEST: DataValidator with Real Market Data")
    print("=" * 80)
    
    # Initialize components
    validator = DataValidator(
        max_gap_minutes=5,
        max_price_change_pct=10.0,
        max_spread_pct=5.0,
        min_quality_score=80.0
    )
    
    data_manager = PolygonDataManager(
        memory_cache_size=50,
        file_cache_hours=24,
        disable_polygon_cache=True
    )
    
    print(f"\nFetching data for {symbol} at {entry_time}...")
    
    # Fetch different types of data
    try:
        # 1. Fetch bar data
        print("\n1. Fetching 1-minute bars...")
        bars_1min = await data_manager.load_bars(
            symbol=symbol,
            start_time=entry_time - timedelta(hours=2),
            end_time=entry_time,
            timeframe='1min'
        )
        
        # 2. Fetch trade data
        print("2. Fetching trade data...")
        trades = await data_manager.load_trades(
            symbol=symbol,
            start_time=entry_time - timedelta(minutes=30),
            end_time=entry_time
        )
        
        # 3. Fetch quote data
        print("3. Fetching quote data...")
        quotes = await data_manager.load_quotes(
            symbol=symbol,
            start_time=entry_time - timedelta(minutes=30),
            end_time=entry_time
        )
        
        # Validate all data
        print("\nValidating data quality...")
        reports = validator.validate_data_quality(
            bars_df=bars_1min,
            trades_df=trades if not trades.empty else None,
            quotes_df=quotes if not quotes.empty else None,
            symbol=symbol
        )
        
        # Print summary
        summary = validator.create_summary_report(reports)
        print(summary)
        
        # Show specific issues if any
        print("\n" + "=" * 80)
        print("DETAILED ISSUE ANALYSIS")
        print("=" * 80)
        
        for data_type, report in reports.items():
            if report.issues:
                print(f"\n{data_type.upper()} Issues:")
                for issue in report.issues[:5]:  # Show first 5
                    print(f"  [{issue.level.value}] {issue.issue_type}: {issue.description}")
                    if issue.timestamp:
                        print(f"    Timestamp: {issue.timestamp}")
                    if issue.metadata:
                        print(f"    Metadata: {issue.metadata}")
        
    except Exception as e:
        print(f"\nError during validation: {e}")
        import traceback
        traceback.print_exc()


async def test_validator_edge_cases(symbol: str):
    """Test validator with problematic data scenarios"""
    print("\n" + "=" * 80)
    print("TEST: DataValidator Edge Cases")
    print("=" * 80)
    
    validator = DataValidator()
    
    # Test Case 1: Data with gaps
    print("\n1. Testing data with gaps...")
    dates = pd.date_range('2025-01-15 09:30:00', '2025-01-15 10:30:00', freq='1min', tz='UTC')
    # Remove 10 minutes of data
    dates = dates[~((dates.hour == 10) & (dates.minute >= 0) & (dates.minute < 10))]
    
    bars_with_gaps = pd.DataFrame({
        'open': 100.0,
        'high': 101.0,
        'low': 99.0,
        'close': 100.5,
        'volume': 1000
    }, index=dates)
    
    report = validator.validate_bars(bars_with_gaps, symbol)
    print(f"Quality Score: {report.quality_score:.1f}%")
    print(f"Issues Found: {len(report.issues)}")
    
    # Test Case 2: Invalid OHLC relationships
    print("\n2. Testing invalid OHLC relationships...")
    invalid_bars = generate_mock_bars(symbol, 
                                    datetime(2025, 1, 15, 9, 30, 0, tzinfo=timezone.utc),
                                    datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc))
    
    # Introduce errors
    invalid_bars.loc[invalid_bars.index[5], 'high'] = invalid_bars.loc[invalid_bars.index[5], 'low'] - 1
    invalid_bars.loc[invalid_bars.index[10], 'low'] = invalid_bars.loc[invalid_bars.index[10], 'high'] + 1
    
    report = validator.validate_bars(invalid_bars, symbol)
    print(f"Quality Score: {report.quality_score:.1f}%")
    print(f"Issues Found: {len(report.issues)}")
    
    # Test Case 3: Crossed quotes
    print("\n3. Testing crossed quotes...")
    quotes = generate_mock_quotes(symbol,
                                datetime(2025, 1, 15, 9, 30, 0, tzinfo=timezone.utc),
                                datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc))
    
    # Create crossed quotes
    crossed_indices = quotes.index[10:15]
    quotes.loc[crossed_indices, 'bid'] = quotes.loc[crossed_indices, 'ask'] + 0.05
    
    report = validator.validate_quotes(quotes, symbol)
    print(f"Quality Score: {report.quality_score:.1f}%")
    print(f"Issues Found: {len(report.issues)}")
    
    # Test Case 4: Price spikes
    print("\n4. Testing price spikes...")
    spike_bars = generate_mock_bars(symbol,
                                  datetime(2025, 1, 15, 9, 30, 0, tzinfo=timezone.utc),
                                  datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc))
    
    # Create 20% spike
    spike_idx = spike_bars.index[15]
    spike_bars.loc[spike_idx, 'close'] = spike_bars.loc[spike_idx, 'close'] * 1.2
    spike_bars.loc[spike_idx, 'high'] = spike_bars.loc[spike_idx, 'high'] * 1.2
    
    report = validator.validate_bars(spike_bars, symbol)
    print(f"Quality Score: {report.quality_score:.1f}%")
    print(f"Issues Found: {len(report.issues)}")
    for issue in report.issues:
        if issue.issue_type == "PRICE_SPIKE":
            print(f"  Price spike: {issue.description}")


async def test_validator_performance(symbol: str, entry_time: datetime):
    """Test validator performance with large datasets"""
    print("\n" + "=" * 80)
    print("TEST: DataValidator Performance")
    print("=" * 80)
    
    validator = DataValidator()
    
    # Generate large datasets
    print("\nGenerating large datasets...")
    
    # 1 day of 1-minute bars (390 bars)
    bars = generate_mock_bars(symbol,
                            entry_time - timedelta(days=1),
                            entry_time,
                            '1min')
    
    # 30k trades
    trades = generate_mock_trades(symbol,
                                entry_time - timedelta(hours=1),
                                entry_time,
                                trades_per_minute=500)
    
    # 60k quotes
    quotes = generate_mock_quotes(symbol,
                                entry_time - timedelta(hours=1),
                                entry_time,
                                quotes_per_minute=1000)
    
    print(f"Generated: {len(bars)} bars, {len(trades)} trades, {len(quotes)} quotes")
    
    # Time validation
    import time
    
    print("\nValidating bars...")
    start = time.time()
    bars_report = validator.validate_bars(bars, symbol)
    bars_time = time.time() - start
    
    print("\nValidating trades...")
    start = time.time()
    trades_report = validator.validate_trades(trades, symbol)
    trades_time = time.time() - start
    
    print("\nValidating quotes...")
    start = time.time()
    quotes_report = validator.validate_quotes(quotes, symbol)
    quotes_time = time.time() - start
    
    print(f"\nValidation Performance:")
    print(f"  Bars: {bars_time:.3f}s ({len(bars)/bars_time:.0f} bars/sec)")
    print(f"  Trades: {trades_time:.3f}s ({len(trades)/trades_time:.0f} trades/sec)")
    print(f"  Quotes: {quotes_time:.3f}s ({len(quotes)/quotes_time:.0f} quotes/sec)")
    
    print(f"\nQuality Scores:")
    print(f"  Bars: {bars_report.quality_score:.1f}%")
    print(f"  Trades: {trades_report.quality_score:.1f}%")
    print(f"  Quotes: {quotes_report.quality_score:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Test DataValidator module")
    parser.add_argument("-s", "--symbol", default="AAPL", help="Stock symbol")
    parser.add_argument("-t", "--time", default="2025-01-15 10:30:00",
                       help="Entry time (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--test", type=int, default=0,
                       help="Specific test (0=all, 1=real data, 2=edge cases, 3=performance)")
    
    args = parser.parse_args()
    
    # Parse entry time
    try:
        entry_time = parse_datetime(args.time)
    except ValueError as e:
        print(f"Error parsing time: {e}")
        sys.exit(1)
    
    # Warn about API usage
    if args.test == 0 or args.test == 1:
        print("\n" + "!" * 60)
        print("WARNING: This will use REAL Polygon API credits!")
        print("!" * 60)
        response = input("\nContinue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    # Run tests
    async def run_tests():
        if args.test == 0 or args.test == 1:
            await test_validator_with_real_data(args.symbol, entry_time)
        
        if args.test == 0 or args.test == 2:
            await test_validator_edge_cases(args.symbol)
        
        if args.test == 0 or args.test == 3:
            await test_validator_performance(args.symbol, entry_time)
    
    asyncio.run(run_tests())


if __name__ == "__main__":
    main()