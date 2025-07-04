# backtest/data/debug/test_request_aggregator.py
"""
Test script for RequestAggregator module
Run with: python -m backtest.data.debug.test_request_aggregator -s AAPL -t "2025-01-15 10:30:00" -d LONG
"""

import argparse
import asyncio
import sys
import os
from datetime import datetime, timedelta, timezone

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.dirname(current_dir)
backtest_dir = os.path.dirname(data_dir)
sys.path.insert(0, backtest_dir)

from backtest.data.polygon_data_manager import PolygonDataManager
from backtest.data.request_aggregator import RequestAggregator, DataNeed, DataType
from backtest.data.debug.test_utils import (
    parse_datetime, MockPolygonDataManager, print_dataframe_summary
)


async def test_aggregator_efficiency(symbol: str, entry_time: datetime):
    """Test how well the aggregator consolidates requests"""
    print("=" * 80)
    print("TEST 1: Request Aggregation Efficiency")
    print("=" * 80)
    
    aggregator = RequestAggregator(extend_window_pct=0.1)
    
    # Simulate multiple overlapping requests
    needs = [
        # Trend analysis modules at different timeframes
        DataNeed("TrendAnalysis_1min", symbol, DataType.BARS, "1min",
                entry_time - timedelta(hours=2), entry_time, priority=8),
        
        DataNeed("TrendAnalysis_5min", symbol, DataType.BARS, "5min",
                entry_time - timedelta(hours=4), entry_time, priority=7),
        
        DataNeed("TrendAnalysis_15min", symbol, DataType.BARS, "15min",
                entry_time - timedelta(hours=6), entry_time, priority=6),
        
        # Multiple modules needing similar 1-min data (should be merged)
        DataNeed("VolumeProfile", symbol, DataType.BARS, "1min",
                entry_time - timedelta(hours=1, minutes=30), entry_time, priority=9),
        
        DataNeed("MarketStructure", symbol, DataType.BARS, "1min",
                entry_time - timedelta(hours=1), entry_time + timedelta(minutes=30), priority=8),
        
        # Order flow needing tick data
        DataNeed("OrderFlow_Trades", symbol, DataType.TRADES, "tick",
                entry_time - timedelta(minutes=30), entry_time, priority=10),
        
        DataNeed("OrderFlow_Quotes", symbol, DataType.QUOTES, "tick",
                entry_time - timedelta(minutes=30), entry_time, priority=10),
        
        # Another module needing slightly different trade window
        DataNeed("LargeOrderDetection", symbol, DataType.TRADES, "tick",
                entry_time - timedelta(minutes=45), entry_time - timedelta(minutes=15), priority=7),
    ]
    
    # Register all needs
    aggregator.register_needs(needs)
    
    # Show the report
    print(aggregator.create_request_report())
    
    # Show aggregated requests
    aggregated = aggregator._aggregate_needs()
    print(f"\n\nFinal Aggregated Requests: {len(aggregated)}")
    print("-" * 60)
    
    for i, req in enumerate(aggregated, 1):
        duration = (req.end_time - req.start_time).total_seconds() / 60
        print(f"{i}. {req.symbol} {req.data_type.value} {req.timeframe}")
        print(f"   Time: {req.start_time.strftime('%H:%M:%S')} to {req.end_time.strftime('%H:%M:%S')} ({duration:.1f} min)")
        print(f"   Modules: {', '.join(req.requesting_modules)}")
        print(f"   Covers {len(req.module_needs)} original needs")
        print()


async def test_data_distribution(symbol: str, entry_time: datetime):
    """Test data fetching and distribution to modules"""
    print("\n" + "=" * 80)
    print("TEST 2: Data Fetching and Distribution")
    print("=" * 80)
    
    # Create aggregator with mock data manager
    mock_manager = MockPolygonDataManager()
    aggregator = RequestAggregator(data_manager=mock_manager, extend_window_pct=0.1)
    
    # Register needs from different modules
    needs = [
        DataNeed("Module_A", symbol, DataType.BARS, "1min",
                entry_time - timedelta(hours=1), entry_time),
        
        DataNeed("Module_B", symbol, DataType.BARS, "1min",
                entry_time - timedelta(minutes=45), entry_time + timedelta(minutes=15)),
        
        DataNeed("Module_C", symbol, DataType.TRADES, "tick",
                entry_time - timedelta(minutes=30), entry_time),
        
        DataNeed("Module_D", symbol, DataType.QUOTES, "tick",
                entry_time - timedelta(minutes=20), entry_time),
    ]
    
    aggregator.register_needs(needs)
    
    # Fetch all data
    print("\nFetching data...")
    module_data = await aggregator.fetch_all_data()
    
    # Show what each module received
    print("\nData Distribution Results:")
    print("-" * 60)
    
    for module_name, data_dict in module_data.items():
        print(f"\n{module_name}:")
        for data_key, df in data_dict.items():
            print(f"  {data_key}: {len(df)} rows")
            if len(df) > 0:
                print(f"    Range: {df.index.min()} to {df.index.max()}")
    
    # Show API efficiency
    print("\n" + "-" * 60)
    print(f"Mock API calls made: {mock_manager.call_count}")
    
    stats = aggregator.get_stats()
    print(f"\nAggregator Statistics:")
    print(f"  Total data points fetched: {stats['data_points_fetched']:,}")
    print(f"  API calls saved: {stats['api_calls_saved']}")


async def test_complex_scenario(symbol: str, entry_time: datetime, direction: str):
    """Test a complex real-world scenario"""
    print("\n" + "=" * 80)
    print("TEST 3: Complex Real-World Scenario")
    print(f"Symbol: {symbol}, Entry: {entry_time}, Direction: {direction}")
    print("=" * 80)
    
    # Create aggregator with mock data manager
    mock_manager = MockPolygonDataManager()
    aggregator = RequestAggregator(data_manager=mock_manager, extend_window_pct=0.15)
    
    # Simulate a complete backtesting system with many modules
    # Each module has different data requirements
    
    if direction == "LONG":
        # For long positions, we might focus more on momentum and support
        needs = [
            # Trend followers need various timeframes
            DataNeed("TrendFollower_1m", symbol, DataType.BARS, "1min",
                    entry_time - timedelta(hours=2), entry_time + timedelta(hours=1)),
            DataNeed("TrendFollower_5m", symbol, DataType.BARS, "5min",
                    entry_time - timedelta(hours=4), entry_time + timedelta(hours=1)),
            DataNeed("TrendFollower_15m", symbol, DataType.BARS, "15min",
                    entry_time - timedelta(hours=8), entry_time + timedelta(hours=1)),
            
            # Support/Resistance calculator
            DataNeed("SupportResistance", symbol, DataType.BARS, "1min",
                    entry_time - timedelta(hours=12), entry_time),
            
            # Volume analysis
            DataNeed("VolumeAnalysis", symbol, DataType.BARS, "1min",
                    entry_time - timedelta(hours=1), entry_time + timedelta(minutes=30)),
            DataNeed("VolumeAnalysis_Trades", symbol, DataType.TRADES, "tick",
                    entry_time - timedelta(minutes=30), entry_time + timedelta(minutes=15)),
            
            # Order flow for entry timing
            DataNeed("OrderFlow", symbol, DataType.TRADES, "tick",
                    entry_time - timedelta(minutes=15), entry_time + timedelta(minutes=5)),
            DataNeed("OrderFlow_Quotes", symbol, DataType.QUOTES, "tick",
                    entry_time - timedelta(minutes=15), entry_time + timedelta(minutes=5)),
            
            # Market microstructure
            DataNeed("Microstructure", symbol, DataType.QUOTES, "tick",
                    entry_time - timedelta(minutes=5), entry_time + timedelta(minutes=30)),
        ]
    else:  # SHORT
        # For short positions, we might focus on resistance and distribution
        needs = [
            # Trend analysis for reversal
            DataNeed("ReversalDetector_1m", symbol, DataType.BARS, "1min",
                    entry_time - timedelta(hours=1), entry_time + timedelta(hours=1)),
            DataNeed("ReversalDetector_5m", symbol, DataType.BARS, "5min",
                    entry_time - timedelta(hours=3), entry_time + timedelta(hours=1)),
            
            # Distribution analysis
            DataNeed("Distribution", symbol, DataType.BARS, "1min",
                    entry_time - timedelta(hours=4), entry_time),
            DataNeed("Distribution_Trades", symbol, DataType.TRADES, "tick",
                    entry_time - timedelta(hours=1), entry_time),
            
            # Selling pressure
            DataNeed("SellingPressure", symbol, DataType.TRADES, "tick",
                    entry_time - timedelta(minutes=45), entry_time + timedelta(minutes=15)),
            DataNeed("SellingPressure_Quotes", symbol, DataType.QUOTES, "tick",
                    entry_time - timedelta(minutes=45), entry_time + timedelta(minutes=15)),
            
            # Resistance levels
            DataNeed("ResistanceLevels", symbol, DataType.BARS, "15min",
                    entry_time - timedelta(hours=24), entry_time),
        ]
    
    # Register all needs
    aggregator.register_needs(needs)
    
    # Show aggregation report
    print(aggregator.create_request_report())
    
    # Fetch all data
    print("\nFetching all data concurrently...")
    start_time = datetime.now()
    module_data = await aggregator.fetch_all_data()
    fetch_duration = (datetime.now() - start_time).total_seconds()
    
    print(f"\nFetch completed in {fetch_duration:.2f} seconds")
    
    # Analyze results
    total_rows = 0
    for module_name, data_dict in module_data.items():
        for data_key, df in data_dict.items():
            total_rows += len(df)
    
    print(f"\nTotal data points distributed: {total_rows:,}")
    print(f"Number of modules served: {len(module_data)}")
    
    # Show efficiency metrics
    stats = aggregator.get_stats()
    efficiency = (stats['api_calls_saved'] / max(1, stats['total_needs'])) * 100
    
    print(f"\nEfficiency Metrics:")
    print(f"  Original requests: {stats['total_needs']}")
    print(f"  Aggregated requests: {stats['aggregated_requests']}")
    print(f"  Efficiency gain: {efficiency:.1f}%")
    
    # Show what would have happened without aggregation
    print("\n" + "-" * 60)
    print("Without Aggregation:")
    print(f"  API calls needed: {stats['total_needs']}")
    print(f"  Estimated time: {stats['total_needs'] * 0.5:.1f} seconds (sequential)")
    print("\nWith Aggregation:")
    print(f"  API calls made: {stats['aggregated_requests']}")
    print(f"  Actual time: {fetch_duration:.2f} seconds (concurrent)")
    print(f"  Time saved: {(stats['total_needs'] * 0.5 - fetch_duration):.1f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Test RequestAggregator module")
    parser.add_argument("-s", "--symbol", default="AAPL", help="Stock symbol")
    parser.add_argument("-t", "--time", default="2025-01-15 10:30:00", 
                       help="Entry time (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("-d", "--direction", choices=["LONG", "SHORT"], 
                       default="LONG", help="Trade direction")
    parser.add_argument("--test", type=int, default=0,
                       help="Run specific test (1-3), 0 for all")
    
    args = parser.parse_args()
    
    # Parse entry time
    try:
        entry_time = parse_datetime(args.time)
    except ValueError as e:
        print(f"Error parsing time: {e}")
        sys.exit(1)
    
    # Run tests
    async def run_tests():
        if args.test == 0 or args.test == 1:
            await test_aggregator_efficiency(args.symbol, entry_time)
        
        if args.test == 0 or args.test == 2:
            await test_data_distribution(args.symbol, entry_time)
        
        if args.test == 0 or args.test == 3:
            await test_complex_scenario(args.symbol, entry_time, args.direction)
    
    # Run the async tests
    asyncio.run(run_tests())


if __name__ == "__main__":
    main()