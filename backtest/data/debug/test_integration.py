# backtest/data/debug/test_integration.py
"""
Integration test for RequestAggregator with PolygonDataManager
Run with: python -m backtest.data.debug.test_integration -s AAPL -t "2025-01-15 10:30:00"
"""

import argparse
import asyncio
import sys
import os
from datetime import datetime, timedelta, timezone
import json

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.dirname(current_dir)
backtest_dir = os.path.dirname(data_dir)
sys.path.insert(0, backtest_dir)

from backtest.data.polygon_data_manager import PolygonDataManager
from backtest.data.request_aggregator import RequestAggregator, DataNeed, DataType
from backtest.data.data_coordinator import DataCoordinator
from backtest.data.debug.test_utils import parse_datetime, print_dataframe_summary


async def test_basic_integration(symbol: str, entry_time: datetime):
    """Test basic integration between RequestAggregator and PolygonDataManager"""
    print("=" * 80)
    print("TEST: Basic Integration with REAL Polygon API")
    print("=" * 80)
    
    # Initialize PolygonDataManager with real API
    print("Initializing PolygonDataManager...")
    data_manager = PolygonDataManager(
        memory_cache_size=50,
        file_cache_hours=24,
        extend_window_bars=100
        # Removed disable_polygon_cache=True
    )
    
    # Create aggregator
    aggregator = RequestAggregator(data_manager=data_manager, extend_window_pct=0.1)
    
    # Register some basic needs
    needs = [
        DataNeed("Module1", symbol, DataType.BARS, "1min",
                entry_time - timedelta(hours=1), entry_time),
        DataNeed("Module2", symbol, DataType.BARS, "1min",
                entry_time - timedelta(minutes=45), entry_time + timedelta(minutes=15)),
        DataNeed("Module3", symbol, DataType.BARS, "5min",
                entry_time - timedelta(hours=2), entry_time),
    ]
    
    aggregator.register_needs(needs)
    
    # Show what will be fetched
    print("\nRequest Aggregation Plan:")
    print(aggregator.create_request_report())
    
    # Fetch data
    print("\nFetching data from Polygon API...")
    data_manager.set_current_plugin("Integration Test")
    
    try:
        module_data = await aggregator.fetch_all_data()
        
        print("\nData Distribution Results:")
        print("-" * 60)
        for module_name, data_dict in module_data.items():
            print(f"\n{module_name}:")
            for data_key, df in data_dict.items():
                print_dataframe_summary(df, f"  {data_key}")
        
        # Show efficiency
        stats = aggregator.get_stats()
        print(f"\nEfficiency Summary:")
        print(f"  Original requests: {stats['total_needs']}")
        print(f"  Aggregated to: {stats['aggregated_requests']}")
        print(f"  API calls saved: {stats['api_calls_saved']}")
        
        cache_stats = data_manager.get_cache_stats()
        print(f"\nCache Performance:")
        print(f"  Cache hits: {cache_stats['api_stats']['cache_hits']}")
        print(f"  API calls: {cache_stats['api_stats']['api_calls']}")
        if cache_stats['api_stats']['total_requests'] > 0:
            print(f"  Hit rate: {cache_stats['api_stats']['cache_hit_rate']:.1f}%")
            
    except Exception as e:
        print(f"\nError during data fetch: {e}")
        import traceback
        traceback.print_exc()


async def test_coordinator_integration(symbol: str, entry_time: datetime):
    """Test the DataCoordinator with full module simulation"""
    print("\n" + "=" * 80)
    print("TEST: DataCoordinator Integration with REAL API")
    print("=" * 80)
    
    # Initialize components with real API
    print("Initializing DataCoordinator with real PolygonDataManager...")
    data_manager = PolygonDataManager(
        memory_cache_size=100,
        file_cache_hours=24,
        extend_window_bars=200
    )
    
    # Create coordinator
    coordinator = DataCoordinator(data_manager)
    
    # Simulate registering calculation modules
    coordinator.register_module("TrendAnalysis", {})
    coordinator.register_module("MarketStructure", {})
    coordinator.register_module("OrderFlow", {})
    coordinator.register_module("VolumeAnalysis", {})
    
    print(f"\nRegistered modules: {list(coordinator.registered_modules.keys())}")
    
    try:
        # Fetch all module data
        print(f"\nFetching data for {symbol} at {entry_time}...")
        start_time = datetime.now()
        
        module_data = await coordinator.fetch_all_module_data(symbol, entry_time, "LONG")
        
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\nFetch completed in {duration:.2f} seconds")
        
        # Show results
        print("\nModule Data Summary:")
        print("-" * 60)
        total_rows = 0
        for module_name, data_dict in module_data.items():
            module_rows = sum(len(df) for df in data_dict.values())
            total_rows += module_rows
            print(f"{module_name}: {len(data_dict)} datasets, {module_rows:,} total rows")
            
            # Show details for each dataset
            for data_key, df in data_dict.items():
                if len(df) > 0:
                    print(f"  {data_key}: {len(df):,} rows [{df.index.min()} to {df.index.max()}]")
        
        print(f"\nTotal data points distributed: {total_rows:,}")
        
        # Show coordinator summary
        print(coordinator.get_summary_report())
        
        # Generate detailed report
        print("\nGenerating detailed data report...")
        json_file, summary_file = coordinator.data_manager.generate_data_report()
        print(f"Reports saved to:")
        print(f"  JSON: {json_file}")
        print(f"  Summary: {summary_file}")
            
    except Exception as e:
        print(f"\nError during coordinator test: {e}")
        import traceback
        traceback.print_exc()


async def test_cache_efficiency(symbol: str, entry_time: datetime):
    """Test cache efficiency with repeated requests"""
    print("\n" + "=" * 80)
    print("TEST: Cache Efficiency with REAL API")
    print("=" * 80)
    
    # Use real API to see actual caching
    data_manager = PolygonDataManager(
        memory_cache_size=50,
        file_cache_hours=24
    )
    
    aggregator = RequestAggregator(data_manager=data_manager)
    
    # Define test needs
    needs = [
        DataNeed("TestModule", symbol, DataType.BARS, "1min",
                entry_time - timedelta(hours=1), entry_time),
    ]
    
    print("Running 3 iterations of the same request...")
    
    for i in range(3):
        print(f"\n--- Iteration {i+1} ---")
        
        # Clear aggregator but not cache
        aggregator.clear_needs()
        aggregator.register_needs(needs)
        
        data_manager.set_current_plugin(f"CacheTest_Iteration_{i+1}")
        
        start_time = datetime.now()
        await aggregator.fetch_all_data()
        duration = (datetime.now() - start_time).total_seconds()
        
        cache_stats = data_manager.get_cache_stats()
        print(f"Fetch time: {duration:.3f} seconds")
        print(f"Cache hits: {cache_stats['api_stats']['cache_hits']}")
        print(f"API calls: {cache_stats['api_stats']['api_calls']}")
        print(f"Memory cache items: {cache_stats['memory_cache']['cached_items']}")
        
        # Calculate hit rate for this iteration
        total_requests = cache_stats['api_stats']['total_requests']
        if total_requests > 0:
            hit_rate = (cache_stats['api_stats']['cache_hits'] / total_requests) * 100
            print(f"Cache hit rate: {hit_rate:.1f}%")
    
    print("\nCache efficiency demonstrated - subsequent requests should be much faster!")


async def test_real_world_scenario(symbol: str, entry_time: datetime):
    """Test a real-world backtesting scenario with actual market hours"""
    print("\n" + "=" * 80)
    print("TEST: Real-World Backtesting Scenario")
    print(f"Symbol: {symbol}, Entry: {entry_time}")
    print("=" * 80)
    
    # Initialize with production-like settings
    data_manager = PolygonDataManager(
        memory_cache_size=200,
        file_cache_hours=24,
        extend_window_bars=500  # Larger extension for production
    )
    
    coordinator = DataCoordinator(data_manager)
    
    # Register modules
    coordinator.register_module("TrendAnalysis", {})
    coordinator.register_module("MarketStructure", {})
    coordinator.register_module("OrderFlow", {})
    coordinator.register_module("VolumeAnalysis", {})
    
    # Test with different entry times throughout the day
    test_scenarios = [
        ("Market Open", entry_time.replace(hour=9, minute=30)),
        ("Mid Morning", entry_time.replace(hour=10, minute=30)),
        ("Lunch Time", entry_time.replace(hour=12, minute=0)),
        ("Power Hour", entry_time.replace(hour=15, minute=0)),
    ]
    
    for scenario_name, test_time in test_scenarios:
        print(f"\n\n>>> Testing {scenario_name} - {test_time} <<<")
        print("-" * 60)
        
        # Clear previous needs
        coordinator.aggregator.clear_needs()
        
        # Set plugin name for tracking
        data_manager.set_current_plugin(f"RealWorld_{scenario_name.replace(' ', '_')}")
        
        # Fetch data
        try:
            start = datetime.now()
            module_data = await coordinator.fetch_all_module_data(symbol, test_time, "LONG")
            duration = (datetime.now() - start).total_seconds()
            
            # Summary
            total_rows = sum(
                len(df) 
                for data_dict in module_data.values() 
                for df in data_dict.values()
            )
            
            print(f"Fetch completed in {duration:.2f}s, {total_rows:,} data points")
            
            # Show cache performance
            cache_stats = data_manager.get_cache_stats()
            if cache_stats['api_stats']['total_requests'] > 0:
                print(f"Cache hit rate: {cache_stats['api_stats']['cache_hit_rate']:.1f}%")
            
        except Exception as e:
            print(f"Error in {scenario_name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Test RequestAggregator integration with PolygonDataManager (REAL API)"
    )
    parser.add_argument("-s", "--symbol", default="AAPL", help="Stock symbol")
    parser.add_argument("-t", "--time", default="2025-01-15 10:30:00",
                       help="Entry time (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--test", type=int, default=0,
                       help="Specific test to run (0=all, 1=basic, 2=coordinator, 3=cache, 4=real-world)")
    
    args = parser.parse_args()
    
    # Parse entry time
    try:
        entry_time = parse_datetime(args.time)
    except ValueError as e:
        print(f"Error parsing time: {e}")
        sys.exit(1)
    
    # Show warning
    print("\n" + "!" * 60)
    print("WARNING: This will use REAL Polygon API credits!")
    print("!" * 60)
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)
    
    # Run selected tests
    async def run_tests():
        if args.test == 0 or args.test == 1:
            await test_basic_integration(args.symbol, entry_time)
        
        if args.test == 0 or args.test == 2:
            await test_coordinator_integration(args.symbol, entry_time)
        
        if args.test == 0 or args.test == 3:
            await test_cache_efficiency(args.symbol, entry_time)
            
        if args.test == 0 or args.test == 4:
            await test_real_world_scenario(args.symbol, entry_time)
    
    asyncio.run(run_tests())


if __name__ == "__main__":
    main()