# Example usage with the new modular structure
import asyncio
from datetime import datetime, timezone
from backtest.data.polygon_data_manager import PolygonDataManager
from backtest.data.data_coordinator import DataCoordinator
from backtest.data.protected_data_manager import ProtectedDataManager

async def main():
    # Option 1: Direct usage
    polygon_manager = PolygonDataManager(
        api_key='your_key',
        cache_dir='./cache',
        memory_cache_size=100,
        file_cache_hours=24
    )
    
    # Option 2: With circuit breaker protection
    protected_manager = ProtectedDataManager(
        polygon_data_manager=polygon_manager,
        circuit_breaker_config={
            'failure_threshold': 0.5,
            'consecutive_failures': 5,
            'recovery_timeout': 60,
            'rate_limits': {
                'bars': {'per_minute': 100, 'burst': 10},
                'trades': {'per_minute': 50, 'burst': 5},
                'quotes': {'per_minute': 50, 'burst': 5}
            }
        }
    )
    
    # Create coordinator
    coordinator = DataCoordinator(protected_manager)  # or polygon_manager
    
    # Register modules (in real usage, these would be actual module instances)
    coordinator.register_module("TrendAnalysis", {})
    coordinator.register_module("MarketStructure", {})
    coordinator.register_module("OrderFlow", {})
    coordinator.register_module("VolumeAnalysis", {})
    
    # Fetch data for all modules
    symbol = 'AAPL'
    entry_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    
    module_data = await coordinator.fetch_all_module_data(symbol, entry_time, "LONG")
    
    # Print results
    for module_name, data_dict in module_data.items():
        print(f"\n{module_name}:")
        for data_key, df in data_dict.items():
            print(f"  {data_key}: {len(df)} rows")
    
    # Get summary report
    print(coordinator.get_summary_report())
    
    # Generate detailed report
    json_file, summary_file = coordinator.generate_data_report()
    print(f"\nReports saved to:")
    print(f"  JSON: {json_file}")
    print(f"  Summary: {summary_file}")

if __name__ == "__main__":
    asyncio.run(main())