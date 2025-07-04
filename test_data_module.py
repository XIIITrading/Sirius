# test_data_module.py
"""
Simple test to verify the data module is working correctly
"""
import asyncio
import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import from the data module
from backtest.data import (
    PolygonDataManager,
    RequestAggregator,
    DataCoordinator,
    ProtectedDataManager,
    DataValidator,
    TradeQuoteAligner,
    DataNeed,
    DataType
)


async def test_basic_functionality():
    """Test basic data module functionality"""
    
    # Check API key
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("❌ POLYGON_API_KEY not found in environment")
        return
    print("✓ API key loaded")
    
    # 1. Test PolygonDataManager initialization
    try:
        data_manager = PolygonDataManager(
            api_key=api_key,
            cache_dir='./test_cache',
            memory_cache_size=10
        )
        print("✓ PolygonDataManager initialized")
    except Exception as e:
        print(f"❌ Failed to initialize PolygonDataManager: {e}")
        return
    
    # 2. Test RequestAggregator
    try:
        aggregator = RequestAggregator(data_manager=data_manager)
        print("✓ RequestAggregator initialized")
    except Exception as e:
        print(f"❌ Failed to initialize RequestAggregator: {e}")
        return
    
    # 3. Test DataCoordinator
    try:
        coordinator = DataCoordinator(data_manager)
        print("✓ DataCoordinator initialized")
    except Exception as e:
        print(f"❌ Failed to initialize DataCoordinator: {e}")
        return
    
    # 4. Test ProtectedDataManager
    try:
        protected_manager = ProtectedDataManager(data_manager)
        print("✓ ProtectedDataManager initialized")
    except Exception as e:
        print(f"❌ Failed to initialize ProtectedDataManager: {e}")
        return
    
    # 5. Test DataValidator
    try:
        validator = DataValidator()
        print("✓ DataValidator initialized")
    except Exception as e:
        print(f"❌ Failed to initialize DataValidator: {e}")
        return
    
    # 6. Test TradeQuoteAligner
    try:
        aligner = TradeQuoteAligner()
        print("✓ TradeQuoteAligner initialized")
    except Exception as e:
        print(f"❌ Failed to initialize TradeQuoteAligner: {e}")
        return
    
    # 7. Test creating a DataNeed
    try:
        need = DataNeed(
            module_name="TestModule",
            symbol="AAPL",
            data_type=DataType.BARS,
            timeframe="1min",
            start_time=datetime.now(timezone.utc) - timedelta(hours=1),
            end_time=datetime.now(timezone.utc)
        )
        print("✓ DataNeed created successfully")
    except Exception as e:
        print(f"❌ Failed to create DataNeed: {e}")
        return
    
    print("\n✅ All components initialized successfully!")
    
    # Optional: Test a simple data fetch (will use API credits)
    test_fetch = input("\nTest data fetching? This will use API credits (y/n): ")
    if test_fetch.lower() == 'y':
        print("\nTesting data fetch...")
        symbol = "AAPL"
        end_time = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        start_time = end_time - timedelta(minutes=5)
        
        try:
            data_manager.set_current_plugin("TestScript")
            bars = await data_manager.load_bars(symbol, start_time, end_time, "1min")
            print(f"✓ Fetched {len(bars)} bars")
            
            # Test cache
            bars2 = await data_manager.load_bars(symbol, start_time, end_time, "1min")
            print(f"✓ Cache working (fetched {len(bars2)} bars from cache)")
            
        except Exception as e:
            print(f"❌ Data fetch failed: {e}")


if __name__ == "__main__":
    print("Testing Backtest Data Module")
    print("=" * 50)
    asyncio.run(test_basic_functionality())