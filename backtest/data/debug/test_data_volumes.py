# backtest/data/debug/test_data_volumes.py
import asyncio
import argparse
from datetime import datetime, timezone, timedelta
import sys
import os

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.dirname(current_dir)
backtest_dir = os.path.dirname(data_dir)
sys.path.insert(0, backtest_dir)

from backtest.data.polygon_data_manager import PolygonDataManager

async def test_bar_volumes(symbol: str, test_date: str):
    """Test how many bars we can pull in a single request"""
    data_manager = PolygonDataManager(
        extend_window_bars=2000  # Default setting
    )
    data_manager.set_current_plugin("BarVolumeTest")
    
    print(f"=== BAR VOLUME TEST for {symbol} ===")
    print(f"extend_window_bars setting: {data_manager.extend_window_bars}")
    
    # Test different time ranges
    test_ranges = [
        ("1 hour", timedelta(hours=1)),
        ("6 hours", timedelta(hours=6)),
        ("12 hours", timedelta(hours=12)),
        ("24 hours", timedelta(hours=24)),
        ("48 hours", timedelta(hours=48)),
        ("5 days", timedelta(days=5)),
    ]
    
    base_time = datetime.strptime(f"{test_date} 16:00:00", "%Y-%m-%d %H:%M:%S")
    base_time = base_time.replace(tzinfo=timezone.utc)  # 11 AM ET
    
    for name, duration in test_ranges:
        start_time = base_time - duration
        end_time = base_time
        
        print(f"\n{name} test:")
        print(f"Requested range: {start_time} to {end_time}")
        
        # Test what actually gets fetched
        bars = await data_manager.load_bars(symbol, start_time, end_time, '1min')
        
        if not bars.empty:
            actual_start = bars.index.min()
            actual_end = bars.index.max()
            print(f"Actual data range: {actual_start} to {actual_end}")
            print(f"Bars fetched: {len(bars):,}")
            
            # Calculate expected vs actual
            expected_minutes = int(duration.total_seconds() / 60)
            print(f"Expected bars (if continuous): {expected_minutes:,}")
            print(f"Coverage: {len(bars)/expected_minutes*100:.1f}%")
            
            # Check if we got extended data
            requested_duration = (end_time - start_time).total_seconds() / 60
            actual_duration = (actual_end - actual_start).total_seconds() / 60
            extra_minutes = actual_duration - requested_duration
            print(f"Extra data fetched: {extra_minutes:.0f} minutes")
        else:
            print("No data returned!")

async def test_tick_volumes(symbol: str, test_date: str):
    """Test how many ticks we can pull in a single request"""
    data_manager = PolygonDataManager()
    data_manager.set_current_plugin("TickVolumeTest")
    
    print(f"\n\n=== TICK VOLUME TEST for {symbol} ===")
    
    # Test during high-volume market hours
    base_time = datetime.strptime(f"{test_date} 15:30:00", "%Y-%m-%d %H:%M:%S")
    base_time = base_time.replace(tzinfo=timezone.utc)  # 10:30 AM ET
    
    test_ranges = [
        ("5 minutes", timedelta(minutes=5)),
        ("15 minutes", timedelta(minutes=15)),
        ("30 minutes", timedelta(minutes=30)),
        ("1 hour", timedelta(hours=1)),
        ("2 hours", timedelta(hours=2)),
    ]
    
    print("\nTRADE DATA:")
    for name, duration in test_ranges:
        start_time = base_time
        end_time = base_time + duration
        
        print(f"\n{name} test ({start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')} UTC):")
        
        trades = await data_manager.load_trades(symbol, start_time, end_time)
        
        if not trades.empty:
            print(f"Trades fetched: {len(trades):,}")
            print(f"Avg trades/minute: {len(trades)/(duration.total_seconds()/60):.1f}")
            
            # Check if we hit any limits
            if len(trades) == 50000:
                print("⚠️  Hit 50,000 limit!")
            elif len(trades) >= 49000:
                print("⚠️  Near 50,000 limit")
        else:
            print("No trades returned!")
    
    print("\n\nQUOTE DATA:")
    for name, duration in test_ranges:
        start_time = base_time
        end_time = base_time + duration
        
        print(f"\n{name} test ({start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')} UTC):")
        
        quotes = await data_manager.load_quotes(symbol, start_time, end_time)
        
        if not quotes.empty:
            print(f"Quotes fetched: {len(quotes):,}")
            print(f"Avg quotes/minute: {len(quotes)/(duration.total_seconds()/60):.1f}")
            
            # Check if we hit any limits
            if len(quotes) == 50000:
                print("⚠️  Hit 50,000 limit!")
            elif len(quotes) >= 49000:
                print("⚠️  Near 50,000 limit")
        else:
            print("No quotes returned!")

async def test_extend_window_effect(symbol: str, test_date: str):
    """Test how extend_window_bars affects data fetching"""
    print(f"\n\n=== EXTEND WINDOW TEST for {symbol} ===")
    
    # Test with different extend_window_bars settings
    extend_settings = [0, 500, 1000, 2000, 5000]
    
    # Request 1 hour of data
    end_time = datetime.strptime(f"{test_date} 16:00:00", "%Y-%m-%d %H:%M:%S")
    end_time = end_time.replace(tzinfo=timezone.utc)
    start_time = end_time - timedelta(hours=1)
    
    print(f"Requesting 1 hour: {start_time} to {end_time}")
    
    for extend_bars in extend_settings:
        data_manager = PolygonDataManager(extend_window_bars=extend_bars)
        data_manager.set_current_plugin(f"ExtendTest_{extend_bars}")
        
        bars = await data_manager.load_bars(symbol, start_time, end_time, '1min')
        
        if not bars.empty:
            actual_start = bars.index.min()
            actual_end = bars.index.max()
            extra_before = (start_time - actual_start).total_seconds() / 60
            extra_after = (actual_end - end_time).total_seconds() / 60
            
            print(f"\nextend_window_bars={extend_bars}:")
            print(f"  Total bars: {len(bars):,}")
            print(f"  Extra before: {extra_before:.0f} min")
            print(f"  Extra after: {extra_after:.0f} min")
            print(f"  Actual range: {actual_start.strftime('%H:%M')} to {actual_end.strftime('%H:%M')}")

async def find_tick_data_limits(symbol: str, test_date: str):
    """Find the maximum tick data we can fetch in one request"""
    print(f"\n\n=== FINDING TICK DATA LIMITS for {symbol} ===")
    
    data_manager = PolygonDataManager()
    data_manager.set_current_plugin("TickLimitTest")
    
    # Start with market open
    base_time = datetime.strptime(f"{test_date} 14:30:00", "%Y-%m-%d %H:%M:%S")
    base_time = base_time.replace(tzinfo=timezone.utc)  # 9:30 AM ET
    
    # Try increasingly larger windows
    print("\nTesting TRADES:")
    for hours in [1, 2, 3, 4, 5, 6]:
        start_time = base_time
        end_time = base_time + timedelta(hours=hours)
        
        trades = await data_manager.load_trades(symbol, start_time, end_time)
        count = len(trades) if not trades.empty else 0
        
        print(f"{hours}h window: {count:,} trades", end="")
        if count >= 50000:
            print(" ← Hit limit!")
            break
        else:
            print()
    
    print("\nTesting QUOTES:")
    for minutes in [15, 30, 45, 60, 90, 120]:
        start_time = base_time
        end_time = base_time + timedelta(minutes=minutes)
        
        quotes = await data_manager.load_quotes(symbol, start_time, end_time)
        count = len(quotes) if not quotes.empty else 0
        
        print(f"{minutes}min window: {count:,} quotes", end="")
        if count >= 50000:
            print(" ← Hit limit!")
            break
        else:
            print()

async def main(symbol: str, test_date: str):
    """Run all volume tests"""
    
    print(f"Running volume tests for {symbol} on {test_date}")
    print("=" * 60)
    
    # Test 1: Bar volumes
    await test_bar_volumes(symbol, test_date)
    
    # Test 2: Tick volumes
    await test_tick_volumes(symbol, test_date)
    
    # Test 3: Extend window effect
    await test_extend_window_effect(symbol, test_date)
    
    # Test 4: Find limits
    await find_tick_data_limits(symbol, test_date)
    
    print("\n\n=== SUMMARY ===")
    print("Current settings in api_client.py:")
    print("- API limit per request: 50,000")
    print("- Default extend_window_bars: 2,000")
    print("\nRecommendations based on tests will appear above.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test data volume capabilities")
    parser.add_argument("-s", "--symbol", default="AAPL", help="Stock symbol")
    parser.add_argument("-t", "--date", default="2025-01-03", help="Test date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    asyncio.run(main(args.symbol, args.date))