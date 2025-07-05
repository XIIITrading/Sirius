# backtest/data/debug/test_bar_limits.py
import asyncio
import argparse
from datetime import datetime, timezone, timedelta
import sys
import os
import logging

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.dirname(current_dir)
backtest_dir = os.path.dirname(data_dir)
sys.path.insert(0, backtest_dir)

from backtest.data.polygon_data_manager import PolygonDataManager

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def debug_bar_fetching(symbol: str, test_date: str):
    """Debug the bar fetching process to understand limits"""
    
    print("=== DEBUGGING BAR FETCH LIMITS ===\n")
    
    # Create data manager with specific settings
    data_manager = PolygonDataManager(
        extend_window_bars=2000,
        memory_cache_size=0,  # Disable cache to see actual API calls
        file_cache_hours=0    # Disable file cache
    )
    data_manager.set_current_plugin("BarLimitDebug")
    
    # Test case: Request 24 hours of data
    end_time = datetime.strptime(f"{test_date} 16:00:00", "%Y-%m-%d %H:%M:%S")
    end_time = end_time.replace(tzinfo=timezone.utc)
    start_time = end_time - timedelta(hours=24)
    
    print(f"Test: Requesting 24 hours of {symbol}")
    print(f"Requested start: {start_time}")
    print(f"Requested end: {end_time}")
    print(f"Expected bars: ~1,440 (if continuous)\n")
    
    # Calculate what SHOULD be requested with extend_window
    minutes_per_bar = 1
    extension_minutes = data_manager.extend_window_bars * minutes_per_bar
    extended_start = start_time - timedelta(minutes=extension_minutes)
    extended_end = end_time + timedelta(minutes=extension_minutes)
    
    print(f"With extend_window_bars={data_manager.extend_window_bars}:")
    print(f"Should request from: {extended_start}")
    print(f"Should request to: {extended_end}")
    print(f"Extension: {extension_minutes} minutes each side\n")
    
    # Monkey-patch to see actual API call
    original_fetch = data_manager.api_client.fetch_bars
    actual_api_calls = []
    
    async def tracked_fetch_bars(symbol, start_time, end_time, timeframe='1min'):
        api_call = {
            'symbol': symbol,
            'start_time': start_time,
            'end_time': end_time,
            'timeframe': timeframe
        }
        actual_api_calls.append(api_call)
        print(f"API CALL INTERCEPTED:")
        print(f"  Symbol: {symbol}")
        print(f"  Start: {start_time}")
        print(f"  End: {end_time}")
        print(f"  Timeframe: {timeframe}")
        
        # Call original
        result = await original_fetch(symbol, start_time, end_time, timeframe)
        if result is not None:
            print(f"  Result: {len(result)} bars returned")
            print(f"  Actual range: {result.index.min()} to {result.index.max()}\n")
        return result
    
    data_manager.api_client.fetch_bars = tracked_fetch_bars
    
    # Make the request
    bars = await data_manager.load_bars(symbol, start_time, end_time, '1min')
    
    print("\n=== RESULTS ===")
    print(f"Bars returned to caller: {len(bars)}")
    if not bars.empty:
        print(f"Data range: {bars.index.min()} to {bars.index.max()}")
        print(f"Duration: {(bars.index.max() - bars.index.min()).total_seconds() / 3600:.1f} hours")
    
    # Check cache behavior
    print(f"\nCache stats:")
    cache_stats = data_manager.get_cache_stats()
    print(f"  Memory cache items: {cache_stats['memory_cache']['cached_items']}")
    print(f"  API calls made: {len(actual_api_calls)}")
    
    # Test the _filter_timerange function
    if not bars.empty and len(actual_api_calls) > 0:
        # Get the raw API result
        api_call = actual_api_calls[0]
        raw_bars = await original_fetch(
            api_call['symbol'], 
            api_call['start_time'], 
            api_call['end_time'], 
            api_call['timeframe']
        )
        
        if raw_bars is not None:
            print(f"\nFilter analysis:")
            print(f"  Raw API result: {len(raw_bars)} bars")
            print(f"  After filtering: {len(bars)} bars")
            print(f"  Filtered out: {len(raw_bars) - len(bars)} bars")

async def test_url_construction(symbol: str, test_date: str):
    """Test the actual URL being constructed"""
    print("\n\n=== URL CONSTRUCTION TEST ===\n")
    
    data_manager = PolygonDataManager(extend_window_bars=2000)
    api_client = data_manager.api_client
    
    # Test URL construction
    end_time = datetime.strptime(f"{test_date} 16:00:00", "%Y-%m-%d %H:%M:%S")
    end_time = end_time.replace(tzinfo=timezone.utc)
    start_time = end_time - timedelta(hours=24)
    
    # Get URL parameters
    from_ts = int(start_time.timestamp() * 1000)
    to_ts = int(end_time.timestamp() * 1000)
    
    url = f"{api_client.base_url}/v2/aggs/ticker/{symbol}/range/1/minute/{from_ts}/{to_ts}"
    
    print(f"Constructed URL:")
    print(f"{url}")
    print(f"\nParameters:")
    print(f"  from_ts: {from_ts} ({datetime.fromtimestamp(from_ts/1000, tz=timezone.utc)})")
    print(f"  to_ts: {to_ts} ({datetime.fromtimestamp(to_ts/1000, tz=timezone.utc)})")
    print(f"  Duration: {(to_ts - from_ts) / 1000 / 3600:.1f} hours")

async def test_pagination_behavior(symbol: str, test_date: str):
    """Test if pagination is happening"""
    print("\n\n=== PAGINATION TEST ===\n")
    
    data_manager = PolygonDataManager()
    data_manager.set_current_plugin("PaginationTest")
    
    # Request a large time range
    end_time = datetime.strptime(f"{test_date} 16:00:00", "%Y-%m-%d %H:%M:%S")
    end_time = end_time.replace(tzinfo=timezone.utc)
    start_time = end_time - timedelta(days=7)  # 7 days
    
    print(f"Requesting 7 days of data for {symbol}")
    
    # Track pagination
    api_client = data_manager.api_client
    original_get = api_client.session.get
    page_count = 0
    total_results = 0
    
    def tracked_get(url, **kwargs):
        nonlocal page_count, total_results
        response = original_get(url, **kwargs)
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            page_count += 1
            total_results += len(results)
            print(f"Page {page_count}: {len(results)} results, has_next: {'next_url' in data}")
        return response
    
    api_client.session.get = tracked_get
    
    bars = await data_manager.load_bars(symbol, start_time, end_time, '1min')
    
    print(f"\nTotal pages fetched: {page_count}")
    print(f"Total results across pages: {total_results}")
    print(f"Final bars returned: {len(bars)}")

async def main(symbol: str, test_date: str):
    """Run all debug tests"""
    await debug_bar_fetching(symbol, test_date)
    await test_url_construction(symbol, test_date)
    await test_pagination_behavior(symbol, test_date)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug bar fetching limits")
    parser.add_argument("-s", "--symbol", default="AAPL", help="Stock symbol")
    parser.add_argument("-t", "--date", default="2025-01-03", help="Test date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    asyncio.run(main(args.symbol, args.date))