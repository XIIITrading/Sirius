# backtest/data/debug/test_extended_hours.py
import argparse
import asyncio
from datetime import datetime, timezone, timedelta
import pandas as pd
import sys
import os

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.dirname(current_dir)
backtest_dir = os.path.dirname(data_dir)
sys.path.insert(0, backtest_dir)

from backtest.data.polygon_data_manager import PolygonDataManager

async def test_extended_hours_data(symbol: str = 'AAPL', test_date: str = '2025-01-03'):
    """
    Test if bars data includes extended hours (pre-market and after-hours)
    """
    # Initialize data manager
    data_manager = PolygonDataManager()
    data_manager.set_current_plugin("ExtendedHoursTest")
    
    # For EST/EDT conversion - we'll assume EST for now
    # Pre-Market: 4:00 AM ET = 9:00 AM UTC
    # After-Hours end: 8:00 PM ET = 1:00 AM UTC (next day)
    
    start_time = datetime.strptime(f"{test_date} 09:00:00", "%Y-%m-%d %H:%M:%S")
    start_time = start_time.replace(tzinfo=timezone.utc)
    
    # End at 1 AM UTC next day (8 PM ET)
    end_time = start_time + timedelta(hours=16)  # 9 AM to 1 AM next day
    
    print(f"Fetching data for {symbol} on {test_date}")
    print(f"UTC range: {start_time} to {end_time}")
    
    # Fetch 1-minute bars
    bars = await data_manager.load_bars(
        symbol=symbol,
        start_time=start_time,
        end_time=end_time,
        timeframe='1min'
    )
    
    if bars.empty:
        print("No data returned!")
        return
    
    print(f"\nTotal bars fetched: {len(bars)}")
    print(f"Data range: {bars.index.min()} to {bars.index.max()}")
    
    # Analyze hourly distribution
    bars['hour_utc'] = bars.index.hour
    hourly_counts = bars.groupby('hour_utc').size()
    
    print("\nBars per UTC hour:")
    for hour in range(24):
        count = hourly_counts.get(hour, 0)
        if count > 0:
            print(f"  Hour {hour:02d}: {count} bars")
    
    # Check for extended hours data
    # Pre-market hours in UTC (EST): 9:00-14:30 UTC
    # Regular hours in UTC (EST): 14:30-21:00 UTC  
    # After-hours in UTC (EST): 21:00-01:00 UTC
    
    pre_market_bars = bars[(bars.index.hour >= 9) & (bars.index.hour < 14) | 
                          ((bars.index.hour == 14) & (bars.index.minute < 30))]
    
    regular_bars = bars[((bars.index.hour == 14) & (bars.index.minute >= 30)) |
                       ((bars.index.hour > 14) & (bars.index.hour < 21))]
    
    after_hours_bars = bars[(bars.index.hour >= 21) | (bars.index.hour < 1)]
    
    print(f"\nExtended Hours Analysis:")
    print(f"Pre-market bars (9:00-14:30 UTC): {len(pre_market_bars)}")
    print(f"Regular hours bars (14:30-21:00 UTC): {len(regular_bars)}")
    print(f"After-hours bars (21:00-01:00 UTC): {len(after_hours_bars)}")
    
    # Show sample timestamps from each period
    if len(pre_market_bars) > 0:
        print(f"\nSample pre-market timestamps:")
        print(pre_market_bars.index[:5].tolist())
    
    if len(after_hours_bars) > 0:
        print(f"\nSample after-hours timestamps:")
        print(after_hours_bars.index[:5].tolist())
    
    # Check data completeness
    print(f"\n✓ Pre-market data present: {len(pre_market_bars) > 0}")
    print(f"✓ After-hours data present: {len(after_hours_bars) > 0}")
    
    return bars

def main():
    parser = argparse.ArgumentParser(description="Test extended hours data fetching")
    parser.add_argument("-s", "--symbol", default="AAPL", help="Stock symbol")
    parser.add_argument("-t", "--date", default="2025-01-03", help="Test date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Run the async test
    asyncio.run(test_extended_hours_data(args.symbol, args.date))

if __name__ == "__main__":
    main()