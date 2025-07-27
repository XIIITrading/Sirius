# fetch_polygon_data.py
"""
Fetch 5-minute OHLC data from Polygon for comparison with TradingView
"""

import pandas as pd
from datetime import datetime
import pytz
from polygon import DataFetcher
from polygon.config import PolygonConfig

def fetch_polygon_bars():
    print("Fetching 5-minute bars from Polygon")
    print("="*80)
    
    # Initialize Polygon fetcher
    fetcher = DataFetcher(config=PolygonConfig({'cache_enabled': True}))
    
    # Define time range in ET
    et_tz = pytz.timezone('US/Eastern')
    utc_tz = pytz.UTC
    
    # Start: July 23, 2025 20:00 ET
    start_et = et_tz.localize(datetime(2025, 7, 23, 20, 0))
    # End: July 25, 2025 13:30 ET  
    end_et = et_tz.localize(datetime(2025, 7, 25, 13, 30))
    
    # Convert to UTC for Polygon
    start_utc = start_et.astimezone(utc_tz)
    end_utc = end_et.astimezone(utc_tz)
    
    print(f"Time range:")
    print(f"  Start: {start_et} ET / {start_utc} UTC")
    print(f"  End: {end_et} ET / {end_utc} UTC")
    print()
    
    # Fetch 5-minute data
    try:
        df = fetcher.fetch_data(
            symbol='TSLA',
            timeframe='5min',
            start_date=start_utc,
            end_date=end_utc,
            use_cache=True,
            validate=True
        )
        
        if df.empty:
            print("No data returned!")
            return
            
        # Convert index to ET for display
        df.index = df.index.tz_convert(et_tz)
        
        # Add Unix timestamp column for comparison with TradingView
        df['unix_time'] = df.index.astype('int64') // 10**9
        
        # Create output dataframe with same format as TradingView
        output_df = pd.DataFrame({
            'time': df['unix_time'],
            'datetime_et': df.index.strftime('%Y-%m-%d %H:%M'),
            'open': df['open'].round(2),
            'high': df['high'].round(2), 
            'low': df['low'].round(2),
            'close': df['close'].round(2),
            'volume': df['volume'].astype(int)
        })
        
        # Save to CSV
        output_df.to_csv('polygon_5min_bars.csv', index=False)
        print(f"Total bars: {len(output_df)}")
        print(f"\nData saved to: polygon_5min_bars.csv")
        
        # Show sample data
        print("\nFirst 10 bars:")
        print(output_df.head(10).to_string())
        
        print("\n" + "="*40)
        
        print("\nLast 10 bars:")
        print(output_df.tail(10).to_string())
        
        # Show specific session highlights
        print("\n" + "="*80)
        print("SESSION ANALYSIS")
        print("="*80)
        
        # July 24 RTH (09:30-16:00 ET)
        july24_start = et_tz.localize(datetime(2025, 7, 24, 9, 30))
        july24_end = et_tz.localize(datetime(2025, 7, 24, 16, 0))
        july24_rth = output_df[
            (pd.to_datetime(output_df['datetime_et']) >= july24_start) & 
            (pd.to_datetime(output_df['datetime_et']) < july24_end)
        ]
        
        if not july24_rth.empty:
            print(f"\nJuly 24 RTH Session (09:30-16:00 ET):")
            print(f"  Bars: {len(july24_rth)}")
            print(f"  High: ${july24_rth['high'].max()}")
            print(f"  Low: ${july24_rth['low'].min()}")
            print(f"  Open: ${july24_rth.iloc[0]['open']}")
            print(f"  Close: ${july24_rth.iloc[-1]['close']}")
            
            # Show bars around 15:55-16:00
            print(f"\n  Bars around RTH close (15:55-16:00):")
            close_bars = july24_rth.tail(3)
            print(close_bars[['datetime_et', 'open', 'high', 'low', 'close']].to_string())
            
    except Exception as e:
        print(f"Error fetching data: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    fetch_polygon_bars()