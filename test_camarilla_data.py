# test_camarilla_data.py
"""
Simple test with CORRECT session times
"""

import pandas as pd
from datetime import datetime, timedelta
import pytz
from polygon import DataFetcher
from polygon.config import PolygonConfig

def test_fetch_data():
    print(f"\n{'='*80}")
    print(f"CAMARILLA DATA TEST FOR TSLA - CORRECTED SESSIONS")
    print(f"{'='*80}\n")
    
    # Initialize Polygon fetcher
    fetcher = DataFetcher(config=PolygonConfig({'cache_enabled': True}))
    
    # HARDCODED: We want pivots for July 25, so we need July 24 data
    data_date = datetime(2025, 7, 24)  # July 24, 2025
    
    print(f"Fetching data for: July 24, 2025")
    print(f"To calculate pivots for: July 25, 2025")
    
    # Convert to UTC for fetching
    utc_tz = pytz.UTC
    et_tz = pytz.timezone('US/Eastern')
    
    # Fetch full day of July 24 data
    start_utc = datetime(2025, 7, 24, 0, 0, tzinfo=utc_tz)
    end_utc = datetime(2025, 7, 24, 23, 59, tzinfo=utc_tz)
    
    print(f"\nFetching from {start_utc} to {end_utc} UTC")
    
    try:
        # Fetch minute data
        df = fetcher.fetch_data(
            symbol='TSLA',
            timeframe='1min',
            start_date=start_utc,
            end_date=end_utc,
            use_cache=True,
            validate=True
        )
        
        print(f"Total bars fetched: {len(df)}")
        
        if df.empty:
            print("ERROR: No data returned!")
            return
        
        # Make sure index is timezone aware
        if df.index.tz is None:
            df.index = df.index.tz_localize(utc_tz)
        else:
            df.index = df.index.tz_convert(utc_tz)
        
        print(f"Data range: {df.index[0]} to {df.index[-1]}")
        
        # CORRECTED SESSIONS:
        # Pre: 08:00-13:29 UTC (04:00-09:29 ET)
        # RTH: 13:30-19:59 UTC (09:30-15:59 ET)  
        # Post: 20:00-23:59 UTC (16:00-19:59 ET)
        
        print(f"\n{'='*60}")
        print("SESSION DATA WITH CORRECT TIMES (July 24, 2025)")
        print(f"{'='*60}")
        
        # Pre-market: 08:00-13:29 UTC
        pre_data = df[(df.index.time >= pd.Timestamp('08:00').time()) & 
                      (df.index.time <= pd.Timestamp('13:29').time())]
        
        if not pre_data.empty:
            print(f"\nPre-Market (08:00-13:29 UTC / 04:00-09:29 ET):")
            print(f"  Bars: {len(pre_data)}")
            print(f"  High: ${pre_data['high'].max():.2f}")
            print(f"  Low: ${pre_data['low'].min():.2f}")
            print(f"  Close: ${pre_data['close'].iloc[-1]:.2f}")
        
        # RTH: 13:30-19:59 UTC
        rth_data = df[(df.index.time >= pd.Timestamp('13:30').time()) & 
                      (df.index.time <= pd.Timestamp('19:59').time())]
        
        if not rth_data.empty:
            print(f"\nRTH (13:30-19:59 UTC / 09:30-15:59 ET):")
            print(f"  Bars: {len(rth_data)}")
            rth_high = rth_data['high'].max()
            rth_low = rth_data['low'].min()
            print(f"  High: ${rth_high:.2f}")
            print(f"  Low: ${rth_low:.2f}")
            
            # Get close at 19:59 UTC (15:59 ET)
            close_1959_utc = df[df.index.time == pd.Timestamp('19:59').time()]
            if not close_1959_utc.empty:
                rth_close = close_1959_utc['close'].iloc[0]
            else:
                # Get last bar before 20:00
                last_rth = rth_data.iloc[-1]
                rth_close = last_rth['close']
            print(f"  Close at 15:59 ET: ${rth_close:.2f}")
            
            # Calculate RTH pivots
            rth_range = rth_high - rth_low
            r6 = rth_close * rth_high / rth_low
            r4 = rth_close + rth_range * 1.1 / 2
            r3 = rth_close + rth_range * 1.1 / 4
            
            print(f"\nRTH-based Pivots for July 25:")
            print(f"  R6: ${r6:.2f}")
            print(f"  R4: ${r4:.2f}")
            print(f"  R3: ${r3:.2f}")
            print(f"  CP: ${rth_close:.2f}")
        
        # Post-market: 20:00-23:59 UTC
        post_data = df[(df.index.time >= pd.Timestamp('20:00').time()) & 
                       (df.index.time <= pd.Timestamp('23:59').time())]
        
        if not post_data.empty:
            print(f"\nPost-Market (20:00-23:59 UTC / 16:00-19:59 ET):")
            print(f"  Bars: {len(post_data)}")
            print(f"  High: ${post_data['high'].max():.2f}")
            print(f"  Low: ${post_data['low'].min():.2f}")
            
            # Get close at 23:59 UTC (19:59 ET)
            close_2359_utc = df[df.index.time == pd.Timestamp('23:59').time()]
            if not close_2359_utc.empty:
                eth_close = close_2359_utc['close'].iloc[0]
            else:
                eth_close = post_data['close'].iloc[-1]
            print(f"  Close at 19:59 ET: ${eth_close:.2f}")
            
            # ETH calculations (RTH + Post)
            eth_high = max(rth_high, post_data['high'].max())
            eth_low = min(rth_low, post_data['low'].min())
            
            print(f"\nETH Combined (RTH + Post):")
            print(f"  High: ${eth_high:.2f}")
            print(f"  Low: ${eth_low:.2f}")
            print(f"  Close: ${eth_close:.2f}")
            
            # ETH pivots
            eth_range = eth_high - eth_low
            eth_r6 = eth_close * eth_high / eth_low
            eth_r4 = eth_close + eth_range * 1.1 / 2
            eth_r3 = eth_close + eth_range * 1.1 / 4
            
            print(f"\nETH-based Pivots for July 25:")
            print(f"  R6: ${eth_r6:.2f}")
            print(f"  R4: ${eth_r4:.2f}")
            print(f"  R3: ${eth_r3:.2f}")
            print(f"  CP: ${eth_close:.2f}")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_fetch_data()