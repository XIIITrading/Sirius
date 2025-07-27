# market_review/calculations/pivots/camarilla_engine.py

from dataclasses import dataclass
from typing import List, Optional, Dict, Protocol
import pandas as pd
from datetime import datetime, timedelta
import pytz

# Protocol for data fetching
class DataFetcherProtocol(Protocol):
    def fetch_data(self, symbol: str, timeframe: str, start_date: datetime, 
                   end_date: datetime, **kwargs) -> pd.DataFrame:
        ...

@dataclass
class CamarillaPivot:
    """Single Camarilla pivot level"""
    level_name: str
    price: float
    strength: int
    timeframe: str

@dataclass 
class CamarillaResult:
    """Complete Camarilla analysis for a timeframe"""
    timeframe: str
    close: float
    high: float
    low: float
    pivots: List[CamarillaPivot]
    range_type: str
    central_pivot: float
    data_type: Optional[str] = 'RTH'
    debug_info: Optional[Dict] = None

class CamarillaEngine:
    """Calculate Camarilla pivot points matching Pine Script exactly"""
    
    def __init__(self, 
                 data_fetcher: DataFetcherProtocol,
                 range_threshold_pct: float = 0.3,
                 switching_tolerance: float = 0.1,
                 calc_mode: str = 'auto',
                 use_close: str = 'auto',
                 use_current_pre: bool = False):
        self.data_fetcher = data_fetcher
        self.range_threshold_pct = range_threshold_pct
        self.switching_tolerance = switching_tolerance
        self.calc_mode = calc_mode
        self.use_close = use_close
        self.use_current_pre = use_current_pre
        
    def calculate_pivots(self, high: float, low: float, close: float) -> dict:
        """Calculate all Camarilla pivot levels - EXACT Pine Script formula"""
        range_val = high - low
        
        # EXACT formulas from Pine Script
        r3 = close + range_val * 1.1 / 4
        r4 = close + range_val * 1.1 / 2
        r6 = close * high / low if low != 0 else close
        
        s3 = close - range_val * 1.1 / 4
        s4 = close - range_val * 1.1 / 2
        s6 = 2 * close - r6
        
        return {
            'R6': r6, 'R4': r4, 'R3': r3,
            'S3': s3, 'S4': s4, 'S6': s6,
            'CP': close
        }
    
    def get_pine_script_data_5min(self, symbol: str, pivots_for_date: datetime) -> Dict:
        """
        Get data exactly as Pine Script does using 5-minute bars.
        
        Pine Script logic:
        - yRthHigh/Low: RTH session high/low from previous day
        - yEthHigh/Low: Full 24h high/low from previous day  
        - preHigh/Low: Pre-market high/low from current day (if using current pre)
        - Switching logic checks if ETH or pre-market exceeded RTH range
        """
        # The date we want pivots FOR
        pivots_date = pivots_for_date.date()
        
        # The date we need data FROM (previous trading day)
        data_date = pivots_date - timedelta(days=1)
        
        # Skip weekends
        while data_date.weekday() >= 5:
            data_date = data_date - timedelta(days=1)
        
        print(f"\n{'='*80}")
        print(f"Pine Script Data Collection for {symbol}")
        print(f"Calculating pivots FOR: {pivots_date}")
        print(f"Using data FROM: {data_date}")
        print(f"{'='*80}")
        
        # Get current time
        now_utc = datetime.now(pytz.UTC)
        
        # Fetch 5-minute data for the data date and possibly current date pre-market
        # We need from start of data_date to current time (or end of pivots_date)
        start_datetime = datetime.combine(data_date, datetime.min.time()).replace(tzinfo=pytz.UTC)
        
        # Don't request future data
        if pivots_for_date > now_utc:
            end_datetime = now_utc
        else:
            # Get up to end of pivots_date or current time, whichever is earlier
            end_of_pivots = datetime.combine(pivots_date, datetime.max.time()).replace(tzinfo=pytz.UTC)
            end_datetime = min(end_of_pivots, now_utc)
        
        print(f"Fetching 5-min data from {start_datetime} to {end_datetime}")
        
        try:
            df = self.data_fetcher.fetch_data(
                symbol=symbol,
                timeframe='5min',
                start_date=start_datetime,
                end_date=end_datetime,
                use_cache=True,
                validate=True
            )
        except Exception as e:
            raise ValueError(f"Failed to fetch data: {str(e)}")
        
        if df.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Ensure index is timezone-aware UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        elif df.index.tz != pytz.UTC:
            df.index = df.index.tz_convert('UTC')
        
        # Convert to ET for session filtering (Pine Script uses ET for US stocks)
        df_et = df.copy()
        df_et.index = df_et.index.tz_convert('US/Eastern')
        
        # Define sessions in ET
        # Pre-market: 04:00 - 09:29 ET
        # RTH: 09:30 - 15:59 ET (16:00 is NOT included in RTH)
        # Post-market: 16:00 - 19:59 ET
        
        # Get data_date sessions
        data_date_df = df_et[df_et.index.date == data_date]
        
        # RTH session (09:30 - 15:59 ET)
        rth_data = data_date_df[
            (data_date_df.index.time >= pd.Timestamp('09:30').time()) & 
            (data_date_df.index.time <= pd.Timestamp('15:59').time())
        ]
        
        # Pre-market session (04:00 - 09:29 ET)
        pre_data = data_date_df[
            (data_date_df.index.time >= pd.Timestamp('04:00').time()) & 
            (data_date_df.index.time <= pd.Timestamp('09:29').time())
        ]
        
        # Post-market session (16:00 - 19:59 ET)
        post_data = data_date_df[
            (data_date_df.index.time >= pd.Timestamp('16:00').time()) & 
            (data_date_df.index.time <= pd.Timestamp('19:59').time())
        ]
        
        # Calculate session values
        print(f"\nSession Data for {data_date} (all times ET):")
        
        # Pre-market
        if not pre_data.empty:
            pre_high = float(pre_data['high'].max())
            pre_low = float(pre_data['low'].min())
            pre_close = float(pre_data['close'].iloc[-1])
            print(f"  Pre-Market (04:00-09:29): H=${pre_high:.2f}, L=${pre_low:.2f}, C=${pre_close:.2f}")
        else:
            pre_high = pre_low = pre_close = None
            print(f"  Pre-Market: No data")
        
        # RTH
        if not rth_data.empty:
            rth_high = float(rth_data['high'].max())
            rth_low = float(rth_data['low'].min())
            # Get close at 15:59 (last bar of RTH)
            rth_close_bar = rth_data[rth_data.index.time == pd.Timestamp('15:55').time()]
            if not rth_close_bar.empty:
                rth_close = float(rth_close_bar['close'].iloc[-1])
            else:
                rth_close = float(rth_data['close'].iloc[-1])
            print(f"  RTH (09:30-15:59): H=${rth_high:.2f}, L=${rth_low:.2f}, C=${rth_close:.2f}")
        else:
            rth_high = rth_low = rth_close = None
            print(f"  RTH: No data")
        
        # Post-market
        if not post_data.empty:
            post_high = float(post_data['high'].max())
            post_low = float(post_data['low'].min())
            # Get close at 19:59 (last bar of post)
            post_close_bar = post_data[post_data.index.time == pd.Timestamp('19:55').time()]
            if not post_close_bar.empty:
                post_close = float(post_close_bar['close'].iloc[-1])
            else:
                post_close = float(post_data['close'].iloc[-1])
            print(f"  Post-Market (16:00-19:59): H=${post_high:.2f}, L=${post_low:.2f}, C=${post_close:.2f}")
        else:
            post_high = post_low = post_close = None
            print(f"  Post-Market: No data")
        
        # ETH values (full day)
        eth_high = float(data_date_df['high'].max())
        eth_low = float(data_date_df['low'].min())
        eth_close = post_close if post_close else rth_close  # Use post close if available, else RTH
        
        print(f"\n  ETH (Full Day): H=${eth_high:.2f}, L=${eth_low:.2f}, C=${eth_close:.2f}")
        
        # Current day pre-market (if use_current_pre is True)
        cd_pre_high = cd_pre_low = None
        if self.use_current_pre and pivots_date <= now_utc.date():
            pivots_date_df = df_et[df_et.index.date == pivots_date]
            cd_pre_data = pivots_date_df[
                (pivots_date_df.index.time >= pd.Timestamp('04:00').time()) & 
                (pivots_date_df.index.time <= pd.Timestamp('08:30').time())
            ]
            if not cd_pre_data.empty:
                cd_pre_high = float(cd_pre_data['high'].max())
                cd_pre_low = float(cd_pre_data['low'].min())
                print(f"\n  Current Day Pre (04:00-08:30): H=${cd_pre_high:.2f}, L=${cd_pre_low:.2f}")
        
        # Apply Pine Script switching logic
        use_eth = False
        
        if self.calc_mode == 'auto' and rth_high and rth_low:
            # Pine Script logic from the code:
            # math.max(yRthHigh, yPostHigh, preHigh) > yRthHigh * (1 + switchingTolerance / 100)
            # or math.min(yRthLow, yPostLow, preLow) < yRthLow * (1 - switchingTolerance / 100)
            
            # Note: In Pine Script, it seems to use pre-market high in the comparison!
            # This explains why we see pre-market high being used
            highs_to_check = [h for h in [rth_high, post_high, pre_high, cd_pre_high] if h is not None]
            lows_to_check = [l for l in [rth_low, post_low, pre_low, cd_pre_low] if l is not None]
            
            if highs_to_check and lows_to_check:
                max_high = max(highs_to_check)
                min_low = min(lows_to_check)
                
                high_threshold = rth_high * (1 + self.switching_tolerance / 100)
                low_threshold = rth_low * (1 - self.switching_tolerance / 100)
                
                high_exceeded = max_high > high_threshold
                low_exceeded = min_low < low_threshold
                
                use_eth = high_exceeded or low_exceeded
                
                print(f"\nSwitching Logic:")
                print(f"  RTH High: ${rth_high:.2f}, Threshold: ${high_threshold:.2f}")
                print(f"  RTH Low: ${rth_low:.2f}, Threshold: ${low_threshold:.2f}")
                print(f"  Max High (all sessions): ${max_high:.2f}")
                print(f"  Min Low (all sessions): ${min_low:.2f}")
                print(f"  High Exceeded: {high_exceeded}")
                print(f"  Low Exceeded: {low_exceeded}")
                print(f"  Using ETH: {use_eth}")
        
        elif self.calc_mode == 'forceETH':
            use_eth = True
        elif self.calc_mode == 'forceRTH':
            use_eth = False
        
        # Determine which values to use
        # IMPORTANT: Pine Script appears to use pre-market high even in "RTH" mode!
        # This matches what we see in the data
        if use_eth:
            calc_high = eth_high
            calc_low = eth_low
            calc_close = eth_close
            data_type = 'ETH'
        else:
            # Pine Script quirk: Uses pre-market high if it exceeds RTH high
            # This explains why Y High = 314.25 (pre-market) not RTH high
            calc_high = max(pre_high, rth_high) if pre_high else rth_high
            calc_low = rth_low
            calc_close = rth_close
            data_type = 'RTH'
        
        print(f"\nFinal Values for Pivot Calculation ({data_type}):")
        print(f"  High: ${calc_high:.2f}")
        print(f"  Low: ${calc_low:.2f}")
        print(f"  Close: ${calc_close:.2f}")
        print(f"  Range: ${calc_high - calc_low:.2f}")
        
        # Calculate pivots
        pivots = self.calculate_pivots(calc_high, calc_low, calc_close)
        
        print(f"\nCalculated Pivots for {pivots_date}:")
        for level in ['R6', 'R4', 'R3', 'CP', 'S3', 'S4', 'S6']:
            print(f"  {level}: ${pivots[level]:.2f}")
        print("="*80 + "\n")
        
        return {
            'pivots_for_date': pivots_date,
            'data_from_date': data_date,
            'data_type': data_type,
            'use_eth': use_eth,
            'rth': {'high': rth_high, 'low': rth_low, 'close': rth_close},
            'pre': {'high': pre_high, 'low': pre_low, 'close': pre_close},
            'post': {'high': post_high, 'low': post_low, 'close': post_close},
            'eth': {'high': eth_high, 'low': eth_low, 'close': eth_close},
            'selected': {'high': calc_high, 'low': calc_low, 'close': calc_close},
            'pivots': pivots
        }
    
    def analyze_daily_pine_script(self, symbol: str, pivots_for_date: datetime) -> CamarillaResult:
        """
        Calculate pivots exactly as Pine Script does using 5-minute data.
        """
        # Get data with Pine Script methodology
        data = self.get_pine_script_data_5min(symbol, pivots_for_date)
        
        # Use the selected values
        selected = data['selected']
        
        # Create pivot objects
        pivots = []
        strength_map = {'R6': 6, 'R4': 4, 'R3': 3, 'S3': 3, 'S4': 4, 'S6': 6}
        
        for name, price in data['pivots'].items():
            if name != 'CP':
                pivots.append(CamarillaPivot(
                    level_name=name,
                    price=price,
                    strength=strength_map.get(name, 3),
                    timeframe='daily'
                ))
        
        # Create debug info
        debug_info = {
            'pivots_for_date': data['pivots_for_date'].strftime('%Y-%m-%d'),
            'data_from_date': data['data_from_date'].strftime('%Y-%m-%d'),
            'data_type': data['data_type'],
            'use_eth': data['use_eth'],
            'rth_data': data['rth'],
            'pre_data': data['pre'],
            'post_data': data['post'],
            'eth_data': data['eth'],
            'selected_data': selected,
            'all_pivots': data['pivots']
        }
        
        return CamarillaResult(
            timeframe='daily',
            close=selected['close'],
            high=selected['high'],
            low=selected['low'],
            pivots=sorted(pivots, key=lambda x: x.price, reverse=True),
            range_type='neutral',
            central_pivot=selected['close'],
            data_type=data['data_type'],
            debug_info=debug_info
        )

    def analyze_timeframe(self, data: pd.DataFrame, timeframe: str = 'daily') -> CamarillaResult:
        """Simple implementation for weekly/monthly using pre-fetched daily data"""
        if data is None or data.empty:
            raise ValueError("No data provided for analysis")
        
        # Ensure columns are lowercase
        data.columns = data.columns.str.lower()
        
        if timeframe == 'daily':
            # Use yesterday's data
            if len(data) >= 2:
                yesterday = data.iloc[-2]
            else:
                yesterday = data.iloc[-1]
                
            high = float(yesterday['high'])
            low = float(yesterday['low'])
            close = float(yesterday['close'])
            
        elif timeframe == 'weekly':
            # Last 5 trading days
            week_data = data.iloc[-5:] if len(data) >= 5 else data
            high = float(week_data['high'].max())
            low = float(week_data['low'].min())
            close = float(week_data['close'].iloc[-1])
            
        elif timeframe == 'monthly':
            # Last 20 trading days
            month_data = data.iloc[-20:] if len(data) >= 20 else data
            high = float(month_data['high'].max())
            low = float(month_data['low'].min())
            close = float(month_data['close'].iloc[-1])
        else:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        
        # Calculate pivots
        pivot_prices = self.calculate_pivots(high, low, close)
        
        # Create pivot objects
        pivots = []
        strength_map = {'R6': 6, 'R4': 4, 'R3': 3, 'S3': 3, 'S4': 4, 'S6': 6}
        
        for name, price in pivot_prices.items():
            if name != 'CP':
                pivots.append(CamarillaPivot(
                    level_name=name,
                    price=price,
                    strength=strength_map.get(name, 3),
                    timeframe=timeframe
                ))
        
        return CamarillaResult(
            timeframe=timeframe,
            close=close,
            high=high,
            low=low,
            pivots=sorted(pivots, key=lambda x: x.price, reverse=True),
            range_type='neutral',
            central_pivot=close,
            data_type='Daily'
        )