"""
Debug Test for Bid/Ask Ratio Plugin to identify data flow issues
"""

import asyncio
import argparse
from datetime import datetime, timedelta, timezone
import logging
import sys
from pathlib import Path
import pandas as pd

# Add parent paths
current_file = Path(__file__).resolve()
sirius_dir = current_file.parent.parent.parent.parent
sys.path.insert(0, str(sirius_dir))

# Import required components
from backtest.data.polygon_data_manager import PolygonDataManager
from backtest.data.trade_quote_aligner import TradeQuoteAligner
from backtest.calculations.order_flow.buy_sell_ratio import BuySellRatioCalculator

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class BidAskRatioDebugger:
    def __init__(self):
        self.data_manager = PolygonDataManager()
        
    async def debug_data_flow(self, symbol: str, test_time: datetime, direction: str):
        """Debug the entire data flow step by step"""
        
        print(f"\n{'='*80}")
        print(f"BID/ASK RATIO DEBUG ANALYSIS")
        print(f"Symbol: {symbol}")
        print(f"Entry Time: {test_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Direction: {direction}")
        print(f"{'='*80}\n")
        
        # Set plugin
        self.data_manager.set_current_plugin("DEBUG_BidAskRatio")
        
        # Time window
        start_time = test_time - timedelta(minutes=35)  # Extra 5 minutes
        end_time = test_time
        
        print(f"Time Window: {start_time.strftime('%H:%M:%S')} to {end_time.strftime('%H:%M:%S')} UTC")
        
        try:
            # Step 1: Fetch raw trades
            print(f"\n{'='*60}")
            print("STEP 1: Fetching Trades")
            print(f"{'='*60}")
            
            trades_df = await self.data_manager.load_trades(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time
            )
            
            print(f"Trades fetched: {len(trades_df)}")
            if not trades_df.empty:
                print(f"Trade columns: {list(trades_df.columns)}")
                print(f"Trade index type: {type(trades_df.index)}")
                print(f"First trade: {trades_df.index[0]}")
                print(f"Last trade: {trades_df.index[-1]}")
                print(f"\nFirst 5 trades:")
                print(trades_df.head())
            
            # Step 2: Fetch raw quotes
            print(f"\n{'='*60}")
            print("STEP 2: Fetching Quotes")
            print(f"{'='*60}")
            
            quotes_df = await self.data_manager.load_quotes(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time
            )
            
            print(f"Quotes fetched: {len(quotes_df)}")
            if not quotes_df.empty:
                print(f"Quote columns: {list(quotes_df.columns)}")
                print(f"Quote index type: {type(quotes_df.index)}")
                print(f"First quote: {quotes_df.index[0]}")
                print(f"Last quote: {quotes_df.index[-1]}")
                print(f"\nFirst 5 quotes:")
                print(quotes_df.head())
            
            if trades_df.empty or quotes_df.empty:
                print("\nERROR: No trade or quote data available")
                return
            
            # Step 3: Align trades with quotes
            print(f"\n{'='*60}")
            print("STEP 3: Aligning Trades with Quotes")
            print(f"{'='*60}")
            
            aligner = TradeQuoteAligner(
                max_quote_age_ms=500,
                min_confidence_threshold=0.5
            )
            
            aligned_df, alignment_report = aligner.align_trades_quotes(trades_df, quotes_df)
            
            print(f"Aligned trades: {len(aligned_df)}")
            print(f"Alignment success rate: {alignment_report.aligned_trades}/{alignment_report.total_trades}")
            
            if not aligned_df.empty:
                print(f"Aligned columns: {list(aligned_df.columns)}")
                print(f"Aligned index type: {type(aligned_df.index)}")
                
                # Check for required columns
                required_cols = ['trade_price', 'trade_size', 'trade_side', 'confidence']
                missing_cols = [col for col in required_cols if col not in aligned_df.columns]
                if missing_cols:
                    print(f"WARNING: Missing columns: {missing_cols}")
                
                # Check trade side distribution
                if 'trade_side' in aligned_df.columns:
                    print(f"\nTrade side distribution:")
                    print(aligned_df['trade_side'].value_counts())
                
                print(f"\nFirst 5 aligned trades:")
                print(aligned_df.head())
            
            # Step 4: Process into minute bars
            print(f"\n{'='*60}")
            print("STEP 4: Processing into Minute Bars")
            print(f"{'='*60}")
            
            calculator = BuySellRatioCalculator(window_minutes=30, min_confidence=0.5)
            
            print(f"Processing {len(aligned_df)} aligned trades...")
            minute_bars = calculator.process_aligned_trades(aligned_df, symbol)
            
            print(f"Minute bars created: {len(minute_bars)}")
            
            if minute_bars:
                print(f"\nAll minute bars:")
                for i, bar in enumerate(minute_bars):
                    print(f"{i+1}. {bar.timestamp.strftime('%H:%M:%S')} - "
                          f"Ratio: {bar.weighted_pressure:+.3f}, "
                          f"Volume: {bar.total_volume:,.0f}, "
                          f"Trades: {bar.trade_count}")
                
                # Step 5: Filter to 30-minute window
                print(f"\n{'='*60}")
                print("STEP 5: Filtering to 30-minute Window")
                print(f"{'='*60}")
                
                filtered_bars = [
                    bar for bar in minute_bars 
                    if bar.timestamp <= test_time and bar.timestamp >= (test_time - timedelta(minutes=30))
                ]
                
                print(f"Filtered bars: {len(filtered_bars)}")
                
                if filtered_bars:
                    print(f"\nFiltered minute bars:")
                    for i, bar in enumerate(filtered_bars):
                        mins_from_entry = int((bar.timestamp - test_time).total_seconds() / 60)
                        print(f"{i+1}. {bar.timestamp.strftime('%H:%M:%S')} ({mins_from_entry}m) - "
                              f"Ratio: {bar.weighted_pressure:+.3f}, "
                              f"Volume: {bar.total_volume:,.0f}")
                    
                    latest_ratio = filtered_bars[-1].weighted_pressure
                    print(f"\nLatest ratio: {latest_ratio:+.3f}")
                else:
                    print("\nERROR: No bars in the 30-minute window!")
                    print(f"Entry time: {test_time}")
                    print(f"Expected window: {test_time - timedelta(minutes=30)} to {test_time}")
                    if minute_bars:
                        print(f"Available bar times: {minute_bars[0].timestamp} to {minute_bars[-1].timestamp}")
            else:
                print("\nERROR: No minute bars created!")
                
                # Debug why no minute bars
                if aligned_df.empty:
                    print("Reason: No aligned trades")
                else:
                    print("Checking confidence filtering...")
                    high_conf = aligned_df[aligned_df['confidence'] >= 0.5]
                    print(f"Trades with confidence >= 0.5: {len(high_conf)}")
                    
                    if not high_conf.empty:
                        print("Checking minute grouping...")
                        print(f"Index type: {type(high_conf.index)}")
                        print(f"First index: {high_conf.index[0]}")
                        print(f"Last index: {high_conf.index[-1]}")
            
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Debug Bid/Ask Ratio data flow'
    )
    
    parser.add_argument(
        '-s', '--symbol',
        type=str,
        required=True,
        help='Stock symbol (e.g., AAPL, TSLA, SPY)'
    )
    
    parser.add_argument(
        '-t', '--time',
        type=str,
        default=None,
        help='Entry time in format "YYYY-MM-DD HH:MM:SS"'
    )
    
    parser.add_argument(
        '-d', '--direction',
        type=str,
        choices=['LONG', 'SHORT'],
        default='LONG',
        help='Trade direction (default: LONG)'
    )
    
    return parser.parse_args()


async def main():
    """Run the debug test"""
    args = parse_arguments()
    
    # Parse datetime
    if args.time:
        try:
            test_time = datetime.strptime(args.time, "%Y-%m-%d %H:%M:%S")
            test_time = test_time.replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"ERROR: Invalid datetime format: {args.time}")
            print("Please use format: YYYY-MM-DD HH:MM:SS")
            return
    else:
        test_time = datetime.now(timezone.utc)
        print(f"Using current time: {test_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # Create debugger and run
    debugger = BidAskRatioDebugger()
    await debugger.debug_data_flow(
        symbol=args.symbol.upper(),
        test_time=test_time,
        direction=args.direction
    )


if __name__ == "__main__":
    asyncio.run(main())