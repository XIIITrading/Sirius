# backtest/plugins/m1_bid_ask_analysis/test.py
"""
Test for M1 Bid/Ask Analysis using enhanced analyzer
"""

import asyncio
import argparse
from datetime import datetime, timedelta, timezone
import logging
import sys
from pathlib import Path

# Add parent paths
current_file = Path(__file__).resolve()
sirius_dir = current_file.parent.parent.parent.parent
sys.path.insert(0, str(sirius_dir))

from modules.calculations.volume.m1_bid_ask_analysis import M1VolumeAnalyzer
from backtest.data.polygon_data_manager import PolygonDataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class BidAskVolumeTest:
    def __init__(self):
        self.analyzer = M1VolumeAnalyzer(lookback_bars=14)
        self.data_manager = PolygonDataManager()
        # Set plugin name for tracking
        self.data_manager.set_current_plugin("M1_BidAsk_Test")
        
    async def run_test(self, symbol: str, test_time: datetime, direction: str):
        """Run bid/ask volume analysis test"""
        
        print(f"\n{'='*60}")
        print(f"M1 BID/ASK VOLUME ANALYSIS TEST")
        print(f"Symbol: {symbol}")
        print(f"Test Time: {test_time}")
        print(f"Direction: {direction}")
        print(f"{'='*60}\n")
        
        # Fetch data
        start_time = test_time - timedelta(minutes=20)
        
        print(f"Fetching trades from {start_time} to {test_time}...")
        # Remove use_cache parameter
        trades_df = await self.data_manager.load_trades(symbol, start_time, test_time)
        
        print(f"Fetching quotes from {start_time} to {test_time}...")
        # Remove use_cache parameter
        quotes_df = await self.data_manager.load_quotes(symbol, start_time, test_time)
        
        if trades_df.empty:
            print("ERROR: No trade data available")
            return
            
        print(f"Processing {len(trades_df)} trades with {len(quotes_df)} quotes...")
        
        # Process trades
        bar_count = 0
        
        for i, (timestamp, trade) in enumerate(trades_df.iterrows()):
            # Find closest quote
            bid, ask = None, None
            if not quotes_df.empty:
                quotes_before = quotes_df[quotes_df.index <= timestamp]
                if not quotes_before.empty:
                    latest_quote = quotes_before.iloc[-1]
                    bid = latest_quote['bid']
                    ask = latest_quote['ask']
            
            # Process trade with bid/ask context
            result = self.analyzer.process_trade_with_context(
                symbol=symbol,
                timestamp=timestamp,
                price=float(trade['price']),
                size=float(trade['size']),
                bid=bid,
                ask=ask
            )
            
            # Check if a bar was completed
            if result and result.get('bar_completed'):
                bar_count += 1
                print(f"Bar {bar_count} completed at {result['bar_timestamp']}")
        
        # Force complete last bar and get final analysis
        print(f"\nForcing completion of final bar...")
        final_analysis = self.analyzer.force_complete_bar(symbol)
        
        # Display results
        if final_analysis:
            self._display_results(final_analysis, direction)
        else:
            print("No analysis results generated")
    
    def _display_results(self, result: dict, direction: str):
        """Display the analysis results"""
        
        # Get all bars for detailed display
        bars = self.analyzer.get_current_bars(result['symbol'])
        
        print(f"\nTotal bars analyzed: {len(bars)}")
        print(f"Time range: {bars[0].timestamp.strftime('%H:%M:%S')} to {bars[-1].timestamp.strftime('%H:%M:%S')}")
        
        # Individual bar results
        print("\nINDIVIDUAL M1 CANDLE RESULTS:")
        print("-" * 100)
        print(f"{'Time':^12} | {'Total Vol':>10} | {'Above Ask':>10} | {'Below Bid':>10} | {'Winner':^10} | {'Strength':>8}")
        print("-" * 100)
        
        bull_wins = 0
        bear_wins = 0
        
        # Show all bars (should be 14)
        for bar in bars:
            winner = 'BULLISH' if bar.above_ask_volume > bar.below_bid_volume else 'BEARISH' if bar.below_bid_volume > bar.above_ask_volume else 'NEUTRAL'
            strength = max(bar.aggressive_buy_ratio, bar.aggressive_sell_ratio)
            
            if winner == 'BULLISH':
                bull_wins += 1
            elif winner == 'BEARISH':
                bear_wins += 1
                
            print(
                f"{bar.timestamp.strftime('%H:%M:%S'):^12} | "
                f"{bar.total_volume:>10.0f} | "
                f"{bar.above_ask_volume:>10.0f} | "
                f"{bar.below_bid_volume:>10.0f} | "
                f"{winner:^10} | "
                f"{strength:>7.1f}%"
            )
        
        print("-" * 100)
        
        # Summary statistics
        metrics = result['metrics']
        print(f"\nSUMMARY ANALYSIS:")
        print(f"Total Candles Analyzed: {result['bars_analyzed']}")
        print(f"Bull Wins: {bull_wins} ({bull_wins/len(bars)*100:.1f}%)")
        print(f"Bear Wins: {bear_wins} ({bear_wins/len(bars)*100:.1f}%)")
        
        print(f"\nVOLUME TOTALS:")
        print(f"Total Above Ask Volume: {metrics['total_above_ask_volume']:,.0f}")
        print(f"Total Below Bid Volume: {metrics['total_below_bid_volume']:,.0f}")
        print(f"Total At Ask Volume: {metrics.get('total_at_ask_volume', 0):,.0f}")
        print(f"Total At Bid Volume: {metrics.get('total_at_bid_volume', 0):,.0f}")
        
        # Overall winner
        print(f"\nOVERALL WINNER:")
        print(f"{result['signal']} - Strength {result['strength']:.1f}%")
        print(f"Reason: {result['reason']}")
        
        # Alignment check
        print(f"\nTRADE DIRECTION ALIGNMENT:")
        print(f"Intended Direction: {direction}")
        
        if direction == "LONG" and result['signal'] == "BULLISH":
            print("✅ ALIGNED - Aggressive buying supports LONG trade")
        elif direction == "SHORT" and result['signal'] == "BEARISH":
            print("✅ ALIGNED - Aggressive selling supports SHORT trade")
        else:
            print("⚠️  WARNING - Signal does not align with intended direction")
        
        # Additional insights
        print(f"\nADDITIONAL INSIGHTS:")
        print(f"Aggressive Buy Ratio: {metrics['aggressive_buy_ratio']:.1f}%")
        print(f"Recent vs Older Buy Ratio: {metrics['recent_buy_ratio']:.1f}% vs {metrics['older_buy_ratio']:.1f}%")
        print(f"Volume Acceleration: {metrics['volume_acceleration']:+.1f}%")
        print(f"Average Spread: {metrics['avg_spread_bps']:.1f} bps")
        print(f"Price Change: {metrics['price_change_pct']:+.2f}%")
        
        print(f"\n{'='*60}\n")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='M1 Bid/Ask Volume Analysis Test'
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
        help='Analysis time in format "YYYY-MM-DD HH:MM:SS"'
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
    """Run the test with CLI arguments"""
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
        test_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        print(f"No time specified, using: {test_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # Create tester and run
    tester = BidAskVolumeTest()
    
    try:
        await tester.run_test(
            symbol=args.symbol.upper(),
            test_time=test_time,
            direction=args.direction
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())