# backtest/plugins/tick_flow_analysis/test.py
"""
Test for Tick Flow Analysis
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

from modules.calculations.volume.tick_flow import TickFlowAnalyzer
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


class TickFlowTest:
    def __init__(self):
        self.analyzer = TickFlowAnalyzer()
        self.data_manager = PolygonDataManager()
        
    async def run_test(self, symbol: str, test_time: datetime, direction: str):
        """Run tick flow analysis test"""
        
        print(f"\n{'='*60}")
        print(f"TICK FLOW ANALYSIS TEST")
        print(f"Symbol: {symbol}")
        print(f"Test Time: {test_time}")
        print(f"Direction: {direction}")
        print(f"{'='*60}\n")
        
        # Fetch data
        start_time = test_time - timedelta(minutes=5)
        
        print(f"Fetching trades from {start_time} to {test_time}...")
        trades_df = await self.data_manager.load_trades(symbol, start_time, test_time)
        
        if trades_df.empty:
            print("ERROR: No trade data available")
            return
            
        print(f"Processing {len(trades_df)} trades...")
        
        # Process trades
        signals_generated = []
        trade_count = 0
        
        for timestamp, trade in trades_df.iterrows():
            if timestamp > test_time:
                break
                
            trade_count += 1
            
            # Convert to expected format
            trade_data = {
                'timestamp': int(timestamp.timestamp() * 1000),
                'price': float(trade['price']),
                'size': float(trade['size']),
                'conditions': trade.get('conditions', [])
            }
            
            # Debug 
            if trade_count <= 5:  # Just print first 5 to see format
                print(f"\nDEBUG Trade #{trade_count}:")
                print(f"  Raw conditions from DataFrame: {trade.get('conditions')} (type: {type(trade.get('conditions'))})")
                print(f"  trade_data conditions: {trade_data['conditions']} (type: {type(trade_data['conditions'])})")
            
            # Process trade
            signal = self.analyzer.process_trade(symbol, trade_data)
            
            if signal:
                signals_generated.append(signal)
                print(f"\nSignal #{len(signals_generated)} at {signal.timestamp.strftime('%H:%M:%S')}")
                print(f"  {signal.signal} - Strength: {signal.strength:.0f}%")
                print(f"  Reason: {signal.reason}")
        
        print(f"\nProcessed {trade_count} trades total")
        
        # Get final analysis
        final_signal = self.analyzer.get_current_analysis(symbol)
        
        # Display results
        if final_signal:
            self._display_results(final_signal, direction, signals_generated)
        else:
            print("Insufficient trades for analysis")
    
    def _display_results(self, signal: dict, direction: str, signals_generated: list):
        """Display the analysis results"""
        
        metrics = signal.metrics
        stats = self.analyzer.get_statistics()
        
        print(f"\n{'='*60}")
        print("FINAL ANALYSIS RESULTS")
        print(f"{'='*60}")
        
        # Trade statistics
        print(f"\nTRADE STATISTICS:")
        print(f"Total Trades Analyzed: {metrics['total_trades']}")
        print(f"Buy Trades: {metrics['buy_trades']} ({metrics['buy_trades']/metrics['total_trades']*100:.1f}%)")
        print(f"Sell Trades: {metrics['sell_trades']} ({metrics['sell_trades']/metrics['total_trades']*100:.1f}%)")
        print(f"Trade Rate: {metrics['trade_rate']:.1f} trades/second")
        
        # Volume analysis
        print(f"\nVOLUME ANALYSIS:")
        print(f"Buy Volume: {metrics['buy_volume']:,.0f} ({metrics['buy_volume_pct']:.1f}%)")
        print(f"Sell Volume: {metrics['sell_volume']:,.0f} ({100-metrics['buy_volume_pct']:.1f}%)")
        print(f"Average Trade Size: {metrics['avg_trade_size']:,.0f}")
        
        # Large trades
        print(f"\nLARGE TRADE ANALYSIS:")
        print(f"Large Buy Trades: {metrics['large_buy_trades']}")
        print(f"Large Sell Trades: {metrics['large_sell_trades']}")
        
        # Signal
        print(f"\nSIGNAL GENERATED:")
        print(f"{signal.signal} - Strength {signal.strength:.1f}%")
        print(f"Momentum Score: {metrics['momentum_score']:+.1f}")
        print(f"Price Trend: {metrics['price_trend']}")
        print(f"Reason: {signal.reason}")
        
        # Direction alignment
        print(f"\nTRADE DIRECTION ALIGNMENT:")
        print(f"Intended Direction: {direction}")
        
        if direction == "LONG" and signal.signal == "BULLISH":
            print("✅ ALIGNED - Buy pressure supports LONG trade")
        elif direction == "SHORT" and signal.signal == "BEARISH":
            print("✅ ALIGNED - Sell pressure supports SHORT trade")
        else:
            print("⚠️  WARNING - Signal does not align with intended direction")
        
        # Signal history
        if signals_generated:
            print(f"\nSIGNAL HISTORY (Last 10):")
            print("-" * 60)
            print(f"{'Time':^12} | {'Signal':^10} | {'Strength':>8} | {'Buy %':>6}")
            print("-" * 60)
            
            for sig in signals_generated[-10:]:
                print(
                    f"{sig.timestamp.strftime('%H:%M:%S'):^12} | "
                    f"{sig.signal:^10} | "
                    f"{sig.strength:>7.0f}% | "
                    f"{sig.metrics['buy_volume_pct']:>5.0f}%"
                )
        
        # Performance stats
        print(f"\nPERFORMANCE STATISTICS:")
        print(f"Trades Processed: {stats['trades_processed']}")
        print(f"Signals Generated: {stats['signals_generated']}")
        print(f"Active Symbols: {', '.join(stats['active_symbols'])}")
        
        print(f"\n{'='*60}\n")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Tick Flow Analysis Test'
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
    tester = TickFlowTest()
    
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