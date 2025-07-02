"""
Test for Bid/Ask Imbalance Analysis Plugin with Bar Index Display
FIXED: Proper quote filtering to prevent cleanup issues
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

from modules.calculations.order_flow.bid_ask_imbal import BidAskImbalance, Trade, Quote
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


class DebugFileWriter:
    """Context manager to write all output to both console and debug file"""
    def __init__(self, filename="debug.txt"):
        self.filename = filename
        self.file = None
        self.terminal = sys.stdout
        
    def write(self, message):
        """Write to both terminal and file"""
        self.terminal.write(message)
        if self.file:
            self.file.write(message)
            self.file.flush()  # Ensure immediate write
    
    def flush(self):
        """Flush both outputs"""
        self.terminal.flush()
        if self.file:
            self.file.flush()
    
    def __enter__(self):
        self.file = open(self.filename, 'w', encoding='utf-8')
        sys.stdout = self
        return self
    
    def __exit__(self, *args):
        sys.stdout = self.terminal
        if self.file:
            self.file.close()


class SimpleTable:
    """Simple table formatter without external dependencies"""
    
    @staticmethod
    def format_table(data, headers=None, col_widths=None):
        """Format data as a simple ASCII table"""
        if not data:
            return ""
        
        # Calculate column widths
        if col_widths is None:
            col_widths = []
            if headers:
                for i, header in enumerate(headers):
                    max_width = len(str(header))
                    for row in data:
                        if i < len(row):
                            max_width = max(max_width, len(str(row[i])))
                    col_widths.append(max_width + 2)
            else:
                # No headers, calculate from data
                num_cols = len(data[0]) if data else 0
                for i in range(num_cols):
                    max_width = 0
                    for row in data:
                        if i < len(row):
                            max_width = max(max_width, len(str(row[i])))
                    col_widths.append(max_width + 2)
        
        # Build separator
        separator = "+" + "+".join("-" * width for width in col_widths) + "+"
        
        lines = [separator]
        
        # Add headers if provided
        if headers:
            header_line = "|"
            for i, header in enumerate(headers):
                width = col_widths[i] if i < len(col_widths) else 10
                header_line += f" {str(header):<{width-2}} |"
            lines.append(header_line)
            lines.append(separator)
        
        # Add data rows
        for row in data:
            row_line = "|"
            for i, cell in enumerate(row):
                width = col_widths[i] if i < len(col_widths) else 10
                row_line += f" {str(cell):<{width-2}} |"
            lines.append(row_line)
        
        lines.append(separator)
        return "\n".join(lines)


class BidAskImbalanceTest:
    def __init__(self):
        self.data_manager = PolygonDataManager()
        
    async def run_test(self, symbol: str, test_time: datetime, direction: str):
        """Run bid/ask imbalance analysis test with proper quote filtering"""
        
        print(f"\n{'='*80}")
        print(f"BID/ASK IMBALANCE ANALYSIS TEST - FIXED VERSION")
        print(f"Symbol: {symbol}")
        print(f"Test Time: {test_time}")
        print(f"Direction: {direction}")
        print(f"{'='*80}\n")
        
        # Initialize analyzer with bar index parameters
        analyzer = BidAskImbalance(
            imbalance_lookback=1000,  # Total trades to track
            trades_per_bar=100,       # Trades per bar index
            spread_history_seconds=600,  # 10 minute quote history
            quote_sync_tolerance_ms=100,
            aggression_threshold=0.7,
            smoothing_alpha=0.1
        )
        
        print(f"Configuration:")
        print(f"- Total trades tracked: {analyzer.imbalance_lookback}")
        print(f"- Trades per bar: {analyzer.trades_per_bar}")
        print(f"- Number of bars: {analyzer.num_bars}")
        print(f"- Quote sync tolerance: {analyzer.quote_sync_tolerance_ms}ms")
        print(f"- Quote history window: {analyzer.spread_history_seconds}s")
        print()
        
        # Fetch data - still get full historical range for trades
        data_start_time = test_time - timedelta(minutes=20)
        
        print(f"Fetching data from {data_start_time} to {test_time}...")
        trades_df = await self.data_manager.load_trades(symbol, data_start_time, test_time)
        quotes_df = await self.data_manager.load_quotes(symbol, data_start_time, test_time)
        
        if trades_df.empty:
            print("ERROR: No trade data available")
            return
            
        print(f"\nDATA SUMMARY:")
        print(f"- Quotes loaded: {len(quotes_df)}")
        print(f"- Trades loaded: {len(trades_df)}")
        print(f"- Quote time range: {quotes_df.index[0]} to {quotes_df.index[-1]}")
        print(f"- Trade time range: {trades_df.index[0]} to {trades_df.index[-1]}")
        
        # CRITICAL FIX: Filter quotes to a reasonable window around test time
        # This prevents the cleanup function from removing quotes we need
        quote_window_minutes = 8  # Process 8 minutes of quotes around test time
        quote_window_start = test_time - timedelta(minutes=quote_window_minutes)
        quote_window_end = test_time + timedelta(minutes=2)  # Little bit forward
        
        quotes_to_process = quotes_df[
            (quotes_df.index >= quote_window_start) & 
            (quotes_df.index <= quote_window_end)
        ]
        
        print(f"\nQUOTE FILTERING:")
        print(f"- Original quotes: {len(quotes_df)}")
        print(f"- Quote window: {quote_window_start} to {quote_window_end}")
        print(f"- Quotes to process: {len(quotes_to_process)}")
        print(f"- This prevents quote cleanup from removing quotes we need for sync")
        
        # Show sample of quotes to process
        print(f"\nFIRST 5 QUOTES TO PROCESS:")
        for i, (timestamp, quote_data) in enumerate(quotes_to_process.head().iterrows()):
            print(f"  {timestamp}: bid={quote_data['bid']:.2f}, ask={quote_data['ask']:.2f}, "
                  f"spread={quote_data['ask']-quote_data['bid']:.4f}")
        
        # Process filtered quotes
        print(f"\n{'='*60}")
        print("PROCESSING QUOTES...")
        print(f"{'='*60}")
        
        quote_count = 0
        for timestamp, quote_data in quotes_to_process.iterrows():
            # Convert timestamp if needed
            if hasattr(timestamp, 'to_pydatetime'):
                py_timestamp = timestamp.to_pydatetime()
            else:
                py_timestamp = timestamp
                
            quote = Quote(
                symbol=symbol,
                bid=float(quote_data['bid']),
                ask=float(quote_data['ask']),
                bid_size=int(quote_data.get('bid_size', 100)),
                ask_size=int(quote_data.get('ask_size', 100)),
                timestamp=py_timestamp
            )
            analyzer.process_quote(quote)
            quote_count += 1
            
            # Show progress every 1000 quotes
            if quote_count % 1000 == 0:
                print(f"  Processed {quote_count} quotes... Latest: {py_timestamp}")
        
        print(f"\nTotal quotes processed: {quote_count}")
        print(f"Quotes in analyzer history: {len(analyzer.quote_history.get(symbol, []))}")
        
        # Check quote history health
        if symbol in analyzer.quote_history and analyzer.quote_history[symbol]:
            history = list(analyzer.quote_history[symbol])
            first_quote_time = history[0].timestamp
            last_quote_time = history[-1].timestamp
            time_span = (last_quote_time - first_quote_time).total_seconds()
            
            print(f"\nQUOTE HISTORY HEALTH CHECK:")
            print(f"- First quote in history: {first_quote_time}")
            print(f"- Last quote in history: {last_quote_time}")
            print(f"- Time span: {time_span:.1f} seconds")
            print(f"- Average quotes per second: {len(history) / max(1, time_span):.1f}")
        
        # Process trades - filter to relevant window
        trades_to_process = trades_df[trades_df.index < test_time]
        print(f"\n{'='*60}")
        print(f"PROCESSING TRADES (up to test time)...")
        print(f"{'='*60}")
        print(f"- Total trades to process: {len(trades_to_process)}")
        
        signals = []
        exceptional_signals = []
        sync_stats = {
            'total': 0,
            'synced': 0,
            'no_sync': 0,
            'first_10_synced': 0
        }
        
        # Process trades
        trades_processed = 0
        first_signal_at = None
        
        for i, (timestamp, trade_data) in enumerate(trades_to_process.iterrows()):
            # Convert timestamp if needed
            if hasattr(timestamp, 'to_pydatetime'):
                trade_timestamp = timestamp.to_pydatetime()
            else:
                trade_timestamp = timestamp
                
            trade = Trade(
                symbol=symbol,
                price=float(trade_data['price']),
                size=int(trade_data['size']),
                timestamp=trade_timestamp
            )
            
            # Check sync before processing (for stats)
            synced_quote = analyzer._get_synchronized_quote(trade)
            sync_stats['total'] += 1
            if synced_quote:
                sync_stats['synced'] += 1
                if i < 10:
                    sync_stats['first_10_synced'] += 1
            else:
                sync_stats['no_sync'] += 1
                
            # Process trade
            signal = analyzer.process_trade(trade)
            trades_processed += 1
            
            # Debug info for first few trades
            if i < 10:
                print(f"\nTrade {i+1} at {trade_timestamp}:")
                print(f"  Price: {trade.price}, Size: {trade.size}")
                if synced_quote:
                    time_diff = (trade.timestamp - synced_quote.timestamp).total_seconds() * 1000
                    print(f"  SYNCED with quote at {synced_quote.timestamp} ({time_diff:+.1f}ms)")
                else:
                    print(f"  NO SYNC - Check quote history coverage")
                    if symbol in analyzer.latest_quotes:
                        latest = analyzer.latest_quotes[symbol]
                        diff = (trade.timestamp - latest.timestamp).total_seconds()
                        print(f"  Latest quote: {latest.timestamp} ({diff:.1f}s away)")
            
            if signal:
                signals.append(signal)
                if first_signal_at is None:
                    first_signal_at = i + 1
                    print(f"\n*** FIRST SIGNAL GENERATED at trade {first_signal_at} ***")
                    print(f"  Signal: {signal.signal_type} - {signal.signal_strength}")
                    print(f"  Reason: {signal.reason}")
                
                if signal.signal_strength == 'EXCEPTIONAL':
                    exceptional_signals.append(signal)
            
            # Show progress
            if trades_processed % 5000 == 0:
                sync_rate = (sync_stats['synced'] / sync_stats['total']) * 100
                print(f"\n--- Progress: {trades_processed} trades, {len(signals)} signals, "
                      f"sync rate: {sync_rate:.1f}% ---")
        
        # Final summary
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE - SUMMARY")
        print(f"{'='*60}")
        print(f"Trades processed: {trades_processed}")
        print(f"Sync rate: {sync_stats['synced']}/{sync_stats['total']} "
              f"({sync_stats['synced']/max(1, sync_stats['total'])*100:.1f}%)")
        print(f"First 10 trades synced: {sync_stats['first_10_synced']}/10")
        print(f"Signals generated: {len(signals)}")
        print(f"First signal at trade: {first_signal_at if first_signal_at else 'NONE'}")
        
        # Display results if we have signals
        if signals:
            self._display_results(signals, exceptional_signals, analyzer, symbol, direction)
        else:
            print("\n*** NO SIGNALS GENERATED ***")
            print("\nPossible reasons:")
            print("1. No synchronized quotes for trades")
            print("2. Not enough trades to meet minimum threshold")
            print("3. Market conditions don't trigger signals")
            
            # Additional debugging
            if symbol in analyzer.classified_trades:
                print(f"\nClassified trades buffer size: {len(analyzer.classified_trades[symbol])}")
            if sync_stats['synced'] == 0:
                print("\nCRITICAL: No trades synchronized with quotes!")
                print("This suggests a data timing issue.")
    
    def _display_results(self, signals: list, exceptional_signals: list, 
                        analyzer: BidAskImbalance, symbol: str, direction: str):
        """Display the analysis results with bar indices"""
        
        final_signal = signals[-1]
        
        # Signal Summary
        print(f"\nSIGNAL SUMMARY:")
        print(f"Total Signals Generated: {len(signals)}")
        print(f"Exceptional Signals: {len(exceptional_signals)}")
        
        # Count signal types
        bull_signals = sum(1 for s in signals if s.bull_score > s.bear_score)
        bear_signals = sum(1 for s in signals if s.bear_score > s.bull_score)
        
        print(f"Bullish Signals: {bull_signals} ({bull_signals/len(signals)*100:.1f}%)")
        print(f"Bearish Signals: {bear_signals} ({bear_signals/len(signals)*100:.1f}%)")
        
        # Final Signal Details
        print(f"\nFINAL SIGNAL AT {final_signal.timestamp.strftime('%H:%M:%S')}:")
        print(f"Type: {final_signal.signal_type}")
        print(f"Strength: {final_signal.signal_strength}")
        print(f"Bull/Bear Score: {final_signal.bull_score}/{final_signal.bear_score}")
        print(f"Confidence: {final_signal.confidence:.1%}")
        print(f"Reason: {final_signal.reason}")
        
        if final_signal.warnings:
            print(f"Warnings: {', '.join(final_signal.warnings)}")
        
        # Bar Index Analysis
        print(f"\nBAR INDEX BREAKDOWN:")
        print("Shows imbalance evolution from oldest to newest (100 trades per bar)")
        
        if final_signal.components.bar_indices:
            bar_data = []
            for bar in reversed(final_signal.components.bar_indices):  # Show oldest first
                bar_data.append([
                    f"Bar {bar.bar_index}",
                    f"{bar.weighted_imbalance:+.1%}",
                    f"{bar.aggression_ratio:.0%}",
                    f"{bar.buy_volume:,.0f}",
                    f"{bar.sell_volume:,.0f}",
                    f"{bar.total_volume:,.0f}",
                    f"{bar.avg_spread:.4f}",
                    bar.time_range[0].strftime('%H:%M:%S')
                ])
            
            table = SimpleTable.format_table(
                bar_data,
                headers=["Bar", "Imbalance", "Aggression", "Buy Vol", "Sell Vol", "Total Vol", "Avg Spread", "Start Time"]
            )
            print(table)
            
            # Show trend analysis
            if len(final_signal.components.bar_indices) >= 3:
                recent_bars = final_signal.components.bar_indices[:3]
                older_bars = final_signal.components.bar_indices[3:] if len(final_signal.components.bar_indices) > 3 else []
                
                if recent_bars and older_bars:
                    recent_avg = sum(b.weighted_imbalance for b in recent_bars) / len(recent_bars)
                    older_avg = sum(b.weighted_imbalance for b in older_bars) / len(older_bars)
                    
                    print(f"\nTREND ANALYSIS:")
                    print(f"Recent bars (1-3) average: {recent_avg:+.1%}")
                    print(f"Older bars (4+) average: {older_avg:+.1%}")
                    
                    if recent_avg > older_avg + 0.1:
                        print("📈 BULLISH MOMENTUM - Buying pressure accelerating")
                    elif recent_avg < older_avg - 0.1:
                        print("📉 BEARISH MOMENTUM - Selling pressure accelerating")
                    else:
                        print("↔️  STEADY FLOW - No significant momentum change")
        
        # Components breakdown
        components = final_signal.components
        print(f"\nIMBALANCE COMPONENTS:")
        components_data = [
            ["Smoothed Imbalance", f"{components.smoothed_imbalance:+.2%}"],
            ["Raw Imbalance", f"{components.raw_imbalance:+.2%}"],
            ["Weighted Imbalance", f"{components.weighted_imbalance:+.2%}"],
            ["Aggression Ratio", f"{components.aggression_ratio:.1%}"],
            ["Buy Volume", f"{components.buy_volume:,.0f}"],
            ["Sell Volume", f"{components.sell_volume:,.0f}"],
            ["Total Volume", f"{components.total_volume:,.0f}"]
        ]
        print(SimpleTable.format_table(components_data, headers=["Metric", "Value"]))
        
        # Spread Analysis
        print(f"\nSPREAD ANALYSIS:")
        spread_data = [
            ["Current Spread", f"{components.current_spread:.4f}"],
            ["1-min Spread Ratio", f"{components.spread_ratio_1min:.2f}x"],
            ["5-min Spread Ratio", f"{components.spread_ratio_5min:.2f}x"],
            ["Spread Volatility", f"{components.spread_volatility:.2%}"],
            ["Spread Trend", f"{components.spread_trend:+.2%}"],
            ["Quote Stability", f"{components.quote_stability:.1%}"],
            ["Liquidity State", components.liquidity_state.upper()]
        ]
        print(SimpleTable.format_table(spread_data, headers=["Metric", "Value"]))
        
        # Recent Signal History
        print(f"\nRECENT SIGNAL HISTORY (Last 10):")
        history_data = []
        for sig in signals[-10:]:
            history_data.append([
                sig.timestamp.strftime('%H:%M:%S'),
                f"{sig.components.smoothed_imbalance:+.1%}",
                f"{sig.bull_score}/{sig.bear_score}",
                sig.signal_strength,
                sig.signal_type
            ])
        print(SimpleTable.format_table(history_data, 
                      headers=["Time", "Imbalance", "Bull/Bear", "Strength", "Type"]))
        
        # Session Summary
        session_summary = analyzer.get_session_summary(symbol)
        if session_summary:
            print(f"\nSESSION SUMMARY:")
            session_data = [
                ["Session Imbalance", f"{session_summary['session_imbalance']:+.2%}"],
                ["Tightest Spread", f"{session_summary['spread_range'][0]:.4f}"],
                ["Widest Spread", f"{session_summary['spread_range'][1]:.4f}"],
                ["Extreme Periods", f"{session_summary['extreme_periods']}"],
                ["Total Volume", f"{session_summary['total_volume']:,.0f}"],
                ["Buy Volume", f"{session_summary['buy_volume']:,.0f}"],
                ["Sell Volume", f"{session_summary['sell_volume']:,.0f}"]
            ]
            print(SimpleTable.format_table(session_data, headers=["Metric", "Value"]))
        
        # Performance Stats
        perf_stats = analyzer.get_performance_stats()
        print(f"\nPERFORMANCE STATISTICS:")
        print(f"Total Calculations: {perf_stats['total_calculations']}")
        print(f"Average Time: {perf_stats['average_time_ms']:.3f}ms")
        
        # Trade Direction Alignment
        print(f"\nTRADE DIRECTION ALIGNMENT:")
        print(f"Intended Direction: {direction}")
        
        # Determine overall signal direction
        if final_signal.bull_score > final_signal.bear_score:
            signal_direction = "BULLISH"
        elif final_signal.bear_score > final_signal.bull_score:
            signal_direction = "BEARISH"
        else:
            signal_direction = "NEUTRAL"
        
        print(f"Signal Direction: {signal_direction}")
        
        if direction == "LONG" and signal_direction == "BULLISH":
            print("✅ ALIGNED - Order flow supports LONG trade")
        elif direction == "SHORT" and signal_direction == "BEARISH":
            print("✅ ALIGNED - Order flow supports SHORT trade")
        else:
            print("⚠️  WARNING - Order flow does not align with intended direction")
        
        print(f"\n{'='*80}\n")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Bid/Ask Imbalance Analysis Test - Fixed Version'
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
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='debug_fixed.txt',
        help='Output debug file name (default: debug_fixed.txt)'
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
    
    print(f"\nWriting debug output to: {args.output}")
    print("Check debug file for full details.\n")
    
    # Create tester and run with debug file output
    tester = BidAskImbalanceTest()
    
    try:
        with DebugFileWriter(args.output):
            await tester.run_test(
                symbol=args.symbol.upper(),
                test_time=test_time,
                direction=args.direction
            )
        
        print(f"\n✅ Debug output written to: {args.output}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())