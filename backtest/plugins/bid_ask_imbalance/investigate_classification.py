"""
Investigate Trade Classification Issues
"""

import asyncio
import argparse
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent paths
current_file = Path(__file__).resolve()
sirius_dir = current_file.parent.parent.parent.parent
sys.path.insert(0, str(sirius_dir))

from backtest.plugins.bid_ask_imbalance import _fetch_data
from modules.calculations.order_flow.bid_ask_imbal import BidAskImbalance, Trade, Quote


async def investigate_classification_issues(symbol: str, entry_time: datetime):
    """Investigate why trades aren't being classified"""
    
    print(f"\n{'='*60}")
    print("TRADE CLASSIFICATION INVESTIGATION")
    print(f"Symbol: {symbol}")
    print(f"Entry Time: {entry_time}")
    print(f"{'='*60}\n")
    
    # Fetch data
    print("Fetching data...")
    trades_df, quotes_df = await _fetch_data(symbol, entry_time)
    
    # Focus on the problematic period (around 13:33:23)
    problem_time = datetime(2025, 6, 27, 13, 33, 23, tzinfo=timezone.utc)
    window_start = problem_time - pd.Timedelta(seconds=10)
    window_end = problem_time + pd.Timedelta(seconds=10)
    
    # Filter data
    problem_trades = trades_df[(trades_df.index >= window_start) & (trades_df.index <= window_end)]
    problem_quotes = quotes_df[(quotes_df.index >= window_start) & (quotes_df.index <= window_end)]
    
    print(f"\nProblem Period Analysis ({problem_time} ± 10s):")
    print(f"- Trades in window: {len(problem_trades)}")
    print(f"- Quotes in window: {len(problem_quotes)}")
    
    # Show quote coverage
    if len(problem_quotes) > 0:
        print(f"\nQuote Coverage:")
        print(f"- First quote: {problem_quotes.index.min()}")
        print(f"- Last quote: {problem_quotes.index.max()}")
        print(f"- Average spread: ${problem_quotes['ask'].mean() - problem_quotes['bid'].mean():.2f}")
    else:
        print("\n❌ CRITICAL: NO QUOTES in problem window!")
        
        # Find nearest quotes
        print("\nSearching for nearest quotes...")
        before_quotes = quotes_df[quotes_df.index < window_start].tail(5)
        after_quotes = quotes_df[quotes_df.index > window_end].head(5)
        
        if len(before_quotes) > 0:
            last_before = before_quotes.index[-1]
            gap_before = (window_start - last_before).total_seconds()
            print(f"- Last quote before window: {last_before} ({gap_before:.1f}s gap)")
        
        if len(after_quotes) > 0:
            first_after = after_quotes.index[0]
            gap_after = (first_after - window_end).total_seconds()
            print(f"- First quote after window: {first_after} ({gap_after:.1f}s gap)")
    
    # Analyze trade prices vs quotes
    if len(problem_trades) > 0 and len(problem_quotes) > 0:
        print(f"\nPrice Analysis:")
        print(f"- Trade price range: ${problem_trades['price'].min():.2f} - ${problem_trades['price'].max():.2f}")
        print(f"- Bid range: ${problem_quotes['bid'].min():.2f} - ${problem_quotes['bid'].max():.2f}")
        print(f"- Ask range: ${problem_quotes['ask'].min():.2f} - ${problem_quotes['ask'].max():.2f}")
        
        # Check if trades are outside bid/ask
        avg_bid = problem_quotes['bid'].mean()
        avg_ask = problem_quotes['ask'].mean()
        trades_below_bid = (problem_trades['price'] < avg_bid).sum()
        trades_above_ask = (problem_trades['price'] > avg_ask).sum()
        trades_at_mid = ((problem_trades['price'] > avg_bid) & (problem_trades['price'] < avg_ask)).sum()
        
        print(f"\nTrade Classification Potential:")
        print(f"- Below bid (sells): {trades_below_bid} ({trades_below_bid/len(problem_trades)*100:.1f}%)")
        print(f"- Above ask (buys): {trades_above_ask} ({trades_above_ask/len(problem_trades)*100:.1f}%)")
        print(f"- At midpoint: {trades_at_mid} ({trades_at_mid/len(problem_trades)*100:.1f}%)")
    
    # Test classification with actual analyzer
    print(f"\n{'='*40}")
    print("TESTING CLASSIFICATION LOGIC")
    print(f"{'='*40}")
    
    # Test with different sync tolerances
    for tolerance in [100, 500, 1000, 5000]:
        analyzer = BidAskImbalance(quote_sync_tolerance_ms=tolerance)
        
        # Process ALL quotes (not just problem window)
        quote_count = 0
        for _, quote_row in quotes_df.iterrows():
            quote = Quote(
                symbol=symbol,
                bid=quote_row['bid'],
                ask=quote_row['ask'],
                bid_size=quote_row.get('bid_size', 100),
                ask_size=quote_row.get('ask_size', 100),
                timestamp=quote_row.name
            )
            analyzer.process_quote(quote)
            quote_count += 1
        
        # Test classification on sample trades
        sample_trades = problem_trades.head(10) if len(problem_trades) > 0 else trades_df.sample(10)
        classified_count = 0
        
        for idx, trade_row in sample_trades.iterrows():
            trade = Trade(
                symbol=symbol,
                price=trade_row['price'],
                size=trade_row['size'],
                timestamp=idx,
                conditions=None
            )
            
            synced_quote = analyzer._get_synchronized_quote(trade)
            if synced_quote:
                classified_count += 1
        
        print(f"\nSync tolerance {tolerance}ms: {classified_count}/{len(sample_trades)} trades classified")
    
    # Analyze entire dataset for patterns
    print(f"\n{'='*40}")
    print("FULL DATASET ANALYSIS")
    print(f"{'='*40}")
    
    # Find quote gaps
    quote_gaps = quotes_df.index.to_series().diff()
    large_gaps = quote_gaps[quote_gaps > pd.Timedelta(seconds=5)]
    
    print(f"\nQuote Gap Analysis:")
    print(f"- Total quotes: {len(quotes_df)}")
    print(f"- Gaps > 5 seconds: {len(large_gaps)}")
    
    if len(large_gaps) > 0:
        print(f"\nLargest quote gaps:")
        for i, (time, gap) in enumerate(large_gaps.nlargest(5).items()):
            print(f"  {time}: {gap.total_seconds():.1f}s gap")
    
    # Group trades by second
    trades_per_second = trades_df.groupby(trades_df.index.floor('S')).agg({
        'price': 'count',
        'size': 'sum'
    }).rename(columns={'price': 'count', 'size': 'volume'})
    
    # Find seconds with high trade count
    high_activity = trades_per_second[trades_per_second['count'] > 1000].sort_values('count', ascending=False)
    
    print(f"\nHigh Activity Periods (>1000 trades/sec):")
    for time, row in high_activity.head(5).iterrows():
        # Check quote availability
        quotes_in_second = quotes_df[(quotes_df.index >= time) & 
                                    (quotes_df.index < time + pd.Timedelta(seconds=1))]
        print(f"{time}: {row['count']:,} trades, {len(quotes_in_second)} quotes")
    
    # Recommendations
    print(f"\n{'='*40}")
    print("RECOMMENDATIONS")
    print(f"{'='*40}")
    
    if len(problem_quotes) == 0:
        print("❌ CRITICAL: No quotes during high-activity period!")
        print("   Solutions:")
        print("   1. Increase quote_sync_tolerance_ms to 5000+ ms")
        print("   2. Enable adaptive sync tolerance")
        print("   3. Extend quote window (before and after)")
        print("   4. Consider using tick rule classification as fallback")
    
    print("\nSuggested configuration changes:")
    print("```python")
    print("_config['quote_sync_tolerance_ms'] = 5000  # 5 seconds")
    print("_config['adaptive_sync'] = True")
    print("_config['quote_window_minutes'] = 30")
    print("_config['quote_forward_minutes'] = 10")
    print("```")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Investigate trade classification issues')
    
    parser.add_argument(
        '-s', '--symbol',
        type=str,
        default='CRCL',
        help='Stock symbol (default: CRCL)'
    )
    
    parser.add_argument(
        '-t', '--time',
        type=str,
        default='2025-06-27 13:35:00',
        help='Entry time (default: 2025-06-27 13:35:00)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Parse time
    entry_time = datetime.strptime(args.time, "%Y-%m-%d %H:%M:%S")
    entry_time = entry_time.replace(tzinfo=timezone.utc)
    
    # Run investigation
    asyncio.run(investigate_classification_issues(args.symbol.upper(), entry_time))