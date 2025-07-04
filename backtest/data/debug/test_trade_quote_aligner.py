# backtest/data/debug/test_trade_quote_aligner.py
"""
Test script for TradeQuoteAligner module
Run with: python -m backtest.data.debug.test_trade_quote_aligner -s AAPL -t "2025-01-15 10:30:00"
"""

import argparse
import asyncio
import sys
import os
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.dirname(current_dir)
backtest_dir = os.path.dirname(data_dir)
sys.path.insert(0, backtest_dir)

from backtest.data.trade_quote_aligner import TradeQuoteAligner, TradeSide
from backtest.data.polygon_data_manager import PolygonDataManager
from backtest.data.debug.test_utils import parse_datetime, print_dataframe_summary


async def test_aligner_with_real_data(symbol: str, entry_time: datetime):
    """Test aligner with real market data"""
    print("=" * 80)
    print("TEST: TradeQuoteAligner with Real Market Data")
    print("=" * 80)
    
    # Initialize components
    aligner = TradeQuoteAligner(
        max_quote_age_ms=1000,
        interpolation_limit_ms=5000,
        min_confidence_threshold=0.5
    )
    
    data_manager = PolygonDataManager(
        memory_cache_size=50,
        file_cache_hours=24
        # Removed disable_polygon_cache=True as it doesn't exist
    )
    
    print(f"\nFetching trades and quotes for {symbol} at {entry_time}...")
    
    try:
        # Fetch 30 minutes of tick data
        start_time = entry_time - timedelta(minutes=30)
        end_time = entry_time
        
        # Set the plugin name for tracking
        data_manager.set_current_plugin("TradeQuoteAligner_Test")
        
        # Fetch trades
        print("Fetching trades...")
        trades_df = await data_manager.load_trades(symbol, start_time, end_time)
        
        # Fetch quotes
        print("Fetching quotes...")
        quotes_df = await data_manager.load_quotes(symbol, start_time, end_time)
        
        if trades_df.empty or quotes_df.empty:
            print("No trade or quote data available for this time period")
            return
        
        print(f"\nData fetched:")
        print(f"  Trades: {len(trades_df):,}")
        print(f"  Quotes: {len(quotes_df):,}")
        
        # Perform alignment
        print("\nAligning trades with quotes...")
        aligned_df, report = aligner.align_trades_quotes(trades_df, quotes_df)
        
        # Calculate order flow metrics
        metrics = aligner.calculate_order_flow_metrics(aligned_df)
        
        # Print summary report
        summary = aligner.create_summary_report(report, metrics)
        print(summary)
        
        # Show sample of aligned data
        if not aligned_df.empty:
            print("\n" + "=" * 80)
            print("SAMPLE ALIGNED TRADES (first 10)")
            print("=" * 80)
            
            sample_cols = ['trade_time', 'trade_price', 'trade_size', 
                          'quote_bid', 'quote_ask', 'spread', 'trade_side', 
                          'confidence', 'quote_age_ms']
            
            # Check which columns actually exist
            available_cols = [col for col in sample_cols if col in aligned_df.columns]
            
            sample_df = aligned_df[available_cols].head(10)
            print(sample_df.to_string(float_format='%.4f'))
        
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()


async def test_aligner_edge_cases(symbol: str):
    """Test aligner with problematic scenarios"""
    print("\n" + "=" * 80)
    print("TEST: TradeQuoteAligner Edge Cases")
    print("=" * 80)
    
    aligner = TradeQuoteAligner()
    
    # Test Case 1: Stale quotes
    print("\n1. Testing with stale quotes...")
    
    # Create sample data with stale quotes
    times = pd.date_range('2025-01-15 09:30:00', '2025-01-15 09:31:00', freq='1s', tz='UTC')
    
    # Trades every 5 seconds
    trade_times = times[::5]
    trades_df = pd.DataFrame({
        'price': 100 + np.random.randn(len(trade_times)) * 0.1,
        'size': np.random.randint(100, 1000, len(trade_times))
    }, index=trade_times)
    
    # Quotes every 10 seconds (creating staleness)
    quote_times = times[::10]
    quotes_df = pd.DataFrame({
        'bid': 99.95 + np.random.randn(len(quote_times)) * 0.05,
        'ask': 100.05 + np.random.randn(len(quote_times)) * 0.05,
        'bid_size': np.random.randint(100, 500, len(quote_times)),
        'ask_size': np.random.randint(100, 500, len(quote_times))
    }, index=quote_times)
    
    aligned_df, report = aligner.align_trades_quotes(trades_df, quotes_df)
    
    print(f"Alignment success rate: {report.aligned_trades}/{report.total_trades} ({report.aligned_trades/report.total_trades*100:.1f}%)")
    print(f"Average quote age: {report.avg_quote_age_ms:.1f} ms")
    print(f"Max quote age: {report.max_quote_age_ms:.1f} ms")
    
    # Test Case 2: Crossed quotes
    print("\n2. Testing with crossed quotes...")
    
    # Create quotes with some crossed markets
    quotes_crossed = quotes_df.copy()
    crossed_indices = quotes_crossed.index[2:4]
    quotes_crossed.loc[crossed_indices, 'bid'] = quotes_crossed.loc[crossed_indices, 'ask'] + 0.02
    
    aligned_df, report = aligner.align_trades_quotes(trades_df, quotes_crossed)
    
    if report.warnings:
        print("Warnings detected:")
        for warning in report.warnings:
            print(f"  - {warning}")
    
    # Test Case 3: Missing quotes (testing interpolation)
    print("\n3. Testing quote interpolation...")
    
    # Remove some quotes to test interpolation
    quotes_sparse = quotes_df.iloc[[0, -1]]  # Only first and last quote
    
    aligned_df, report = aligner.align_trades_quotes(trades_df, quotes_sparse)
    
    methods_used = report.alignment_methods_used
    print(f"Alignment methods used: {methods_used}")
    
    # Test Case 4: Midpoint trades
    print("\n4. Testing midpoint trade classification...")
    
    # Create trades at midpoint
    midpoint_trades = trades_df.copy()
    midpoint = (quotes_df['bid'].iloc[0] + quotes_df['ask'].iloc[0]) / 2
    midpoint_trades['price'] = midpoint
    
    aligned_df, report = aligner.align_trades_quotes(midpoint_trades, quotes_df)
    
    print(f"Trade side distribution: {report.side_distribution}")


async def test_aligner_performance(symbol: str, entry_time: datetime):
    """Test aligner performance with large datasets"""
    print("\n" + "=" * 80)
    print("TEST: TradeQuoteAligner Performance")
    print("=" * 80)
    
    aligner = TradeQuoteAligner()
    
    # Generate large datasets
    print("\nGenerating large datasets...")
    
    # 1 hour of trades (avg 10 trades/second = 36,000 trades)
    trade_times = []
    current = entry_time - timedelta(hours=1)
    end = entry_time
    
    while current < end:
        # Random inter-trade time (exponential distribution)
        current += timedelta(seconds=np.random.exponential(0.1))
        if current < end:
            trade_times.append(current)
    
    trades_df = pd.DataFrame({
        'price': 100 + np.random.randn(len(trade_times)) * 0.5,
        'size': np.random.randint(100, 5000, len(trade_times))
    }, index=pd.DatetimeIndex(trade_times))
    
    # Quotes (avg 20 quotes/second = 72,000 quotes)
    quote_times = []
    current = entry_time - timedelta(hours=1)
    
    while current < end:
        current += timedelta(seconds=np.random.exponential(0.05))
        if current < end:
            quote_times.append(current)
    
    spreads = np.random.uniform(0.01, 0.05, len(quote_times))
    midpoints = 100 + np.random.randn(len(quote_times)) * 0.5
    
    quotes_df = pd.DataFrame({
        'bid': midpoints - spreads/2,
        'ask': midpoints + spreads/2,
        'bid_size': np.random.randint(100, 1000, len(quote_times)),
        'ask_size': np.random.randint(100, 1000, len(quote_times))
    }, index=pd.DatetimeIndex(quote_times))
    
    print(f"Generated: {len(trades_df):,} trades, {len(quotes_df):,} quotes")
    
    # Time alignment
    import time
    print("\nPerforming alignment...")
    start_time = time.time()
    
    aligned_df, report = aligner.align_trades_quotes(trades_df, quotes_df)
    
    elapsed = time.time() - start_time
    
    # Calculate order flow metrics
    metrics_start = time.time()
    metrics = aligner.calculate_order_flow_metrics(aligned_df)
    metrics_elapsed = time.time() - metrics_start
    
    print(f"\nPerformance Results:")
    print(f"  Alignment time: {elapsed:.3f} seconds")
    print(f"  Trades per second: {len(trades_df)/elapsed:,.0f}")
    print(f"  Metrics calculation: {metrics_elapsed:.3f} seconds")
    print(f"  Total time: {elapsed + metrics_elapsed:.3f} seconds")
    
    print(f"\nAlignment Results:")
    print(f"  Success rate: {report.aligned_trades/report.total_trades*100:.1f}%")
    print(f"  Average confidence: {aligned_df['confidence'].mean():.3f}")
    
    # Show confidence distribution
    print(f"\nConfidence Distribution:")
    for level, count in sorted(report.confidence_distribution.items()):
        print(f"  {level}: {count:,}")


async def test_order_flow_analysis(symbol: str, entry_time: datetime):
    """Test complete order flow analysis pipeline"""
    print("\n" + "=" * 80)
    print("TEST: Complete Order Flow Analysis")
    print("=" * 80)
    
    # This test shows how the aligner integrates with your order flow modules
    aligner = TradeQuoteAligner(
        max_quote_age_ms=500,      # Stricter for HFT analysis
        min_confidence_threshold=0.7
    )
    
    data_manager = PolygonDataManager(
        memory_cache_size=50,
        file_cache_hours=24
    )
    
    print(f"\nAnalyzing order flow for {symbol} at {entry_time}...")
    
    try:
        # Set plugin name for tracking
        data_manager.set_current_plugin("OrderFlowAnalysis_Test")
        
        # Test different time windows
        windows = [5, 15, 30]  # minutes
        
        for window in windows:
            print(f"\n--- {window}-Minute Window ---")
            
            start_time = entry_time - timedelta(minutes=window)
            
            # Fetch data
            trades_df = await data_manager.load_trades(symbol, start_time, entry_time)
            quotes_df = await data_manager.load_quotes(symbol, start_time, entry_time)
            
            if trades_df.empty or quotes_df.empty:
                print(f"No data available for {window}-minute window")
                continue
            
            # Align and analyze
            aligned_df, report = aligner.align_trades_quotes(trades_df, quotes_df)
            metrics = aligner.calculate_order_flow_metrics(aligned_df)
            
            # Print key metrics
            print(f"Trades: {len(trades_df):,}")
            print(f"Buy/Sell Ratio: {metrics.get('buy_sell_ratio', 0):.2f}")
            print(f"Net Order Flow: {metrics.get('net_order_flow', 0):,}")
            print(f"Large Buy %: {metrics.get('large_buy_volume', 0) / max(1, metrics.get('buy_volume', 1)) * 100:.1f}%")
            print(f"Large Sell %: {metrics.get('large_sell_volume', 0) / max(1, metrics.get('sell_volume', 1)) * 100:.1f}%")
            
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Test TradeQuoteAligner module")
    parser.add_argument("-s", "--symbol", default="AAPL", help="Stock symbol")
    parser.add_argument("-t", "--time", default="2025-01-15 10:30:00",
                       help="Entry time (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--test", type=int, default=0,
                       help="Specific test (0=all, 1=real data, 2=edge cases, 3=performance, 4=order flow)")
    
    args = parser.parse_args()
    
    # Parse entry time
    try:
        entry_time = parse_datetime(args.time)
    except ValueError as e:
        print(f"Error parsing time: {e}")
        sys.exit(1)
    
    # Warn about API usage
    if args.test in [0, 1, 4]:
        print("\n" + "!" * 60)
        print("WARNING: This will use REAL Polygon API credits!")
        print("!" * 60)
        response = input("\nContinue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    # Run tests
    async def run_tests():
        if args.test == 0 or args.test == 1:
            await test_aligner_with_real_data(args.symbol, entry_time)
        
        if args.test == 0 or args.test == 2:
            await test_aligner_edge_cases(args.symbol)
        
        if args.test == 0 or args.test == 3:
            await test_aligner_performance(args.symbol, entry_time)
            
        if args.test == 0 or args.test == 4:
            await test_order_flow_analysis(args.symbol, entry_time)
    
    asyncio.run(run_tests())


if __name__ == "__main__":
    main()