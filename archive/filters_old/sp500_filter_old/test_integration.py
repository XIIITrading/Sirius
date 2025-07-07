# modules/filters/sp500_filter/test_integration.py
"""
Integration test for S&P 500 market filter system.
Tests the complete pipeline from data fetching to filtering and scoring.
"""

import sys
import os
from datetime import datetime, timedelta, timezone
import pandas as pd
import logging
import traceback
from typing import Dict, List

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sp500_filter_dir = current_dir
filters_dir = os.path.dirname(sp500_filter_dir)
modules_dir = os.path.dirname(filters_dir)
sirius_root = os.path.dirname(modules_dir)
sys.path.insert(0, sirius_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required modules can be imported."""
    print("=" * 60)
    print("TESTING IMPORTS")
    print("=" * 60)
    
    imports_ok = True
    
    # Test local imports
    try:
        from sp500_tickers import get_sp500_tickers, check_update_status
        print("✓ sp500_tickers imported successfully")
    except Exception as e:
        print(f"✗ Failed to import sp500_tickers: {e}")
        imports_ok = False
    
    try:
        from market_filter import MarketFilter, FilterCriteria
        print("✓ market_filter imported successfully")
    except Exception as e:
        print(f"✗ Failed to import market_filter: {e}")
        imports_ok = False
    
    try:
        from sp500_bridge import SP500Bridge
        print("✓ sp500_bridge imported successfully")
    except Exception as e:
        print(f"✗ Failed to import sp500_bridge: {e}")
        imports_ok = False
    
    # Test Polygon imports
    try:
        from polygon import DataFetcher
        from polygon.config import PolygonConfig
        print("✓ Polygon modules imported successfully")
    except Exception as e:
        print(f"✗ Failed to import Polygon modules: {e}")
        imports_ok = False
    
    print()
    return imports_ok


def test_sp500_tickers():
    """Test the S&P 500 ticker list functionality."""
    print("=" * 60)
    print("TESTING S&P 500 TICKER LIST")
    print("=" * 60)
    
    try:
        from sp500_tickers import get_sp500_tickers, check_update_status, LAST_UPDATED
        
        # Check update status
        print(f"Last updated: {LAST_UPDATED}")
        is_current = check_update_status()
        
        # Get tickers
        tickers = get_sp500_tickers(check_staleness=False)
        print(f"✓ Loaded {len(tickers)} tickers")
        print(f"  Sample tickers: {tickers[:5]}")
        print(f"  Contains AAPL: {'AAPL' in tickers}")
        print(f"  Contains MSFT: {'MSFT' in tickers}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing S&P 500 tickers: {e}")
        traceback.print_exc()
        return False


def test_market_filter():
    """Test the market filter engine with sample data."""
    print("\n" + "=" * 60)
    print("TESTING MARKET FILTER ENGINE")
    print("=" * 60)
    
    try:
        from market_filter import MarketFilter, FilterCriteria
        
        # Create sample data
        sample_data = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'LOW_PRICE', 'LOW_VOL'],
            'price': [150.0, 300.0, 120.0, 180.0, 250.0, 3.0, 100.0],
            'avg_daily_volume': [50_000_000, 25_000_000, 20_000_000, 40_000_000, 80_000_000, 1_000_000, 100_000],
            'premarket_volume': [6_000_000, 3_000_000, 1_000_000, 5_000_000, 12_000_000, 100_000, 5_000],
            'dollar_volume': [7_500_000_000, 7_500_000_000, 2_400_000_000, 7_200_000_000, 20_000_000_000, 3_000_000, 10_000_000],
            'atr': [3.5, 5.2, 2.8, 4.1, 8.5, 0.5, 2.0],
            'atr_percent': [2.3, 1.7, 2.3, 2.3, 3.4, 16.7, 2.0]
        })
        
        print("Sample data created:")
        print(f"  {len(sample_data)} stocks")
        
        # Initialize filter with criteria
        criteria = FilterCriteria(
            min_price=5.0,
            max_price=500.0,
            min_avg_volume=500_000
        )
        
        filter_engine = MarketFilter(criteria=criteria)
        print(f"\n✓ MarketFilter initialized with criteria:")
        for key, value in criteria.to_dict().items():
            print(f"  {key}: {value}")
        
        # Apply filters
        filtered = filter_engine.apply_filters(sample_data)
        print(f"\n✓ Filters applied: {len(filtered)}/{len(sample_data)} passed")
        
        # Rank by interest
        ranked = filter_engine.rank_by_interest(filtered)
        print(f"\n✓ Ranking complete. Top 3:")
        if not ranked.empty:
            for _, row in ranked.head(3).iterrows():
                print(f"  #{row['rank']} {row['ticker']}: Score = {row['interest_score']:.2f}")
        
        # Test score explanation
        if not ranked.empty:
            explanation = filter_engine.explain_score(ranked.iloc[0])
            print(f"\n✓ Score explanation for {ranked.iloc[0]['ticker']}:")
            for comp, details in explanation['components'].items():
                print(f"  {comp}: {details['contribution']:.2f} points")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing market filter: {e}")
        traceback.print_exc()
        return False


def test_polygon_connection():
    """Test connection to Polygon API."""
    print("\n" + "=" * 60)
    print("TESTING POLYGON CONNECTION")
    print("=" * 60)
    
    try:
        from polygon import DataFetcher
        from polygon.config import PolygonConfig
        
        # Initialize fetcher
        fetcher = DataFetcher(config=PolygonConfig({'cache_enabled': True}))
        print("✓ DataFetcher initialized")
        
        # Test with a known ticker
        test_ticker = 'AAPL'
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=5)
        
        print(f"\nFetching test data for {test_ticker}...")
        df = fetcher.fetch_data(
            symbol=test_ticker,
            timeframe='1d',
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )
        
        if not df.empty:
            print(f"✓ Successfully fetched {len(df)} days of data")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
            print(f"  Latest close: ${df['close'].iloc[-1]:.2f}")
            return True
        else:
            print("✗ No data returned from Polygon")
            return False
            
    except Exception as e:
        print(f"✗ Error testing Polygon connection: {e}")
        print("\nPossible issues:")
        print("  1. API key not configured")
        print("  2. Network connectivity")
        print("  3. Polygon module not properly installed")
        traceback.print_exc()
        return False


def test_sp500_bridge_mini():
    """Test SP500Bridge with a small subset of tickers."""
    print("\n" + "=" * 60)
    print("TESTING SP500 BRIDGE (MINI SCAN)")
    print("=" * 60)
    
    try:
        from sp500_bridge import SP500Bridge
        from market_filter import FilterCriteria
        
        # Create bridge with test criteria
        criteria = FilterCriteria(
            min_price=10.0,
            max_price=500.0,
            min_avg_volume=1_000_000,
            min_premarket_volume_ratio=0.05  # Lower for testing
        )
        
        bridge = SP500Bridge(filter_criteria=criteria, parallel_workers=3)
        print("✓ SP500Bridge initialized")
        
        # Override with small ticker list for testing
        test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        bridge.tickers = test_tickers
        print(f"\nTesting with {len(test_tickers)} tickers: {test_tickers}")
        
        # Progress callback
        def progress(completed, total, ticker):
            print(f"  Processing {ticker} ({completed}/{total})")
        
        # Run mini scan
        print("\nRunning mini scan...")
        scan_time = datetime.now(timezone.utc).replace(hour=9, minute=0, second=0)
        
        results = bridge.run_morning_scan(
            scan_time=scan_time,
            lookback_days=14,
            progress_callback=progress
        )
        
        if not results.empty:
            print(f"\n✓ Scan complete: {len(results)} stocks passed filters")
            print("\nResults:")
            display_cols = ['ticker', 'price', 'interest_score', 'premarket_volume', 'atr_percent']
            print(results[display_cols].to_string(index=False))
            
            # Get summary
            summary = bridge.get_summary_stats(results)
            print("\nSummary Stats:")
            print(f"  Pass rate: {summary['pass_rate']}")
            print(f"  Avg interest score: {summary.get('avg_interest_score', 0):.2f}")
            
            return True
        else:
            print("\n⚠ No stocks passed the filters")
            print("This might be normal if:")
            print("  - Market is closed (no pre-market volume)")
            print("  - Criteria are too strict")
            print("  - Data fetching issues")
            return True  # Not necessarily an error
            
    except Exception as e:
        print(f"\n✗ Error testing SP500Bridge: {e}")
        traceback.print_exc()
        return False


def test_full_integration():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("S&P 500 MARKET FILTER - INTEGRATION TEST")
    print("=" * 60)
    print(f"Test Time: {datetime.now()}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Track test results
    results = {
        'imports': False,
        'tickers': False,
        'filter': False,
        'polygon': False,
        'bridge': False
    }
    
    # Run tests
    results['imports'] = test_imports()
    
    if results['imports']:
        results['tickers'] = test_sp500_tickers()
        results['filter'] = test_market_filter()
        results['polygon'] = test_polygon_connection()
        
        if results['polygon']:
            results['bridge'] = test_sp500_bridge_mini()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name.ljust(15)}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED - System is ready!")
        print("\nNext steps:")
        print("1. Run a full S&P 500 scan")
        print("2. Create the file output formatter")
        print("3. Schedule for automated morning runs")
    else:
        print("✗ SOME TESTS FAILED - Please check the errors above")
        print("\nCommon issues:")
        print("1. Missing Polygon API key")
        print("2. Import path issues")
        print("3. Network connectivity")
        print("4. Market closed (no pre-market data)")
    
    return all_passed


if __name__ == "__main__":
    # Change to the sp500_filter directory
    os.chdir(current_dir)
    
    # Run integration test
    success = test_full_integration()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)