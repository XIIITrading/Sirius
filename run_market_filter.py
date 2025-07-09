# run_market_filter.py
"""
Market Filter Launcher - S&P 500 Pre-Market Scanner
Purpose: Run pre-market scans to identify high-interest trading opportunities
Usage: 
    python run_market_filter.py                    # Run with default criteria
    python run_market_filter.py --relaxed         # Run with relaxed criteria
    python run_market_filter.py --strict          # Run with strict criteria
    python run_market_filter.py --push            # Auto-push to Supabase
    python run_market_filter.py --top 20          # Show only top 20 results
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
log_dir = os.path.join(project_root, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f'market_filter_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='S&P 500 Pre-Market Scanner - Find high-interest trading opportunities'
    )
    
    # Filter criteria options
    criteria_group = parser.add_mutually_exclusive_group()
    criteria_group.add_argument(
        '--relaxed', 
        action='store_true',
        help='Use relaxed filter criteria (more results)'
    )
    criteria_group.add_argument(
        '--strict', 
        action='store_true',
        help='Use strict filter criteria (fewer, higher quality results)'
    )
    criteria_group.add_argument(
        '--custom',
        action='store_true',
        help='Use custom filter criteria (interactive)'
    )
    
    # Output options
    parser.add_argument(
        '--top',
        type=int,
        default=50,
        help='Number of top results to display (default: 50)'
    )
    
    parser.add_argument(
        '--push',
        action='store_true',
        help='Automatically push results to Supabase'
    )
    
    parser.add_argument(
        '--no-markdown',
        action='store_true',
        help='Skip creating markdown report'
    )
    
    # Performance options
    parser.add_argument(
        '--workers',
        type=int,
        default=10,
        help='Number of parallel workers for data fetching (default: 10)'
    )
    
    parser.add_argument(
        '--test',
        type=int,
        metavar='N',
        help='Test mode: scan only first N tickers'
    )
    
    return parser.parse_args()


def check_environment():
    """Check if all required packages and credentials are available"""
    print("\n🔍 Checking environment...")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'dotenv': 'python-dotenv',
        'polygon': 'polygon-api-client'
    }
    
    optional_packages = {
        'supabase': 'supabase'
    }
    
    missing_required = []
    missing_optional = []
    
    # Check required packages
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_required.append(pip_name)
            logger.error(f"✗ {package} is NOT installed")
    
    # Check optional packages
    for package, pip_name in optional_packages.items():
        try:
            __import__(package)
            logger.info(f"✓ {package} is installed (optional)")
        except ImportError:
            missing_optional.append(pip_name)
            logger.warning(f"⚠ {package} is NOT installed (optional)")
    
    if missing_required:
        print("\n❌ Missing required packages!")
        print("Install them using:")
        print(f"pip install {' '.join(missing_required)}")
        return False
    
    if missing_optional:
        print("\n⚠️  Missing optional packages:")
        print(f"   pip install {' '.join(missing_optional)}")
        print("   (Required for Supabase integration)")
    
    # Check for API credentials
    from dotenv import load_dotenv
    load_dotenv()
    
    polygon_key = os.getenv('POLYGON_API_KEY')
    if not polygon_key:
        print("\n❌ POLYGON_API_KEY not found in environment variables!")
        print("   Please add it to your .env file")
        return False
    else:
        print("✓ Polygon API key found")
    
    # Check Supabase credentials if push is requested
    if '--push' in sys.argv:
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            print("\n⚠️  Supabase credentials not found")
            print("   Add SUPABASE_URL and SUPABASE_KEY to .env for database features")
    
    return True


def get_filter_criteria(args):
    """Get filter criteria based on command line arguments"""
    from market_review.pre_market.sp500_filter.market_filter import FilterCriteria
    
    if args.strict:
        print("\n📊 Using STRICT filter criteria")
        return FilterCriteria(
            min_price=10.0,
            max_price=500.0,
            min_avg_volume=1_000_000,
            min_premarket_volume=500_000,
            min_premarket_volume_ratio=0.20,  # 20% of avg daily
            min_dollar_volume=10_000_000,
            min_atr=3.0,
            min_atr_percent=15.0
        )
    
    elif args.relaxed:
        print("\n📊 Using RELAXED filter criteria")
        return FilterCriteria(
            min_price=5.0,
            max_price=500.0,
            min_avg_volume=500_000,
            min_premarket_volume=100_000,
            min_premarket_volume_ratio=0.001,  # 0.1%
            min_dollar_volume=1_000_000,
            min_atr=0.5,
            min_atr_percent=0.5
        )
    
    elif args.custom:
        print("\n📊 Custom filter criteria setup")
        return get_custom_criteria()
    
    else:
        print("\n📊 Using DEFAULT filter criteria")
        return FilterCriteria()  # Default values


def get_custom_criteria():
    """Interactive custom criteria setup"""
    from market_review.pre_market.sp500_filter.market_filter import FilterCriteria
    
    print("\nEnter custom filter criteria (press Enter for default):")
    
    def get_float(prompt, default):
        value = input(f"  {prompt} [{default}]: ").strip()
        return float(value) if value else default
    
    def get_int(prompt, default):
        value = input(f"  {prompt} [{default:,}]: ").strip()
        return int(value.replace(',', '')) if value else default
    
    criteria = FilterCriteria()
    
    criteria.min_price = get_float("Min price ($)", criteria.min_price)
    criteria.max_price = get_float("Max price ($)", criteria.max_price)
    criteria.min_avg_volume = get_int("Min avg daily volume", int(criteria.min_avg_volume))
    criteria.min_premarket_volume = get_int("Min pre-market volume", int(criteria.min_premarket_volume))
    criteria.min_premarket_volume_ratio = get_float("Min PM volume ratio (0-1)", criteria.min_premarket_volume_ratio)
    criteria.min_dollar_volume = get_float("Min dollar volume ($)", criteria.min_dollar_volume)
    criteria.min_atr = get_float("Min ATR ($)", criteria.min_atr)
    criteria.min_atr_percent = get_float("Min ATR %", criteria.min_atr_percent)
    
    return criteria


def display_market_hours_info():
    """Display current market status and timing info"""
    from datetime import time
    
    now = datetime.now(timezone.utc)
    et_offset = -5 if now.month < 3 or now.month > 11 else -4  # EST vs EDT
    et_time = now.replace(tzinfo=None) + timedelta(hours=et_offset)
    
    print("\n⏰ Market Hours Information:")
    print(f"   Current UTC: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Current ET:  {et_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Pre-market: 4:00 AM - 9:30 AM ET (09:00 - 14:30 UTC)
    pre_market_start = time(9, 0)   # 09:00 UTC
    pre_market_end = time(14, 30)   # 14:30 UTC
    
    # Regular hours: 9:30 AM - 4:00 PM ET (14:30 - 21:00 UTC)
    regular_start = time(14, 30)    # 14:30 UTC
    regular_end = time(21, 0)       # 21:00 UTC
    
    current_time = now.time()
    
    if pre_market_start <= current_time < pre_market_end:
        print("   📈 Status: PRE-MARKET OPEN")
        print("   ✅ Perfect time for pre-market scanning!")
    elif regular_start <= current_time < regular_end:
        print("   📊 Status: REGULAR TRADING HOURS")
        print("   ⚠️  Pre-market data may be stale")
    elif now.weekday() in [5, 6]:  # Saturday or Sunday
        print("   🚫 Status: WEEKEND - Markets closed")
        print("   ❌ No pre-market data available")
    else:
        print("   🌙 Status: AFTER HOURS or CLOSED")
        print("   ⚠️  Limited or no pre-market data")
    
    print()


def progress_callback(completed: int, total: int, ticker: str):
    """Progress callback for scan"""
    if completed % 25 == 0 or completed == total:
        percentage = (completed / total) * 100
        print(f"   Progress: {completed}/{total} ({percentage:.1f}%) - Processing: {ticker}")


def main():
    """Main launcher function"""
    args = parse_arguments()
    
    print("=" * 70)
    print("S&P 500 PRE-MARKET SCANNER")
    print("=" * 70)
    
    # Check environment
    if not check_environment():
        return 1
    
    # Display market hours info
    display_market_hours_info()
    
    try:
        # Import required modules
        from market_review.pre_market.sp500_filter.sp500_bridge import SP500Bridge
        from market_review.pre_market.sp500_filter.sp500_tickers import get_sp500_tickers
        from market_review.pre_market.sp500_filter.market_filter import (
            push_to_supabase, create_real_data_markdown, get_user_confirmation
        )
        
        logger.info("✓ Successfully imported all required modules")
        
        # Get filter criteria
        criteria = get_filter_criteria(args)
        
        # Get tickers to scan
        if args.test:
            all_tickers = get_sp500_tickers()[:args.test]
            print(f"\n🧪 TEST MODE: Scanning first {len(all_tickers)} tickers only")
        else:
            all_tickers = get_sp500_tickers()
            print(f"\n📊 Scanning ALL {len(all_tickers)} S&P 500 stocks")
        
        print(f"🔧 Using {args.workers} parallel workers")
        print("\n⏳ This may take several minutes...\n")
        
        # Initialize bridge
        bridge = SP500Bridge(
            filter_criteria=criteria,
            parallel_workers=args.workers
        )
        
        # Run scan
        scan_start = datetime.now()
        scan_results = bridge.run_morning_scan(
            progress_callback=progress_callback
        )
        scan_duration = (datetime.now() - scan_start).total_seconds()
        
        print(f"\n✅ Scan completed in {scan_duration:.1f} seconds")
        
        # Get summary
        summary = bridge.get_summary_stats(scan_results)
        
        # Display results
        if scan_results.empty:
            print("\n❌ No stocks passed the filters")
            print("\nPossible reasons:")
            print("  1. Market timing - Pre-market data only available 4:00-9:30 AM ET")
            print("  2. Criteria too strict - Try --relaxed option")
            print("  3. Weekend/Holiday - No trading data")
            print("\nTry: python run_market_filter.py --relaxed")
        else:
            print(f"\n✅ Found {len(scan_results)} stocks that passed filters")
            
            # Display top results
            display_count = min(args.top, len(scan_results))
            print(f"\n🏆 Top {display_count} Stocks by Interest Score:")
            print("-" * 100)
            print(f"{'Rank':<5} {'Ticker':<8} {'Price':<10} {'Score':<8} "
                  f"{'PM Volume':<12} {'PM %':<8} {'ATR %':<8} {'$ Volume':<12}")
            print("-" * 100)
            
            for _, row in scan_results.head(display_count).iterrows():
                pm_pct = (row['premarket_volume'] / row['avg_daily_volume'] * 100)
                
                # Emoji for top 3
                rank_display = str(row['rank'])
                if row['rank'] == 1:
                    rank_display = "🥇"
                elif row['rank'] == 2:
                    rank_display = "🥈"
                elif row['rank'] == 3:
                    rank_display = "🥉"
                
                # Color coding for scores
                score_display = f"{row['interest_score']:.2f}"
                if row['interest_score'] > 70:
                    score_display = f"⭐ {score_display}"
                
                print(f"{rank_display:<5} {row['ticker']:<8} ${row['price']:<9.2f} "
                      f"{score_display:<8} {row['premarket_volume']:>11,.0f} "
                      f"{pm_pct:>7.2f}% {row['atr_percent']:>7.2f}% "
                      f"${row['dollar_volume']/1e6:>10.1f}M")
            
            # Score breakdown for top stock
            if len(scan_results) > 0:
                print("\n📊 Score Breakdown - Top Stock:")
                top_stock = scan_results.iloc[0]
                explanation = bridge.filter_engine.explain_score(top_stock)
                
                print(f"\n   {top_stock['ticker']} - Total Score: {top_stock['interest_score']:.2f}")
                print("   " + "-" * 50)
                
                for component, details in explanation['components'].items():
                    contribution = details['contribution']
                    print(f"   {component:<20} {contribution:>6.2f} points  ({details['raw_value']})")
        
        # Create markdown report
        if not args.no_markdown:
            create_real_data_markdown(
                scan_results, 
                summary, 
                criteria, 
                len(all_tickers)
            )
        
        # Push to Supabase
        if args.push and not scan_results.empty:
            print("\n📤 Pushing results to Supabase...")
            success = push_to_supabase(scan_results, datetime.now())
            if not success:
                print("   Failed to push to Supabase")
        elif not args.push and not scan_results.empty:
            if get_user_confirmation("Would you like to push results to Supabase?"):
                success = push_to_supabase(scan_results, datetime.now())
        
        print("\n" + "=" * 70)
        print("SCAN COMPLETE")
        print("=" * 70)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Scan interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())