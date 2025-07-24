# run_ranking_custom.py

import argparse
import os
from datetime import date, datetime
from dotenv import load_dotenv

from src.services.ranking_engine import RankingEngine
from src.services.level_service import LevelService

# Load environment variables
load_dotenv()

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Run ranking engine with Polygon data')
    
    parser.add_argument(
        '--date', '-d',
        type=str,
        help='Date to process (format: YYYY-MM-DD). Defaults to today.',
        default=None
    )
    
    parser.add_argument(
        '--tickers', '-t',
        type=str,
        nargs='+',
        help='Specific tickers to process. If not provided, processes all tickers with levels.',
        default=None
    )
    
    parser.add_argument(
        '--api-key', '-k',
        type=str,
        help='Polygon API key. Can also be set via POLYGON_API_KEY env var.',
        default=os.getenv('POLYGON_API_KEY')
    )

    parser.add_argument(
        '--atr', '-a',
        type=str,
        choices=['5min', '15min', '30min', '1hour'],
        default='15min',
        help='ATR timeframe to use for zone calculations'
    )
    
    # Parse arguments AFTER all add_argument calls
    args = parser.parse_args()
    
    # Validate API key
    if not args.api_key:
        print("Error: Polygon API key required. Set via --api-key or POLYGON_API_KEY env var.")
        print("Example: python run_ranking_custom.py -d 2025-07-23 -t TSLA -k YOUR_API_KEY")
        print("Or set POLYGON_API_KEY in your .env file")
        return
    
    # Parse date
    if args.date:
        try:
            target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        except ValueError:
            print(f"Error: Invalid date format '{args.date}'. Use YYYY-MM-DD format.")
            return
    else:
        target_date = date.today()
    
    print(f"Running ranking for date: {target_date}")
    print(f"Using Polygon API for market data")
    print(f"Using {args.atr} ATR for zone calculations")
    
    # Initialize services with ATR timeframe
    ranking_engine = RankingEngine(
        polygon_api_key=args.api_key,
        atr_timeframe=args.atr
    )
    level_service = LevelService()
    
    # Get tickers
    if args.tickers:
        tickers = args.tickers
        print(f"Processing specified tickers: {', '.join(tickers)}")
    else:
        # Get all tickers with levels for the date
        tickers = level_service.get_unique_tickers_with_levels(target_date)
        print(f"Found {len(tickers)} tickers with premarket levels")
    
    if not tickers:
        print("No tickers to process")
        return
    
    # Process all tickers - the ranking engine will fetch market data from Polygon
    results = ranking_engine.process_multiple_tickers(
        tickers=tickers,
        target_date=target_date
    )
    
    # Display results
    print("\n" + "="*60)
    print(f"RANKING RESULTS FOR {target_date}")
    print("="*60)
    
    total_levels = 0
    for ticker, ranked_levels in results.items():
        if ranked_levels:
            total_levels += len(ranked_levels)
            print(f"\n{ticker}: {len(ranked_levels)} levels ranked")
            print("-" * 40)
            
            # Show top 3 levels
            for level in ranked_levels[:3]:
                print(f"  Rank {level.rank}:")
                print(f"    Score: {level.confluence_score:.2f}")
                print(f"    Zone: ${level.zone_low:.2f} - ${level.zone_high:.2f}")
                print(f"    TV Variable: {level.tv_variable}")
    
    print(f"\n✅ Total: {total_levels} levels ranked across {len(results)} tickers")

if __name__ == "__main__":
    main()