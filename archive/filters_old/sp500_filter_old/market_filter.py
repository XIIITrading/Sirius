# modules/filters/sp500_filter/market_filter.py
"""
Market Filter Engine - Pure Calculation Module
Applies filtering criteria and calculates interest scores for stock screening.
No data fetching - receives DataFrames and returns filtered/ranked results.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class FilterCriteria:
    """Configurable filter criteria with defaults."""
    min_price: float = 5.0
    max_price: float = 500.0
    min_avg_volume: float = 500_000
    min_premarket_volume: float = 300_000
    min_premarket_volume_ratio: float = 0.10  # 10% of avg daily volume
    min_dollar_volume: float = 5_000_000
    min_atr: float = 2.0
    min_atr_percent: float = 10.0  # 10% of price
    
    def to_dict(self) -> Dict:
        """Convert criteria to dictionary for logging/display."""
        return {
            'Price Range': f'${self.min_price} - ${self.max_price}',
            'Min Avg Volume': f'{self.min_avg_volume:,.0f} shares',
            'Min PM Volume Ratio': f'{self.min_premarket_volume_ratio:.1%}',
            'Min Dollar Volume': f'${self.min_dollar_volume:,.0f}',
            'Min ATR': f'${self.min_atr}',
            'Min ATR %': f'{self.min_atr_percent}%'
        }


@dataclass
class InterestScoreWeights:
    """Weights for interest score calculation."""
    premarket_volume_ratio: float = 0.40
    atr_percentage: float = 0.25
    dollar_volume_score: float = 0.20
    premarket_volume_absolute: float = 0.10
    price_atr_bonus: float = 0.05
    
    def validate(self):
        """Ensure weights sum to 1.0."""
        total = (self.premarket_volume_ratio + self.atr_percentage + 
                self.dollar_volume_score + self.premarket_volume_absolute + 
                self.price_atr_bonus)
        if not np.isclose(total, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass
class FilterResult:
    """Container for filter results."""
    ticker: str
    passed_filters: bool
    failed_criteria: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    interest_score: float = 0.0
    interest_components: Dict[str, float] = field(default_factory=dict)


class MarketFilter:
    """
    Pure calculation engine for market filtering and ranking.
    No data fetching - operates on provided DataFrames.
    """
    
    def __init__(self, 
                 criteria: Optional[FilterCriteria] = None,
                 weights: Optional[InterestScoreWeights] = None):
        """
        Initialize the market filter.
        
        Args:
            criteria: Filter criteria (uses defaults if None)
            weights: Interest score weights (uses defaults if None)
        """
        self.criteria = criteria or FilterCriteria()
        self.weights = weights or InterestScoreWeights()
        self.weights.validate()
        
        logger.info("MarketFilter initialized with criteria: %s", self.criteria.to_dict())
    
    def apply_filters(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all filter criteria to market data.
        """
        if market_data.empty:
            logger.warning("Empty market data provided")
            return pd.DataFrame()
        
        # Validate required columns
        required_columns = ['ticker', 'price', 'avg_daily_volume', 'premarket_volume',
                        'dollar_volume', 'atr', 'atr_percent']
        missing_columns = set(required_columns) - set(market_data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Start with copy to avoid modifying original
        df = market_data.copy()
        initial_count = len(df)
        
        # Track filtering results
        filter_results = []
        
        # Apply each filter and track what passes/fails
        filters = {
            'price_min': df['price'] >= self.criteria.min_price,
            'price_max': df['price'] <= self.criteria.max_price,
            'avg_volume': df['avg_daily_volume'] >= self.criteria.min_avg_volume,
            'pm_volume_min': df['premarket_volume'] >= self.criteria.min_premarket_volume,  # Add this line
            'pm_volume_ratio': (df['premarket_volume'] / df['avg_daily_volume']) >= self.criteria.min_premarket_volume_ratio,
            'dollar_volume': df['dollar_volume'] >= self.criteria.min_dollar_volume,
            'atr_min': df['atr'] >= self.criteria.min_atr,
            'atr_percent': df['atr_percent'] >= self.criteria.min_atr_percent
        }
        
        # Combine all filters
        combined_filter = pd.Series(True, index=df.index)
        for filter_name, filter_mask in filters.items():
            combined_filter &= filter_mask
        
        # Log filtering stats
        for filter_name, filter_mask in filters.items():
            passed = filter_mask.sum()
            logger.debug(f"Filter '{filter_name}': {passed}/{initial_count} passed")
        
        # Apply combined filter
        filtered_df = df[combined_filter].copy()
        
        logger.info(f"Filtering complete: {len(filtered_df)}/{initial_count} stocks passed all filters")
        
        # Calculate interest scores for filtered stocks
        if not filtered_df.empty:
            filtered_df = self._calculate_interest_scores(filtered_df)
        
        return filtered_df
    
    def _calculate_interest_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate interest scores for all stocks in DataFrame.
        
        Args:
            df: DataFrame with filtered stocks
            
        Returns:
            DataFrame with added interest score columns
        """
        # Calculate individual components
        
        # 1. Pre-market Volume Ratio (0-100 scale, capped at 100)
        df['pm_vol_ratio_score'] = (
            (df['premarket_volume'] / df['avg_daily_volume']) * 100
        ).clip(upper=100)
        
        # 2. ATR Percentage (already in percentage form)
        df['atr_percent_score'] = df['atr_percent'].clip(upper=100)
        
        # 3. Dollar Volume Score (normalized to $5M baseline)
        df['dollar_vol_score'] = (
            (df['dollar_volume'] / self.criteria.min_dollar_volume) * 100
        ).clip(upper=100)
        
        # 4. Pre-market Volume Absolute Score (normalized, log scale)
        # Use log scale to handle large volume variations
        df['pm_vol_abs_score'] = (
            np.log10(df['premarket_volume'] + 1) / np.log10(1_000_000) * 100
        ).clip(upper=100)
        
        # 5. Price-ATR Sweet Spot Bonus
        # Bonus for stocks with ATR between 2-5% of price (optimal volatility)
        df['price_atr_bonus'] = 0.0
        sweet_spot_mask = (df['atr_percent'] >= 2.0) & (df['atr_percent'] <= 5.0)
        df.loc[sweet_spot_mask, 'price_atr_bonus'] = 100.0
        
        # Calculate weighted composite score
        df['interest_score'] = (
            (df['pm_vol_ratio_score'] * self.weights.premarket_volume_ratio) +
            (df['atr_percent_score'] * self.weights.atr_percentage) +
            (df['dollar_vol_score'] * self.weights.dollar_volume_score) +
            (df['pm_vol_abs_score'] * self.weights.premarket_volume_absolute) +
            (df['price_atr_bonus'] * self.weights.price_atr_bonus)
        )
        
        # Round for display
        df['interest_score'] = df['interest_score'].round(2)
        
        logger.debug(f"Interest scores calculated. Range: {df['interest_score'].min():.2f} - {df['interest_score'].max():.2f}")
        
        return df
    
    def rank_by_interest(self, filtered_df: pd.DataFrame, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Rank filtered stocks by interest score.
        
        Args:
            filtered_df: DataFrame with filtered stocks and interest scores
            top_n: Return only top N stocks (None for all)
            
        Returns:
            DataFrame sorted by interest score (descending)
        """
        if filtered_df.empty:
            return filtered_df
        
        # Sort by interest score descending
        ranked_df = filtered_df.sort_values('interest_score', ascending=False)
        
        # Add rank column
        ranked_df['rank'] = range(1, len(ranked_df) + 1)
        
        # Return top N if specified
        if top_n and top_n < len(ranked_df):
            logger.info(f"Returning top {top_n} stocks by interest score")
            return ranked_df.head(top_n)
        
        return ranked_df
    
    def get_filter_summary(self, market_data: pd.DataFrame, filtered_data: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for the filtering process.
        
        Args:
            market_data: Original market data
            filtered_data: Filtered results
            
        Returns:
            Dictionary with summary statistics
        """
        total_stocks = len(market_data)
        passed_stocks = len(filtered_data)
        
        summary = {
            'total_stocks': total_stocks,
            'passed_filters': passed_stocks,
            'pass_rate': f"{(passed_stocks / total_stocks * 100):.1f}%" if total_stocks > 0 else "0.0%",
            'filter_criteria': self.criteria.to_dict()
        }
        
        if not filtered_data.empty:
            summary.update({
                'interest_score_range': f"{filtered_data['interest_score'].min():.2f} - {filtered_data['interest_score'].max():.2f}",
                'avg_interest_score': f"{filtered_data['interest_score'].mean():.2f}",
                'top_ticker': filtered_data.iloc[0]['ticker'] if len(filtered_data) > 0 else None,
                'top_score': f"{filtered_data.iloc[0]['interest_score']:.2f}" if len(filtered_data) > 0 else None
            })
        
        return summary
    
    def explain_score(self, row: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Explain the interest score calculation for a single stock.
        
        Args:
            row: Series containing stock data with calculated scores
            
        Returns:
            Dictionary with score breakdown
        """
        explanation = {
            'components': {
                'PM Volume Ratio': {
                    'raw_value': f"{row['premarket_volume'] / row['avg_daily_volume']:.2%}",
                    'score': row['pm_vol_ratio_score'],
                    'weight': self.weights.premarket_volume_ratio,
                    'contribution': row['pm_vol_ratio_score'] * self.weights.premarket_volume_ratio
                },
                'ATR Percentage': {
                    'raw_value': f"{row['atr_percent']:.2f}%",
                    'score': row['atr_percent_score'],
                    'weight': self.weights.atr_percentage,
                    'contribution': row['atr_percent_score'] * self.weights.atr_percentage
                },
                'Dollar Volume': {
                    'raw_value': f"${row['dollar_volume']:,.0f}",
                    'score': row['dollar_vol_score'],
                    'weight': self.weights.dollar_volume_score,
                    'contribution': row['dollar_vol_score'] * self.weights.dollar_volume_score
                },
                'PM Volume Absolute': {
                    'raw_value': f"{row['premarket_volume']:,.0f}",
                    'score': row['pm_vol_abs_score'],
                    'weight': self.weights.premarket_volume_absolute,
                    'contribution': row['pm_vol_abs_score'] * self.weights.premarket_volume_absolute
                },
                'Price-ATR Bonus': {
                    'raw_value': f"{'Yes' if row['price_atr_bonus'] > 0 else 'No'}",
                    'score': row['price_atr_bonus'],
                    'weight': self.weights.price_atr_bonus,
                    'contribution': row['price_atr_bonus'] * self.weights.price_atr_bonus
                }
            },
            'total_score': row['interest_score']
        }
        
        return explanation
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate market data before processing.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check for required columns
        required_columns = ['ticker', 'price', 'avg_daily_volume', 'premarket_volume',
                          'dollar_volume', 'atr', 'atr_percent']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing columns: {missing_columns}")
        
        # Check for data types and ranges
        if 'price' in df.columns:
            if (df['price'] < 0).any():
                errors.append("Negative prices found")
        
        if 'avg_daily_volume' in df.columns:
            if (df['avg_daily_volume'] < 0).any():
                errors.append("Negative volumes found")
        
        if 'atr' in df.columns:
            if (df['atr'] < 0).any():
                errors.append("Negative ATR values found")
        
        # Check for missing values
        if df.isnull().any().any():
            null_counts = df.isnull().sum()
            null_cols = null_counts[null_counts > 0]
            errors.append(f"Missing values found: {null_cols.to_dict()}")
        
        return len(errors) == 0, errors


def push_to_supabase(scan_results: pd.DataFrame, scan_date: datetime):
    """
    Push scan results to Supabase.
    
    Args:
        scan_results: DataFrame with scan results
        scan_date: Date of the scan
    """
    try:
        from supabase import create_client, Client
    except ImportError:
        print("\n❌ Supabase client not installed. Please install with: pip install supabase")
        return False
    
    # Get Supabase credentials from environment variables
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    
    if not supabase_url or not supabase_key:
        print("\n❌ Supabase credentials not found in environment variables.")
        print("   Please set SUPABASE_URL and SUPABASE_KEY in your .env file")
        return False
    
    try:
        # Initialize Supabase client
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Format scan date for database
        scan_date_str = scan_date.strftime('%Y-%m-%d')
        
        # Check if records exist for this date
        existing_records = supabase.table('premarket_scans').select('ticker').eq('scan_date', scan_date_str).execute()
        
        if existing_records.data and len(existing_records.data) > 0:
            print(f"\n⚠️  Found {len(existing_records.data)} existing records for {scan_date_str}")
            print("   This will cause duplicate key errors.")
            
            if get_user_confirmation(f"Would you like to DELETE all existing records for {scan_date_str} before inserting new ones?"):
                # Delete existing records
                print(f"🗑️  Deleting existing records for {scan_date_str}...")
                delete_response = supabase.table('premarket_scans').delete().eq('scan_date', scan_date_str).execute()
                print(f"✅ Deleted {len(existing_records.data)} existing records")
            else:
                print("\n⏭️  Skipping push to avoid duplicate key errors.")
                return False
        
        # Prepare data for insertion
        records = []
        for _, row in scan_results.iterrows():
            record = {
                'ticker': row['ticker'],
                'price': float(row['price']),
                'rank': int(row['rank']),
                'premarket_volume': int(row['premarket_volume']),
                'avg_daily_volume': int(row['avg_daily_volume']),
                'dollar_volume': float(row['dollar_volume']),
                'atr': float(row['atr']),
                'atr_percent': float(row['atr_percent']),
                'interest_score': float(row['interest_score']),
                'pm_vol_ratio_score': float(row['pm_vol_ratio_score']),
                'atr_percent_score': float(row['atr_percent_score']),
                'dollar_vol_score': float(row['dollar_vol_score']),
                'pm_vol_abs_score': float(row['pm_vol_abs_score']),
                'price_atr_bonus': float(row['price_atr_bonus']),
                'scan_date': scan_date_str,
                'passed_filters': True,  # All results passed filters
                'market_session': 'pre-market'
            }
            records.append(record)
        
        # Insert records
        print(f"\n📤 Pushing {len(records)} records to Supabase...")
        
        # Insert in batches to handle large datasets
        batch_size = 100
        total_inserted = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            response = supabase.table('premarket_scans').insert(batch).execute()
            total_inserted += len(batch)
            print(f"   Inserted batch {i//batch_size + 1}/{(len(records) + batch_size - 1)//batch_size} ({total_inserted}/{len(records)} records)")
        
        print(f"\n✅ Successfully pushed {len(records)} records to Supabase!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error pushing to Supabase: {e}")
        return False


def get_user_confirmation(prompt: str) -> bool:
    """Get yes/no confirmation from user."""
    while True:
        response = input(f"\n{prompt} (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no.")


def example_usage():
    """Example usage with real data from SP500Bridge."""
    import os
    import sys
    from datetime import datetime, timezone
    
    # Add current directory to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    try:
        from sp500_bridge import SP500Bridge
        from sp500_tickers import get_sp500_tickers
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Make sure sp500_bridge.py and sp500_tickers.py are in the same directory")
        return
    
    print("Market Filter - S&P 500 Full Scan")
    print("=" * 70)
    print(f"Scan Time: {datetime.now()}")
    
    # Get all S&P 500 tickers
    all_tickers = get_sp500_tickers()
    print(f"\nScanning ALL {len(all_tickers)} S&P 500 stocks...")
    print("⚠️  This will take several minutes to complete...\n")
    
    # Use relaxed criteria for better results
    relaxed_criteria = FilterCriteria(
        min_price=5.0,
        max_price=500.0,
        min_avg_volume=500_000,
        min_premarket_volume=300_000,         # Add minimum PM volume requirement
        min_premarket_volume_ratio=0.001,      # 0.1% - very relaxed for testing
        min_dollar_volume=1_000_000,           # $1M instead of $5M
        min_atr=0.5,                           # $0.50 instead of $2
        min_atr_percent=0.5                    # 0.5% instead of 10%
    )

    # Initialize bridge with relaxed criteria
    try:
        # Use more parallel workers for full scan
        bridge = SP500Bridge(filter_criteria=relaxed_criteria, parallel_workers=10)
        
        # Progress callback
        def progress(completed, total, ticker):
            if completed % 10 == 0 or completed == total:  # Print every 10 stocks
                print(f"  Progress: {completed}/{total} ({completed/total*100:.1f}%) - Current: {ticker}")
        
        # Run scan
        print("Starting scan...")
        scan_results = bridge.run_morning_scan(progress_callback=progress)
        
        # Get scan date
        scan_date = datetime.now()
        
        if scan_results.empty:
            print("\nNo stocks passed filters. This could be because:")
            print("  - Market is closed (no pre-market data)")
            print("  - Criteria still too strict")
            print("  - Try running during market hours (4 AM - 9:30 AM ET)")
            
            # Still create markdown with empty results
            summary = bridge.get_summary_stats(scan_results)
            create_real_data_markdown(scan_results, summary, relaxed_criteria, len(all_tickers))
        else:
            print(f"\n✓ Found {len(scan_results)} stocks that passed filters")
            
            # Get summary stats
            summary = bridge.get_summary_stats(scan_results)
            
            # Show console output
            print("\nTop Stocks by Interest Score:")
            print("-" * 80)
            print(f"{'Rank':<5} {'Ticker':<8} {'Price':<10} {'Score':<8} {'PM Vol':<12} {'PM %':<8} {'ATR %':<8}")
            print("-" * 80)
            
            for _, row in scan_results.head(10).iterrows():
                pm_pct = (row['premarket_volume'] / row['avg_daily_volume'] * 100)
                print(f"{row['rank']:<5} {row['ticker']:<8} ${row['price']:<9.2f} "
                      f"{row['interest_score']:<8.2f} {row['premarket_volume']:>11,.0f} "
                      f"{pm_pct:>7.2f}% {row['atr_percent']:>7.2f}%")
            
            # Get the MarketFilter instance to explain scores
            filter_engine = bridge.filter_engine
            
            # Explain top stock
            if len(scan_results) > 0:
                top_stock = scan_results.iloc[0]
                print(f"\nScore Breakdown for {top_stock['ticker']}:")
                explanation = filter_engine.explain_score(top_stock)
                for component, details in explanation['components'].items():
                    print(f"  {component}: {details['contribution']:.2f} points "
                          f"(raw: {details['raw_value']})")
            
            # Ask if user wants to push to Supabase
            if get_user_confirmation("Would you like to push these results to Supabase?"):
                success = push_to_supabase(scan_results, scan_date)
                if success:
                    print("\n🎉 Data successfully stored in Supabase!")
                    print("   You can now track your scanner history over time.")
            else:
                print("\n⏭️  Skipping Supabase push.")
            
            # Create markdown report
            create_real_data_markdown(scan_results, summary, relaxed_criteria, len(all_tickers))
        
    except Exception as e:
        print(f"\nError during scan: {e}")
        import traceback
        traceback.print_exc()


def create_real_data_markdown(scan_results, summary, criteria, total_tickers_scanned):
    """Create a markdown file with real scan results."""
    import os
    from datetime import datetime
    
    # Create temp directory if it doesn't exist
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(temp_dir, f"market_filter_scan_{timestamp}.md")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("# Market Filter - S&P 500 Full Scan Results\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Data Source:** Polygon.io via SP500Bridge\n")
        f.write(f"**Scan Type:** Full S&P 500 Scan\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total S&P 500 Stocks | {total_tickers_scanned} |\n")
        f.write(f"| Stocks Scanned | {summary.get('total_scanned', total_tickers_scanned)} |\n")
        f.write(f"| Passed Filters | {summary.get('passed_filters', 0)} |\n")
        f.write(f"| Pass Rate | {summary.get('pass_rate', '0.0%')} |\n")
        if summary.get('avg_interest_score'):
            f.write(f"| Avg Interest Score | {summary.get('avg_interest_score'):.2f} |\n")
            f.write(f"| Score Std Dev | {summary.get('interest_score_std', 0):.2f} |\n")
        f.write("\n")
        
        # Filter Criteria
        f.write("## Filter Criteria\n\n")
        f.write("| Criterion | Value |\n")
        f.write("|-----------|-------|\n")
        f.write(f"| Price Range | ${criteria.min_price} - ${criteria.max_price} |\n")
        f.write(f"| Min Avg Volume | {criteria.min_avg_volume:,.0f} shares |\n")
        f.write(f"| Min PM Volume | {criteria.min_premarket_volume:,.0f} shares |\n")  # Add this line
        f.write(f"| Min PM Volume Ratio | {criteria.min_premarket_volume_ratio:.1%} |\n")
        f.write(f"| Min Dollar Volume | ${criteria.min_dollar_volume:,.0f} |\n")
        f.write(f"| Min ATR | ${criteria.min_atr} |\n")
        f.write(f"| Min ATR % | {criteria.min_atr_percent}% |\n")
        f.write("\n")
        
        # Results
        if scan_results.empty:
            f.write("## Results\n\n")
            f.write("❌ **No stocks passed all filters**\n\n")
            f.write("### Possible Reasons:\n\n")
            f.write("1. **Market Timing**: Pre-market data is only available 4:00 AM - 9:30 AM ET\n")
            f.write("2. **Weekend/Holiday**: No trading data on non-market days\n")
            f.write("3. **Data Latency**: Polygon data might have a slight delay\n")
            f.write("4. **Criteria**: Even relaxed criteria might be too strict\n\n")
            f.write("**Try running during pre-market hours for best results!**\n")
        else:
            f.write(f"## Top {min(50, len(scan_results))} Stocks by Interest Score\n\n")
            
            # Detailed results table
            f.write("| Rank | Ticker | Price | Score | PM Volume | PM % | Avg Volume | ATR | ATR % | $ Volume |\n")
            f.write("|:----:|:------:|------:|------:|----------:|-----:|-----------:|----:|------:|---------:|\n")
            
            for _, row in scan_results.head(50).iterrows():
                pm_vol_pct = (row['premarket_volume'] / row['avg_daily_volume'] * 100)
                
                # Highlight top 3
                ticker = row['ticker']
                if row['rank'] == 1:
                    ticker = f"🥇 **{ticker}**"
                elif row['rank'] == 2:
                    ticker = f"🥈 **{ticker}**"
                elif row['rank'] == 3:
                    ticker = f"🥉 **{ticker}**"
                
                # Score coloring
                score = row['interest_score']
                if score > 70:
                    score_str = f"**{score:.1f}**"
                else:
                    score_str = f"{score:.1f}"
                
                f.write(f"| {row['rank']} | {ticker} | ${row['price']:.2f} | {score_str} | ")
                f.write(f"{row['premarket_volume']:,.0f} | {pm_vol_pct:.2f}% | ")
                f.write(f"{row['avg_daily_volume']/1e6:.1f}M | ${row['atr']:.2f} | ")
                f.write(f"{row['atr_percent']:.2f}% | ${row['dollar_volume']/1e6:.1f}M |\n")
            
            # Score components breakdown
            if len(scan_results) > 0:
                f.write("\n## Score Components (Top Stock)\n\n")
                top = scan_results.iloc[0]
                f.write(f"**{top['ticker']}** - Total Score: {top['interest_score']:.2f}\n\n")
                f.write("| Component | Raw Value | Score | Weight | Contribution |\n")
                f.write("|-----------|-----------|------:|-------:|-------------:|\n")
                f.write(f"| PM Vol Ratio | {top['premarket_volume']/top['avg_daily_volume']:.2%} | ")
                f.write(f"{top['pm_vol_ratio_score']:.1f} | 40% | {top['pm_vol_ratio_score']*0.4:.1f} |\n")
                f.write(f"| ATR % | {top['atr_percent']:.2f}% | ")
                f.write(f"{top['atr_percent_score']:.1f} | 25% | {top['atr_percent_score']*0.25:.1f} |\n")
                f.write(f"| Dollar Vol | ${top['dollar_volume']:,.0f} | ")
                f.write(f"{top['dollar_vol_score']:.1f} | 20% | {top['dollar_vol_score']*0.2:.1f} |\n")
                f.write(f"| PM Vol Abs | {top['premarket_volume']:,.0f} | ")
                f.write(f"{top['pm_vol_abs_score']:.1f} | 10% | {top['pm_vol_abs_score']*0.1:.1f} |\n")
                f.write(f"| Price-ATR | {'Yes' if top['price_atr_bonus'] > 0 else 'No'} | ")
                f.write(f"{top['price_atr_bonus']:.1f} | 5% | {top['price_atr_bonus']*0.05:.1f} |\n")
            
            # Distribution stats
            if len(scan_results) > 10:
                f.write("\n## Score Distribution\n\n")
                f.write("| Range | Count | Percentage |\n")
                f.write("|-------|------:|-----------:|\n")
                
                ranges = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
                for low, high in ranges:
                    count = len(scan_results[(scan_results['interest_score'] >= low) & 
                                           (scan_results['interest_score'] < high)])
                    pct = count / len(scan_results) * 100
                    f.write(f"| {low}-{high} | {count} | {pct:.1f}% |\n")
        
        # Timestamp
        f.write("\n---\n")
        f.write(f"*Full S&P 500 scan completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    print(f"\n✅ Scan results markdown created: {output_path}")
    print(f"   Open in VS Code for formatted view")
    
    # Try to open in default editor
    try:
        if os.name == 'nt':  # Windows
            os.startfile(output_path)
    except:
        pass


if __name__ == "__main__":
    example_usage()