# modules/data/supabase_client.py
"""
Module: Supabase Client for Premarket Scanner Results
Purpose: Handle all database operations for premarket scan data
Dependencies: supabase, python-dotenv
"""

"""
- Full Supabase connectivity with environment variable support
- All the query methods we discussed
- Comprehensive error handling and logging
- UTC time handling throughout
- A standalone test script to verify functionality
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Any
import logging
from dotenv import load_dotenv

try:
    from supabase import create_client, Client
except ImportError:
    print("Error: supabase package not installed. Run: pip install supabase")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SupabaseClient:
    """
    Client for interacting with Supabase premarket_scans table.
    All timestamps are handled in UTC.
    """
    
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        """
        Initialize Supabase client.
        
        Args:
            url: Supabase project URL (optional, defaults to env var)
            key: Supabase API key (optional, defaults to env var)
        """
        # Load environment variables
        load_dotenv()
        
        # Get credentials
        self.url = url or os.getenv('SUPABASE_URL')
        self.key = key or os.getenv('SUPABASE_KEY')
        
        if not self.url or not self.key:
            raise ValueError(
                "Supabase credentials not found. Please set SUPABASE_URL and "
                "SUPABASE_KEY environment variables or pass them to the constructor."
            )
        
        # Initialize client
        try:
            self.client: Client = create_client(self.url, self.key)
            logger.info("Supabase client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise
    
    def get_today_scans(self, 
                       passed_filters_only: bool = True,
                       sort_by: str = 'rank',
                       ascending: bool = True) -> List[Dict[str, Any]]:
        """
        Get today's premarket scan results.
        
        Args:
            passed_filters_only: Only return scans that passed filters
            sort_by: Column to sort by ('rank' or 'interest_score')
            ascending: Sort order
            
        Returns:
            List of scan results
        """
        # Get today's date in UTC
        today = datetime.now(timezone.utc).date()
        logger.info(f"Fetching scans for today: {today}")
        
        return self.get_scans_by_date(
            scan_date=today,
            passed_filters_only=passed_filters_only,
            sort_by=sort_by,
            ascending=ascending
        )
    
    def get_scans_by_date(self,
                         scan_date: Any,
                         passed_filters_only: bool = True,
                         sort_by: str = 'rank',
                         ascending: bool = True) -> List[Dict[str, Any]]:
        """
        Get premarket scan results for a specific date.
        
        Args:
            scan_date: Date to query (date object or string)
            passed_filters_only: Only return scans that passed filters
            sort_by: Column to sort by
            ascending: Sort order
            
        Returns:
            List of scan results
        """
        try:
            # Convert date to string if needed
            if hasattr(scan_date, 'strftime'):
                date_str = scan_date.strftime('%Y-%m-%d')
            else:
                date_str = str(scan_date)
            
            logger.info(f"Querying scans for date: {date_str}")
            
            # Build query
            query = self.client.table('premarket_scans').select('*')
            query = query.eq('scan_date', date_str)
            
            if passed_filters_only:
                query = query.eq('passed_filters', True)
            
            # Apply sorting
            query = query.order(sort_by, desc=not ascending)
            
            # Execute query
            response = query.execute()
            
            logger.info(f"Retrieved {len(response.data)} scans for {date_str}")
            return response.data
            
        except Exception as e:
            logger.error(f"Error fetching scans for {scan_date}: {e}")
            return []
    
    def get_recent_scans(self,
                        days: int = 5,
                        passed_filters_only: bool = True,
                        sort_by: str = 'scan_date',
                        include_today: bool = True) -> List[Dict[str, Any]]:
        """
        Get premarket scan results from the last N days.
        
        Args:
            days: Number of days to look back
            passed_filters_only: Only return scans that passed filters
            sort_by: Column to sort by
            include_today: Whether to include today's date
            
        Returns:
            List of scan results
        """
        try:
            # Calculate date range
            end_date = datetime.now(timezone.utc).date()
            if not include_today:
                end_date = end_date - timedelta(days=1)
            
            start_date = end_date - timedelta(days=days-1)
            
            logger.info(f"Querying scans from {start_date} to {end_date}")
            
            # Build query
            query = self.client.table('premarket_scans').select('*')
            query = query.gte('scan_date', start_date.strftime('%Y-%m-%d'))
            query = query.lte('scan_date', end_date.strftime('%Y-%m-%d'))
            
            if passed_filters_only:
                query = query.eq('passed_filters', True)
            
            # Apply sorting - for multiple days, also sort by rank within each date
            if sort_by == 'scan_date':
                query = query.order('scan_date', desc=True).order('rank', desc=False)
            else:
                query = query.order(sort_by, desc=sort_by == 'interest_score')
            
            # Execute query
            response = query.execute()
            
            logger.info(f"Retrieved {len(response.data)} scans for {days} days")
            return response.data
            
        except Exception as e:
            logger.error(f"Error fetching recent scans: {e}")
            return []
    
    def get_scan_by_ticker(self,
                          ticker: str,
                          scan_date: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """
        Get scan data for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            scan_date: Specific date (optional, defaults to today)
            
        Returns:
            Scan data for the ticker or None if not found
        """
        try:
            # Default to today if no date provided
            if scan_date is None:
                scan_date = datetime.now(timezone.utc).date()
            
            # Convert date to string if needed
            if hasattr(scan_date, 'strftime'):
                date_str = scan_date.strftime('%Y-%m-%d')
            else:
                date_str = str(scan_date)
            
            logger.info(f"Querying scan for {ticker} on {date_str}")
            
            # Build query
            query = self.client.table('premarket_scans').select('*')
            query = query.eq('ticker', ticker.upper())
            query = query.eq('scan_date', date_str)
            
            # Execute query
            response = query.execute()
            
            if response.data:
                logger.info(f"Found scan data for {ticker}")
                return response.data[0]
            else:
                logger.info(f"No scan data found for {ticker} on {date_str}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching scan for {ticker}: {e}")
            return None
    
    def get_top_movers(self,
                      scan_date: Optional[Any] = None,
                      limit: int = 10,
                      min_interest_score: float = 50.0) -> List[Dict[str, Any]]:
        """
        Get top movers based on interest score.
        
        Args:
            scan_date: Date to query (optional, defaults to today)
            limit: Maximum number of results
            min_interest_score: Minimum interest score threshold
            
        Returns:
            List of top movers
        """
        try:
            # Default to today if no date provided
            if scan_date is None:
                scan_date = datetime.now(timezone.utc).date()
            
            # Convert date to string if needed
            if hasattr(scan_date, 'strftime'):
                date_str = scan_date.strftime('%Y-%m-%d')
            else:
                date_str = str(scan_date)
            
            logger.info(f"Querying top {limit} movers for {date_str}")
            
            # Build query
            query = self.client.table('premarket_scans').select('*')
            query = query.eq('scan_date', date_str)
            query = query.eq('passed_filters', True)
            query = query.gte('interest_score', min_interest_score)
            query = query.order('interest_score', desc=True)
            query = query.limit(limit)
            
            # Execute query
            response = query.execute()
            
            logger.info(f"Retrieved {len(response.data)} top movers")
            return response.data
            
        except Exception as e:
            logger.error(f"Error fetching top movers: {e}")
            return []
    
    def test_connection(self) -> bool:
        """
        Test the Supabase connection.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Try a simple query
            query = self.client.table('premarket_scans').select('ticker').limit(1)
            response = query.execute()
            logger.info("Supabase connection test successful")
            return True
        except Exception as e:
            logger.error(f"Supabase connection test failed: {e}")
            return False


# ============= STANDALONE TEST SCRIPT =============
if __name__ == "__main__":
    print("=== Testing Supabase Client ===\n")
    
    try:
        # Initialize client
        print("1. Initializing Supabase client...")
        client = SupabaseClient()
        print("✓ Client initialized successfully\n")
        
        # Test connection
        print("2. Testing connection...")
        if client.test_connection():
            print("✓ Connection test passed\n")
        else:
            print("✗ Connection test failed\n")
            sys.exit(1)
        
        # Get today's scans
        print("3. Fetching today's scans...")
        today_scans = client.get_today_scans()
        print(f"✓ Found {len(today_scans)} scans for today")
        
        if today_scans:
            # Display first 5
            print("\nTop 5 by rank:")
            print(f"{'Rank':<6} {'Ticker':<8} {'Price':<10} {'Interest':<10} {'PM Volume':<15}")
            print("-" * 60)
            
            for scan in today_scans[:5]:
                print(f"{scan['rank']:<6} {scan['ticker']:<8} "
                      f"${scan['price']:<9.2f} {scan['interest_score']:<10.2f} "
                      f"{scan['premarket_volume']:>14,}")
        
        # Get specific ticker
        print("\n4. Testing ticker lookup...")
        test_ticker = today_scans[0]['ticker'] if today_scans else 'AAPL'
        ticker_data = client.get_scan_by_ticker(test_ticker)
        
        if ticker_data:
            print(f"✓ Found data for {test_ticker}:")
            print(f"  - Price: ${ticker_data['price']}")
            print(f"  - Interest Score: {ticker_data['interest_score']}")
            print(f"  - ATR%: {ticker_data['atr_percent']}%")
        
        # Get recent scans
        print("\n5. Testing recent scans (last 3 days)...")
        recent_scans = client.get_recent_scans(days=3)
        
        # Count by date
        dates_count = {}
        for scan in recent_scans:
            date = scan['scan_date']
            dates_count[date] = dates_count.get(date, 0) + 1
        
        print(f"✓ Found {len(recent_scans)} total scans")
        for date, count in sorted(dates_count.items(), reverse=True):
            print(f"  - {date}: {count} scans")
        
        # Get top movers
        print("\n6. Testing top movers...")
        top_movers = client.get_top_movers(limit=5)
        
        if top_movers:
            print(f"✓ Top {len(top_movers)} movers by interest score:")
            for i, mover in enumerate(top_movers, 1):
                print(f"  {i}. {mover['ticker']}: Score={mover['interest_score']:.2f}, "
                      f"Rank={mover['rank']}")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()