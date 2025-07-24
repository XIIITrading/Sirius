# src/services/level_service.py

from typing import List, Optional, Dict, Any, Tuple
from datetime import date, datetime
from levels_config.database import supabase_connection
from src.models.level_models import PremarketLevel, RankedLevel, RankedLevelWithDetails

class LevelService:
    """
    Service class for managing levels in the database.
    Handles all database operations for the ranking system workflow.
    """
    
    def __init__(self):
        """Initialize with Supabase client."""
        self.client = supabase_connection.get_client()
    
    # ========== Read Operations (For Analyzer) ==========
    
    def get_premarket_levels_for_analysis(self, ticker: str, target_date: date) -> List[PremarketLevel]:
        """
        Get all active premarket levels for a ticker on a specific date.
        This is used by the analyzer to fetch levels from Notion->Supabase.
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            target_date: Date to analyze
            
        Returns:
            List of PremarketLevel objects ready for ranking
        """
        try:
            result = self.client.table('premarket_levels')\
                .select("*")\
                .eq('ticker', ticker.upper())\
                .eq('date', target_date.isoformat())\
                .eq('active', True)\
                .order('level_type', desc=False)\
                .order('price', desc=True)\
                .execute()
            
            levels = []
            for row in result.data:
                level = PremarketLevel(
                    id=row['id'],
                    date=date.fromisoformat(row['date']),
                    ticker=row['ticker'],
                    level_type=row['level_type'],
                    price=float(row['price']),
                    notes=row.get('notes'),
                    active=row['active'],
                    created_at=datetime.fromisoformat(row['created_at'].replace('Z', '+00:00'))
                )
                levels.append(level)
            
            print(f"📊 Retrieved {len(levels)} premarket levels for {ticker} on {target_date}")
            
            return levels
            
        except Exception as e:
            print(f"❌ Error retrieving premarket levels: {e}")
            return []
    
    def get_all_tickers_for_date(self, target_date: date) -> List[str]:
        """
        Get all unique tickers that have levels for a specific date.
        Used to process multiple tickers in batch.
        
        Args:
            target_date: Date to check
            
        Returns:
            List of unique ticker symbols
        """
        try:
            result = self.client.table('premarket_levels')\
                .select('ticker')\
                .eq('date', target_date.isoformat())\
                .eq('active', True)\
                .execute()
            
            # Extract unique tickers
            tickers = list(set(row['ticker'] for row in result.data))
            tickers.sort()
            
            print(f"📊 Found {len(tickers)} tickers with levels for {target_date}")
            return tickers
            
        except Exception as e:
            print(f"❌ Error retrieving tickers: {e}")
            return []
    
    def get_unique_tickers_with_levels(self, target_date: date) -> List[str]:
        """
        Get all unique tickers that have levels for a specific date.
        Alias for get_all_tickers_for_date for compatibility.
        """
        return self.get_all_tickers_for_date(target_date)
    
    # ========== Write Operations (For Analyzer Output) ==========
    
    def save_ranked_levels(self, ranked_levels: List[RankedLevel]) -> List[RankedLevel]:
        """
        Save ranked levels after analysis.
        This is the output from the Python analyzer.
        
        Args:
            ranked_levels: List of RankedLevel objects to save
            
        Returns:
            List of saved RankedLevel objects with IDs populated
        """
        saved_levels = []
        
        for level in ranked_levels:
            try:
                # Prepare data for insertion
                data = {
                    'premarket_level_id': level.premarket_level_id,
                    'date': level.date.isoformat(),
                    'ticker': level.ticker.upper(),
                    'rank': level.rank,
                    'confluence_score': level.confluence_score,
                    'zone_high': level.zone_high,
                    'zone_low': level.zone_low,
                    'tv_variable': level.tv_variable
                }
                
                # Add optional fields if present
                if level.current_price is not None:
                    data['current_price'] = level.current_price
                if level.atr_value is not None:
                    data['atr_value'] = level.atr_value
                
                # Insert into database
                result = self.client.table('ranked_levels').insert(data).execute()
                
                # Update the level object with generated values
                created_data = result.data[0]
                level.id = created_data['id']
                level.created_at = datetime.fromisoformat(created_data['created_at'].replace('Z', '+00:00'))
                
                saved_levels.append(level)
                
            except Exception as e:
                print(f"⚠️  Error saving ranked level: {e}")
                # Continue with other levels even if one fails
        
        print(f"✅ Saved {len(saved_levels)} ranked levels")
        return saved_levels
    
    def clear_existing_rankings(self, ticker: str, target_date: date) -> bool:
        """
        Clear existing rankings for a ticker/date before creating new ones.
        This ensures we don't have duplicate rankings.
        
        Args:
            ticker: Stock symbol
            target_date: Date of rankings to clear
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete existing rankings
            result = self.client.table('ranked_levels')\
                .delete()\
                .eq('ticker', ticker.upper())\
                .eq('date', target_date.isoformat())\
                .execute()
            
            print(f"🗑️  Cleared existing rankings for {ticker} on {target_date}")
            return True
            
        except Exception as e:
            print(f"❌ Error clearing rankings: {e}")
            return False
    
    # ========== Read Ranked Levels (For Notion Output) ==========
    
    def get_ranked_levels_with_details(self, ticker: str, target_date: date) -> List[RankedLevelWithDetails]:
        """
        Get ranked levels with full details for export to Notion.
        Joins ranked_levels with premarket_levels for complete information.
        
        Args:
            ticker: Stock symbol
            target_date: Date to retrieve
            
        Returns:
            List of RankedLevelWithDetails objects sorted by rank
        """
        try:
            # First get ranked levels
            ranked_result = self.client.table('ranked_levels')\
                .select("*")\
                .eq('ticker', ticker.upper())\
                .eq('date', target_date.isoformat())\
                .order('rank', desc=False)\
                .execute()
            
            if not ranked_result.data:
                print(f"⚠️  No ranked levels found for {ticker} on {target_date}")
                return []
            
            # Get all premarket level IDs
            premarket_ids = [row['premarket_level_id'] for row in ranked_result.data]
            
            # Fetch corresponding premarket levels
            premarket_result = self.client.table('premarket_levels')\
                .select("*")\
                .in_('id', premarket_ids)\
                .execute()
            
            # Create lookup dictionary
            premarket_lookup = {row['id']: row for row in premarket_result.data}
            
            # Combine the data
            detailed_levels = []
            for ranked_row in ranked_result.data:
                premarket_row = premarket_lookup.get(ranked_row['premarket_level_id'])
                
                if premarket_row:
                    # Create RankedLevel object
                    ranked_level = RankedLevel(
                        id=ranked_row['id'],
                        premarket_level_id=ranked_row['premarket_level_id'],
                        date=date.fromisoformat(ranked_row['date']),
                        ticker=ranked_row['ticker'],
                        rank=ranked_row['rank'],
                        confluence_score=float(ranked_row['confluence_score']),
                        zone_high=float(ranked_row['zone_high']),
                        zone_low=float(ranked_row['zone_low']),
                        tv_variable=ranked_row['tv_variable'],
                        current_price=float(ranked_row['current_price']) if ranked_row.get('current_price') else None,
                        atr_value=float(ranked_row['atr_value']) if ranked_row.get('atr_value') else None,
                        created_at=datetime.fromisoformat(ranked_row['created_at'].replace('Z', '+00:00'))
                    )
                    
                    # Create PremarketLevel object
                    premarket_level = PremarketLevel(
                        id=premarket_row['id'],
                        date=date.fromisoformat(premarket_row['date']),
                        ticker=premarket_row['ticker'],
                        level_type=premarket_row['level_type'],
                        price=float(premarket_row['price']),
                        notes=premarket_row.get('notes'),
                        active=premarket_row['active'],
                        created_at=datetime.fromisoformat(premarket_row['created_at'].replace('Z', '+00:00'))
                    )
                    
                    # Create combined object
                    detailed = RankedLevelWithDetails(
                        ranked_level=ranked_level,
                        premarket_level=premarket_level
                    )
                    detailed_levels.append(detailed)
            
            print(f"📊 Retrieved {len(detailed_levels)} ranked levels with details")
            return detailed_levels
            
        except Exception as e:
            print(f"❌ Error retrieving ranked levels: {e}")
            return []
    
    # ========== Utility Methods ==========
    
    def get_analysis_summary(self, target_date: date) -> Dict[str, Any]:
        """
        Get a summary of analysis status for a given date.
        Shows which tickers have been entered and which have been analyzed.
        
        Args:
            target_date: Date to check
            
        Returns:
            Dictionary with summary information
        """
        try:
            # Get tickers with premarket levels
            premarket_result = self.client.table('premarket_levels')\
                .select('ticker')\
                .eq('date', target_date.isoformat())\
                .eq('active', True)\
                .execute()
            
            premarket_tickers = list(set(row['ticker'] for row in premarket_result.data))
            
            # Get tickers with ranked levels  
            ranked_result = self.client.table('ranked_levels')\
                .select('ticker')\
                .eq('date', target_date.isoformat())\
                .execute()
            
            ranked_tickers = list(set(row['ticker'] for row in ranked_result.data))
            
            # Calculate summary
            summary = {
                'date': target_date.isoformat(),
                'tickers_with_levels': sorted(premarket_tickers),
                'tickers_analyzed': sorted(ranked_tickers),
                'tickers_pending': sorted(set(premarket_tickers) - set(ranked_tickers)),
                'total_entered': len(premarket_tickers),
                'total_analyzed': len(ranked_tickers),
                'total_pending': len(premarket_tickers) - len(ranked_tickers)
            }
            
            return summary
            
        except Exception as e:
            print(f"❌ Error getting analysis summary: {e}")
            return {}