# journal/trades/database.py
"""Database integration for trades."""

import os
from typing import List, Optional

# Handle both package and script imports
try:
    from .models import Trade
except ImportError:
    from models import Trade

# Only import supabase if it's being used
def get_supabase_client():
    try:
        from supabase import create_client, Client
        return create_client, Client
    except ImportError:
        return None, None


class SupabaseClient:
    """Client for interacting with Supabase."""
    
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        """Initialize Supabase client."""
        self.url = url or os.environ.get('SUPABASE_URL')
        self.key = key or os.environ.get('SUPABASE_KEY')
        
        if not self.url or not self.key:
            raise ValueError("Supabase URL and key must be provided")
        
        create_client, Client = get_supabase_client()
        if not create_client:
            raise ImportError("Supabase package not installed. Run: pip install supabase")
            
        self.client: Client = create_client(self.url, self.key)
        self.table_name = 'trades'
    
    def insert_trades(self, trades: List[Trade]) -> dict:
        """Insert trades into Supabase."""
        data = [trade.to_dict() for trade in trades]
        
        try:
            result = self.client.table(self.table_name).insert(data).execute()
            return {'success': True, 'count': len(result.data)}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_trades(self, start_date: str = None, end_date: str = None) -> List[dict]:
        """Retrieve trades from Supabase with optional date filtering."""
        query = self.client.table(self.table_name).select("*")
        
        if start_date:
            query = query.gte('entry_time', start_date)
        if end_date:
            query = query.lte('entry_time', end_date)
            
        result = query.execute()
        return result.data