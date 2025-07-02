# journal/trades/plugin.py
"""Main plugin interface for trade processing."""

from pathlib import Path
from datetime import date
from typing import List, Optional

# Handle both package and script imports
try:
    from .parser import TradeParser
    from .processor import TradeProcessor
    from .database import SupabaseClient
    from .models import Trade
except ImportError:
    from parser import TradeParser
    from processor import TradeProcessor
    from database import SupabaseClient
    from models import Trade


class TradePlugin:
    """Main plugin for processing trades."""
    
    def __init__(self, supabase_url: Optional[str] = None, 
                 supabase_key: Optional[str] = None):
        """Initialize the trade plugin."""
        self.processor = TradeProcessor()
        self.db = SupabaseClient(supabase_url, supabase_key) if supabase_url and supabase_key else None
    
    def process_file(self, file_path: Path, trade_date: Optional[date] = None) -> List[Trade]:
        """
        Process a broker statement file and return trades.
        
        Args:
            file_path: Path to the CSV file
            trade_date: The date when these trades occurred (for proper timezone conversion)
        """
        # Create parser with the specific trade date
        parser = TradeParser(trade_date=trade_date)
        
        # Parse executions from CSV
        executions = parser.parse_csv(file_path)
        
        # Process into trades
        trades = self.processor.process_executions(executions)
        
        return trades
    
    def save_trades(self, trades: List[Trade]) -> dict:
        """Save trades to Supabase."""
        if not self.db:
            return {'success': False, 'error': 'Database not configured'}
        return self.db.insert_trades(trades)
    
    def process_and_save(self, file_path: Path, trade_date: Optional[date] = None) -> dict:
        """
        Process file and save trades to database.
        
        Args:
            file_path: Path to the CSV file
            trade_date: The date when these trades occurred
        """
        trades = self.process_file(file_path, trade_date)
        result = self.save_trades(trades) if self.db else {'success': False, 'error': 'No database'}
        
        return {
            'trades_processed': len(trades),
            'database_result': result
        }