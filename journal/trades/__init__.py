# journal/trades/__init__.py
"""Trade processing module for the trading journal."""

# Handle both package and direct imports
try:
    from .parser import TradeParser
    from .processor import TradeProcessor
    from .models import Trade, Execution
    from .database import SupabaseClient
    from .plugin import TradePlugin
except ImportError:
    # When running scripts directly from the trades directory
    from parser import TradeParser
    from processor import TradeProcessor
    from models import Trade, Execution
    from database import SupabaseClient
    from plugin import TradePlugin

__all__ = ['TradeParser', 'TradeProcessor', 'Trade', 'Execution', 'SupabaseClient', 'TradePlugin']