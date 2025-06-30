# backtest/debug/base_debug.py
"""
Base class for calculation-specific debug tools
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class BaseCalculationDebugger(ABC):
    """Base class for calculation debugging"""
    
    def __init__(self, symbol: str, entry_time: datetime, 
                 lookback_minutes: int = 120, forward_minutes: int = 5):
        """
        Initialize debugger
        
        Args:
            symbol: Stock symbol
            entry_time: Entry time (UTC)
            lookback_minutes: Minutes of historical data to analyze
            forward_minutes: Minutes of forward data (for validation)
        """
        self.symbol = symbol
        self.entry_time = entry_time.replace(tzinfo=timezone.utc) if entry_time.tzinfo is None else entry_time
        self.lookback_minutes = lookback_minutes
        self.forward_minutes = forward_minutes
        
    @property
    @abstractmethod
    def calculation_name(self) -> str:
        """Name of the calculation being debugged"""
        pass
    
    @property
    @abstractmethod
    def timeframe(self) -> str:
        """Timeframe for this calculation"""
        pass
    
    @abstractmethod
    async def run_debug(self) -> Dict[str, Any]:
        """Run the debug analysis"""
        pass
    
    @abstractmethod
    def display_results(self, results: Dict[str, Any]) -> None:
        """Display debug results in readable format"""
        pass
    
    def save_debug_report(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """Save debug report to file"""
        if not output_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"debug_{self.calculation_name}_{self.symbol}_{timestamp}.json"
        
        # Convert datetime objects to strings
        def serialize(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=serialize)
        
        logger.info(f"\nDebug report saved to: {output_path}")
        return output_path
    
    def format_price_bar(self, time: str, o: float, h: float, l: float, c: float, 
                        v: int, markers: List[str] = None) -> str:
        """Format a price bar for display"""
        bar_str = f"{time:^10} {o:^8.2f} {h:^8.2f} {l:^8.2f} {c:^8.2f} {v:^10,d}"
        if markers:
            bar_str += " " + " ".join(markers)
        return bar_str