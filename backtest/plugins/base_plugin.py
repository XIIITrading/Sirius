"""
Base plugin interface for backtest system.
All plugins must implement this interface.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional


class BacktestPlugin(ABC):
    """Base class for all backtest plugins"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the plugin"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version"""
        pass
    
    @abstractmethod
    async def run_analysis(self, symbol: str, entry_time: datetime, direction: str) -> Dict[str, Any]:
        """
        Run the complete analysis for this plugin.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            entry_time: Entry time in UTC
            direction: 'LONG' or 'SHORT'
            
        Returns:
            Dictionary containing:
            - plugin_name: str
            - timestamp: datetime
            - signal: dict with direction, strength, confidence
            - details: dict with plugin-specific details
            - display_data: dict with pre-formatted UI data
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get plugin configuration"""
        pass
    
    @abstractmethod
    def validate_inputs(self, symbol: str, entry_time: datetime, direction: str) -> bool:
        """Validate input parameters"""
        pass