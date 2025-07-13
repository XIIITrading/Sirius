# live_monitor/data/hist_request/fetch_coordinator.py
"""
Coordinator for managing all historical data fetchers
Handles parallel fetching and progress tracking
"""
import logging
from typing import Dict, List, Set, Optional
from datetime import datetime, timezone

from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from .ema_fetchers import M1EMAFetcher, M5EMAFetcher, M15EMAFetcher
from .market_structure_fetchers import (
    M1MarketStructureFetcher, 
    M5MarketStructureFetcher, 
    M15MarketStructureFetcher
)
from .trend_fetchers import (
    M1StatisticalTrendFetcher,
    M5StatisticalTrendFetcher,
    M15StatisticalTrendFetcher
)
from .zone_fetchers import HVNFetcher, OrderBlocksFetcher

logger = logging.getLogger(__name__)


class HistoricalFetchCoordinator(QObject):
    """
    Coordinates all historical data fetching
    Manages parallel fetches and tracks progress
    """
    
    # Signals
    fetch_started = pyqtSignal(str)         # symbol
    fetch_progress = pyqtSignal(dict)       # {'symbol': str, 'completed': int, 'total': int}
    all_fetches_completed = pyqtSignal(str) # symbol
    fetch_error = pyqtSignal(dict)          # {'symbol': str, 'errors': List[str]}
    
    # Individual data ready signals for different consumers
    ema_data_ready = pyqtSignal(dict)       # {'M1': df, 'M5': df, 'M15': df}
    structure_data_ready = pyqtSignal(dict)  # {'M1': df, 'M5': df, 'M15': df}
    trend_data_ready = pyqtSignal(dict)      # {'M1': df, 'M5': df, 'M15': df}
    zone_data_ready = pyqtSignal(dict)       # {'HVN': df, 'OrderBlocks': df}
    
    def __init__(self, rest_client):
        """
        Initialize coordinator with REST client
        
        Args:
            rest_client: PolygonRESTClient instance
        """
        super().__init__()
        self.rest_client = rest_client
        
        # Initialize all fetchers
        self._init_fetchers()
        
        # Tracking
        self.current_symbol: Optional[str] = None
        self.active_fetches: Set[str] = set()
        self.completed_fetches: Set[str] = set()
        self.fetch_errors: List[str] = []
        self.fetch_results: Dict[str, pd.DataFrame] = {}
        
        # Priority groups for phased fetching
        self.priority_groups = {
            'high': ['M1_EMA', 'M5_EMA'],  # Fastest signals
            'medium': ['M15_EMA', 'M1_MarketStructure', 'M5_MarketStructure'],
            'low': ['M15_MarketStructure', 'M1_StatisticalTrend', 
                   'M5_StatisticalTrend', 'M15_StatisticalTrend', 'HVN', 'OrderBlocks']
        }
        
        logger.info("HistoricalFetchCoordinator initialized")
    
    def _init_fetchers(self):
        """Initialize all fetcher instances"""
        # EMA Fetchers
        self.m1_ema_fetcher = M1EMAFetcher(self.rest_client)
        self.m5_ema_fetcher = M5EMAFetcher(self.rest_client)
        self.m15_ema_fetcher = M15EMAFetcher(self.rest_client)
        
        # Market Structure Fetchers
        self.m1_structure_fetcher = M1MarketStructureFetcher(self.rest_client)
        self.m5_structure_fetcher = M5MarketStructureFetcher(self.rest_client)
        self.m15_structure_fetcher = M15MarketStructureFetcher(self.rest_client)
        
        # Trend Fetchers
        self.m1_trend_fetcher = M1StatisticalTrendFetcher(self.rest_client)
        self.m5_trend_fetcher = M5StatisticalTrendFetcher(self.rest_client)
        self.m15_trend_fetcher = M15StatisticalTrendFetcher(self.rest_client)
        
        # Zone Fetchers
        self.hvn_fetcher = HVNFetcher(self.rest_client)
        self.order_blocks_fetcher = OrderBlocksFetcher(self.rest_client)
        
        # Create fetcher map
        self.fetchers = {
            'M1_EMA': self.m1_ema_fetcher,
            'M5_EMA': self.m5_ema_fetcher,
            'M15_EMA': self.m15_ema_fetcher,
            'M1_MarketStructure': self.m1_structure_fetcher,
            'M5_MarketStructure': self.m5_structure_fetcher,
            'M15_MarketStructure': self.m15_structure_fetcher,
            'M1_StatisticalTrend': self.m1_trend_fetcher,
            'M5_StatisticalTrend': self.m5_trend_fetcher,
            'M15_StatisticalTrend': self.m15_trend_fetcher,
            'HVN': self.hvn_fetcher,
            'OrderBlocks': self.order_blocks_fetcher
        }
        
        # Connect all fetcher signals
        for name, fetcher in self.fetchers.items():
            fetcher.fetch_completed.connect(self._on_fetch_completed)
            fetcher.fetch_failed.connect(self._on_fetch_failed)
    
    def fetch_all_for_symbol(self, symbol: str, priority_mode: bool = True):
        """
        Fetch all historical data for a symbol
        
        Args:
            symbol: Stock symbol
            priority_mode: If True, fetch in priority groups
        """
        if not symbol:
            return
        
        # Clear previous state
        self._reset_state()
        self.current_symbol = symbol
        
        # Clear all fetcher caches for new symbol
        for fetcher in self.fetchers.values():
            fetcher.clear_cache()
        
        # Track all fetches
        self.active_fetches = set(self.fetchers.keys())
        
        # Emit start signal
        self.fetch_started.emit(symbol)
        
        logger.info(f"Starting historical fetch for {symbol} with {len(self.active_fetches)} fetchers")
        
        if priority_mode:
            # Start with high priority fetches
            self._start_priority_group('high', symbol)
            # Others will be triggered as high priority completes
        else:
            # Start all fetches at once
            for name, fetcher in self.fetchers.items():
                fetcher.fetch_for_symbol(symbol)
    
    def _start_priority_group(self, priority: str, symbol: str):
        """Start fetches for a priority group"""
        group_fetchers = self.priority_groups.get(priority, [])
        
        for fetcher_name in group_fetchers:
            if fetcher_name in self.fetchers and fetcher_name in self.active_fetches:
                self.fetchers[fetcher_name].fetch_for_symbol(symbol)
        
        logger.info(f"Started {priority} priority fetches: {group_fetchers}")
    
    def _on_fetch_completed(self, data: dict):
        """Handle successful fetch completion"""
        fetcher_name = data.get('fetcher')
        symbol = data.get('symbol')
        
        # Verify this is for current symbol
        if symbol != self.current_symbol:
            logger.warning(f"Ignoring fetch result for {symbol}, current is {self.current_symbol}")
            return
        
        # Store result
        self.fetch_results[fetcher_name] = data['dataframe']
        self.completed_fetches.add(fetcher_name)
        
        logger.info(f"Completed fetch: {fetcher_name} ({len(data['dataframe'])} bars)")
        
        # Update progress
        self._update_progress()
        
        # Check if we should start next priority group
        self._check_priority_progression()
        
        # Emit specific data ready signals
        self._emit_data_ready_signals(fetcher_name)
        
        # Check if all complete
        if self.completed_fetches == self.active_fetches:
            self._on_all_fetches_completed()
    
    def _on_fetch_failed(self, error_data: dict):
        """Handle fetch failure"""
        fetcher_name = error_data.get('fetcher')
        error_msg = error_data.get('error')
        
        logger.error(f"Fetch failed: {fetcher_name} - {error_msg}")
        
        # Track error
        self.fetch_errors.append(f"{fetcher_name}: {error_msg}")
        self.completed_fetches.add(fetcher_name)  # Mark as complete even if failed
        
        # Update progress
        self._update_progress()
        
        # Check if all complete
        if self.completed_fetches == self.active_fetches:
            self._on_all_fetches_completed()
    
    def _update_progress(self):
        """Emit progress update"""
        progress = {
            'symbol': self.current_symbol,
            'completed': len(self.completed_fetches),
            'total': len(self.active_fetches),
            'percentage': (len(self.completed_fetches) / len(self.active_fetches) * 100) 
                         if self.active_fetches else 0
        }
        self.fetch_progress.emit(progress)
    
    def _check_priority_progression(self):
        """Check if we should start next priority group"""
        # Check high priority completion
        high_priority = set(self.priority_groups['high'])
        if high_priority.issubset(self.completed_fetches) and not hasattr(self, '_medium_started'):
            self._medium_started = True
            self._start_priority_group('medium', self.current_symbol)
        
        # Check medium priority completion
        medium_priority = set(self.priority_groups['medium'])
        if medium_priority.issubset(self.completed_fetches) and not hasattr(self, '_low_started'):
            self._low_started = True
            self._start_priority_group('low', self.current_symbol)
    
    def _emit_data_ready_signals(self, fetcher_name: str):
        """Emit specific data ready signals based on fetcher type"""
        # Check if we have complete sets for each type
        
        # EMA data
        if fetcher_name.endswith('_EMA'):
            ema_data = {}
            for timeframe in ['M1', 'M5', 'M15']:
                key = f"{timeframe}_EMA"
                if key in self.fetch_results:
                    ema_data[timeframe] = self.fetch_results[key]
            
            if ema_data:
                self.ema_data_ready.emit(ema_data)
        
        # Market Structure data
        elif fetcher_name.endswith('_MarketStructure'):
            structure_data = {}
            for timeframe in ['M1', 'M5', 'M15']:
                key = f"{timeframe}_MarketStructure"
                if key in self.fetch_results:
                    structure_data[timeframe] = self.fetch_results[key]
            
            if structure_data:
                self.structure_data_ready.emit(structure_data)
        
        # Statistical Trend data
        elif fetcher_name.endswith('_StatisticalTrend'):
            trend_data = {}
            for timeframe in ['M1', 'M5', 'M15']:
                key = f"{timeframe}_StatisticalTrend"
                if key in self.fetch_results:
                    trend_data[timeframe] = self.fetch_results[key]
            
            if trend_data:
                self.trend_data_ready.emit(trend_data)
        
        # Zone data
        elif fetcher_name in ['HVN', 'OrderBlocks']:
            zone_data = {}
            if 'HVN' in self.fetch_results:
                zone_data['HVN'] = self.fetch_results['HVN']
            if 'OrderBlocks' in self.fetch_results:
                zone_data['OrderBlocks'] = self.fetch_results['OrderBlocks']
            
            if zone_data:
                self.zone_data_ready.emit(zone_data)
    
    def _on_all_fetches_completed(self):
        """Handle completion of all fetches"""
        logger.info(f"All fetches completed for {self.current_symbol}")
        
        # Emit error signal if any errors
        if self.fetch_errors:
            self.fetch_error.emit({
                'symbol': self.current_symbol,
                'errors': self.fetch_errors
            })
        
        # Emit completion signal
        self.all_fetches_completed.emit(self.current_symbol)
    
    def _reset_state(self):
        """Reset internal state for new fetch"""
        self.active_fetches.clear()
        self.completed_fetches.clear()
        self.fetch_errors.clear()
        self.fetch_results.clear()
        
        # Reset priority flags
        if hasattr(self, '_medium_started'):
            delattr(self, '_medium_started')
        if hasattr(self, '_low_started'):
            delattr(self, '_low_started')
    
    def get_fetcher(self, name: str):
        """Get a specific fetcher instance"""
        return self.fetchers.get(name)
    
    def get_fetch_status(self) -> Dict:
        """Get current fetch status"""
        return {
            'symbol': self.current_symbol,
            'active': len(self.active_fetches),
            'completed': len(self.completed_fetches),
            'errors': len(self.fetch_errors),
            'results': list(self.fetch_results.keys()),
            'in_progress': self.active_fetches - self.completed_fetches
        }