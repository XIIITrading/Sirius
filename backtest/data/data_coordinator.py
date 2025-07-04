# backtest/data/data_coordinator.py
"""
Module: Data Coordinator
Purpose: Coordinates data fetching between calculation modules and PolygonDataManager
Features: Request aggregation, plugin tracking, centralized data distribution
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import pandas as pd

from .polygon_data_manager import PolygonDataManager
from .request_aggregator import RequestAggregator, DataNeed, DataType

logger = logging.getLogger(__name__)


class DataCoordinator:
    """
    Central coordinator for all data requests in the backtesting system.
    
    This class acts as the interface between calculation modules and the data layer,
    ensuring efficient data fetching through request aggregation.
    """
    
    def __init__(self, polygon_data_manager: PolygonDataManager):
        """
        Initialize the coordinator.
        
        Args:
            polygon_data_manager: Configured PolygonDataManager instance
        """
        self.data_manager = polygon_data_manager
        self.aggregator = RequestAggregator(
            data_manager=polygon_data_manager,
            extend_window_pct=0.1
        )
        
        # Track active calculation modules
        self.registered_modules = {}
        
        logger.info("DataCoordinator initialized")
    
    def register_module(self, module_name: str, module_instance: Any):
        """Register a calculation module"""
        self.registered_modules[module_name] = module_instance
        logger.info(f"Registered module: {module_name}")
    
    def collect_data_needs(self, symbol: str, entry_time: datetime, 
                          direction: str = "LONG") -> List[DataNeed]:
        """
        Collect data needs from all registered modules.
        
        This method would be called by each module to register their needs,
        or modules could call it directly through a standard interface.
        """
        needs = []
        
        # Example of how modules might specify their needs
        # In practice, each module would have a get_data_needs() method
        
        # Simulate collection from various modules based on the architecture doc
        if "TrendAnalysis" in self.registered_modules or True:  # Simulating
            # 1/5/15-min statistical trends
            for timeframe, lookback_hours in [("1min", 2), ("5min", 4), ("15min", 6)]:
                needs.append(DataNeed(
                    module_name=f"TrendAnalysis_{timeframe}",
                    symbol=symbol,
                    data_type=DataType.BARS,
                    timeframe=timeframe,
                    start_time=entry_time - timedelta(hours=lookback_hours),
                    end_time=entry_time + timedelta(hours=1),  # Forward looking
                    priority=7
                ))
        
        if "MarketStructure" in self.registered_modules or True:
            # Fractal-based BOS/CHoCH detection
            needs.append(DataNeed(
                module_name="MarketStructure",
                symbol=symbol,
                data_type=DataType.BARS,
                timeframe="5min",
                start_time=entry_time - timedelta(hours=4),
                end_time=entry_time + timedelta(hours=1),
                priority=8
            ))
        
        if "OrderFlow" in self.registered_modules or True:
            # Large order detection, bid/ask imbalance
            needs.extend([
                DataNeed(
                    module_name="OrderFlow_Trades",
                    symbol=symbol,
                    data_type=DataType.TRADES,
                    timeframe="tick",
                    start_time=entry_time - timedelta(minutes=30),
                    end_time=entry_time + timedelta(minutes=15),
                    priority=9
                ),
                DataNeed(
                    module_name="OrderFlow_Quotes",
                    symbol=symbol,
                    data_type=DataType.QUOTES,
                    timeframe="tick",
                    start_time=entry_time - timedelta(minutes=30),
                    end_time=entry_time + timedelta(minutes=15),
                    priority=9
                )
            ])
        
        if "VolumeAnalysis" in self.registered_modules or True:
            # HVN engine needs 14 days!
            needs.extend([
                DataNeed(
                    module_name="VolumeAnalysis_HVN",
                    symbol=symbol,
                    data_type=DataType.BARS,
                    timeframe="5min",
                    start_time=entry_time - timedelta(days=14),
                    end_time=entry_time,
                    priority=6
                ),
                DataNeed(
                    module_name="VolumeAnalysis_M1",
                    symbol=symbol,
                    data_type=DataType.BARS,
                    timeframe="1min",
                    start_time=entry_time - timedelta(hours=2),
                    end_time=entry_time + timedelta(hours=1),
                    priority=8
                )
            ])
        
        return needs
    
    async def fetch_all_module_data(self, symbol: str, entry_time: datetime,
                                   direction: str = "LONG") -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Main method to fetch all data for registered modules.
        
        Returns:
            Dict mapping module_name -> data_key -> DataFrame
        """
        # Clear any previous needs
        self.aggregator.clear_needs()
        
        # Collect needs from all modules
        needs = self.collect_data_needs(symbol, entry_time, direction)
        
        # Register all needs with aggregator
        self.aggregator.register_needs(needs)
        
        # Set current plugin for tracking
        self.data_manager.set_current_plugin("DataCoordinator")
        
        # Fetch all data concurrently
        logger.info(f"Fetching data for {len(needs)} module needs...")
        module_data = await self.aggregator.fetch_all_data()
        
        return module_data
    
    def get_summary_report(self) -> str:
        """Get a summary report of the data fetching operation"""
        lines = []
        lines.append("=" * 80)
        lines.append("DATA COORDINATOR SUMMARY")
        lines.append("=" * 80)
        
        # Aggregator stats
        agg_stats = self.aggregator.get_stats()
        lines.append(f"\nAggregation Statistics:")
        lines.append(f"  Total needs: {agg_stats['total_needs']}")
        lines.append(f"  Aggregated requests: {agg_stats['aggregated_requests']}")
        lines.append(f"  API calls saved: {agg_stats['api_calls_saved']}")
        lines.append(f"  Efficiency: {(agg_stats['api_calls_saved'] / max(1, agg_stats['total_needs']) * 100):.1f}%")
        
        # Data manager stats
        dm_stats = self.data_manager.get_cache_stats()
        lines.append(f"\nCache Statistics:")
        lines.append(f"  Memory cache hit rate: {dm_stats['memory_cache']['hit_rate']:.1f}%")
        lines.append(f"  File cache size: {dm_stats['file_cache']['total_size_mb']:.1f} MB")
        lines.append(f"  API calls made: {dm_stats['api_stats']['api_calls']}")
        
        return "\n".join(lines)