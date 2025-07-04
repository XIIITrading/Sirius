# backtest/data/data_coordinator.py
"""
Module: Data Coordinator
Purpose: Coordinates data requests from multiple calculation modules
Features: Module registration, need aggregation, concurrent fetching, report generation
"""

import logging
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import defaultdict
import json
from pathlib import Path

from .request_aggregator import RequestAggregator, DataNeed, DataType
from .polygon_data_manager import PolygonDataManager

logger = logging.getLogger(__name__)


class DataCoordinator:
    """
    Coordinates data requests from multiple calculation modules.
    
    This is the main interface between calculation modules and the data layer.
    It handles:
    1. Module registration
    2. Data need collection
    3. Aggregated fetching
    4. Data distribution
    5. Performance reporting
    """
    
    def __init__(self, data_manager: Union[PolygonDataManager, Any]):
        """
        Initialize the coordinator.
        
        Args:
            data_manager: PolygonDataManager or ProtectedDataManager instance
        """
        self.data_manager = data_manager
        self.aggregator = RequestAggregator(data_manager=data_manager)
        self.registered_modules: Dict[str, Dict[str, Any]] = {}
        self.module_data_needs: Dict[str, List[DataNeed]] = defaultdict(list)
        
        logger.info("DataCoordinator initialized")
    
    def register_module(self, module_name: str, config: Dict[str, Any]):
        """
        Register a calculation module with the coordinator.
        
        Args:
            module_name: Name of the module
            config: Module configuration (timeframes, data types needed, etc.)
        """
        self.registered_modules[module_name] = config
        logger.info(f"Registered module: {module_name}")
    
    def unregister_module(self, module_name: str):
        """Unregister a module"""
        if module_name in self.registered_modules:
            del self.registered_modules[module_name]
            if module_name in self.module_data_needs:
                del self.module_data_needs[module_name]
            logger.info(f"Unregistered module: {module_name}")
    
    async def fetch_all_module_data(self, symbol: str, entry_time: datetime, 
                                   direction: str = "LONG") -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch all data needed by registered modules.
        
        Args:
            symbol: Trading symbol
            entry_time: Entry time for the trade
            direction: Trade direction (LONG/SHORT)
            
        Returns:
            Dict mapping module_name -> data_key -> DataFrame
        """
        # Clear previous needs
        self.aggregator.clear_needs()
        self.module_data_needs.clear()
        
        # Collect needs from all modules
        self._collect_module_needs(symbol, entry_time, direction)
        
        # Register all needs with aggregator
        all_needs = []
        for module_needs in self.module_data_needs.values():
            all_needs.extend(module_needs)
        
        self.aggregator.register_needs(all_needs)
        
        # Fetch all data
        logger.info(f"Fetching data for {len(self.registered_modules)} modules")
        module_data = await self.aggregator.fetch_all_data()
        
        return module_data
    
    def _collect_module_needs(self, symbol: str, entry_time: datetime, direction: str):
        """
        Collect data needs from all registered modules.
        
        This method defines the standard data requirements for each module type.
        """
        # Ensure timezone aware
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        
        for module_name, config in self.registered_modules.items():
            needs = []
            
            if module_name == "TrendAnalysis":
                # Trend analysis needs multiple timeframes
                needs.extend([
                    DataNeed(module_name, symbol, DataType.BARS, "1min",
                            entry_time - timedelta(hours=2), entry_time + timedelta(hours=1)),
                    DataNeed(module_name, symbol, DataType.BARS, "5min",
                            entry_time - timedelta(hours=4), entry_time + timedelta(hours=1)),
                    DataNeed(module_name, symbol, DataType.BARS, "15min",
                            entry_time - timedelta(hours=8), entry_time + timedelta(hours=2)),
                ])
            
            elif module_name == "MarketStructure":
                # Market structure needs bars and volume
                needs.extend([
                    DataNeed(module_name, symbol, DataType.BARS, "1min",
                            entry_time - timedelta(hours=6), entry_time + timedelta(minutes=30)),
                    DataNeed(module_name, symbol, DataType.BARS, "5min",
                            entry_time - timedelta(hours=12), entry_time + timedelta(hours=1)),
                ])
            
            elif module_name == "OrderFlow":
                # Order flow needs tick data
                needs.extend([
                    DataNeed(module_name, symbol, DataType.TRADES, "tick",
                            entry_time - timedelta(minutes=30), entry_time + timedelta(minutes=15)),
                    DataNeed(module_name, symbol, DataType.QUOTES, "tick",
                            entry_time - timedelta(minutes=30), entry_time + timedelta(minutes=15)),
                ])
            
            elif module_name == "VolumeAnalysis":
                # Volume analysis needs bars and trades
                needs.extend([
                    DataNeed(module_name, symbol, DataType.BARS, "1min",
                            entry_time - timedelta(hours=3), entry_time + timedelta(minutes=30)),
                    DataNeed(module_name, symbol, DataType.TRADES, "tick",
                            entry_time - timedelta(minutes=15), entry_time + timedelta(minutes=5)),
                ])
            
            else:
                # Default needs for unknown modules
                needs.append(
                    DataNeed(module_name, symbol, DataType.BARS, "1min",
                            entry_time - timedelta(hours=1), entry_time)
                )
            
            # Store needs for this module
            self.module_data_needs[module_name] = needs
    
    def get_module_needs(self, module_name: str) -> List[DataNeed]:
        """Get the data needs for a specific module"""
        return self.module_data_needs.get(module_name, [])
    
    def get_summary_report(self) -> str:
        """Get a summary report of the coordinator's activity"""
        lines = []
        lines.append("=" * 60)
        lines.append("DATA COORDINATOR SUMMARY")
        lines.append("=" * 60)
        
        lines.append(f"\nRegistered Modules: {len(self.registered_modules)}")
        for module_name in self.registered_modules:
            needs = self.module_data_needs.get(module_name, [])
            lines.append(f"  - {module_name}: {len(needs)} data needs")
        
        # Get aggregator stats
        stats = self.aggregator.get_stats()
        lines.append(f"\nAggregation Statistics:")
        lines.append(f"  Total Needs: {stats['total_needs']}")
        lines.append(f"  Aggregated Requests: {stats['aggregated_requests']}")
        lines.append(f"  API Calls Saved: {stats['api_calls_saved']}")
        lines.append(f"  Data Points Fetched: {stats['data_points_fetched']:,}")
        
        return "\n".join(lines)
    
    def generate_data_report(self) -> Tuple[str, str]:
        """Generate comprehensive data usage report"""
        return self.data_manager.generate_data_report()


# Example usage
async def main():
    import asyncio
    
    # Option 1: Direct usage
    polygon_manager = PolygonDataManager(
        api_key='your_key',
        cache_dir='./cache',
        memory_cache_size=100,
        file_cache_hours=24
    )
    
    # Option 2: With circuit breaker protection
    from .protected_data_manager import ProtectedDataManager
    from .circuit_breaker import CircuitBreaker
    
    protected_manager = ProtectedDataManager(
        polygon_data_manager=polygon_manager,
        circuit_breaker_config={
            'failure_threshold': 0.5,
            'consecutive_failures': 5,
            'recovery_timeout': 60,
            'rate_limits': {
                'bars': {'per_minute': 100, 'burst': 10},
                'trades': {'per_minute': 50, 'burst': 5},
                'quotes': {'per_minute': 50, 'burst': 5}
            }
        }
    )
    
    # Create coordinator
    coordinator = DataCoordinator(protected_manager)  # or polygon_manager
    
    # Register modules (in real usage, these would be actual module instances)
    coordinator.register_module("TrendAnalysis", {})
    coordinator.register_module("MarketStructure", {})
    coordinator.register_module("OrderFlow", {})
    coordinator.register_module("VolumeAnalysis", {})
    
    # Fetch data for all modules
    symbol = 'AAPL'
    entry_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    
    module_data = await coordinator.fetch_all_module_data(symbol, entry_time, "LONG")
    
    # Print results
    for module_name, data_dict in module_data.items():
        print(f"\n{module_name}:")
        for data_key, df in data_dict.items():
            print(f"  {data_key}: {len(df)} rows")
    
    # Get summary report
    print(coordinator.get_summary_report())
    
    # Generate detailed report
    json_file, summary_file = coordinator.generate_data_report()
    print(f"\nReports saved to:")
    print(f"  JSON: {json_file}")
    print(f"  Summary: {summary_file}")


if __name__ == "__main__":
    asyncio.run(main())