# backtest/data/data_coordinator.py
"""
Module: Data Coordinator
Purpose: Coordinates data requests from multiple calculation modules
Features: Module registration, need aggregation, concurrent fetching, report generation
"""

import logging
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import json
from pathlib import Path

from .request_aggregator import RequestAggregator, DataNeed, DataType
from .polygon_data_manager import PolygonDataManager
from .protected_data_manager import ProtectedDataManager

logger = logging.getLogger(__name__)


class DataCoordinator:
    """
    Coordinates data requests from multiple calculation modules.
    Acts as the central hub for data management in the backtesting system.
    """
    
    def __init__(self, data_manager: Union[PolygonDataManager, ProtectedDataManager]):
        """
        Initialize coordinator with a data manager.
        
        Args:
            data_manager: Either PolygonDataManager or ProtectedDataManager instance
        """
        self.data_manager = data_manager
        self.aggregator = RequestAggregator(data_manager=data_manager)
        self.registered_modules = {}
        
        logger.info("DataCoordinator initialized")
    
    def register_module(self, module_name: str, module_config: Dict[str, Any]):
        """Register a calculation module"""
        self.registered_modules[module_name] = module_config
        logger.info(f"Registered module: {module_name}")
    
    def _generate_module_needs(self, symbol: str, entry_time: datetime, 
                              direction: str) -> List[DataNeed]:
        """Generate data needs for all registered modules"""
        needs = []
        
        # Generate needs based on module types
        for module_name in self.registered_modules:
            if module_name == "TrendAnalysis":
                needs.extend([
                    DataNeed(module_name, symbol, DataType.BARS, "1min",
                            entry_time - timedelta(hours=2), entry_time),
                    DataNeed(module_name, symbol, DataType.BARS, "5min",
                            entry_time - timedelta(hours=4), entry_time),
                    DataNeed(module_name, symbol, DataType.BARS, "15min",
                            entry_time - timedelta(hours=6), entry_time),
                ])
            
            elif module_name == "MarketStructure":
                needs.append(
                    DataNeed(module_name, symbol, DataType.BARS, "1min",
                            entry_time - timedelta(hours=1), 
                            entry_time + timedelta(minutes=30))
                )
            
            elif module_name == "OrderFlow":
                needs.extend([
                    DataNeed(module_name, symbol, DataType.TRADES, "tick",
                            entry_time - timedelta(minutes=30), entry_time),
                    DataNeed(module_name, symbol, DataType.QUOTES, "tick",
                            entry_time - timedelta(minutes=30), entry_time),
                ])
            
            elif module_name == "VolumeAnalysis":
                needs.extend([
                    DataNeed(module_name, symbol, DataType.BARS, "1min",
                            entry_time - timedelta(hours=1, minutes=30), entry_time),
                    DataNeed(module_name, symbol, DataType.TRADES, "tick",
                            entry_time - timedelta(minutes=45), entry_time),
                ])
        
        return needs
    
    async def fetch_all_module_data(self, symbol: str, entry_time: datetime, 
                                   direction: str) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch all data needed by registered modules.
        
        Returns:
            Dict mapping module_name -> data_key -> DataFrame
        """
        # Clear previous needs
        self.aggregator.clear_needs()
        
        # Generate needs for all modules
        needs = self._generate_module_needs(symbol, entry_time, direction)
        
        # Register needs with aggregator
        self.aggregator.register_needs(needs)
        
        # Fetch all data efficiently
        module_data = await self.aggregator.fetch_all_data()
        
        return module_data
    
    def get_summary_report(self) -> str:
        """Get summary report from aggregator"""
        return self.aggregator.create_request_report()
    
    def generate_data_report(self) -> Tuple[str, str]:
        """Generate detailed data report"""
        if hasattr(self.data_manager, 'generate_data_report'):
            return self.data_manager.generate_data_report()
        else:
            # Fallback for when using basic PolygonDataManager
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = Path('./temp')
            report_dir.mkdir(exist_ok=True)
            
            # Create simple report
            json_file = report_dir / f"data_report_{timestamp}.json"
            summary_file = report_dir / f"data_report_{timestamp}_summary.txt"
            
            report_data = {
                "timestamp": timestamp,
                "aggregator_stats": self.aggregator.get_stats(),
                "registered_modules": list(self.registered_modules.keys())
            }
            
            with open(json_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            with open(summary_file, 'w') as f:
                f.write(self.get_summary_report())
            
            return str(json_file), str(summary_file)