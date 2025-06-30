# backtest/plugins/base_plugin.py
"""
Enhanced base plugin interface for backtesting calculations.
Provides abstract base for self-contained calculation modules with storage.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BacktestPlugin(ABC):
    """
    Abstract base class for backtest calculation plugins.
    Each plugin is self-contained with its own adapter, storage, and schema.
    """
    
    # ==================== METADATA ====================
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for the calculation"""
        pass
    
    @property
    @abstractmethod
    def adapter_name(self) -> str:
        """Unique identifier for the adapter"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version for compatibility tracking"""
        pass
    
    # ==================== ADAPTER ====================
    
    @property
    @abstractmethod
    def adapter_class(self) -> Type:
        """The adapter class that implements calculation logic"""
        pass
    
    @abstractmethod
    def get_adapter_config(self) -> Dict[str, Any]:
        """Configuration parameters for the adapter instance"""
        pass
    
    # ==================== STORAGE ====================
    
    @property
    @abstractmethod
    def storage_table(self) -> str:
        """Supabase table name for storing results"""
        pass
    
    @abstractmethod
    def get_storage_mapping(self, signal_data: Dict) -> Dict[str, Any]:
        """Convert signal data to storage format for Supabase"""
        pass
    
    @abstractmethod
    async def store_results(self, supabase_client, uid: str, signal_data: Dict) -> bool:
        """
        Store calculation results to Supabase.
        Returns True if successful, False otherwise.
        """
        pass
    
    # ==================== VALIDATION ====================
    
    def validate_signal_data(self, signal_data: Dict) -> bool:
        """Validate signal data before storage"""
        required_fields = ['direction', 'strength', 'confidence']
        return all(field in signal_data for field in required_fields)
    
    def validate_storage_data(self, storage_data: Dict) -> bool:
        """Validate storage data is JSON serializable"""
        try:
            import json
            json.dumps(storage_data)
            return True
        except (TypeError, ValueError) as e:
            logger.error(f"Storage data not JSON serializable: {e}")
            return False
    
    # ==================== UTILITIES ====================
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata for reporting"""
        return {
            'name': self.name,
            'adapter_name': self.adapter_name,
            'version': self.version,
            'storage_table': self.storage_table,
            'adapter_class': self.adapter_class.__name__
        }
    
    @staticmethod
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization"""
        import numpy as np
        import pandas as pd
        
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.to_pydatetime().isoformat()
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: BacktestPlugin.convert_numpy_types(value) 
                   for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [BacktestPlugin.convert_numpy_types(item) for item in obj]
        return obj


class PluginRegistry:
    """Registry for managing loaded plugins"""
    
    def __init__(self):
        self.plugins: Dict[str, BacktestPlugin] = {}
        self.storage_mappings: Dict[str, str] = {}  # adapter_name -> table_name
        
    def register(self, plugin: BacktestPlugin) -> None:
        """Register a plugin in the registry"""
        adapter_name = plugin.adapter_name
        
        if adapter_name in self.plugins:
            logger.warning(f"Overwriting existing plugin: {adapter_name}")
        
        self.plugins[adapter_name] = plugin
        self.storage_mappings[adapter_name] = plugin.storage_table
        
        logger.info(f"Registered plugin: {plugin.name} (v{plugin.version})")
    
    def get_plugin(self, adapter_name: str) -> Optional[BacktestPlugin]:
        """Get a plugin by adapter name"""
        return self.plugins.get(adapter_name)
    
    def get_all_plugins(self) -> Dict[str, BacktestPlugin]:
        """Get all registered plugins"""
        return self.plugins.copy()
    
    def get_storage_table(self, adapter_name: str) -> Optional[str]:
        """Get storage table name for an adapter"""
        return self.storage_mappings.get(adapter_name)
    
    def get_plugin_by_calculation_name(self, calculation_name: str) -> Optional[BacktestPlugin]:
        """Find a plugin by its human-readable calculation name"""
        for plugin in self.plugins.values():
            if plugin.name == calculation_name:
                return plugin
        return None