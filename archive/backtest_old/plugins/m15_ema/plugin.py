# backtest/plugins/m15_ema/plugin.py
"""
M15 EMA Crossover Plugin
Complete self-contained plugin for 15-minute EMA crossover analysis.
"""

from typing import Dict, Any, Type
from plugins.base_plugin import BacktestPlugin
from .adapter import M15EMABackAdapter
from .storage import M15EMAStorage


class M15EMAPlugin(BacktestPlugin):
    """Plugin for 15-minute EMA crossover calculation"""
    
    # ==================== METADATA ====================
    
    @property
    def name(self) -> str:
        return "15-Min EMA Crossover"
    
    @property
    def adapter_name(self) -> str:
        return "m15_ema_crossover"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    # ==================== ADAPTER ====================
    
    @property
    def adapter_class(self) -> Type:
        return M15EMABackAdapter
    
    def get_adapter_config(self) -> Dict[str, Any]:
        """Configuration for M15 EMA adapter"""
        return {
            'buffer_size': 40  # Number of 15-minute candles to maintain
        }
    
    # ==================== STORAGE ====================
    
    @property
    def storage_table(self) -> str:
        return "bt_m15_ema"
    
    def get_storage_mapping(self, signal_data: Dict) -> Dict[str, Any]:
        """Convert signal data to storage format"""
        metadata = signal_data.get('metadata', {})
        
        return {
            'signal_direction': signal_data.get('direction'),
            'signal_strength': float(signal_data.get('strength', 0)),
            'signal_confidence': float(signal_data.get('confidence', 0)),
            'ema_9': float(metadata.get('ema_9', 0)),
            'ema_21': float(metadata.get('ema_21', 0)),
            'ema_spread': float(metadata.get('ema_spread', 0)),
            'ema_spread_pct': float(metadata.get('ema_spread_pct', 0)),
            'price_vs_ema9': metadata.get('price_vs_ema9', 'unknown'),
            'trend_strength': float(metadata.get('trend_strength', 0)),
            'reason': metadata.get('reason', '')
        }
    
    async def store_results(self, supabase_client, uid: str, signal_data: Dict) -> bool:
        """Store M15 EMA results to Supabase"""
        # Validate signal data
        if not self.validate_signal_data(signal_data):
            self.handle_storage_error(
                ValueError("Invalid signal data"), 
                uid
            )
            return False
        
        # Convert numpy types
        signal_data_clean = self.convert_numpy_types(signal_data)
        
        # Delegate to storage handler
        return await M15EMAStorage.store(supabase_client, uid, signal_data_clean)


# Export plugin instance
plugin = M15EMAPlugin()