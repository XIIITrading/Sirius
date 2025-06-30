# backtest/plugins/m5_market_structure/plugin.py
"""
M5 Market Structure Plugin
Complete self-contained plugin for 5-minute fractal-based market structure analysis.
"""

from typing import Dict, Any, Type
from plugins.base_plugin import BacktestPlugin
from .adapter import M5MarketStructureBackAdapter
from .storage import M5MarketStructureStorage


class M5MarketStructurePlugin(BacktestPlugin):
    """Plugin for 5-minute market structure calculation using fractals"""
    
    # ==================== METADATA ====================
    
    @property
    def name(self) -> str:
        return "5-Min Market Structure"
    
    @property
    def adapter_name(self) -> str:
        return "m5_market_structure"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    # ==================== ADAPTER ====================
    
    @property
    def adapter_class(self) -> Type:
        return M5MarketStructureBackAdapter
    
    def get_adapter_config(self) -> Dict[str, Any]:
        """Configuration for M5 Market Structure adapter"""
        return {
            'fractal_length': 3,       # Fewer bars for 5-min fractal detection
            'buffer_size': 100,        # 100 5-min candles = 500 minutes
            'min_candles_required': 15 # Minimum for valid signal
        }
    
    # ==================== STORAGE ====================
    
    @property
    def storage_table(self) -> str:
        return "bt_m5_market_structure"
    
    def get_storage_mapping(self, signal_data: Dict) -> Dict[str, Any]:
        """Convert signal data to storage format"""
        metadata = signal_data.get('metadata', {})
        
        return {
            'signal_direction': signal_data.get('direction'),
            'signal_strength': float(signal_data.get('strength', 0)),
            'signal_confidence': float(signal_data.get('confidence', 0)),
            'structure_type': metadata.get('structure_type', 'NONE'),  # BOS, CHoCH, or NONE
            'current_trend': metadata.get('current_trend', 'NEUTRAL'),
            'last_high_fractal': metadata.get('last_high_fractal'),
            'last_low_fractal': metadata.get('last_low_fractal'),
            'last_break_type': metadata.get('last_break_type'),
            'last_break_time': metadata.get('last_break_time'),
            'last_break_price': metadata.get('last_break_price'),
            'fractal_count': int(metadata.get('fractal_count', 0)),
            'structure_breaks': int(metadata.get('structure_breaks', 0)),
            'trend_changes': int(metadata.get('trend_changes', 0)),
            'candles_processed': int(metadata.get('candles_processed', 0)),
            'timeframe': '5-minute',
            'reason': metadata.get('reason', '')
        }
    
    async def store_results(self, supabase_client, uid: str, signal_data: Dict) -> bool:
        """Store M5 Market Structure results to Supabase"""
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
        return await M5MarketStructureStorage.store(supabase_client, uid, signal_data_clean)


# Export plugin instance
plugin = M5MarketStructurePlugin()