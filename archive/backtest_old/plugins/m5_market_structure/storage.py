# backtest/plugins/m5_market_structure/storage.py
"""
Storage handler for M5 Market Structure calculation results.
Handles Supabase storage operations for 5-minute market structure analysis.
"""

import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class M5MarketStructureStorage:
    """Handles storage operations for M5 Market Structure results"""
    
    @staticmethod
    def prepare_storage_data(uid: str, signal_data: Dict) -> Dict[str, Any]:
        """
        Prepare data for storage in bt_m5_market_structure table.
        
        Args:
            uid: Unique identifier for the backtest
            signal_data: Signal data from the adapter
            
        Returns:
            Dict ready for Supabase storage
        """
        # Extract metadata
        metadata = signal_data.get('metadata', {})
        
        # Build storage record
        storage_data = {
            'uid': uid,
            'signal_direction': signal_data.get('direction', 'NEUTRAL'),
            'signal_strength': float(signal_data.get('strength', 0)),
            'signal_confidence': float(signal_data.get('confidence', 0)),
            'structure_type': metadata.get('structure_type', 'NONE'),
            'current_trend': metadata.get('current_trend', 'NEUTRAL'),
            'last_high_fractal': float(metadata['last_high_fractal']) if metadata.get('last_high_fractal') is not None else None,
            'last_low_fractal': float(metadata['last_low_fractal']) if metadata.get('last_low_fractal') is not None else None,
            'last_break_type': metadata.get('last_break_type'),
            'last_break_price': float(metadata['last_break_price']) if metadata.get('last_break_price') is not None else None,
            'fractal_count': int(metadata.get('fractal_count', 0)),
            'structure_breaks': int(metadata.get('structure_breaks', 0)),
            'trend_changes': int(metadata.get('trend_changes', 0)),
            'candles_processed': int(metadata.get('candles_processed', 0)),
            'timeframe': '5-minute',
            'reason': metadata.get('reason', '')
        }
        
        # Handle last_break_time
        if metadata.get('last_break_time'):
            break_time = metadata['last_break_time']
            # Convert ISO string back to timestamp if needed
            if isinstance(break_time, str):
                storage_data['last_break_time'] = break_time
            elif hasattr(break_time, 'isoformat'):
                storage_data['last_break_time'] = break_time.isoformat()
            else:
                storage_data['last_break_time'] = str(break_time)
        else:
            storage_data['last_break_time'] = None
        
        return storage_data
    
    @staticmethod
    async def store(supabase_client, uid: str, signal_data: Dict) -> bool:
        """
        Store M5 Market Structure results to Supabase.
        
        Args:
            supabase_client: Initialized Supabase client
            uid: Unique identifier for the backtest
            signal_data: Signal data from the calculation
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Prepare data
            storage_data = M5MarketStructureStorage.prepare_storage_data(uid, signal_data)
            
            # Upsert to handle re-runs
            response = supabase_client.table('bt_m5_market_structure').upsert(storage_data).execute()
            
            if response.data:
                logger.debug(f"Successfully stored M5 Market Structure results for {uid}")
                return True
            else:
                logger.error(f"Failed to store M5 Market Structure results for {uid}: No data returned")
                return False
                
        except Exception as e:
            logger.error(f"Error storing M5 Market Structure results for {uid}: {e}")
            return False