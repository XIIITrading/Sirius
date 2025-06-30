# backtest/plugins/m15_ema/storage.py
"""
Storage handler for M15 EMA calculation results.
Handles Supabase storage operations.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class M15EMAStorage:
    """Handles storage operations for M15 EMA results"""
    
    @staticmethod
    def prepare_storage_data(uid: str, signal_data: Dict) -> Dict[str, Any]:
        """
        Prepare data for storage in bt_m15_ema table.
        
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
            'ema_9': float(metadata.get('ema_9', 0)),
            'ema_21': float(metadata.get('ema_21', 0)),
            'ema_spread': float(metadata.get('ema_spread', 0)),
            'ema_spread_pct': float(metadata.get('ema_spread_pct', 0)),
            'price_vs_ema9': metadata.get('price_vs_ema9', 'unknown'),
            'trend_strength': float(metadata.get('trend_strength', 0)),
            'reason': metadata.get('reason', '')
        }
        
        return storage_data
    
    @staticmethod
    async def store(supabase_client, uid: str, signal_data: Dict) -> bool:
        """
        Store M15 EMA results to Supabase.
        
        Args:
            supabase_client: Initialized Supabase client
            uid: Unique identifier for the backtest
            signal_data: Signal data from the calculation
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Prepare data
            storage_data = M15EMAStorage.prepare_storage_data(uid, signal_data)
            
            # Upsert to handle re-runs
            response = supabase_client.table('bt_m15_ema').upsert(storage_data).execute()
            
            if response.data:
                logger.debug(f"Successfully stored M15 EMA results for {uid}")
                return True
            else:
                logger.error(f"Failed to store M15 EMA results for {uid}: No data returned")
                return False
                
        except Exception as e:
            logger.error(f"Error storing M15 EMA results for {uid}: {e}")
            return False