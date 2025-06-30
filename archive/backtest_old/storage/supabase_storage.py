# backtest/storage/supabase_storage.py
"""
Supabase storage module for backtest results.
Handles UID generation, bar storage, and result persistence.
Uses plugin-based storage for calculation results.
"""

import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from supabase import create_client, Client

logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
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
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj


@dataclass
class BacktestStorageResult:
    """Result of storage operation"""
    success: bool
    uid: str
    error: Optional[str] = None
    rows_inserted: Dict[str, int] = None


class BacktestStorage:
    """
    Manages storage of backtest results to Supabase.
    Delegates calculation-specific storage to plugins.
    """
    
    def __init__(self, supabase_url: Optional[str] = None, 
                 supabase_key: Optional[str] = None,
                 plugin_registry: Optional[Any] = None):
        """
        Initialize storage with Supabase connection.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase anon/service key
            plugin_registry: Registry of loaded plugins
        """
        # Get credentials from environment if not provided
        self.supabase_url = supabase_url or os.getenv('SUPABASE_URL')
        self.supabase_key = supabase_key or os.getenv('SUPABASE_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase credentials required (SUPABASE_URL and SUPABASE_KEY)")
            
        # Initialize Supabase client
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Plugin registry for delegating storage
        self.plugin_registry = plugin_registry
        
        # Track statistics
        self.storage_operations = 0
        self.successful_stores = 0
        self.failed_stores = 0
        
        logger.info("Initialized Supabase backtest storage with plugin support")
    
    def generate_uid(self, config: 'BacktestConfig') -> str:
        """
        Generate UID from config: TICKER.MMDDYY.HHMM.SIDE
        
        Args:
            config: Backtest configuration
            
        Returns:
            UID string (e.g., "AAPL.062724.1335.L")
        """
        # Ensure UTC timezone
        entry_time = config.entry_time
        if entry_time.tzinfo != timezone.utc:
            entry_time = entry_time.astimezone(timezone.utc)
        
        # Format components
        ticker = config.symbol.upper()
        date_str = entry_time.strftime('%m%d%y')  # MMDDYY
        time_str = entry_time.strftime('%H%M')    # HHMM (24-hour)
        side = 'L' if config.direction == 'LONG' else 'S'
        
        uid = f"{ticker}.{date_str}.{time_str}.{side}"
        
        logger.debug(f"Generated UID: {uid}")
        return uid
    
    async def store_backtest_data(self, uid: str, config: 'BacktestConfig',
                                  bars_df: pd.DataFrame, results: 'BacktestResult') -> BacktestStorageResult:
        """
        Store complete backtest data to Supabase.
        
        Args:
            uid: Unique identifier for this backtest
            config: Backtest configuration
            bars_df: DataFrame with exactly 75 bars
            results: BacktestResult object
            
        Returns:
            BacktestStorageResult indicating success/failure
        """
        self.storage_operations += 1
        
        try:
            # 1. Store/update master record in bt_index
            calculations_run = [signal.name for signal in results.entry_signals]
            await self._store_bt_index(uid, config, calculations_run)
            
            # 2. Store bar data
            bars_stored = await self._store_bt_bars(uid, bars_df, config.entry_time)
            
            # 3. Store calculation results using plugins
            calculations_stored = 0
            for signal in results.entry_signals:
                if await self._store_calculation_result(uid, signal):
                    calculations_stored += 1
            
            # 4. Store aggregated results
            await self._store_aggregated_results(
                uid, 
                results.aggregated_signal, 
                results.forward_analysis
            )
            
            self.successful_stores += 1
            
            return BacktestStorageResult(
                success=True,
                uid=uid,
                rows_inserted={
                    'bt_index': 1,
                    'bt_bars': bars_stored,
                    'calculations': calculations_stored,
                    'aggregated': 1
                }
            )
            
        except Exception as e:
            self.failed_stores += 1
            logger.error(f"Failed to store backtest data: {e}")
            return BacktestStorageResult(
                success=False,
                uid=uid,
                error=str(e)
            )
    
    async def _store_bt_index(self, uid: str, config: 'BacktestConfig', 
                             calculations_run: List[str]) -> None:
        """Store/update master record in bt_index table"""
        # Parse UID components for storage
        uid_parts = uid.split('.')
        ticker = uid_parts[0]
        
        # Convert entry time to date and time
        entry_date = config.entry_time.date()
        entry_time = config.entry_time.time()
        
        # Prepare data
        index_data = {
            'uid': uid,
            'ticker': ticker,
            'date': entry_date.isoformat(),
            'time': entry_time.isoformat(),
            'side': 'L' if config.direction == 'LONG' else 'S',
            'calculations_run': calculations_run
        }
        
        # Try to update first (in case of re-run)
        response = self.supabase.table('bt_index').upsert(index_data).execute()
        
        if not response.data:
            raise Exception(f"Failed to store bt_index record for {uid}")
            
        logger.info(f"Stored bt_index record for {uid}")
    
    async def _store_bt_bars(self, uid: str, bars_df: pd.DataFrame, 
                            entry_time: datetime) -> int:
        """
        Store exactly 75 bars to bt_bars table.
        
        Args:
            uid: Unique identifier
            bars_df: DataFrame with bar data
            entry_time: Entry time for bar indexing
            
        Returns:
            Number of bars stored
        """
        if len(bars_df) != 75:
            logger.warning(f"Expected 75 bars, got {len(bars_df)}. Adjusting...")
            
        # Prepare bar data with proper indexing
        bars_data = []
        
        # Convert entry_time to pandas timestamp for comparison
        entry_ts = pd.Timestamp(entry_time).tz_convert('UTC') if entry_time.tzinfo else pd.Timestamp(entry_time).tz_localize('UTC')
        
        # Find the entry bar index more robustly
        time_diffs = abs(bars_df.index - entry_ts)
        entry_idx = time_diffs.argmin()
        
        logger.debug(f"Entry time: {entry_ts}, Entry index: {entry_idx}")
        
        # Calculate bar indices (-15 to +59)
        for i, (timestamp, row) in enumerate(bars_df.iterrows()):
            bar_index = i - entry_idx
            
            # Ensure we're within the valid range
            if bar_index < -15 or bar_index > 59:
                logger.debug(f"Skipping bar at index {bar_index} (outside range)")
                continue
                
            # Convert all values to ensure JSON serialization
            bar_data = {
                'uid': uid,
                'bar_index': int(bar_index),  # Ensure Python int
                'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': int(row['volume']) if not pd.isna(row['volume']) else 0  # Convert numpy int to Python int
            }
            bars_data.append(bar_data)
        
        logger.info(f"Prepared {len(bars_data)} bars for storage (expected 75)")
        
        # Delete existing bars for this UID (in case of re-run)
        delete_response = self.supabase.table('bt_bars').delete().eq('uid', uid).execute()
        logger.debug(f"Deleted {len(delete_response.data) if delete_response.data else 0} existing bars")
        
        # Insert new bars in batches
        batch_size = 100
        total_inserted = 0
        
        for i in range(0, len(bars_data), batch_size):
            batch = bars_data[i:i + batch_size]
            # Apply numpy conversion to the entire batch
            batch = convert_numpy_types(batch)
            
            response = self.supabase.table('bt_bars').insert(batch).execute()
            
            if response.data:
                total_inserted += len(response.data)
            else:
                logger.error(f"Failed to insert bars batch {i//batch_size + 1}")
        
        logger.info(f"Stored {total_inserted} bars for {uid}")
        return total_inserted
    
    async def _store_calculation_result(self, uid: str, signal: 'StandardSignal') -> bool:
        """
        Store calculation-specific results using plugin storage.
        
        Args:
            uid: Unique identifier
            signal: StandardSignal from calculation
            
        Returns:
            True if stored successfully, False otherwise
        """
        if not self.plugin_registry:
            logger.warning("No plugin registry available, skipping calculation storage")
            return False
        
        # Find plugin by calculation name
        plugin = self.plugin_registry.get_plugin_by_calculation_name(signal.name)
        
        if not plugin:
            logger.debug(f"No plugin found for calculation: {signal.name}")
            return False
        
        # Convert signal to dict format for plugin
        signal_data = {
            'name': signal.name,
            'timestamp': signal.timestamp,
            'direction': signal.direction,
            'strength': signal.strength,
            'confidence': signal.confidence,
            'metadata': signal.metadata
        }
        
        # Delegate storage to plugin
        try:
            success = await plugin.store_results(self.supabase, uid, signal_data)
            if success:
                logger.debug(f"Stored {signal.name} results via plugin")
            else:
                logger.warning(f"Plugin failed to store {signal.name} results")
            return success
            
        except Exception as e:
            logger.error(f"Error in plugin storage for {signal.name}: {e}")
            return False
    
    async def _store_aggregated_results(self, uid: str, aggregated_signal: Dict[str, Any],
                                       forward_analysis: Dict[str, Any]) -> None:
        """
        Store aggregated results and forward analysis.
        
        Args:
            uid: Unique identifier
            aggregated_signal: Aggregated signal data
            forward_analysis: Forward analysis results
        """
        # Prepare aggregated data - convert all numpy types
        aggregated_data = convert_numpy_types({
            'uid': uid,
            'consensus_direction': aggregated_signal.get('consensus_direction', 'NEUTRAL'),
            'agreement_score': float(aggregated_signal.get('agreement_score', 0)),
            'average_strength': float(aggregated_signal.get('average_strength', 0)),
            'average_confidence': float(aggregated_signal.get('average_confidence', 0)),
            'participating_calculations': aggregated_signal.get('participating_calculations', 0),
            'total_calculations': aggregated_signal.get('total_calculations', 0),
            # Forward analysis
            'entry_price': float(forward_analysis.get('entry_price', 0)),
            'exit_price': float(forward_analysis.get('exit_price', 0)),
            'final_pnl': float(forward_analysis.get('final_pnl', 0)),
            'max_favorable_move': float(forward_analysis.get('max_favorable_move', 0)),
            'max_adverse_move': float(forward_analysis.get('max_adverse_move', 0)),
            'signal_matched_user': forward_analysis.get('signal_accuracy', {}).get('consensus_matched_user', False),
            'trade_profitable': forward_analysis.get('signal_accuracy', {}).get('profitable', False),
            'signal_aligned_outcome': forward_analysis.get('signal_accuracy', {}).get('signal_aligned_with_outcome', False)
        })
        
        # Upsert the aggregated data
        response = self.supabase.table('bt_aggregated').upsert(aggregated_data).execute()
        
        if not response.data:
            logger.error(f"Failed to store aggregated results for {uid}")
        else:
            logger.debug(f"Stored aggregated results for {uid}")
    
    def _get_calculation_table_name(self, signal_name: str) -> Optional[str]:
        """
        Get table name from plugin registry.
        Replaces hardcoded mapping.
        """
        if not self.plugin_registry:
            return None
            
        plugin = self.plugin_registry.get_plugin_by_calculation_name(signal_name)
        return plugin.storage_table if plugin else None
    
    async def check_uid_exists(self, uid: str) -> bool:
        """Check if a UID already exists in the database"""
        try:
            response = self.supabase.table('bt_index').select('uid').eq('uid', uid).execute()
            return len(response.data) > 0 if response.data else False
        except Exception as e:
            logger.error(f"Error checking UID existence: {e}")
            return False
    
    async def get_backtest_by_uid(self, uid: str) -> Optional[Dict[str, Any]]:
        """Retrieve complete backtest data by UID"""
        try:
            # Get index record
            index_response = self.supabase.table('bt_index').select('*').eq('uid', uid).execute()
            
            if not index_response.data:
                return None
                
            result = index_response.data[0]
            
            # Get bars
            bars_response = self.supabase.table('bt_bars').select('*').eq('uid', uid).order('bar_index').execute()
            result['bars'] = bars_response.data if bars_response.data else []
            
            # Get aggregated results
            agg_response = self.supabase.table('bt_aggregated').select('*').eq('uid', uid).execute()
            result['aggregated'] = agg_response.data[0] if agg_response.data else None
            
            # Get calculation-specific results based on calculations_run
            result['calculations'] = {}
            for calc_name in result.get('calculations_run', []):
                table_name = self._get_calculation_table_name(calc_name)
                if table_name:
                    calc_response = self.supabase.table(table_name).select('*').eq('uid', uid).execute()
                    if calc_response.data:
                        result['calculations'][calc_name] = calc_response.data[0]
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving backtest {uid}: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        success_rate = (
            self.successful_stores / self.storage_operations * 100
            if self.storage_operations > 0 else 0
        )
        
        return {
            'total_operations': self.storage_operations,
            'successful_stores': self.successful_stores,
            'failed_stores': self.failed_stores,
            'success_rate': round(success_rate, 2),
            'supabase_url': self.supabase_url.split('.')[0] if self.supabase_url else 'Not configured'
        }


def prepare_bars_for_storage(historical_df: pd.DataFrame, forward_df: pd.DataFrame,
                           entry_time: datetime) -> pd.DataFrame:
    """
    Prepare exactly 75 bars for storage.
    
    Args:
        historical_df: Historical bars before entry
        forward_df: Forward bars including and after entry
        entry_time: Entry time
        
    Returns:
        DataFrame with exactly 75 bars indexed -15 to +59
    """
    # Ensure entry_time is timezone aware
    if entry_time.tzinfo is None:
        entry_ts = pd.Timestamp(entry_time).tz_localize('UTC')
    else:
        entry_ts = pd.Timestamp(entry_time).tz_convert('UTC')
    
    # Get 15 bars before entry
    lookback = historical_df[historical_df.index < entry_ts].tail(15).copy()
    
    # Get 60 bars including and after entry  
    forward = forward_df[forward_df.index >= entry_ts].head(60).copy()
    
    # Combine
    all_bars = pd.concat([lookback, forward])
    
    # Remove any duplicates (keeping first occurrence)
    all_bars = all_bars[~all_bars.index.duplicated(keep='first')]
    
    # Sort by index to ensure proper order
    all_bars = all_bars.sort_index()
    
    # Log the actual composition
    logger.debug(f"Prepared bars: {len(lookback)} historical + {len(forward)} forward = {len(all_bars)} total")
    
    # Ensure we have exactly 75 bars
    if len(all_bars) < 75:
        logger.warning(f"Only {len(all_bars)} bars available, expected 75")
    elif len(all_bars) > 75:
        # Trim to exactly 75
        all_bars = all_bars.iloc[:75]
        logger.debug(f"Trimmed to 75 bars from {len(all_bars)}")
    
    return all_bars