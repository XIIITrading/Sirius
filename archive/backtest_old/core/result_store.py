# backtest/core/result_store.py
"""
Storage and retrieval of backtest results.
Provides structured storage for analysis and AI training.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging
import uuid

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
class BacktestResult:
    """Complete result from a single backtest run"""
    # Configuration (required fields first)
    config: Any  # BacktestConfig
    
    # Entry analysis
    entry_signals: List[Any]  # List of StandardSignal
    aggregated_signal: Dict[str, Any]  # From SignalAggregator
    
    # Forward analysis
    forward_data: pd.DataFrame  # 60 bars of forward data
    forward_analysis: Dict[str, Any]  # Performance metrics
    
    # Metadata
    runtime_seconds: float
    
    # Optional fields with defaults (must come after required fields)
    historical_bars: Optional[pd.DataFrame] = None  # Historical bars used
    bars_for_storage: Optional[pd.DataFrame] = None  # Prepared 75 bars
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        # Handle entry_time which might be a pandas Timestamp
        entry_time = self.config.entry_time
        if isinstance(entry_time, pd.Timestamp):
            entry_time_str = entry_time.to_pydatetime().isoformat()
        elif hasattr(entry_time, 'isoformat'):
            entry_time_str = entry_time.isoformat()
        else:
            entry_time_str = str(entry_time)
        
        # Convert all values to ensure JSON serialization
        return convert_numpy_types({
            'result_id': self.result_id,
            'timestamp': self.timestamp.isoformat(),
            'config': {
                'symbol': self.config.symbol,
                'entry_time': entry_time_str,
                'direction': self.config.direction,
                'historical_lookback_hours': self.config.historical_lookback_hours,
                'forward_bars': self.config.forward_bars,
                'store_to_supabase': getattr(self.config, 'store_to_supabase', False)
            },
            'entry_signals': [
                convert_numpy_types(signal.to_dict()) for signal in self.entry_signals
            ],
            'aggregated_signal': convert_numpy_types(self.aggregated_signal),
            'forward_analysis': convert_numpy_types(self.forward_analysis),
            'runtime_seconds': float(self.runtime_seconds)
        })
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary for quick analysis"""
        # Handle entry_time which might be a pandas Timestamp
        entry_time = self.config.entry_time
        if isinstance(entry_time, pd.Timestamp):
            entry_time_str = entry_time.to_pydatetime().isoformat()
        elif hasattr(entry_time, 'isoformat'):
            entry_time_str = entry_time.isoformat()
        else:
            entry_time_str = str(entry_time)
        
        return convert_numpy_types({
            'result_id': self.result_id,
            'symbol': self.config.symbol,
            'entry_time': entry_time_str,
            'direction': self.config.direction,
            'consensus_direction': self.aggregated_signal.get('consensus_direction'),
            'signal_agreement': self.aggregated_signal.get('agreement_score', 0),
            'final_pnl': self.forward_analysis.get('final_pnl', 0),
            'max_favorable_move': self.forward_analysis.get('max_favorable_move', 0),
            'max_adverse_move': self.forward_analysis.get('max_adverse_move', 0),
            'signal_accuracy': self.forward_analysis.get('signal_accuracy', {})
        })


class BacktestResultStore:
    """
    Manages storage and retrieval of backtest results.
    Provides methods for analysis and bulk operations.
    """
    
    def __init__(self, results_dir: Optional[str] = None):
        """
        Initialize result store.
        
        Args:
            results_dir: Directory to store results
        """
        self.results_dir = Path(results_dir) if results_dir else Path('backtest/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.daily_dir = self.results_dir / 'daily'
        self.daily_dir.mkdir(exist_ok=True)
        
        self.summary_file = self.results_dir / 'summary.json'
        
        # In-memory cache of recent results
        self.recent_results: List[BacktestResult] = []
        self.max_recent = 100
        
        logger.info(f"Initialized result store at {self.results_dir}")
        
    def set_results_dir(self, results_dir: str) -> None:
        """Update results directory"""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.daily_dir = self.results_dir / 'daily'
        self.daily_dir.mkdir(exist_ok=True)
        
    def store_result(self, result: BacktestResult) -> str:
        """
        Store a backtest result.
        
        Args:
            result: BacktestResult to store
            
        Returns:
            Result ID
        """
        # Add to recent results
        self.recent_results.append(result)
        if len(self.recent_results) > self.max_recent:
            self.recent_results.pop(0)
            
        # Create daily directory
        date_str = result.config.entry_time.strftime('%Y%m%d')
        day_dir = self.daily_dir / date_str
        day_dir.mkdir(exist_ok=True)
        
        # Store detailed result (without DataFrames to save space)
        result_file = day_dir / f"{result.result_id}.json"
        with open(result_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
            
        # Store forward data separately (as parquet for efficiency)
        if not result.forward_data.empty:
            data_file = day_dir / f"{result.result_id}_forward.parquet"
            result.forward_data.to_parquet(data_file)
            
        # Optionally store historical bars if available
        if result.historical_bars is not None and not result.historical_bars.empty:
            hist_file = day_dir / f"{result.result_id}_historical.parquet"
            result.historical_bars.to_parquet(hist_file)
            
        # Optionally store prepared bars if available
        if result.bars_for_storage is not None and not result.bars_for_storage.empty:
            bars_file = day_dir / f"{result.result_id}_bars_storage.parquet"
            result.bars_for_storage.to_parquet(bars_file)
            
        # Update summary
        self._update_summary(result)
        
        logger.info(f"Stored result {result.result_id} for {result.config.symbol}")
        return result.result_id
        
    def _update_summary(self, result: BacktestResult) -> None:
        """Update summary file with new result"""
        # Load existing summary
        summaries = []
        if self.summary_file.exists():
            try:
                with open(self.summary_file, 'r') as f:
                    summaries = json.load(f)
            except:
                summaries = []
                
        # Add new summary
        summaries.append(result.get_summary())
        
        # Keep last 10000 summaries
        if len(summaries) > 10000:
            summaries = summaries[-10000:]
            
        # Save updated summary
        with open(self.summary_file, 'w') as f:
            json.dump(summaries, f, indent=2, default=str)
            
    def get_result(self, result_id: str) -> Optional[BacktestResult]:
        """Retrieve a specific result by ID"""
        # Check recent results first
        for result in self.recent_results:
            if result.result_id == result_id:
                return result
                
        # Search in files
        for day_dir in self.daily_dir.iterdir():
            result_file = day_dir / f"{result_id}.json"
            if result_file.exists():
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        
                    # Load forward data if exists
                    data_file = day_dir / f"{result_id}_forward.parquet"
                    if data_file.exists():
                        forward_data = pd.read_parquet(data_file)
                    else:
                        forward_data = pd.DataFrame()
                        
                    # Load historical bars if exists
                    hist_file = day_dir / f"{result_id}_historical.parquet"
                    historical_bars = None
                    if hist_file.exists():
                        historical_bars = pd.read_parquet(hist_file)
                        
                    # Load prepared bars if exists
                    bars_file = day_dir / f"{result_id}_bars_storage.parquet"
                    bars_for_storage = None
                    if bars_file.exists():
                        bars_for_storage = pd.read_parquet(bars_file)
                        
                    # Reconstruct result (simplified - would need proper deserialization)
                    return data
                    
                except Exception as e:
                    logger.error(f"Failed to load result {result_id}: {e}")
                    
        return None
        
    def query_results(self, 
                     symbol: Optional[str] = None,
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None,
                     direction: Optional[str] = None,
                     min_pnl: Optional[float] = None,
                     max_results: int = 1000) -> List[Dict[str, Any]]:
        """
        Query results with filters.
        
        Returns:
            List of result summaries matching criteria
        """
        # Load summary
        if not self.summary_file.exists():
            return []
            
        try:
            with open(self.summary_file, 'r') as f:
                summaries = json.load(f)
        except:
            return []
            
        # Apply filters
        filtered = []
        for summary in summaries:
            # Symbol filter
            if symbol and summary.get('symbol') != symbol:
                continue
                
            # Date filters
            entry_time = datetime.fromisoformat(summary['entry_time'])
            if start_date and entry_time < start_date:
                continue
            if end_date and entry_time > end_date:
                continue
                
            # Direction filter
            if direction and summary.get('direction') != direction:
                continue
                
            # P&L filter
            if min_pnl is not None and summary.get('final_pnl', 0) < min_pnl:
                continue
                
            filtered.append(summary)
            
            if len(filtered) >= max_results:
                break
                
        return filtered
        
    def get_performance_stats(self, results: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Calculate performance statistics from results"""
        if results is None:
            # Use all results
            if self.summary_file.exists():
                with open(self.summary_file, 'r') as f:
                    results = json.load(f)
            else:
                results = []
                
        if not results:
            return {}
            
        # Convert to DataFrame for easy analysis
        df = pd.DataFrame(results)
        
        # Basic stats
        total_trades = len(df)
        profitable = df['final_pnl'] > 0
        win_rate = (profitable.sum() / total_trades * 100) if total_trades > 0 else 0
        
        # P&L stats
        avg_pnl = df['final_pnl'].mean()
        avg_win = df.loc[profitable, 'final_pnl'].mean() if profitable.any() else 0
        avg_loss = df.loc[~profitable, 'final_pnl'].mean() if (~profitable).any() else 0
        
        # Signal accuracy
        signal_accuracy = pd.DataFrame(df['signal_accuracy'].tolist())
        consensus_accuracy = (
            signal_accuracy['consensus_matched_user'].sum() / len(signal_accuracy) * 100
            if 'consensus_matched_user' in signal_accuracy else 0
        )
        
        # By symbol
        by_symbol = df.groupby('symbol').agg({
            'final_pnl': ['count', 'mean', 'sum'],
            'max_favorable_move': 'mean',
            'max_adverse_move': 'mean'
        }).round(2)
        
        return {
            'total_backtests': total_trades,
            'win_rate': round(win_rate, 2),
            'average_pnl': round(avg_pnl, 2),
            'average_win': round(avg_win, 2),
            'average_loss': round(avg_loss, 2),
            'profit_factor': round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0,
            'consensus_accuracy': round(consensus_accuracy, 2),
            'by_symbol': by_symbol.to_dict() if not by_symbol.empty else {},
            'best_performing_signals': self._analyze_best_signals(df),
            'worst_performing_signals': self._analyze_worst_signals(df)
        }
        
    def _analyze_best_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Analyze which signal combinations perform best"""
        # This would analyze entry_signals to find patterns
        # Simplified for now
        return []
        
    def _analyze_worst_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Analyze which signal combinations perform worst"""
        # This would analyze entry_signals to find patterns
        # Simplified for now
        return []
        
    def export_for_ai_analysis(self, output_file: str, 
                              filters: Optional[Dict] = None) -> int:
        """
        Export results in format suitable for AI analysis.
        
        Args:
            output_file: Output file path
            filters: Optional filters to apply
            
        Returns:
            Number of results exported
        """
        # Query results with filters
        results = self.query_results(**(filters or {}))
        
        # Prepare data for AI
        ai_data = []
        for summary in results:
            # Flatten signals and create feature vector
            entry_signals = summary.get('entry_signals', [])
            
            features = {
                'symbol': summary['symbol'],
                'entry_time': summary['entry_time'],
                'user_direction': summary['direction'],
                'consensus_direction': summary['consensus_direction'],
                'signal_agreement': summary['signal_agreement'],
                'final_pnl': summary['final_pnl'],
                'max_favorable_move': summary['max_favorable_move'],
                'max_adverse_move': summary['max_adverse_move']
            }
            
            # Add individual signal features
            # This would be expanded to include all signal details
            
            ai_data.append(features)
            
        # Save as parquet for efficient loading
        df = pd.DataFrame(ai_data)
        df.to_parquet(output_file, index=False)
        
        logger.info(f"Exported {len(ai_data)} results for AI analysis to {output_file}")
        return len(ai_data)
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics"""
        total_files = sum(1 for _ in self.daily_dir.glob('**/*.json'))
        total_size = sum(
            f.stat().st_size for f in self.daily_dir.glob('**/*') if f.is_file()
        ) / (1024 * 1024)  # MB
        
        return {
            'total_results': total_files,
            'recent_results_cached': len(self.recent_results),
            'storage_size_mb': round(total_size, 2),
            'daily_directories': len(list(self.daily_dir.iterdir()))
        }