"""Parquet-based file cache for market data"""
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class FileCache:
    """Manages parquet file caching for market data"""
    
    def __init__(self, cache_dir: Path, cache_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_hours = cache_hours
        self.metadata_file = self.cache_dir / 'cache_metadata.json'
        self.metadata = self._load_metadata()
        
    def get(self, symbol: str, data_type: str, timeframe: str,
            start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        """Retrieve data from file cache if available and valid"""
        symbol_dir = self.cache_dir / symbol.lower() / data_type
        if not symbol_dir.exists():
            return None
        
        # Check each parquet file in symbol directory
        for parquet_file in symbol_dir.glob(f"{timeframe}_*.parquet"):
            try:
                # Extract metadata from filename
                parts = parquet_file.stem.split('_')
                if len(parts) < 3:
                    continue
                
                file_meta_key = parts[2]  # Hash in filename
                
                # Check if this file might contain our data
                if file_meta_key in self.metadata:
                    meta = self.metadata[file_meta_key]
                    if self._is_cache_valid(meta, start_time, end_time):
                        df = pd.read_parquet(parquet_file)
                        df.index = pd.to_datetime(df.index, utc=True)
                        
                        # Verify actual data coverage
                        if not df.empty and df.index.min() <= start_time and df.index.max() >= end_time:
                            logger.info(f"File cache hit for {symbol} {data_type}")
                            return df
                            
            except Exception as e:
                logger.error(f"Error reading cache file {parquet_file}: {e}")
                continue
        
        return None
    
    def put(self, symbol: str, data_type: str, timeframe: str,
            start_time: datetime, end_time: datetime, 
            df: pd.DataFrame, cache_key: str):
        """Store data in file cache"""
        try:
            symbol_dir = self.cache_dir / symbol.lower() / data_type
            symbol_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename with metadata
            filename = f"{timeframe}_{start_time.strftime('%Y%m%d')}_{cache_key[:8]}.parquet"
            filepath = symbol_dir / filename
            
            # Save to parquet with compression
            compression = 'snappy' if data_type == 'bars' else 'gzip'
            df.to_parquet(filepath, compression=compression)
            
            # Update metadata
            self.metadata[cache_key] = {
                'symbol': symbol,
                'data_type': data_type,
                'timeframe': timeframe,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'cached_at': datetime.now(timezone.utc).isoformat(),
                'data_points': len(df)
            }
            
            self._save_metadata()
            logger.info(f"Cached {len(df)} {data_type} to {filepath}")
            
        except Exception as e:
            logger.error(f"Error caching to file: {e}")
    
    def _is_cache_valid(self, meta: Dict, start_time: datetime, end_time: datetime) -> bool:
        """Check if cache entry is valid for requested range"""
        cached_start = datetime.fromisoformat(meta['start_time'])
        cached_end = datetime.fromisoformat(meta['end_time'])
        cached_at = datetime.fromisoformat(meta['cached_at'])
        
        # Ensure timezone aware
        if cached_start.tzinfo is None:
            cached_start = cached_start.replace(tzinfo=timezone.utc)
        if cached_end.tzinfo is None:
            cached_end = cached_end.replace(tzinfo=timezone.utc)
        
        # Check age
        cache_age = datetime.now(timezone.utc) - cached_at
        if cache_age.total_seconds() > self.cache_hours * 3600:
            return False
        
        # Check coverage
        return cached_start <= start_time and cached_end >= end_time
    
    def _load_metadata(self) -> Dict[str, Dict]:
        """Load cache metadata from file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save cache metadata to file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")
    
    def clear(self, symbol: Optional[str] = None, data_type: Optional[str] = None):
        """Clear cache files"""
        if symbol:
            symbol_dir = self.cache_dir / symbol.lower()
            if symbol_dir.exists():
                if data_type:
                    type_dir = symbol_dir / data_type
                    if type_dir.exists():
                        for file in type_dir.glob("*.parquet"):
                            file.unlink()
                else:
                    for type_dir in symbol_dir.iterdir():
                        if type_dir.is_dir():
                            for file in type_dir.glob("*.parquet"):
                                file.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get file cache statistics"""
        stats = {
            'bars': {'files': 0, 'size_mb': 0},
            'trades': {'files': 0, 'size_mb': 0},
            'quotes': {'files': 0, 'size_mb': 0}
        }
        
        total_files = 0
        total_size_mb = 0
        
        for symbol_dir in self.cache_dir.iterdir():
            if symbol_dir.is_dir() and symbol_dir.name != 'cache_metadata.json':
                for type_dir in symbol_dir.iterdir():
                    if type_dir.is_dir() and type_dir.name in stats:
                        for file in type_dir.glob("*.parquet"):
                            stats[type_dir.name]['files'] += 1
                            size_mb = file.stat().st_size / (1024 * 1024)
                            stats[type_dir.name]['size_mb'] += size_mb
                            total_files += 1
                            total_size_mb += size_mb
        
        # Round sizes
        for data_type in stats:
            stats[data_type]['size_mb'] = round(stats[data_type]['size_mb'], 2)
        
        return {
            'total_files': total_files,
            'total_size_mb': round(total_size_mb, 2),
            'by_type': stats,
            'metadata_entries': len(self.metadata)
        }