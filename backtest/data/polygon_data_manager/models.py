"""Data models for the Polygon data manager"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class CacheMetadata:
    """Metadata for cached data"""
    symbol: str
    data_type: str  # 'bars', 'trades', 'quotes'
    timeframe: str
    start_time: datetime
    end_time: datetime
    cached_at: datetime
    data_points: int
    cache_key: str


@dataclass
class DataRequest:
    """Track individual data requests for reporting"""
    request_id: str
    plugin_name: str
    symbol: str
    data_type: str  # 'bars', 'trades', 'quotes'
    timeframe: str
    start_time: datetime
    end_time: datetime
    requested_at: datetime
    source: str  # 'memory_cache', 'file_cache', 'polygon_api'
    success: bool
    returned_count: int
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    error: Optional[str] = None
    processing_time_ms: float = 0.0
    cache_key: Optional[str] = None