"""
Polygon Data Manager - Modular data fetching with intelligent caching
"""
import os
from dotenv import load_dotenv

# Load environment variables when the module is imported
load_dotenv()

from .data_manager import PolygonDataManager
from .models import DataRequest, CacheMetadata

__all__ = ['PolygonDataManager', 'DataRequest', 'CacheMetadata']