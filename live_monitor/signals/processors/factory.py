"""
Factory for creating signal processors
"""

from typing import Dict, Type
from .base_processor import BaseSignalProcessor
from .ema import M1EMAProcessor, M5EMAProcessor, M15EMAProcessor
from .trend import M1TrendProcessor, M5TrendProcessor, M15TrendProcessor
from .market_structure import (
    M1MarketStructureProcessor, 
    M5MarketStructureProcessor, 
    M15MarketStructureProcessor
)


class ProcessorFactory:
    """Factory for creating signal processors"""
    
    _processors: Dict[str, Type[BaseSignalProcessor]] = {
        'M1_EMA': M1EMAProcessor,
        'M5_EMA': M5EMAProcessor,
        'M15_EMA': M15EMAProcessor,
        'STATISTICAL_TREND': M1TrendProcessor,
        'STATISTICAL_TREND_5M': M5TrendProcessor,
        'STATISTICAL_TREND_15M': M15TrendProcessor,
        'M1_MARKET_STRUCTURE': M1MarketStructureProcessor,
        'M5_MARKET_STRUCTURE': M5MarketStructureProcessor,
        'M15_MARKET_STRUCTURE': M15MarketStructureProcessor,
    }
    
    @classmethod
    def create_processor(cls, processor_type: str) -> BaseSignalProcessor:
        """Create a processor instance"""
        if processor_type not in cls._processors:
            raise ValueError(f"Unknown processor type: {processor_type}")
        return cls._processors[processor_type]()
    
    @classmethod
    def register_processor(cls, name: str, processor_class: Type[BaseSignalProcessor]):
        """Register a new processor type"""
        cls._processors[name] = processor_class
    
    @classmethod
    def get_available_processors(cls) -> list:
        """Get list of available processor types"""
        return list(cls._processors.keys())