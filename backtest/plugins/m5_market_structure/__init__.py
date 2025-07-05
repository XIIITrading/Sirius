# modules/calculations/market_structure/__init__.py
"""
Market Structure Analysis Module
Provides fractal-based market structure detection for trend analysis across multiple timeframes
"""

from .m1_market_structure import (
    MarketStructureAnalyzer,
    MarketStructureSignal,
    MarketStructureMetrics,
    Fractal,
    Candle
)

from .m5_market_structure import (
    M5MarketStructureAnalyzer
)

__all__ = [
    # M1 (1-minute) exports
    'MarketStructureAnalyzer',
    'MarketStructureSignal', 
    'MarketStructureMetrics',
    'Fractal',
    'Candle',
    # M5 (5-minute) exports
    'M5MarketStructureAnalyzer'
]

# Module metadata
__version__ = '1.0.0'
__author__ = 'Trading System'