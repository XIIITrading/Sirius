# modules/calculations/market_structure/__init__.py
"""
Market Structure Analysis Module
Provides fractal-based market structure detection for trend analysis
"""

from .m1_market_structure import (
    MarketStructureAnalyzer,
    MarketStructureSignal,
    MarketStructureMetrics,
    Fractal,
    Candle
)

__all__ = [
    'MarketStructureAnalyzer',
    'MarketStructureSignal', 
    'MarketStructureMetrics',
    'Fractal',
    'Candle'
]

# Module metadata
__version__ = '1.0.0'
__author__ = 'Trading System'