# live_monitor/signals/utils/__init__.py
from .signal_descriptions import (
    generate_ema_description,
    generate_trend_description,
    generate_market_structure_description,  # ADD THIS LINE
    get_source_identifier
)

__all__ = [
    'generate_ema_description',
    'generate_trend_description',
    'generate_market_structure_description',  # ADD THIS LINE
    'get_source_identifier'
]