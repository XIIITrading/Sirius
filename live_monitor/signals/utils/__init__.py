# live_monitor/signals/utils/__init__.py
from .signal_descriptions import (
    generate_ema_description,
    generate_trend_description,
    get_source_identifier
)

__all__ = [
    'generate_ema_description',
    'generate_trend_description',
    'get_source_identifier'
]