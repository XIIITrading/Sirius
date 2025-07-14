# live_monitor/signals/processors/trend/__init__.py
from .m1_trend_processor import M1TrendProcessor
from .m5_trend_processor import M5TrendProcessor
from .m15_trend_processor import M15TrendProcessor

__all__ = ['M1TrendProcessor', 'M5TrendProcessor', 'M15TrendProcessor']