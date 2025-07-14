# live_monitor/signals/processors/ema/__init__.py
from .m1_ema_processor import M1EMAProcessor
from .m5_ema_processor import M5EMAProcessor
from .m15_ema_processor import M15EMAProcessor

__all__ = ['M1EMAProcessor', 'M5EMAProcessor', 'M15EMAProcessor']