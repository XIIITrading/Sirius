# live_monitor/dashboard/segments/signal_display_segment.py
"""
Signal Display Segment - Methods for updating signal displays
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SignalDisplaySegment:
    """Dashboard segment for signal display updates"""
    
    def update_signal_display(self, signal_value: float, category: str, source: str):
        """Update the signal label with color coding for specific source"""
        # Determine which label to update based on source
        if source == 'M1':
            label = self.m1_signal_label
            prefix = 'M1'
        elif source == 'M5':
            label = self.m5_signal_label
            prefix = 'M5'
        elif source == 'M15':
            label = self.m15_signal_label
            prefix = 'M15'
        elif source == 'STAT':
            label = self.stat_signal_label
            prefix = 'STAT'
        elif source == 'M1 MSTRUCT':
            label = self.m1_mstruct_label
            prefix = 'M1 MSTRUCT'
        elif source == 'M5 MSTRUCT':
            label = self.m5_mstruct_label
            prefix = 'M5 MSTRUCT'
        elif source == 'M15 MSTRUCT':
            label = self.m15_mstruct_label
            prefix = 'M15 MSTRUCT'
        elif source == 'M5 TREND':
            label = self.m5_trend_label
            prefix = 'M5 TREND'
        elif source == 'M15 TREND':
            label = self.m15_trend_label
            prefix = 'M15 TREND'
        else:
            logger.warning(f"Unknown signal source: {source}")
            return
        
        # Format the signal display
        label.setText(f"{prefix}: {category} ({signal_value:+.1f})")
        
        # Apply color based on signal value
        if signal_value >= 25:
            # Bullish - Green
            color = "#26a69a"
        elif signal_value > 0:
            # Weak Bullish - Light Green
            color = "#66bb6a"
        elif signal_value > -25:
            # Weak Bearish - Light Red
            color = "#ef5350"
        else:
            # Bearish - Red
            color = "#d32f2f"
        
        # Apply the style
        label.setStyleSheet(f"QLabel {{ color: {color}; font-weight: bold; margin-left: 10px; }}")
        
        # Update the last update time
        self.update_time_label.setText(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
        
        # Log the update
        logger.debug(f"Updated {source} signal: {category} ({signal_value:+.1f})")