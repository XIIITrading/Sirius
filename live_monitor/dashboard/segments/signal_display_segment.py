# live_monitor/dashboard/segments/signal_display_segment.py
"""
Signal Display Segment - Methods for updating signal displays
"""

class SignalDisplaySegment:
    """Dashboard segment for signal display updates"""
    
    def update_signal_display(self, signal_value: float, category: str, timeframe: str):
        """Update the signal label with color coding for specific timeframe"""
        # Determine which label to update
        if timeframe == 'M1':
            label = self.m1_signal_label
        elif timeframe == 'M5':
            label = self.m5_signal_label
        elif timeframe == 'M15':
            label = self.m15_signal_label
        else:
            return
        
        # Format the signal display
        label.setText(f"{timeframe}: {category} ({signal_value:+.1f})")
        
        # Apply color based on category
        if signal_value >= 25:
            # Bullish - Green
            label.setStyleSheet("QLabel { color: #26a69a; font-weight: bold; margin-left: 10px; }")
        elif signal_value > 0:
            # Weak Bullish - Light Green
            label.setStyleSheet("QLabel { color: #66bb6a; font-weight: bold; margin-left: 10px; }")
        elif signal_value > -25:
            # Weak Bearish - Light Red
            label.setStyleSheet("QLabel { color: #ef5350; font-weight: bold; margin-left: 10px; }")
        else:
            # Bearish - Red
            label.setStyleSheet("QLabel { color: #d32f2f; font-weight: bold; margin-left: 10px; }")