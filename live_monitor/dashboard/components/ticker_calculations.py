# live_monitor/dashboard/components/ticker_calculations.py

import logging
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QFrame, QGridLayout)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from ...styles import TickerCalcStyles

logger = logging.getLogger(__name__)


class TickerCalculations(QWidget):
    """Widget for displaying ticker calculations"""
    
    # Signals
    calculation_updated = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_data = {}
        self.init_ui()
        self.apply_styles()
        
    def init_ui(self):
        """Initialize the UI"""
        self.setObjectName("ticker_calc_container")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Header
        header = QLabel("Ticker Calculations")
        header.setObjectName("calc_header")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # Price section
        price_frame = self.create_price_section()
        layout.addWidget(price_frame)
        
        # Divider
        divider = QFrame()
        divider.setObjectName("calc_divider")
        divider.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(divider)
        
        # Volume section
        volume_frame = self.create_volume_section()
        layout.addWidget(volume_frame)
        
        # Divider
        divider2 = QFrame()
        divider2.setObjectName("calc_divider")
        divider2.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(divider2)
        
        # Range section
        range_frame = self.create_range_section()
        layout.addWidget(range_frame)
        
        # Add stretch
        layout.addStretch()
        
    def create_price_section(self):
        """Create price information section"""
        frame = QFrame()
        layout = QGridLayout(frame)
        layout.setSpacing(5)
        
        # Last price
        layout.addWidget(self.create_label("Last Price:", "calc_label"), 0, 0)
        self.last_price_label = self.create_label("--", "calc_value")
        layout.addWidget(self.last_price_label, 0, 1)
        
        # Change
        layout.addWidget(self.create_label("Change:", "calc_label"), 1, 0)
        self.change_label = self.create_label("--", "calc_value")
        layout.addWidget(self.change_label, 1, 1)
        
        # Bid/Ask
        layout.addWidget(self.create_label("Bid/Ask:", "calc_label"), 2, 0)
        self.bid_ask_label = self.create_label("--", "calc_value")
        layout.addWidget(self.bid_ask_label, 2, 1)
        
        # Spread
        layout.addWidget(self.create_label("Spread:", "calc_label"), 3, 0)
        self.spread_label = self.create_label("--", "calc_value")
        layout.addWidget(self.spread_label, 3, 1)
        
        return frame
        
    def create_volume_section(self):
        """Create volume information section"""
        frame = QFrame()
        layout = QGridLayout(frame)
        layout.setSpacing(5)
        
        # Volume
        layout.addWidget(self.create_label("Volume:", "calc_label"), 0, 0)
        self.volume_label = self.create_label("--", "calc_value")
        layout.addWidget(self.volume_label, 0, 1)
        
        # Average volume
        layout.addWidget(self.create_label("Avg Volume:", "calc_label"), 1, 0)
        self.avg_volume_label = self.create_label("--", "calc_value")
        layout.addWidget(self.avg_volume_label, 1, 1)
        
        # Volume ratio
        layout.addWidget(self.create_label("Vol Ratio:", "calc_label"), 2, 0)
        self.vol_ratio_label = self.create_label("--", "calc_value")
        layout.addWidget(self.vol_ratio_label, 2, 1)
        
        return frame
        
    def create_range_section(self):
        """Create range information section"""
        frame = QFrame()
        layout = QGridLayout(frame)
        layout.setSpacing(5)
        
        # Day range
        layout.addWidget(self.create_label("Day Range:", "calc_label"), 0, 0)
        self.day_range_label = self.create_label("--", "calc_value")
        layout.addWidget(self.day_range_label, 0, 1)
        
        # ATR
        layout.addWidget(self.create_label("ATR (14):", "calc_label"), 1, 0)
        self.atr_label = self.create_label("--", "calc_value")
        layout.addWidget(self.atr_label, 1, 1)
        
        # Position in range
        layout.addWidget(self.create_label("Range Pos:", "calc_label"), 2, 0)
        self.range_pos_label = self.create_label("--", "calc_value")
        layout.addWidget(self.range_pos_label, 2, 1)
        
        return frame
        
    def create_label(self, text, object_name):
        """Create a styled label"""
        label = QLabel(text)
        label.setObjectName(object_name)
        return label
        
    def apply_styles(self):
        """Apply styles to the widget"""
        self.setStyleSheet(TickerCalcStyles.get_stylesheet())
        
    def update_calculations(self, data: dict):
        """Update calculations with new market data"""
        self.current_data.update(data)
        
        # Update price info
        if 'last_price' in data and data['last_price'] is not None:
            self.last_price_label.setText(f"${data['last_price']:.2f}")
            
        # Update change - with None checking
        if 'change' in data and 'change_pct' in data:
            change = data['change']
            change_pct = data['change_pct']
            
            if change is not None and change_pct is not None:
                change_label = self.change_label
                if change >= 0:
                    change_label.setObjectName("calc_value_positive")
                    change_label.setText(f"+${change:.2f} ({change_pct:.2f}%)")
                else:
                    change_label.setObjectName("calc_value_negative")
                    change_label.setText(f"${change:.2f} ({change_pct:.2f}%)")
                change_label.setStyleSheet(TickerCalcStyles.get_stylesheet())
            else:
                # Handle None values
                self.change_label.setText("--")
        
        # Update bid/ask
        if 'bid' in data and 'ask' in data:
            bid = data['bid']
            ask = data['ask']
            
            if bid is not None and ask is not None:
                self.bid_ask_label.setText(f"${bid:.2f} / ${ask:.2f}")
                
                # Calculate spread
                spread = ask - bid
                spread_pct = (spread / ask * 100) if ask > 0 else 0
                self.spread_label.setText(f"${spread:.3f} ({spread_pct:.2f}%)")
            else:
                self.bid_ask_label.setText("--")
                self.spread_label.setText("--")
        
        # Update volume
        if 'volume' in data and data['volume'] is not None:
            volume = data['volume']
            self.volume_label.setText(f"{volume:,}")
            
            # Update volume ratio if we have average volume
            if 'avg_volume' in data and data['avg_volume'] is not None:
                avg_vol = data['avg_volume']
                self.avg_volume_label.setText(f"{avg_vol:,}")
                
                if avg_vol > 0:
                    vol_ratio = volume / avg_vol
                    self.vol_ratio_label.setText(f"{vol_ratio:.2f}x")
                else:
                    self.vol_ratio_label.setText("--")
        
        # Update range
        if 'day_high' in data and 'day_low' in data:
            high = data['day_high']
            low = data['day_low']
            
            if high is not None and low is not None:
                self.day_range_label.setText(f"${low:.2f} - ${high:.2f}")
                
                # Calculate position in range
                if 'last_price' in data and data['last_price'] is not None:
                    last = data['last_price']
                    if high > low:
                        range_pos = ((last - low) / (high - low)) * 100
                        self.range_pos_label.setText(f"{range_pos:.1f}%")
                    else:
                        self.range_pos_label.setText("--")
        
        # Update ATR if available
        if 'atr' in data and data['atr'] is not None:
            self.atr_label.setText(f"${data['atr']:.2f}")
            
        # Emit update signal
        self.calculation_updated.emit(self.current_data)
        
    def clear_calculations(self):
        """Clear all calculations"""
        self.current_data.clear()
        
        # Reset all labels
        self.last_price_label.setText("--")
        self.change_label.setText("--")
        self.change_label.setObjectName("calc_value")
        self.bid_ask_label.setText("--")
        self.spread_label.setText("--")
        self.volume_label.setText("--")
        self.avg_volume_label.setText("--")
        self.vol_ratio_label.setText("--")
        self.day_range_label.setText("--")
        self.atr_label.setText("--")
        self.range_pos_label.setText("--")
        
        # Reapply styles
        self.apply_styles()