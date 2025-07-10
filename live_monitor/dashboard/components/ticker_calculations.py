"""
Ticker Calculations Display Module
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QFrame, QGridLayout)
from PyQt6.QtCore import Qt, pyqtSlot
from ...styles import TickerCalcStyles, BaseStyles


class TickerCalculations(QWidget):
    """Display widget for ticker calculations"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.calculation_labels = {}
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
        layout.addWidget(header)
        
        # Divider
        divider = QFrame()
        divider.setObjectName("calc_divider")
        divider.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(divider)
        
        # Calculations grid
        grid = QGridLayout()
        grid.setSpacing(8)
        
        # Create calculation rows
        calculations = [
            ("Last Price", "last_price", "0.00"),
            ("Bid/Ask", "bid_ask", "0.00 / 0.00"),
            ("Day Change", "day_change", "0.00 (0.00%)"),
            ("Volume", "volume", "0"),
            ("Avg Volume", "avg_volume", "0"),
            ("Day Range", "day_range", "0.00 - 0.00"),
            ("52W Range", "52w_range", "0.00 - 0.00"),
            ("Market Cap", "market_cap", "0"),
        ]
        
        for i, (label_text, key, default) in enumerate(calculations):
            # Label
            label = QLabel(f"{label_text}:")
            label.setObjectName("calc_label")
            grid.addWidget(label, i, 0, Qt.AlignmentFlag.AlignLeft)
            
            # Value
            value = QLabel(default)
            value.setObjectName("calc_value")
            self.calculation_labels[key] = value
            grid.addWidget(value, i, 1, Qt.AlignmentFlag.AlignRight)
            
        layout.addLayout(grid)
        layout.addStretch()
        
    def apply_styles(self):
        """Apply styles to the widget"""
        self.setStyleSheet(TickerCalcStyles.get_stylesheet())
        
    @pyqtSlot(dict)
    def update_calculations(self, data):
        """Update calculation displays with new data"""
        # Update last price
        if 'last_price' in data:
            self.calculation_labels['last_price'].setText(f"${data['last_price']:.2f}")
            
        # Update bid/ask
        if 'bid' in data and 'ask' in data:
            self.calculation_labels['bid_ask'].setText(
                f"${data['bid']:.2f} / ${data['ask']:.2f}"
            )
            
        # Update day change
        if 'change' in data and 'change_percent' in data:
            change_label = self.calculation_labels['day_change']
            change = data['change']
            change_pct = data['change_percent']
            
            change_label.setText(f"${change:.2f} ({change_pct:.2f}%)")
            
            # Apply color based on positive/negative
            if change > 0:
                change_label.setObjectName("calc_value_positive")
            elif change < 0:
                change_label.setObjectName("calc_value_negative")
            else:
                change_label.setObjectName("calc_value")
                
        # Update volume
        if 'volume' in data:
            volume = data['volume']
            self.calculation_labels['volume'].setText(f"{volume:,.0f}")
            
        # Update other fields as needed...
        
    def clear_calculations(self):
        """Clear all calculation displays"""
        for label in self.calculation_labels.values():
            label.setText("--")
            label.setObjectName("calc_value")