"""
Ticker Entry Widget
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QLabel
from PyQt6.QtCore import pyqtSignal, Qt
from ...styles import BaseStyles


class TickerEntry(QWidget):
    """Widget for entering ticker symbols"""
    
    # Signal emitted when ticker is changed
    ticker_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.apply_styles()
        
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        # Label
        self.label = QLabel("Ticker Symbol")
        self.label.setObjectName("ticker_label")
        layout.addWidget(self.label)
        
        # Input field
        self.ticker_input = QLineEdit()
        self.ticker_input.setObjectName("ticker_input")
        self.ticker_input.setPlaceholderText("Enter ticker (e.g., AAPL)")
        self.ticker_input.setMaxLength(10)
        self.ticker_input.textChanged.connect(self._on_ticker_changed)
        layout.addWidget(self.ticker_input)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
    def _on_ticker_changed(self, text):
        """Handle ticker text changes"""
        # Convert to uppercase
        upper_text = text.upper()
        if upper_text != text:
            self.ticker_input.setText(upper_text)
        
        # Emit signal
        self.ticker_changed.emit(upper_text)
        
    def apply_styles(self):
        """Apply styles to the widget"""
        self.setObjectName("ticker_entry_widget")
        
        # Additional specific styles
        style = f"""
        QWidget#ticker_entry_widget {{
            background-color: {BaseStyles.BACKGROUND_SECONDARY};
            border: 1px solid {BaseStyles.BORDER_COLOR};
            border-radius: 5px;
        }}
        
        QLabel#ticker_label {{
            font-weight: bold;
            font-size: {BaseStyles.FONT_SIZE_NORMAL};
            color: {BaseStyles.TEXT_SECONDARY};
        }}
        
        QLineEdit#ticker_input {{
            font-size: {BaseStyles.FONT_SIZE_LARGE};
            font-weight: bold;
            padding: 8px;
            text-transform: uppercase;
        }}
        """
        self.setStyleSheet(style)
        
    def get_ticker(self):
        """Get the current ticker value"""
        return self.ticker_input.text()
        
    def set_ticker(self, ticker):
        """Set the ticker value"""
        self.ticker_input.setText(ticker)