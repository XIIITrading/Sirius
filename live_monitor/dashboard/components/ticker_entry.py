# live_monitor/dashboard/components/ticker_entry.py
"""
Ticker Entry Component for Live Monitor Dashboard
"""

import logging
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QPushButton
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QFont

from ...styles import BaseStyles

logger = logging.getLogger(__name__)


class TickerEntry(QWidget):
    """
    Widget for entering and displaying ticker symbols
    Submits on Enter key or button click
    """
    
    # Signal emitted when ticker is submitted
    ticker_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_ticker = None
        self.init_ui()
        self.apply_styles()
        
    def init_ui(self):
        """Initialize the UI components"""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Container widget
        container = QWidget()
        container.setObjectName("ticker_entry_container")
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(10, 10, 10, 10)
        container_layout.setSpacing(10)
        
        # Title
        title_label = QLabel("TICKER SYMBOL")
        title_label.setObjectName("section_title")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(title_label)
        
        # Entry section
        entry_widget = QWidget()
        entry_layout = QVBoxLayout(entry_widget)
        entry_layout.setContentsMargins(0, 0, 0, 0)
        entry_layout.setSpacing(5)
        
        # Create horizontal layout for input and button
        input_layout = QHBoxLayout()
        input_layout.setSpacing(5)
        
        # Ticker input field
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Enter ticker symbol...")
        self.ticker_input.setMaxLength(10)
        self.ticker_input.setObjectName("ticker_input")
        
        # Connect Enter key to submit
        self.ticker_input.returnPressed.connect(self.submit_ticker)
        
        # Submit button
        self.submit_button = QPushButton("Submit")
        self.submit_button.setObjectName("submit_button")
        self.submit_button.clicked.connect(self.submit_ticker)
        self.submit_button.setFixedWidth(80)
        
        # Style the button
        self.submit_button.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                border: none;
                border-radius: 4px;
                padding: 6px;
                font-weight: bold;
                color: white;
            }
            QPushButton:hover {
                background-color: #059669;
            }
            QPushButton:pressed {
                background-color: #047857;
            }
        """)
        
        # Add widgets to layout
        input_layout.addWidget(self.ticker_input)
        input_layout.addWidget(self.submit_button)
        
        entry_layout.addLayout(input_layout)
        
        # Current ticker display
        display_layout = QHBoxLayout()
        display_layout.setContentsMargins(0, 5, 0, 0)
        
        label = QLabel("Current:")
        label.setObjectName("ticker_label")
        display_layout.addWidget(label)
        
        self.ticker_display = QLabel("None")
        self.ticker_display.setObjectName("ticker_value")
        self.ticker_display.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        display_layout.addWidget(self.ticker_display)
        
        display_layout.addStretch()
        
        entry_layout.addLayout(display_layout)
        container_layout.addWidget(entry_widget)
        
        # Add container to main layout
        layout.addWidget(container)
        
    def submit_ticker(self):
        """Submit the current ticker"""
        ticker = self.ticker_input.text().strip().upper()
        
        # Validate ticker
        if not ticker:
            return
            
        if ticker == self.current_ticker:
            return
            
        # Update current ticker
        self.current_ticker = ticker
        
        # Update display
        self.ticker_display.setText(ticker)
        self.ticker_display.setStyleSheet("QLabel { color: #10b981; }")
        
        # Clear input
        self.ticker_input.clear()
        
        # Emit signal
        self.ticker_changed.emit(ticker)
        
        # Visual feedback on button
        self.submit_button.setText("✓")
        self.submit_button.setStyleSheet("""
            QPushButton {
                background-color: #059669;
                border: none;
                border-radius: 4px;
                padding: 6px;
                font-weight: bold;
                color: white;
            }
        """)
        
        # Reset button after delay
        QTimer.singleShot(1000, self.reset_button)
        
        # Log the change
        logger.info(f"Ticker submitted: {ticker}")
    
    def reset_button(self):
        """Reset submit button appearance"""
        self.submit_button.setText("Submit")
        self.submit_button.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                border: none;
                border-radius: 4px;
                padding: 6px;
                font-weight: bold;
                color: white;
            }
            QPushButton:hover {
                background-color: #059669;
            }
            QPushButton:pressed {
                background-color: #047857;
            }
        """)
    
    def apply_styles(self):
        """Apply component styles"""
        self.setStyleSheet(f"""
            #ticker_entry_container {{
                background-color: {BaseStyles.CARD_BACKGROUND};
                border: 1px solid {BaseStyles.BORDER_COLOR};
                border-radius: 8px;
            }}
            
            #section_title {{
                color: {BaseStyles.TEXT_PRIMARY};
                font-size: 12px;
                font-weight: bold;
                letter-spacing: 1px;
            }}
            
            #ticker_input {{
                background-color: {BaseStyles.INPUT_BACKGROUND};
                border: 1px solid {BaseStyles.BORDER_COLOR};
                border-radius: 4px;
                padding: 8px;
                color: {BaseStyles.TEXT_PRIMARY};
                font-size: 14px;
            }}
            
            #ticker_input:focus {{
                border: 1px solid {BaseStyles.ACCENT_COLOR};
                outline: none;
            }}
            
            #ticker_label {{
                color: {BaseStyles.TEXT_SECONDARY};
                font-size: 12px;
            }}
            
            #ticker_value {{
                color: {BaseStyles.TEXT_PRIMARY};
            }}
        """)
    
    def get_current_ticker(self) -> str:
        """Get the current ticker symbol"""
        return self.current_ticker or ""
    
    def set_ticker(self, ticker: str):
        """Set ticker programmatically"""
        if ticker:
            self.ticker_input.setText(ticker)
            self.submit_ticker()