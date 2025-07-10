"""
Entry/Size Calculations Module
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QGridLayout, QFrame)
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt
from ...styles import EntryCalcStyles, BaseStyles


class EntryCalculations(QWidget):
    """Widget for entry and position size calculations"""
    
    # Signal emitted when calculations are complete
    calculation_complete = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.inputs = {}
        self.results = {}
        self.init_ui()
        self.apply_styles()
        
    def init_ui(self):
        """Initialize the UI"""
        self.setObjectName("entry_calc_container")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Header
        header = QLabel("Entry / Size Calculations")
        header.setObjectName("calc_header")
        header.setStyleSheet(f"font-size: {BaseStyles.FONT_SIZE_LARGE}; font-weight: bold;")
        layout.addWidget(header)
        
        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(f"background-color: {BaseStyles.BORDER_COLOR};")
        layout.addWidget(divider)
        
        # Input fields
        input_grid = QGridLayout()
        input_grid.setSpacing(8)
        
        # Create input fields
        input_fields = [
            ("Account Size", "account_size", "100000"),
            ("Risk %", "risk_percent", "1.0"),
            ("Entry Price", "entry_price", "0.00"),
            ("Stop Loss", "stop_loss", "0.00"),
        ]
        
        for i, (label_text, key, placeholder) in enumerate(input_fields):
            # Label
            label = QLabel(f"{label_text}:")
            label.setObjectName("entry_label")
            input_grid.addWidget(label, i, 0, Qt.AlignmentFlag.AlignLeft)
            
            # Input
            input_field = QLineEdit()
            input_field.setObjectName("entry_input")
            input_field.setPlaceholderText(placeholder)
            self.inputs[key] = input_field
            input_grid.addWidget(input_field, i, 1)
            
        layout.addLayout(input_grid)
        
        # Calculate button
        self.calculate_btn = QPushButton("Calculate")
        self.calculate_btn.setObjectName("calculate_button")
        self.calculate_btn.clicked.connect(self.calculate_position)
        layout.addWidget(self.calculate_btn)
        
        # Results section
        results_label = QLabel("Results:")
        results_label.setObjectName("entry_label")
        results_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(results_label)
        
        # Results grid
        results_grid = QGridLayout()
        results_grid.setSpacing(8)
        
        result_fields = [
            ("Position Size", "position_size"),
            ("Shares/Contracts", "shares"),
            ("Risk Amount", "risk_amount"),
            ("Risk/Reward", "risk_reward"),
        ]
        
        for i, (label_text, key) in enumerate(result_fields):
            # Label
            label = QLabel(f"{label_text}:")
            label.setObjectName("entry_label")
            results_grid.addWidget(label, i, 0, Qt.AlignmentFlag.AlignLeft)
            
            # Result
            result = QLabel("--")
            result.setObjectName("entry_result")
            self.results[key] = result
            results_grid.addWidget(result, i, 1)
            
        layout.addLayout(results_grid)
        layout.addStretch()
        
    def apply_styles(self):
        """Apply styles to the widget"""
        self.setStyleSheet(EntryCalcStyles.get_stylesheet())
        
    @pyqtSlot()
    def calculate_position(self):
        """Calculate position size based on inputs"""
        try:
            # Get input values
            account_size = float(self.inputs['account_size'].text() or 0)
            risk_percent = float(self.inputs['risk_percent'].text() or 0)
            entry_price = float(self.inputs['entry_price'].text() or 0)
            stop_loss = float(self.inputs['stop_loss'].text() or 0)
            
            if account_size <= 0 or risk_percent <= 0 or entry_price <= 0 or stop_loss <= 0:
                return
                
            # Calculate risk amount
            risk_amount = account_size * (risk_percent / 100)
            
            # Calculate position size
            price_difference = abs(entry_price - stop_loss)
            if price_difference > 0:
                shares = risk_amount / price_difference
                position_size = shares * entry_price
                
                # Update results
                self.results['position_size'].setText(f"${position_size:,.2f}")
                self.results['shares'].setText(f"{shares:,.0f}")
                self.results['risk_amount'].setText(f"${risk_amount:,.2f}")
                
                # Calculate R:R if we had a target (placeholder)
                self.results['risk_reward'].setText("--")
                
                # Apply risk level styling
                if risk_percent <= 1:
                    self.results['risk_amount'].setObjectName("risk_low")
                elif risk_percent <= 2:
                    self.results['risk_amount'].setObjectName("risk_medium")
                else:
                    self.results['risk_amount'].setObjectName("risk_high")
                    
                # Emit signal with calculation results
                self.calculation_complete.emit({
                    'account_size': account_size,
                    'risk_percent': risk_percent,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'risk_amount': risk_amount,
                    'position_size': position_size,
                    'shares': shares
                })
                
        except ValueError:
            # Handle invalid input
            for result in self.results.values():
                result.setText("Invalid input")
                
    def clear_results(self):
        """Clear all result displays"""
        for result in self.results.values():
            result.setText("--")
            
    @pyqtSlot(float)
    def update_entry_price(self, price):
        """Update entry price from external source"""
        self.inputs['entry_price'].setText(f"{price:.2f}")