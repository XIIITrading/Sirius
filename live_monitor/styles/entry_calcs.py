"""
Styles for entry/size calculations module
"""

from .base_styles import BaseStyles

class EntryCalcStyles:
    
    @staticmethod
    def get_stylesheet():
        return f"""
        /* Container */
        QWidget#entry_calc_container {{
            background-color: {BaseStyles.BACKGROUND_SECONDARY};
            border: 1px solid {BaseStyles.BORDER_COLOR};
            border-radius: 5px;
            padding: 10px;
        }}
        
        /* Input fields */
        QLineEdit#entry_input {{
            background-color: {BaseStyles.BACKGROUND_TERTIARY};
            border: 1px solid {BaseStyles.BORDER_COLOR};
            padding: 8px;
            font-size: {BaseStyles.FONT_SIZE_NORMAL};
            border-radius: 3px;
        }}
        
        QLineEdit#entry_input:focus {{
            border: 1px solid {BaseStyles.ACCENT_PRIMARY};
            background-color: {BaseStyles.BACKGROUND_SECONDARY};
        }}
        
        /* Labels */
        QLabel#entry_label {{
            color: {BaseStyles.TEXT_SECONDARY};
            font-size: {BaseStyles.FONT_SIZE_SMALL};
            margin-bottom: 3px;
        }}
        
        /* Result displays */
        QLabel#entry_result {{
            background-color: {BaseStyles.BACKGROUND_TERTIARY};
            border: 1px solid {BaseStyles.BORDER_COLOR};
            padding: 8px;
            border-radius: 3px;
            font-weight: bold;
            font-size: {BaseStyles.FONT_SIZE_NORMAL};
        }}
        
        /* Calculate button */
        QPushButton#calculate_button {{
            background-color: {BaseStyles.ACCENT_PRIMARY};
            font-size: {BaseStyles.FONT_SIZE_NORMAL};
            padding: 10px 20px;
            font-weight: bold;
            margin-top: 10px;
        }}
        
        QPushButton#calculate_button:hover {{
            background-color: {BaseStyles.ACCENT_HOVER};
        }}
        
        /* Risk level indicators */
        QLabel#risk_low {{
            color: {BaseStyles.POSITIVE};
            font-weight: bold;
        }}
        
        QLabel#risk_medium {{
            color: {BaseStyles.WARNING};
            font-weight: bold;
        }}
        
        QLabel#risk_high {{
            color: {BaseStyles.NEGATIVE};
            font-weight: bold;
        }}
        """