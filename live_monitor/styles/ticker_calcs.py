"""
Styles for ticker calculations module
"""

from .base_styles import BaseStyles

class TickerCalcStyles:
    
    @staticmethod
    def get_stylesheet():
        return f"""
        /* Container styling */
        QWidget#ticker_calc_container {{
            background-color: {BaseStyles.BACKGROUND_SECONDARY};
            border: 1px solid {BaseStyles.BORDER_COLOR};
            border-radius: 5px;
            padding: 10px;
        }}
        
        /* Section headers */
        QLabel#calc_header {{
            font-size: {BaseStyles.FONT_SIZE_LARGE};
            font-weight: bold;
            color: {BaseStyles.TEXT_PRIMARY};
            padding: 5px 0;
        }}
        
        /* Calculation labels */
        QLabel#calc_label {{
            color: {BaseStyles.TEXT_SECONDARY};
            font-size: {BaseStyles.FONT_SIZE_SMALL};
        }}
        
        /* Calculation values */
        QLabel#calc_value {{
            font-size: {BaseStyles.FONT_SIZE_NORMAL};
            font-weight: bold;
            color: {BaseStyles.TEXT_PRIMARY};
        }}
        
        /* Positive values */
        QLabel#calc_value_positive {{
            color: {BaseStyles.POSITIVE};
            font-weight: bold;
        }}
        
        /* Negative values */
        QLabel#calc_value_negative {{
            color: {BaseStyles.NEGATIVE};
            font-weight: bold;
        }}
        
        /* Percentage values */
        QLabel#calc_percentage {{
            font-size: {BaseStyles.FONT_SIZE_SMALL};
            padding-left: 5px;
        }}
        
        /* Divider lines */
        QFrame#calc_divider {{
            background-color: {BaseStyles.BORDER_COLOR};
            max-height: 1px;
            margin: 5px 0;
        }}
        """