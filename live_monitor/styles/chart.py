"""
Styles for chart component
"""

from .base_styles import BaseStyles

class ChartStyles:
    
    # Chart color scheme
    CHART_BACKGROUND = "#1a1a1a"
    CHART_GRID = "#333333"
    CHART_TEXT = "#cccccc"
    
    # Candlestick colors
    CANDLE_BULL_BODY = "#26a69a"
    CANDLE_BULL_WICK = "#26a69a"
    CANDLE_BEAR_BODY = "#ef5350"
    CANDLE_BEAR_WICK = "#ef5350"
    
    # Indicator colors
    HVN_COLOR = "#ffa726"
    HVN_ALPHA = 0.3
    
    ORDER_BLOCK_BULL = "#26a69a"
    ORDER_BLOCK_BEAR = "#ef5350"
    ORDER_BLOCK_ALPHA = 0.2
    
    CAMARILLA_PIVOT = "#9c27b0"
    CAMARILLA_SUPPORT = "#2196f3"
    CAMARILLA_RESISTANCE = "#f44336"
    
    # Entry/Exit markers
    ENTRY_MARKER = "#00ff00"
    EXIT_MARKER = "#ff0000"
    
    @staticmethod
    def get_stylesheet():
        return f"""
        /* Chart container */
        QWidget#chart_container {{
            background-color: {BaseStyles.BACKGROUND_SECONDARY};
            border: 1px solid {BaseStyles.BORDER_COLOR};
            border-radius: 5px;
            padding: 5px;
        }}
        
        /* Chart header */
        QLabel#chart_header {{
            font-size: {BaseStyles.FONT_SIZE_LARGE};
            font-weight: bold;
            padding: 10px;
            background-color: {BaseStyles.BACKGROUND_TERTIARY};
            border-bottom: 1px solid {BaseStyles.BORDER_COLOR};
        }}
        
        /* Chart controls */
        QWidget#chart_controls {{
            background-color: {BaseStyles.BACKGROUND_TERTIARY};
            padding: 5px;
            border-bottom: 1px solid {BaseStyles.BORDER_COLOR};
        }}
        
        QPushButton#chart_button {{
            padding: 5px 10px;
            margin: 0 2px;
            font-size: {BaseStyles.FONT_SIZE_SMALL};
        }}
        
        /* Timeframe selector */
        QComboBox#timeframe_selector {{
            padding: 5px 10px;
            margin: 0 5px;
        }}
        
        /* Indicator toggles */
        QCheckBox#indicator_toggle {{
            padding: 5px;
            margin: 0 5px;
        }}
        
        QCheckBox#indicator_toggle::indicator {{
            width: 16px;
            height: 16px;
        }}
        
        QCheckBox#indicator_toggle::indicator:checked {{
            background-color: {BaseStyles.ACCENT_PRIMARY};
            border: 2px solid {BaseStyles.ACCENT_PRIMARY};
        }}
        
        QCheckBox#indicator_toggle::indicator:unchecked {{
            background-color: {BaseStyles.BACKGROUND_SECONDARY};
            border: 2px solid {BaseStyles.BORDER_COLOR};
        }}
        """