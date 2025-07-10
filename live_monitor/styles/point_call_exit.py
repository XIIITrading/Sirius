"""
Styles for point & call exit system
"""

from .base_styles import BaseStyles

class PointCallExitStyles:
    
    @staticmethod
    def get_stylesheet():
        return f"""
        /* Container */
        QWidget#point_call_exit_container {{
            background-color: {BaseStyles.BACKGROUND_SECONDARY};
            border: 1px solid {BaseStyles.BORDER_COLOR};
            border-radius: 5px;
        }}
        
        /* Header */
        QLabel#point_call_exit_header {{
            font-size: {BaseStyles.FONT_SIZE_LARGE};
            font-weight: bold;
            padding: 10px;
            background-color: {BaseStyles.BACKGROUND_TERTIARY};
            border-bottom: 1px solid {BaseStyles.BORDER_COLOR};
        }}
        
        /* Table styling */
        QTableWidget#exit_signals_table {{
            background-color: {BaseStyles.BACKGROUND_SECONDARY};
            alternate-background-color: {BaseStyles.BACKGROUND_TERTIARY};
            gridline-color: {BaseStyles.BORDER_COLOR};
            border: none;
        }}
        
        QTableWidget#exit_signals_table::item {{
            padding: 5px;
        }}
        
        QTableWidget#exit_signals_table::item:selected {{
            background-color: {BaseStyles.ACCENT_PRIMARY};
        }}
        
        /* Exit urgency indicators */
        QLabel#exit_urgent {{
            color: {BaseStyles.NEGATIVE};
            font-weight: bold;
            background-color: rgba(239, 83, 80, 0.2);
            padding: 2px 6px;
            border-radius: 3px;
        }}
        
        QLabel#exit_warning {{
            color: {BaseStyles.WARNING};
            font-weight: bold;
            background-color: rgba(255, 167, 38, 0.2);
            padding: 2px 6px;
            border-radius: 3px;
        }}
        
        QLabel#exit_normal {{
            color: {BaseStyles.TEXT_SECONDARY};
            padding: 2px 6px;
        }}
        
        /* Profit/Loss indicators */
        QLabel#pnl_positive {{
            color: {BaseStyles.POSITIVE};
            font-weight: bold;
            font-size: {BaseStyles.FONT_SIZE_NORMAL};
        }}
        
        QLabel#pnl_negative {{
            color: {BaseStyles.NEGATIVE};
            font-weight: bold;
            font-size: {BaseStyles.FONT_SIZE_NORMAL};
        }}
        
        /* Target/Stop badges */
        QLabel#target_reached {{
            background-color: {BaseStyles.POSITIVE};
            color: {BaseStyles.TEXT_PRIMARY};
            padding: 2px 8px;
            border-radius: 3px;
            font-weight: bold;
            font-size: {BaseStyles.FONT_SIZE_SMALL};
        }}
        
        QLabel#stop_triggered {{
            background-color: {BaseStyles.NEGATIVE};
            color: {BaseStyles.TEXT_PRIMARY};
            padding: 2px 8px;
            border-radius: 3px;
            font-weight: bold;
            font-size: {BaseStyles.FONT_SIZE_SMALL};
        }}
        """