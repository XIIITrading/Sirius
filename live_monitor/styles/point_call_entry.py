"""
Styles for point & call entry system
"""

from .base_styles import BaseStyles

class PointCallEntryStyles:
    
    @staticmethod
    def get_stylesheet():
        return f"""
        /* Container */
        QWidget#point_call_entry_container {{
            background-color: {BaseStyles.BACKGROUND_SECONDARY};
            border: 1px solid {BaseStyles.BORDER_COLOR};
            border-radius: 5px;
        }}
        
        /* Header */
        QLabel#point_call_entry_header {{
            font-size: {BaseStyles.FONT_SIZE_LARGE};
            font-weight: bold;
            padding: 10px;
            background-color: {BaseStyles.BACKGROUND_TERTIARY};
            border-bottom: 1px solid {BaseStyles.BORDER_COLOR};
        }}
        
        /* Table styling */
        QTableWidget#entry_signals_table {{
            background-color: {BaseStyles.BACKGROUND_SECONDARY};
            alternate-background-color: {BaseStyles.BACKGROUND_TERTIARY};
            gridline-color: {BaseStyles.BORDER_COLOR};
            border: none;
        }}
        
        QTableWidget#entry_signals_table::item {{
            padding: 5px;
        }}
        
        QTableWidget#entry_signals_table::item:selected {{
            background-color: {BaseStyles.ACCENT_PRIMARY};
        }}
        
        QHeaderView::section {{
            background-color: {BaseStyles.BACKGROUND_TERTIARY};
            border: 1px solid {BaseStyles.BORDER_COLOR};
            padding: 5px;
            font-weight: bold;
        }}
        
        /* Signal strength indicators */
        QLabel#signal_strong {{
            color: {BaseStyles.POSITIVE};
            font-weight: bold;
            font-size: {BaseStyles.FONT_SIZE_NORMAL};
        }}
        
        QLabel#signal_medium {{
            color: {BaseStyles.WARNING};
            font-weight: bold;
            font-size: {BaseStyles.FONT_SIZE_NORMAL};
        }}
        
        QLabel#signal_weak {{
            color: {BaseStyles.TEXT_TERTIARY};
            font-size: {BaseStyles.FONT_SIZE_NORMAL};
        }}
        
        /* Entry type badges */
        QLabel#entry_type_long {{
            background-color: {BaseStyles.POSITIVE};
            color: {BaseStyles.TEXT_PRIMARY};
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: bold;
            font-size: {BaseStyles.FONT_SIZE_SMALL};
        }}
        
        QLabel#entry_type_short {{
            background-color: {BaseStyles.NEGATIVE};
            color: {BaseStyles.TEXT_PRIMARY};
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: bold;
            font-size: {BaseStyles.FONT_SIZE_SMALL};
        }}
        """