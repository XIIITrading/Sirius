"""
Base styles and common theme definitions
"""

class BaseStyles:
    # Dark theme color palette
    BACKGROUND_PRIMARY = "#1e1e1e"
    BACKGROUND_SECONDARY = "#2b2b2b"
    BACKGROUND_TERTIARY = "#323232"
    
    BORDER_COLOR = "#444"
    BORDER_LIGHT = "#555"
    
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#cccccc"
    TEXT_TERTIARY = "#888888"
    
    # Accent colors
    ACCENT_PRIMARY = "#0d7377"
    ACCENT_HOVER = "#14a085"
    ACCENT_PRESSED = "#0a5d61"
    
    # Status colors
    POSITIVE = "#26a69a"
    NEGATIVE = "#ef5350"
    NEUTRAL = "#888888"
    WARNING = "#ffa726"
    
    # Font definitions
    FONT_FAMILY = "Arial"
    FONT_SIZE_SMALL = "11px"
    FONT_SIZE_NORMAL = "13px"
    FONT_SIZE_LARGE = "16px"
    FONT_SIZE_XLARGE = "20px"
    
    @staticmethod
    def get_base_stylesheet():
        return f"""
        QWidget {{
            background-color: {BaseStyles.BACKGROUND_PRIMARY};
            color: {BaseStyles.TEXT_PRIMARY};
            font-family: {BaseStyles.FONT_FAMILY};
            font-size: {BaseStyles.FONT_SIZE_NORMAL};
        }}
        
        QGroupBox {{
            border: 1px solid {BaseStyles.BORDER_COLOR};
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: bold;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }}
        
        QPushButton {{
            background-color: {BaseStyles.ACCENT_PRIMARY};
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
            color: {BaseStyles.TEXT_PRIMARY};
        }}
        
        QPushButton:hover {{
            background-color: {BaseStyles.ACCENT_HOVER};
        }}
        
        QPushButton:pressed {{
            background-color: {BaseStyles.ACCENT_PRESSED};
        }}
        
        QPushButton:disabled {{
            background-color: {BaseStyles.BACKGROUND_TERTIARY};
            color: {BaseStyles.TEXT_TERTIARY};
        }}
        
        QLineEdit {{
            background-color: {BaseStyles.BACKGROUND_SECONDARY};
            border: 1px solid {BaseStyles.BORDER_COLOR};
            padding: 6px;
            border-radius: 3px;
        }}
        
        QLineEdit:focus {{
            border: 1px solid {BaseStyles.ACCENT_PRIMARY};
        }}
        
        QScrollBar:vertical {{
            background-color: {BaseStyles.BACKGROUND_SECONDARY};
            width: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {BaseStyles.BORDER_COLOR};
            border-radius: 6px;
            min-height: 20px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: {BaseStyles.BORDER_LIGHT};
        }}
        """