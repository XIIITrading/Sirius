# live_monitor/styles/base_styles.py
"""
Base styles and common theme definitions
"""

class BaseStyles:
    # Dark theme color palette
    BACKGROUND_PRIMARY = "#1e1e1e"
    BACKGROUND_SECONDARY = "#2b2b2b"
    BACKGROUND_TERTIARY = "#323232"
    
    # Add missing attributes that dashboard components expect
    BACKGROUND = "#1e1e1e"  # Same as BACKGROUND_PRIMARY
    CARD_BACKGROUND = "#2b2b2b"  # Same as BACKGROUND_SECONDARY
    HEADER_BACKGROUND = "#323232"  # Same as BACKGROUND_TERTIARY
    INPUT_BACKGROUND = "#2b2b2b"  # Same as BACKGROUND_SECONDARY
    
    BORDER_COLOR = "#444"
    BORDER_LIGHT = "#555"
    BORDER_HOVER = "#555"  # Same as BORDER_LIGHT
    
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#cccccc"
    TEXT_TERTIARY = "#888888"
    TEXT_MUTED = "#888888"  # Same as TEXT_TERTIARY
    
    # Accent colors
    ACCENT_PRIMARY = "#0d7377"
    ACCENT_HOVER = "#14a085"
    ACCENT_PRESSED = "#0a5d61"
    
    # Add missing accent attributes
    ACCENT_COLOR = "#0d7377"  # Same as ACCENT_PRIMARY
    ACCENT_ACTIVE = "#0a5d61"  # Same as ACCENT_PRESSED
    
    # Status colors
    POSITIVE = "#26a69a"
    NEGATIVE = "#ef5350"
    NEUTRAL = "#888888"
    WARNING = "#ffa726"
    
    # Add missing status color mappings
    SUCCESS_COLOR = "#26a69a"  # Same as POSITIVE
    ERROR_COLOR = "#ef5350"    # Same as NEGATIVE
    WARNING_COLOR = "#ffa726"  # Same as WARNING
    INFO_COLOR = "#3b82f6"     # Blue
    
    # Chart colors
    BULLISH_COLOR = "#26a69a"  # Same as POSITIVE
    BEARISH_COLOR = "#ef5350"  # Same as NEGATIVE
    NEUTRAL_COLOR = "#888888"  # Same as NEUTRAL
    
    # Font definitions
    FONT_FAMILY = "Arial"
    FONT_SIZE_SMALL = "11px"
    FONT_SIZE_NORMAL = "13px"
    FONT_SIZE_LARGE = "16px"
    FONT_SIZE_XLARGE = "20px"
    
    # Add font family for monospace
    FONT_FAMILY_MONO = "Consolas, 'Courier New', monospace"
    
    # Spacing
    SPACING_XS = "4px"
    SPACING_SM = "8px"
    SPACING_MD = "12px"
    SPACING_LG = "16px"
    SPACING_XL = "24px"
    
    # Border radius
    BORDER_RADIUS_SM = "4px"
    BORDER_RADIUS_MD = "6px"
    BORDER_RADIUS_LG = "8px"
    
    @staticmethod
    def get_base_stylesheet():
        return f"""
        QWidget {{
            background-color: {BaseStyles.BACKGROUND_PRIMARY};
            color: {BaseStyles.TEXT_PRIMARY};
            font-family: {BaseStyles.FONT_FAMILY};
            font-size: {BaseStyles.FONT_SIZE_NORMAL};
        }}
        
        QMainWindow {{
            background-color: {BaseStyles.BACKGROUND};
        }}
        
        QFrame {{
            background-color: {BaseStyles.CARD_BACKGROUND};
            border: 1px solid {BaseStyles.BORDER_COLOR};
            border-radius: {BaseStyles.BORDER_RADIUS_MD};
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
        
        /* Tables */
        QTableWidget {{
            background-color: {BaseStyles.CARD_BACKGROUND};
            border: 1px solid {BaseStyles.BORDER_COLOR};
            gridline-color: {BaseStyles.BORDER_COLOR};
        }}
        
        QTableWidget::item {{
            padding: {BaseStyles.SPACING_SM};
        }}
        
        QTableWidget::item:selected {{
            background-color: {BaseStyles.ACCENT_COLOR};
        }}
        
        QHeaderView::section {{
            background-color: {BaseStyles.HEADER_BACKGROUND};
            padding: {BaseStyles.SPACING_SM};
            border: none;
            border-bottom: 2px solid {BaseStyles.BORDER_COLOR};
        }}
        
        /* Combo Boxes */
        QComboBox {{
            background-color: {BaseStyles.INPUT_BACKGROUND};
            border: 1px solid {BaseStyles.BORDER_COLOR};
            border-radius: {BaseStyles.BORDER_RADIUS_SM};
            padding: {BaseStyles.SPACING_SM};
            color: {BaseStyles.TEXT_PRIMARY};
        }}
        
        QComboBox::drop-down {{
            border: none;
        }}
        
        QComboBox::down-arrow {{
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid {BaseStyles.TEXT_SECONDARY};
            margin-right: {BaseStyles.SPACING_SM};
        }}
        
        /* Status Bar */
        QStatusBar {{
            background-color: {BaseStyles.HEADER_BACKGROUND};
            color: {BaseStyles.TEXT_SECONDARY};
            border-top: 1px solid {BaseStyles.BORDER_COLOR};
        }}
        
        /* Checkboxes */
        QCheckBox {{
            color: {BaseStyles.TEXT_PRIMARY};
            spacing: {BaseStyles.SPACING_SM};
        }}
        
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border: 2px solid {BaseStyles.BORDER_COLOR};
            border-radius: 3px;
            background-color: {BaseStyles.INPUT_BACKGROUND};
        }}
        
        QCheckBox::indicator:checked {{
            background-color: {BaseStyles.ACCENT_COLOR};
            border-color: {BaseStyles.ACCENT_COLOR};
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