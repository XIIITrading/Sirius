# live_monitor/dashboard/segments/ui_builder_segment.py
"""
UI Builder Segment - Contains all UI construction methods
"""

from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QSplitter, 
                             QStatusBar, QLabel)
from PyQt6.QtCore import Qt

from ...styles import BaseStyles
from ..components import (TickerEntry, TickerCalculations, EntryCalculations,
                         PointCallEntry, PointCallExit, HVNTableWidget,
                         SupplyDemandTableWidget, OrderBlocksTableWidget)
from ..widgets.server_status import ServerStatusWidget


class UIBuilderSegment:
    """Dashboard segment for building the UI"""
    
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("Live Monitor - Trading Dashboard")
        self.setGeometry(100, 100, 1600, 900)
        
        # Create main widget that will hold everything
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Main vertical layout to hold header and content
        main_vertical_layout = QVBoxLayout(main_widget)
        main_vertical_layout.setContentsMargins(0, 0, 0, 0)
        main_vertical_layout.setSpacing(0)
        
        # Create header
        header_widget = self._create_header()
        main_vertical_layout.addWidget(header_widget)
        
        # Create content
        content_widget = self._create_content()
        main_vertical_layout.addWidget(content_widget, 1)
        
        # Add status bar
        self.setup_status_bar()
    
    def _create_header(self):
        """Create the header widget"""
        header_widget = QWidget()
        header_widget.setFixedHeight(40)
        header_widget.setStyleSheet("QWidget { background-color: #2a2a2a; border-bottom: 1px solid #444; }")
        
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(10, 5, 10, 5)
        
        # Add title label to header
        title_label = QLabel("Live Monitor Dashboard")
        title_label.setStyleSheet("QLabel { color: white; font-size: 16px; font-weight: bold; }")
        header_layout.addWidget(title_label)
        
        # Add stretch to push indicator to the right
        header_layout.addStretch()
        
        # Add server status indicator
        self.server_status_widget = ServerStatusWidget()
        header_layout.addWidget(self.server_status_widget)
        
        return header_widget
    
    def _create_content(self):
        """Create the main content area"""
        content_widget = QWidget()
        main_layout = QHBoxLayout(content_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Create main horizontal splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Create left column
        left_widget = self._create_left_column()
        
        # Create middle/right section
        middle_right_widget = self._create_middle_right_section()
        
        # Add widgets to main horizontal splitter
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(middle_right_widget)
        
        # Set horizontal splitter proportions (25% left, 75% middle/right)
        main_splitter.setSizes([400, 1200])
        
        # Add splitter to main layout
        main_layout.addWidget(main_splitter)
        
        return content_widget
    
    def _create_left_column(self):
        """Create the left column widgets"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(5)
        
        # Ticker Entry
        self.ticker_entry = TickerEntry()
        left_layout.addWidget(self.ticker_entry)
        
        # Ticker Calculations
        self.ticker_calculations = TickerCalculations()
        left_layout.addWidget(self.ticker_calculations, 1)
        
        # Entry/Size Calculations
        self.entry_calculations = EntryCalculations()
        left_layout.addWidget(self.entry_calculations, 1)
        
        return left_widget
    
    def _create_middle_right_section(self):
        """Create the middle and right section"""
        middle_right_widget = QWidget()
        middle_right_layout = QVBoxLayout(middle_right_widget)
        middle_right_layout.setContentsMargins(0, 0, 0, 0)
        middle_right_layout.setSpacing(5)
        
        # Create vertical splitter for top/bottom sections
        vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Create top section
        top_widget = self._create_top_section()
        
        # Create bottom section
        bottom_widget = self._create_bottom_section()
        
        # Add top and bottom to vertical splitter
        vertical_splitter.addWidget(top_widget)
        vertical_splitter.addWidget(bottom_widget)
        
        # Set vertical splitter proportions (40% top, 60% bottom)
        vertical_splitter.setSizes([360, 540])
        
        # Add vertical splitter to middle_right layout
        middle_right_layout.addWidget(vertical_splitter)
        
        return middle_right_widget
    
    def _create_top_section(self):
        """Create the top section with entry/exit systems"""
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(5)
        
        # Point & Call Entry (left side of top)
        self.point_call_entry = PointCallEntry()
        top_layout.addWidget(self.point_call_entry, 1)
        
        # Point & Call Exit (right side of top)
        self.point_call_exit = PointCallExit()
        top_layout.addWidget(self.point_call_exit, 1)
        
        return top_widget
    
    def _create_bottom_section(self):
        """Create the bottom section with three tables"""
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(5)
        
        # HVN Table
        self.hvn_table = HVNTableWidget()
        bottom_layout.addWidget(self.hvn_table, 1)
        
        # Supply/Demand Table
        self.supply_demand_table = SupplyDemandTableWidget()
        bottom_layout.addWidget(self.supply_demand_table, 1)
        
        # Order Blocks Table
        self.order_blocks_table = OrderBlocksTableWidget()
        bottom_layout.addWidget(self.order_blocks_table, 1)
        
        return bottom_widget
    
    def setup_status_bar(self):
        """Setup status bar with connection indicator and separate signal displays"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Connection status label
        self.connection_label = QLabel("● Disconnected")
        self.connection_label.setStyleSheet("QLabel { color: #ff4444; font-weight: bold; }")
        self.status_bar.addWidget(self.connection_label)
        
        # Current symbol label
        self.symbol_label = QLabel("Symbol: None")
        self.status_bar.addWidget(self.symbol_label)
        
        # Signal status labels for different timeframes
        self.m1_signal_label = QLabel("M1: --")
        self.m1_signal_label.setStyleSheet("QLabel { font-weight: bold; margin-left: 10px; }")
        self.status_bar.addWidget(self.m1_signal_label)
        
        self.m5_signal_label = QLabel("M5: --")
        self.m5_signal_label.setStyleSheet("QLabel { font-weight: bold; margin-left: 10px; }")
        self.status_bar.addWidget(self.m5_signal_label)
        
        self.m15_signal_label = QLabel("M15: --")
        self.m15_signal_label.setStyleSheet("QLabel { font-weight: bold; margin-left: 10px; }")
        self.status_bar.addWidget(self.m15_signal_label)

        self.stat_signal_label = QLabel("STAT: --")
        self.stat_signal_label.setStyleSheet("QLabel { font-weight: bold; margin-left: 10px; }")
        self.status_bar.addWidget(self.stat_signal_label)
        
        self.m1_mstruct_label = QLabel("M1 MSTRUCT: --")
        self.m1_mstruct_label.setStyleSheet("QLabel { font-weight: bold; margin-left: 10px; }")
        self.status_bar.addWidget(self.m1_mstruct_label)

        self.m5_mstruct_label = QLabel("M5 MSTRUCT: --")
        self.m5_mstruct_label.setStyleSheet("QLabel { font-weight: bold; margin-left: 10px; }")
        self.status_bar.addWidget(self.m5_mstruct_label)
        
        self.m15_mstruct_label = QLabel("M15 MSTRUCT: --")
        self.m15_mstruct_label.setStyleSheet("QLabel { font-weight: bold; margin-left: 10px; }")
        self.status_bar.addWidget(self.m15_mstruct_label)
        
        self.m5_trend_label = QLabel("M5 TREND: --")
        self.m5_trend_label.setStyleSheet("QLabel { font-weight: bold; margin-left: 10px; }")
        self.status_bar.addWidget(self.m5_trend_label)

        # FIX: Correct the M15 trend label
        self.m15_trend_label = QLabel("M15 TREND: --")
        self.m15_trend_label.setStyleSheet("QLabel { font-weight: bold; margin-left: 10px; }")
        self.status_bar.addWidget(self.m15_trend_label)
        
        # Last update time
        self.update_time_label = QLabel("Last Update: Never")
        self.status_bar.addPermanentWidget(self.update_time_label)
    
    def apply_styles(self):
        """Apply dark theme styles"""
        self.setStyleSheet(BaseStyles.get_base_stylesheet())