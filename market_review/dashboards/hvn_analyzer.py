# hvn_analyzer.py
"""
HVN Peak Analyzer GUI
A PyQt6 application for analyzing volume peaks across multiple timeframes
"""

import sys
import traceback
from datetime import datetime, time, timedelta
from typing import Dict, Optional, List

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QDateEdit, QTimeEdit,
    QTableWidget, QTableWidgetItem, QGroupBox, QSplitter,
    QMessageBox, QHeaderView, QComboBox, QTextEdit
)
from PyQt6.QtCore import Qt, QDate, QTime, QThread, pyqtSignal
from PyQt6.QtGui import QFont

# Import your modules
from market_review.data.polygon_bridge import PolygonHVNBridge
from market_review.calculations.volume.hvn_engine import TimeframeResult
from market_review.calculations.confluence.hvn_confluence import (
    HVNConfluenceCalculator, ConfluenceAnalysis, ConfluenceZone
)
from market_review.calculations.zones.m15_atr_zones import ATRZoneEnhancer, ATRZoneAnalysis


class HVNCalculationThread(QThread):
    """Background thread for HVN calculations"""
    
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    current_price = pyqtSignal(float)
    
    def __init__(self, symbol: str, end_datetime: datetime):
        super().__init__()
        self.symbol = symbol
        self.end_datetime = end_datetime
        self.bridge = PolygonHVNBridge(
            hvn_levels=100,
            hvn_percentile=80.0,
            lookback_days=120,
            update_interval_minutes=5,
            cache_enabled=True
        )
        # Initialize ATR zone enhancer
        self.atr_enhancer = ATRZoneEnhancer(
            periods=14,
            lookback_bars=100,
            cache_enabled=True
        )
        
    def run(self):
        try:
            self.progress.emit(f"Fetching data for {self.symbol}...")
            
            # Calculate HVN using bridge
            state = self.bridge.calculate_hvn(
                self.symbol,
                end_date=self.end_datetime,
                timeframe='5min'
            )
            
            # Emit current price
            self.current_price.emit(state.current_price)
            
            self.progress.emit("Running multi-timeframe analysis...")
            
            # Get multi-timeframe results
            results = self.bridge.hvn_engine.analyze_multi_timeframe(
                state.recent_bars,
                timeframes=[120, 60, 15]
            )
            
            self.progress.emit("Calculating confluence zones...")
            
            # Calculate confluence zones using the imported module
            confluence_calculator = HVNConfluenceCalculator(
                confluence_threshold_percent=0.5,
                min_peaks_for_zone=2,
                max_peaks_per_timeframe=10
            )
            
            confluence_analysis = confluence_calculator.calculate(
                results=results,
                current_price=state.current_price,
                max_zones=5
            )
            
            # NEW: Calculate ATR zones
            self.progress.emit("Calculating 15-minute ATR zones...")
            
            atr_zone_analysis = self.atr_enhancer.enhance_confluence_zones(
                confluence_analysis,
                self.symbol,
                self.end_datetime
            )
            
            # Emit enhanced results
            enhanced_results = {
                'timeframe_results': results,
                'confluence_analysis': confluence_analysis,
                'atr_zone_analysis': atr_zone_analysis,
                'current_price': state.current_price
            }
            
            self.finished.emit(enhanced_results)
            
        except Exception as e:
            self.error.emit(f"Error: {str(e)}\n{traceback.format_exc()}")


class TimeframeTable(QWidget):
    """Widget for displaying peaks for a single timeframe"""
    
    def __init__(self, timeframe: int):
        super().__init__()
        self.timeframe = timeframe
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Group box with title
        group = QGroupBox(f"{self.timeframe}-Day Analysis")
        group.setStyleSheet("""
            QGroupBox {
                color: #ffffff;
                border: 2px solid #7B68EE;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 15px;
                background-color: #1e1e1e;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px 0 10px;
                color: #ffffff;
                font-weight: bold;
                font-size: 14px;
            }
        """)
        group_layout = QVBoxLayout()
        
        # Info labels
        self.info_label = QLabel("No data loaded")
        self.info_label.setStyleSheet("color: #888888; padding: 5px; font-size: 11px;")
        group_layout.addWidget(self.info_label)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Rank", "Price", "Volume %"])
        
        # Style the table with purple theme
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #7B68EE;
                border-radius: 5px;
                gridline-color: #4a4a6a;
                font-size: 12px;
            }
            QTableWidget::item {
                padding: 4px;
                border: none;
            }
            QTableWidget::item:selected {
                background-color: #7B68EE;
            }
            QTableWidget::item:alternate {
                background-color: #333333;
            }
            QHeaderView::section {
                background-color: #5a4a8a;
                color: #ffffff;
                padding: 6px;
                border: none;
                font-weight: bold;
                font-size: 12px;
            }
            QTableCornerButton::section {
                background-color: #5a4a8a;
            }
        """)
        
        # Set column widths
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 50)
        self.table.setColumnWidth(2, 80)
        
        group_layout.addWidget(self.table)
        group.setLayout(group_layout)
        
        layout.addWidget(group)
        self.setLayout(layout)
        
    def update_data(self, result: TimeframeResult):
        """Update table with timeframe results"""
        # Update info label
        info_text = f"Range: ${result.price_range[0]:.2f} - ${result.price_range[1]:.2f} | "
        info_text += f"Peaks: {len(result.peaks)}"
        self.info_label.setText(info_text)
        self.info_label.setStyleSheet("color: #ffffff; padding: 5px; font-size: 11px;")
        
        # Update table
        self.table.setRowCount(len(result.peaks))
        
        for row, peak in enumerate(result.peaks):
            # Rank
            rank_item = QTableWidgetItem(f"#{peak.rank}")
            rank_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 0, rank_item)
            
            # Price
            price_item = QTableWidgetItem(f"${peak.price:.2f}")
            price_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 1, price_item)
            
            # Volume %
            volume_item = QTableWidgetItem(f"{peak.volume_percent:.2f}%")
            volume_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 2, volume_item)
            
            # Highlight top 3 ranks
            if peak.rank <= 3:
                font = QFont()
                font.setBold(True)
                for col in range(3):
                    self.table.item(row, col).setFont(font)
                    
        self.table.resizeRowsToContents()
        
    def clear_data(self):
        """Clear the table"""
        self.table.setRowCount(0)
        self.info_label.setText("No data loaded")
        self.info_label.setStyleSheet("color: #888888; padding: 5px; font-size: 11px;")


class ConfluenceSection(QWidget):
    """Widget for displaying confluence zones analysis with ATR enhancements"""
    
    def __init__(self):
        super().__init__()
        self.current_price = 0.0
        self.atr_zone_analysis = None  # Store ATR analysis
        self.confluence_calculator = HVNConfluenceCalculator(
            confluence_threshold_percent=0.5,
            min_peaks_for_zone=2,
            max_peaks_per_timeframe=10
        )
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Group box
        group = QGroupBox("Confluence Analysis - Top 5 Zones")
        group.setStyleSheet("""
            QGroupBox {
                color: #ffffff;
                border: 2px solid #7B68EE;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 15px;
                background-color: #1e1e1e;
                font-size: 16px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px 0 10px;
            }
        """)
        
        group_layout = QVBoxLayout()
        
        # Current price label
        self.price_label = QLabel("Current Price: --")
        self.price_label.setStyleSheet("""
            QLabel {
                color: #7B68EE;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                background-color: #2b2b2b;
                border-radius: 5px;
                margin-bottom: 10px;
            }
        """)
        group_layout.addWidget(self.price_label)
        
        # Confluence results
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #7B68EE;
                border-radius: 5px;
                padding: 10px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 13px;
                line-height: 1.5;
            }
        """)
        
        group_layout.addWidget(self.results_text)
        group.setLayout(group_layout)
        
        layout.addWidget(group)
        self.setLayout(layout)
    
    def update_analysis(self, results: Dict[int, TimeframeResult], 
                       current_price: float,
                       atr_zone_analysis: Optional[ATRZoneAnalysis] = None):
        """Update confluence analysis using the calculator"""
        self.current_price = current_price
        self.atr_zone_analysis = atr_zone_analysis  # Store ATR analysis
        self.price_label.setText(f"Current Price: ${current_price:.2f}")
        
        # Calculate confluence using the imported module
        analysis = self.confluence_calculator.calculate(
            results=results,
            current_price=current_price,
            max_zones=5
        )
        
        # Display the analysis with ATR zones
        self.display_analysis(analysis)
        
    def display_analysis(self, analysis: ConfluenceAnalysis):
        """Display the confluence analysis results with ATR zones"""
        if not analysis.zones:
            self.results_text.setHtml("<p style='color: #888888;'>No confluence zones found</p>")
            return
            
        html = ""
        
        # Add ATR value in header if available
        if self.atr_zone_analysis:
            atr_value = self.atr_zone_analysis.atr_result.atr_value
            atr_pct = self.atr_zone_analysis.atr_result.atr_percentage
            html += f"""
            <div style='margin-bottom: 10px; padding: 8px; background-color: #3a3a5a; border-radius: 5px;'>
                <p style='color: #9370DB; margin: 0; font-size: 14px;'>
                    📊 15-Min ATR: ${atr_value:.2f} ({atr_pct:.2f}% of price)
                </p>
            </div>
            """
        
        # Add summary if price is in a zone
        if analysis.price_in_zone:
            # Check if also in ATR zone
            atr_zone_msg = ""
            if self.atr_zone_analysis:
                atr_zone = self.atr_zone_analysis.get_zone_by_id(analysis.price_in_zone.zone_id)
                if atr_zone and atr_zone.contains_price(self.current_price):
                    position = atr_zone.get_price_position(self.current_price)
                    atr_zone_msg = f" | {position['atr_percentage']:.1f}% of ATR from center"
            
            html += f"""
            <div style='margin-bottom: 15px; padding: 10px; background-color: #4a3a6a; border-radius: 5px;'>
                <p style='color: #50C878; margin: 0;'>
                    ⚡ Current price is INSIDE Zone #{analysis.price_in_zone.zone_id}{atr_zone_msg}
                </p>
            </div>
            """
        
        # Display each zone with ATR enhancement
        for zone in analysis.zones[:5]:
            # Get corresponding ATR zone if available
            atr_zone = None
            if self.atr_zone_analysis:
                atr_zone = self.atr_zone_analysis.get_zone_by_id(zone.zone_id)
            
            # Determine strength emoji and color
            strength_display = {
                'Strong': ('🟢', '#50C878'),
                'Moderate': ('🟡', '#FFD700'),
                'Weak': ('🔵', '#87CEEB')
            }
            emoji, color = strength_display.get(zone.strength, ('🔵', '#87CEEB'))
            
            # Direction from current price
            direction = "above" if zone.center_price > self.current_price else "below"
            
            # Format zone info
            zone_html = f"""
            <div style='margin-bottom: 20px; padding: 15px; background-color: #333333; border-radius: 8px; border-left: 4px solid {color};'>
                <h3 style='color: #9370DB; margin: 0 0 10px 0;'>
                    Zone #{zone.zone_id} - {emoji} {zone.strength} 
                    <span style='font-size: 12px; color: #888888;'>(Score: {zone.strength_score:.1f})</span>
                </h3>
                
                <!-- Volume-based zone info -->
                <div style='margin-bottom: 10px; padding: 10px; background-color: #2a2a2a; border-radius: 5px;'>
                    <h4 style='color: #7B68EE; margin: 0 0 8px 0; font-size: 13px;'>📊 VOLUME-BASED ZONE</h4>
                    <table style='width: 100%; color: #ffffff; font-size: 12px;'>
                        <tr>
                            <td style='padding: 3px; width: 30%;'><b>Center:</b></td>
                            <td style='color: #7B68EE;'>${zone.center_price:.2f}</td>
                            <td style='padding: 3px; width: 30%;'><b>Distance:</b></td>
                            <td style='color: {'#50C878' if zone.distance_percentage < 1.0 else '#FFA500'};'>
                                ${zone.distance_from_current:.2f} ({zone.distance_percentage:.2f}% {direction})
                            </td>
                        </tr>
                        <tr>
                            <td style='padding: 3px;'><b>Zone Range:</b></td>
                            <td>${zone.zone_low:.2f} - ${zone.zone_high:.2f}</td>
                            <td style='padding: 3px;'><b>Width:</b></td>
                            <td>${zone.zone_width:.2f}</td>
                        </tr>
                        <tr>
                            <td style='padding: 3px;'><b>Timeframes:</b></td>
                            <td>{', '.join(f'{tf}d' for tf in sorted(zone.timeframes))}</td>
                            <td style='padding: 3px;'><b>Volume:</b></td>
                            <td>{zone.total_volume_weight:.2f}%</td>
                        </tr>
                    </table>
                </div>
            """
            
            # Add ATR zone info if available
            if atr_zone:
                # Zone relationship icon
                if atr_zone.zones_aligned:
                    relationship_icon = "✅"
                    relationship_text = "Aligned"
                elif atr_zone.atr_to_volume_ratio > 2.0:
                    relationship_icon = "⚡"
                    relationship_text = "High Volatility"
                else:
                    relationship_icon = "📍"
                    relationship_text = "Tight Range"
                
                zone_html += f"""
                <!-- ATR-based zone info -->
                <div style='margin-bottom: 10px; padding: 10px; background-color: #2a2a2a; border-radius: 5px;'>
                    <h4 style='color: #7B68EE; margin: 0 0 8px 0; font-size: 13px;'>
                        📈 15-MIN ATR ZONE {relationship_icon} {relationship_text}
                    </h4>
                    <table style='width: 100%; color: #ffffff; font-size: 12px;'>
                        <tr>
                            <td style='padding: 3px; width: 30%;'><b>ATR Upper:</b></td>
                            <td style='color: #87CEEB;'>${atr_zone.atr_upper:.2f}</td>
                            <td style='padding: 3px; width: 30%;'><b>Extension:</b></td>
                            <td style='color: {'#FFA500' if atr_zone.atr_extends_above else '#50C878'};'>
                                {'+$' + f'{atr_zone.upper_extension:.2f}' if atr_zone.atr_extends_above else 'Within'}
                            </td>
                        </tr>
                        <tr>
                            <td style='padding: 3px;'><b>ATR Lower:</b></td>
                            <td style='color: #87CEEB;'>${atr_zone.atr_lower:.2f}</td>
                            <td style='padding: 3px;'><b>Extension:</b></td>
                            <td style='color: {'#FFA500' if atr_zone.atr_extends_below else '#50C878'};'>
                                {'-$' + f'{atr_zone.lower_extension:.2f}' if atr_zone.atr_extends_below else 'Within'}
                            </td>
                        </tr>
                        <tr>
                            <td style='padding: 3px;'><b>ATR Width:</b></td>
                            <td>${atr_zone.atr_zone_width:.2f}</td>
                            <td style='padding: 3px;'><b>Ratio:</b></td>
                            <td>{atr_zone.atr_to_volume_ratio:.2f}x volume zone</td>
                        </tr>
                    </table>
                """
                
                # Add price position if inside ATR zone
                if atr_zone.contains_price(self.current_price):
                    position = atr_zone.get_price_position(self.current_price)
                    zone_html += f"""
                    <div style='margin-top: 8px; padding: 5px; background-color: #3a3a5a; border-radius: 3px;'>
                        <p style='margin: 0; font-size: 11px; color: #50C878;'>
                            💹 Price is {position['atr_percentage']:.1f}% of ATR from center
                            (${abs(position['distance_from_center']):.2f} {'above' if position['distance_from_center'] > 0 else 'below'})
                        </p>
                    </div>
                    """
                
                zone_html += "</div>"
            
            # Add peak details (existing code)
            zone_html += f"""
                <div style='margin-top: 8px; font-size: 11px; color: #bbbbbb;'>
                    <b>Contributing Peaks:</b><br>
            """
            
            for tf, price, vol in zone.peaks:
                zone_html += f"• {tf}d peak at ${price:.2f} ({vol:.2f}%)<br>"
                
            zone_html += "</div></div>"
            html += zone_html
            
        # Add analysis footer
        footer_time = analysis.analysis_time.strftime('%Y-%m-%d %H:%M:%S')
        if self.atr_zone_analysis:
            footer_time = self.atr_zone_analysis.analysis_time.strftime('%Y-%m-%d %H:%M:%S')
            
        html += f"""
        <div style='margin-top: 15px; padding: 10px; background-color: #2a2a2a; border-radius: 5px; font-size: 11px; color: #888888;'>
            <p style='margin: 0;'>
                Analysis Time: {footer_time} | 
                Total Zones: {analysis.total_zones_found} | 
                Strongest: Zone #{analysis.strongest_zone.zone_id if analysis.strongest_zone else 'N/A'} | 
                Nearest: Zone #{analysis.nearest_zone.zone_id if analysis.nearest_zone else 'N/A'}
            </p>
        </div>
        """
            
        self.results_text.setHtml(html)


class HVNPeakAnalyzer(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.calculation_thread = None
        self.current_price = 0.0
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("HVN Peak Analyzer")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set dark theme with purple accents
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QGroupBox {
                color: #ffffff;
                border: 2px solid #7B68EE;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLabel {
                color: #ffffff;
            }
            QLineEdit, QDateEdit, QTimeEdit, QComboBox {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #7B68EE;
                border-radius: 3px;
                padding: 5px;
            }
            QLineEdit:focus, QDateEdit:focus, QTimeEdit:focus, QComboBox:focus {
                border: 2px solid #9370DB;
            }
            QSplitter::handle:vertical {
                background-color: #7B68EE;
                height: 3px;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout - using splitter for proportional sizing
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        central_widget.setLayout(main_layout)
        
        # Main vertical splitter
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Input section (15%)
        input_widget = QWidget()
        input_layout = QVBoxLayout()
        input_widget.setLayout(input_layout)
        
        input_group = QGroupBox("Analysis Parameters")
        input_group_layout = QHBoxLayout()
        
        # Symbol input
        input_group_layout.addWidget(QLabel("Symbol:"))
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("TSLA")
        self.symbol_input.setText("TSLA")
        self.symbol_input.setMaximumWidth(100)
        input_group_layout.addWidget(self.symbol_input)
        
        input_group_layout.addSpacing(20)
        
        # Date input
        input_group_layout.addWidget(QLabel("Date:"))
        self.date_input = QDateEdit()
        self.date_input.setCalendarPopup(True)
        self.date_input.setDate(QDate.currentDate())
        self.date_input.setDisplayFormat("yyyy-MM-dd")
        input_group_layout.addWidget(self.date_input)
        
        input_group_layout.addSpacing(20)
        
        # Time input
        input_group_layout.addWidget(QLabel("Time:"))
        self.time_input = QTimeEdit()
        self.time_input.setTime(QTime.currentTime())
        self.time_input.setDisplayFormat("HH:mm:ss")
        input_group_layout.addWidget(self.time_input)
        
        input_group_layout.addSpacing(20)
        
        # Timezone selector
        input_group_layout.addWidget(QLabel("Timezone:"))
        self.timezone_combo = QComboBox()
        self.timezone_combo.addItems(["UTC", "US/Eastern", "US/Central", "US/Pacific"])
        self.timezone_combo.setCurrentText("US/Eastern")
        input_group_layout.addWidget(self.timezone_combo)
        
        input_group_layout.addSpacing(20)
        
        # Now - 30min button
        self.now_minus_button = QPushButton("Now - 30min")
        self.now_minus_button.clicked.connect(self.set_recent_time)
        self.now_minus_button.setToolTip("Set time to 30 minutes ago")
        self.now_minus_button.setStyleSheet("""
            QPushButton {
                background-color: #5a4a8a;
                color: white;
                padding: 6px 12px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #6a5a9a;
            }
        """)
        input_group_layout.addWidget(self.now_minus_button)
        
        input_group_layout.addStretch()
        
        # Run button
        self.run_button = QPushButton("Run Analysis")
        self.run_button.clicked.connect(self.run_analysis)
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #7B68EE;
                color: white;
                padding: 8px 20px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #9370DB;
            }
            QPushButton:pressed {
                background-color: #6A5ACD;
            }
            QPushButton:disabled {
                background-color: #4a4a4a;
                color: #888888;
            }
        """)
        input_group_layout.addWidget(self.run_button)
        
        input_group.setLayout(input_group_layout)
        input_layout.addWidget(input_group)
        
        # Tables section (35%)
        tables_widget = QWidget()
        tables_layout = QHBoxLayout()
        tables_widget.setLayout(tables_layout)
        
        # Create three timeframe tables
        self.table_120 = TimeframeTable(120)
        self.table_60 = TimeframeTable(60)
        self.table_15 = TimeframeTable(15)
        
        tables_layout.addWidget(self.table_120)
        tables_layout.addWidget(self.table_60)
        tables_layout.addWidget(self.table_15)
        
        # Confluence section (50%)
        self.confluence_section = ConfluenceSection()
        
        # Add all sections to splitter
        main_splitter.addWidget(input_widget)
        main_splitter.addWidget(tables_widget)
        main_splitter.addWidget(self.confluence_section)
        
        # Set proportional sizes (15%, 35%, 50%)
        total_height = 900
        main_splitter.setSizes([
            int(total_height * 0.15),
            int(total_height * 0.35),
            int(total_height * 0.50)
        ])
        
        main_layout.addWidget(main_splitter)
        
    def set_recent_time(self):
        """Set time to 30 minutes ago in the selected timezone"""
        import pytz
        tz_name = self.timezone_combo.currentText()
        if tz_name == "UTC":
            tz = pytz.UTC
        else:
            tz = pytz.timezone(tz_name)
        
        # Get current time in selected timezone
        now = datetime.now(tz)
        recent_time = now - timedelta(minutes=30)
        
        # Update date and time inputs
        self.date_input.setDate(QDate(recent_time.year, recent_time.month, recent_time.day))
        self.time_input.setTime(QTime(recent_time.hour, recent_time.minute, recent_time.second))
        
    def run_analysis(self):
        """Run the HVN analysis"""
        # Validate inputs
        symbol = self.symbol_input.text().strip().upper()
        if not symbol:
            QMessageBox.warning(self, "Input Error", "Please enter a symbol")
            return
            
        # Get datetime
        date = self.date_input.date().toPyDate()
        time = self.time_input.time().toPyTime()
        end_datetime = datetime.combine(date, time)
        
        # Apply timezone
        import pytz
        tz_name = self.timezone_combo.currentText()
        if tz_name == "UTC":
            tz = pytz.UTC
        else:
            tz = pytz.timezone(tz_name)
        end_datetime = tz.localize(end_datetime)
        
        # Convert to UTC for comparison
        end_datetime_utc = end_datetime.astimezone(pytz.UTC)
        now_utc = datetime.now(pytz.UTC)
        
        # Check if the selected time is in the future
        if end_datetime_utc > now_utc:
            QMessageBox.warning(
                self, 
                "Invalid Time", 
                f"Selected time is in the future!\n\n"
                f"Selected: {end_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
                f"UTC: {end_datetime_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
                f"Current UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}\n\n"
                f"Please select a time in the past."
            )
            return
        
        # Clear previous results
        self.table_120.clear_data()
        self.table_60.clear_data()
        self.table_15.clear_data()
        self.confluence_section.results_text.clear()
        
        # Disable run button
        self.run_button.setEnabled(False)
        self.setWindowTitle(f"HVN Peak Analyzer - Running analysis for {symbol}...")
        
        # Start calculation thread
        self.calculation_thread = HVNCalculationThread(symbol, end_datetime)
        self.calculation_thread.finished.connect(self.on_calculation_finished)
        self.calculation_thread.error.connect(self.on_calculation_error)
        self.calculation_thread.progress.connect(self.on_progress_update)
        self.calculation_thread.current_price.connect(self.on_current_price)
        self.calculation_thread.start()
        
    def on_progress_update(self, message: str):
        """Update status with progress in title bar"""
        self.setWindowTitle(f"HVN Peak Analyzer - {message}")
        
    def on_current_price(self, price: float):
        """Store current price for confluence calculation"""
        self.current_price = price
        
    def on_calculation_finished(self, results: Dict):
        """Handle calculation completion with ATR zones"""
        # Extract components from enhanced results
        timeframe_results = results.get('timeframe_results', {})
        confluence_analysis = results.get('confluence_analysis')
        atr_zone_analysis = results.get('atr_zone_analysis')
        current_price = results.get('current_price', self.current_price)
        
        # Update tables
        if 120 in timeframe_results:
            self.table_120.update_data(timeframe_results[120])
        if 60 in timeframe_results:
            self.table_60.update_data(timeframe_results[60])
        if 15 in timeframe_results:
            self.table_15.update_data(timeframe_results[15])
            
        # Update confluence analysis with ATR zones
        if confluence_analysis:
            self.confluence_section.update_analysis(
                timeframe_results, 
                current_price,
                atr_zone_analysis
            )
        else:
            # Fallback to original method if no confluence analysis
            self.confluence_section.update_analysis(timeframe_results, current_price)
            
        # Update window title with ATR info if available
        if atr_zone_analysis:
            atr_value = atr_zone_analysis.atr_result.atr_value
            self.setWindowTitle(f"HVN Peak Analyzer - 15m ATR: ${atr_value:.2f}")
        else:
            self.setWindowTitle("HVN Peak Analyzer")
        
        # Re-enable run button
        self.run_button.setEnabled(True)
        
    def on_calculation_error(self, error_msg: str):
        """Handle calculation error"""
        QMessageBox.critical(self, "Analysis Error", error_msg)
        self.setWindowTitle("HVN Peak Analyzer - Error occurred")
        self.run_button.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = HVNPeakAnalyzer()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()