# market_review/dashboards/camarilla_analyzer.py
"""
Camarilla Pivot Analyzer GUI
A PyQt6 application for analyzing Camarilla pivot points across multiple timeframes
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
from PyQt6.QtGui import QFont, QColor

import pandas as pd
import yfinance as yf
import pytz

# Import your modules
from market_review.calculations.pivots.camarilla_engine import CamarillaEngine, CamarillaResult
from market_review.calculations.confluence.camarilla_confluence import (
    CamarillaConfluenceCalculator, CamarillaConfluenceAnalysis, CamarillaConfluenceZone
)
from market_review.data.polygon_bridge import PolygonHVNBridge

class CamarillaCalculationThread(QThread):
    """Background thread for Camarilla calculations"""
    
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    current_price = pyqtSignal(float)
    
    def __init__(self, symbol: str, end_datetime: datetime):
        super().__init__()
        self.symbol = symbol
        self.end_datetime = end_datetime
        
        # Create polygon bridge instance
        self.polygon_bridge = PolygonHVNBridge(
            cache_enabled=True
        )
        
        # Initialize engine with polygon data fetcher
        self.engine = CamarillaEngine(
            data_fetcher=self.polygon_bridge.fetcher,  # Pass the polygon fetcher
            range_threshold_pct=0.3,
            switching_tolerance=0.1,
            calc_mode='auto',
            use_close='auto',
            use_current_pre=False
        )
        
    def run(self):
        try:
            self.progress.emit(f"Fetching data for {self.symbol}...")
            
            # Get current price from polygon
            try:
                # Use polygon bridge to get latest price
                latest_state = self.polygon_bridge.calculate_hvn(self.symbol, end_date=datetime.now())
                current = latest_state.current_price
            except:
                # Fallback to yfinance for current price only
                ticker = yf.Ticker(self.symbol)
                info = ticker.info
                current = info.get('regularMarketPrice', info.get('currentPrice', 0))
                
                if not current:
                    recent_data = ticker.history(period='1d')
                    if not recent_data.empty:
                        current = float(recent_data['Close'].iloc[-1])
                    else:
                        current = 0.0
                    
            self.current_price.emit(float(current))
            
            self.progress.emit("Calculating Camarilla pivots...")
            
            # Convert end_datetime from ET to UTC for the engine
            et_tz = pytz.timezone('US/Eastern')
            utc_tz = pytz.UTC
            
            # If end_datetime is naive, assume it's ET
            if self.end_datetime.tzinfo is None:
                end_datetime_et = et_tz.localize(self.end_datetime)
            else:
                end_datetime_et = self.end_datetime
                
            # Convert to UTC
            end_datetime_utc = end_datetime_et.astimezone(utc_tz)
            
            print(f"Converting time: {self.end_datetime} ET -> {end_datetime_utc} UTC")
            
            # Use the new Pine Script exact method
            results = {}
            
            # Daily with Pine Script exact logic
            try:
                daily_result = self.engine.analyze_daily_pine_script(
                    self.symbol, 
                    end_datetime_utc  # Pass UTC time
                )
                results['daily'] = daily_result
                
                # Debug output
                if daily_result.debug_info:
                    print(f"\nDEBUG - Daily Pivots Calculation:")
                    print(f"  Data Type Used: {daily_result.data_type}")
                    print(f"  High: ${daily_result.high:.2f}")
                    print(f"  Low: ${daily_result.low:.2f}")
                    print(f"  Close: ${daily_result.close:.2f}")
                    
                    # Find R6 pivot
                    r6_pivots = [p for p in daily_result.pivots if p.level_name == 'R6']
                    if r6_pivots:
                        print(f"  R6: ${r6_pivots[0].price:.2f}")
                
            except Exception as e:
                self.error.emit(f"Error calculating daily pivots: {str(e)}")
                return
            
            # For weekly/monthly, we still need to implement polygon-based fetching
            self.progress.emit("Calculating weekly and monthly pivots...")
            
            # Get daily data from polygon for weekly/monthly
            try:
                # Fetch 45 days of daily data
                start_date = end_datetime_utc - timedelta(days=45)
                
                daily_data = self.polygon_bridge.fetcher.fetch_data(
                    symbol=self.symbol,
                    timeframe='day',
                    start_date=start_date,
                    end_date=end_datetime_utc,
                    use_cache=True,
                    validate=True
                )
                
                if not daily_data.empty:
                    # Weekly
                    try:
                        weekly_result = self.engine.analyze_timeframe(daily_data, 'weekly')
                        results['weekly'] = weekly_result
                    except Exception as e:
                        self.error.emit(f"Error calculating weekly pivots: {str(e)}")
                    
                    # Monthly
                    try:
                        monthly_result = self.engine.analyze_timeframe(daily_data, 'monthly')
                        results['monthly'] = monthly_result
                    except Exception as e:
                        self.error.emit(f"Error calculating monthly pivots: {str(e)}")
                        
            except Exception as e:
                self.error.emit(f"Error fetching daily data for weekly/monthly: {str(e)}")
            
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(f"Error: {str(e)}\n{traceback.format_exc()}")


class PivotTable(QWidget):
    """Widget for displaying pivots for a single timeframe"""
    
    def __init__(self, timeframe: str):
        super().__init__()
        self.timeframe = timeframe
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Group box with title
        title = self.timeframe.capitalize()
        group = QGroupBox(f"{title} Pivots")
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
        self.table.setHorizontalHeaderLabels(["Level", "Price", "Type"])
        
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
        self.table.setColumnWidth(0, 60)
        self.table.setColumnWidth(2, 100)
        
        group_layout.addWidget(self.table)
        group.setLayout(group_layout)
        
        layout.addWidget(group)
        self.setLayout(layout)
        
    def update_data(self, result: CamarillaResult, current_price: float):
        """Update table with Camarilla results"""
        if not result:
            self.clear_data()
            return
            
        # Update info label
        info_text = f"H: ${result.high:.2f} L: ${result.low:.2f} C: ${result.close:.2f} | "
        info_text += f"Range: {result.range_type.capitalize()} | CP: ${result.central_pivot:.2f}"
        
        # Add data type indicator for daily timeframe
        if self.timeframe == 'daily' and hasattr(result, 'data_type'):
            info_text += f" | Data: {result.data_type}"
        
        self.info_label.setText(info_text)
        self.info_label.setStyleSheet("color: #ffffff; padding: 5px; font-size: 11px;")
        
        # Prepare pivot data for table
        pivot_data = []
        
        # Add central pivot
        pivot_data.append(('CP', result.central_pivot, 'Central', None))
        
        # Add all pivots
        for pivot in result.pivots:
            level_type = 'Resistance' if pivot.level_name.startswith('R') else 'Support'
            pivot_data.append((pivot.level_name, pivot.price, level_type, pivot.strength))
        
        # Sort by price descending
        pivot_data.sort(key=lambda x: x[1], reverse=True)
        
        # Update table
        self.table.setRowCount(len(pivot_data))
        
        for row, (level, price, level_type, strength) in enumerate(pivot_data):
            # Level name
            level_item = QTableWidgetItem(level)
            level_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # Color code based on type
            if level == 'CP':
                level_item.setForeground(QColor('#87CEEB'))  # Light blue for CP
            elif level_type == 'Resistance':
                level_item.setForeground(QColor('#FF6B6B'))  # Red for resistance
            else:
                level_item.setForeground(QColor('#4ECDC4'))  # Green for support
                
            self.table.setItem(row, 0, level_item)
            
            # Price
            price_item = QTableWidgetItem(f"${price:.2f}")
            price_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            
            # Highlight if price is near current
            if current_price > 0:
                distance_pct = abs(price - current_price) / current_price * 100
                if distance_pct <= 0.5:
                    price_item.setBackground(QColor('#4a3a6a'))
                    font = QFont()
                    font.setBold(True)
                    price_item.setFont(font)
            
            self.table.setItem(row, 1, price_item)
            
            # Type
            type_item = QTableWidgetItem(level_type)
            type_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # Add strength indicator for non-CP levels
            if strength:
                type_text = f"{level_type}"
                if strength >= 6:
                    type_text += " ⚡"
                elif strength >= 4:
                    type_text += " •"
                type_item.setText(type_text)
                
            self.table.setItem(row, 2, type_item)
            
        self.table.resizeRowsToContents()
        
    def clear_data(self):
        """Clear the table"""
        self.table.setRowCount(0)
        self.info_label.setText("No data loaded")
        self.info_label.setStyleSheet("color: #888888; padding: 5px; font-size: 11px;")


class CamarillaConfluenceSection(QWidget):
    """Widget for displaying Camarilla confluence analysis"""
    
    def __init__(self):
        super().__init__()
        self.current_price = 0.0
        self.confluence_calculator = CamarillaConfluenceCalculator(
            confluence_threshold_percent=0.3,
            min_pivots_for_zone=2,
            max_pivots_per_timeframe=6
        )
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Group box
        group = QGroupBox("Camarilla Confluence Analysis - Key Support/Resistance Zones")
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
        
        # Current price and nearest levels
        self.summary_label = QLabel("Current Price: --")
        self.summary_label.setStyleSheet("""
            QLabel {
                color: #7B68EE;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                background-color: #2b2b2b;
                border-radius: 5px;
                margin-bottom: 10px;
            }
        """)
        group_layout.addWidget(self.summary_label)
        
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
        
    def update_analysis(self, results: Dict[str, CamarillaResult], current_price: float):
        """Update confluence analysis using the calculator"""
        self.current_price = current_price
        
        # Check if we have valid results
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            self.summary_label.setText(f"Current Price: ${current_price:.2f} | No valid pivot data")
            self.results_text.setHtml("<p style='color: #888888;'>No valid pivot data available</p>")
            return
        
        # Calculate confluence
        analysis = self.confluence_calculator.calculate(
            results=valid_results,
            current_price=current_price,
            max_zones=10
        )
        
        # Update summary
        summary = f"Current Price: ${current_price:.2f}"
        if analysis.nearest_resistance:
            summary += f" | Next R: ${analysis.nearest_resistance.center_price:.2f}"
        if analysis.nearest_support:
            summary += f" | Next S: ${analysis.nearest_support.center_price:.2f}"
        
        self.summary_label.setText(summary)
        
        # Display the analysis
        self.display_analysis(analysis)
        
    def display_analysis(self, analysis: CamarillaConfluenceAnalysis):
        """Display the confluence analysis results"""
        if not analysis.zones:
            self.results_text.setHtml("<p style='color: #888888;'>No confluence zones found</p>")
            return
            
        html = ""
        
        # Add summary if price is in a zone
        if analysis.price_in_zone:
            zone_type = analysis.price_in_zone.zone_type
            color = '#FF6B6B' if 'resistance' in zone_type else '#4ECDC4'
            html += f"""
            <div style='margin-bottom: 15px; padding: 10px; background-color: #4a3a6a; border-radius: 5px;'>
                <p style='color: {color}; margin: 0; font-weight: bold;'>
                    ⚡ Price is currently testing {zone_type.upper()} Zone #{analysis.price_in_zone.zone_id}
                </p>
            </div>
            """
        
        # Separate resistance and support zones
        resistance_zones = [z for z in analysis.zones if z.is_resistance()]
        support_zones = [z for z in analysis.zones if z.is_support()]
        
        # Display resistance zones
        if resistance_zones:
            html += "<h3 style='color: #FF6B6B; margin: 15px 0 10px 0;'>🔻 Resistance Zones (Above)</h3>"
            for zone in resistance_zones[:3]:  # Top 3 resistance zones
                html += self._format_zone_html(zone)
        
        # Display support zones
        if support_zones:
            html += "<h3 style='color: #4ECDC4; margin: 20px 0 10px 0;'>🔺 Support Zones (Below)</h3>"
            for zone in support_zones[:3]:  # Top 3 support zones
                html += self._format_zone_html(zone)
        
        # Add analysis footer
        html += f"""
        <div style='margin-top: 15px; padding: 10px; background-color: #2a2a2a; border-radius: 5px; font-size: 11px; color: #888888;'>
            <p style='margin: 0;'>
                Analysis Time: {analysis.analysis_time.strftime('%Y-%m-%d %H:%M:%S')} | 
                Total Zones: {analysis.total_zones_found} | 
                R Zones: {len(analysis.resistance_zones)} | 
                S Zones: {len(analysis.support_zones)}
            </p>
        </div>
        """
            
        self.results_text.setHtml(html)
        
    def _format_zone_html(self, zone: CamarillaConfluenceZone) -> str:
        """Format a single zone as HTML"""
        # Determine strength emoji and color
        strength_display = {
            'Strong': ('💪', '#50C878'),
            'Moderate': ('✓', '#FFD700'),
            'Weak': ('○', '#87CEEB')
        }
        emoji, color = strength_display.get(zone.strength_classification, ('○', '#87CEEB'))
        
        # Zone type color
        if zone.zone_type == 'resistance':
            type_color = '#FF6B6B'
        elif zone.zone_type == 'support':
            type_color = '#4ECDC4'
        else:
            type_color = '#9370DB'
        
        # Distance styling
        distance_color = '#50C878' if zone.distance_percentage < 1.0 else '#FFA500'
        
        zone_html = f"""
        <div style='margin-bottom: 15px; padding: 12px; background-color: #333333; border-radius: 8px; border-left: 4px solid {type_color};'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <h4 style='color: #9370DB; margin: 0;'>
                    Zone #{zone.zone_id} {emoji} 
                    <span style='color: {color}; font-size: 13px;'>{zone.strength_classification}</span>
                </h4>
                <span style='color: {distance_color}; font-size: 12px;'>
                    {zone.distance_percentage:.2f}% away
                </span>
            </div>
            <table style='width: 100%; margin-top: 8px; font-size: 12px;'>
                <tr>
                    <td style='color: #888888; padding: 2px;'>Range:</td>
                    <td style='color: #ffffff;'>${zone.zone_low:.2f} - ${zone.zone_high:.2f}</td>
                    <td style='color: #888888; padding: 2px;'>Width:</td>
                    <td style='color: #ffffff;'>${zone.zone_width:.2f}</td>
                </tr>
                <tr>
                    <td style='color: #888888; padding: 2px;'>Center:</td>
                    <td style='color: {type_color}; font-weight: bold;'>${zone.center_price:.2f}</td>
                    <td style='color: #888888; padding: 2px;'>Levels:</td>
                    <td style='color: #ffffff;'>{', '.join(sorted(zone.level_names))}</td>
                </tr>
            </table>
            <div style='margin-top: 6px; font-size: 11px; color: #bbbbbb;'>
                Timeframes: {', '.join(zone.timeframes)} | Strength: {zone.total_strength:.0f}
            </div>
        </div>
        """
        
        return zone_html


class CamarillaPivotAnalyzer(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.calculation_thread = None
        self.current_price = 0.0
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Camarilla Pivot Analyzer - Pine Script Compatible")
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
        input_group_layout.addWidget(QLabel("End Date:"))
        self.date_input = QDateEdit()
        self.date_input.setCalendarPopup(True)
        self.date_input.setDate(QDate.currentDate())
        self.date_input.setDisplayFormat("yyyy-MM-dd")
        input_group_layout.addWidget(self.date_input)
        
        input_group_layout.addSpacing(20)
        
        # Quick date buttons
        self.today_button = QPushButton("Today")
        self.today_button.clicked.connect(lambda: self.date_input.setDate(QDate.currentDate()))
        self.today_button.setStyleSheet("""
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
        input_group_layout.addWidget(self.today_button)
        
        self.yesterday_button = QPushButton("Yesterday")
        self.yesterday_button.clicked.connect(
            lambda: self.date_input.setDate(QDate.currentDate().addDays(-1))
        )
        self.yesterday_button.setStyleSheet("""
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
        input_group_layout.addWidget(self.yesterday_button)
        
        input_group_layout.addStretch()
        
        # Run button
        self.run_button = QPushButton("Calculate Pivots")
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
        self.table_daily = PivotTable('daily')
        self.table_weekly = PivotTable('weekly')
        self.table_monthly = PivotTable('monthly')
        
        tables_layout.addWidget(self.table_daily)
        tables_layout.addWidget(self.table_weekly)
        tables_layout.addWidget(self.table_monthly)
        
        # Confluence section (50%)
        self.confluence_section = CamarillaConfluenceSection()
        
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
        
    def run_analysis(self):
        """Run the Camarilla analysis"""
        # Validate inputs
        symbol = self.symbol_input.text().strip().upper()
        if not symbol:
            QMessageBox.warning(self, "Input Error", "Please enter a symbol")
            return
            
        # Get date
        date = self.date_input.date().toPyDate()
        end_datetime = datetime.combine(date, time(23, 59, 59))
        
        # Check if the selected date is in the future
        if date > datetime.now().date():
            QMessageBox.warning(
                self, 
                "Invalid Date", 
                f"Selected date is in the future!\n\n"
                f"Please select today's date or earlier."
            )
            return
        
        # Clear previous results
        self.table_daily.clear_data()
        self.table_weekly.clear_data()
        self.table_monthly.clear_data()
        self.confluence_section.results_text.clear()
        
        # Disable run button
        self.run_button.setEnabled(False)
        self.setWindowTitle(f"Camarilla Pivot Analyzer - Calculating pivots for {symbol}...")
        
        # Start calculation thread
        self.calculation_thread = CamarillaCalculationThread(symbol, end_datetime)
        self.calculation_thread.finished.connect(self.on_calculation_finished)
        self.calculation_thread.error.connect(self.on_calculation_error)
        self.calculation_thread.progress.connect(self.on_progress_update)
        self.calculation_thread.current_price.connect(self.on_current_price)
        self.calculation_thread.start()
        
    def on_progress_update(self, message: str):
        """Update status with progress in title bar"""
        self.setWindowTitle(f"Camarilla Pivot Analyzer - {message}")
        
    def on_current_price(self, price: float):
        """Store current price for confluence calculation"""
        self.current_price = price
        
    def on_calculation_finished(self, results: Dict[str, CamarillaResult]):
        """Handle calculation completion"""
        # Check if we have valid results
        if not results:
            self.on_calculation_error("No results returned from calculation")
            return
            
        # Update tables with None checks
        if 'daily' in results and results['daily']:
            self.table_daily.update_data(results['daily'], self.current_price)
        else:
            self.table_daily.clear_data()
            
        if 'weekly' in results and results['weekly']:
            self.table_weekly.update_data(results['weekly'], self.current_price)
        else:
            self.table_weekly.clear_data()
            
        if 'monthly' in results and results['monthly']:
            self.table_monthly.update_data(results['monthly'], self.current_price)
        else:
            self.table_monthly.clear_data()
            
        # Update confluence analysis
        self.confluence_section.update_analysis(results, self.current_price)
            
        # Reset window title
        self.setWindowTitle("Camarilla Pivot Analyzer - Pine Script Compatible")
        
        # Re-enable run button
        self.run_button.setEnabled(True)
        
    def on_calculation_error(self, error_msg: str):
        """Handle calculation error"""
        QMessageBox.critical(self, "Analysis Error", error_msg)
        self.setWindowTitle("Camarilla Pivot Analyzer - Error occurred")
        self.run_button.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = CamarillaPivotAnalyzer()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()